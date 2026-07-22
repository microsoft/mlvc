# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn
import torch.nn.functional as F

from .block_mc import block_mc_func
from ..utils.model_compat_support import RenameModule_LoadHook


def bilinearupscaling(inputfeature):
    outfeature = F.interpolate(inputfeature, scale_factor=(2, 2), mode="bilinear", align_corners=False)

    return outfeature


class MEBasic(nn.Module):
    def __init__(self, complexity_level=0, in_ch=8):
        super().__init__()
        self.relu = nn.ReLU()
        self.by_pass = False
        if complexity_level < 0:
            self.by_pass = True
        elif complexity_level == 0:
            self.conv1 = nn.Conv2d(in_ch, 32, 7, 1, padding=3)
            self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
            self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
            self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
            self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)
        elif complexity_level == 1:
            self.conv1 = nn.Conv2d(in_ch, 32, 7, 1, padding=3)
            self.conv2 = nn.Conv2d(32, 32, 7, 1, padding=3)
            self.conv3 = nn.Conv2d(32, 32, 7, 1, padding=3)
            self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
            self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)
        elif complexity_level == 2:
            self.conv1 = nn.Conv2d(in_ch, 16, 7, 1, padding=3)
            self.conv2 = nn.Conv2d(16, 32, 7, 1, padding=3)
            self.conv3 = nn.Conv2d(32, 16, 7, 1, padding=3)
            self.conv4 = nn.Conv2d(16, 8, 7, 1, padding=3)
            self.conv5 = nn.Conv2d(8, 2, 7, 1, padding=3)
        elif complexity_level == 3:
            self.conv1 = nn.Conv2d(in_ch, 32, 5, 1, padding=2)
            self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
            self.conv3 = nn.Conv2d(64, 32, 5, 1, padding=2)
            self.conv4 = nn.Conv2d(32, 16, 5, 1, padding=2)
            self.conv5 = nn.Conv2d(16, 2, 5, 1, padding=2)

    def forward(self, x):
        if self.by_pass:
            return x[:, -2:, :, :]

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class ME_Spynet(nn.Module):
    def __init__(self, com_1x=-1, com_2x=2, com_4x=1, com_8x=0, out_2x=True):
        super().__init__()
        self.me_8x = MEBasic(com_8x)
        self.me_4x = MEBasic(com_4x)
        self.me_2x = MEBasic(com_2x)
        self.me_1x = MEBasic(com_1x)
        self.out_2x = out_2x

        # Support module list naming scheme
        RenameModule_LoadHook(
            {
                "moduleBasic.0": "me_8x",
            }
        ).register(self)

    def forward(self, im1, im2):
        batchsize = im1.size()[0]

        im1_1x = im1
        im1_2x = F.avg_pool2d(im1_1x, kernel_size=2, stride=2)
        im1_4x = F.avg_pool2d(im1_2x, kernel_size=2, stride=2)
        im1_8x = F.avg_pool2d(im1_4x, kernel_size=2, stride=2)
        im2_1x = im2
        im2_2x = F.avg_pool2d(im2_1x, kernel_size=2, stride=2)
        im2_4x = F.avg_pool2d(im2_2x, kernel_size=2, stride=2)
        im2_8x = F.avg_pool2d(im2_4x, kernel_size=2, stride=2)

        shape_fine = im1_8x.size()
        zero_shape = [batchsize, 2, shape_fine[2], shape_fine[3]]
        flow_8x = torch.zeros(zero_shape, dtype=im1.dtype, device=im1.device)
        flow_8x = self.me_8x(torch.cat((im1_8x, im2_8x, flow_8x), dim=1))

        flow_4x = bilinearupscaling(flow_8x) * 2.0
        flow_4x = flow_4x + self.me_4x(
            torch.cat((im1_4x, block_mc_func(im2_4x, flow_4x, self.training), flow_4x), dim=1)
        )

        flow_2x = bilinearupscaling(flow_4x) * 2.0
        flow_2x = flow_2x + self.me_2x(
            torch.cat((im1_2x, block_mc_func(im2_2x, flow_2x, self.training), flow_2x), dim=1)
        )

        if self.out_2x:
            return flow_2x

        flow_1x = bilinearupscaling(flow_2x) * 2.0
        flow_1x = flow_1x + self.me_1x(
            torch.cat((im1_1x, block_mc_func(im2_1x, flow_1x, self.training), flow_1x), dim=1)
        )
        return flow_1x
