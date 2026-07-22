# Copyright (c) Richard Zhang, Phillip Isola, Alexei A. Efros,
# Eli Shechtman, Oliver Wang and other contributors.
# Origin: https://github.com/richzhang/PerceptualSimilarity
#
# Licensed under the BSD 2-Clause "Simplified" License.
# See https://github.com/richzhang/PerceptualSimilarity/blob/master/LICENSE
#
# Modifications Copyright (c) Microsoft Corporation

import logging
from collections import namedtuple
from typing import cast
import torch
from torchvision import models as tv  # type: ignore[reportMissingImports]


class vgg16(torch.nn.Module):
    slice1: torch.nn.Sequential
    slice2: torch.nn.Sequential
    slice3: torch.nn.Sequential
    slice4: torch.nn.Sequential
    slice5: torch.nn.Sequential

    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()

        weights = None
        if pretrained is True:
            from torchvision.models.vgg import VGG16_Weights  # type: ignore[reportMissingImports]

            weights = VGG16_Weights.DEFAULT

        self.feature_resolutions = [1.0, 0.5, 0.25, 0.125, 0.0625]
        self.feature_channels = [64, 128, 256, 512, 512]

        logging.info(f"Using VGG16 model with weights: {weights}")

        vgg_pretrained_features = cast(torch.nn.Sequential, tv.vgg16(weights=weights).features)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out
