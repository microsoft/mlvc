# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn


class MS_SSIM(nn.Module):
    _FILTER_SIZE = 11
    _FILTER_SIGMA = 1.5
    _filter_w: torch.Tensor
    _filter_h: torch.Tensor
    _C: torch.Tensor
    _weight5: torch.Tensor
    _weight4: torch.Tensor

    def __init__(self, *, channels=1, data_range=1):
        super().__init__()

        self.channels = channels

        filter1d = self._gaussian_1d(self._FILTER_SIZE, self._FILTER_SIGMA)[None, None, ...]
        if channels > 1:
            filter1d = filter1d.tile((channels, 1, 1))
        self.register_buffer("_filter_w", filter1d[..., None, :], False)
        self.register_buffer("_filter_h", filter1d[..., :, None], False)

        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        self.register_buffer("_C", torch.asarray((C1, C2), dtype=torch.float))

        # scale weigths
        self.register_buffer("_weight5", torch.asarray((0.0448, 0.2856, 0.3001, 0.2363, 0.1333), dtype=torch.float))
        # scale weights for small images according to HM implementation
        self.register_buffer("_weight4", torch.asarray((0.0517, 0.3295, 0.3462, 0.2726), dtype=torch.float))

    @staticmethod
    def _gaussian_1d(size, sigma):
        x = torch.arange(size, dtype=torch.float) - (size - 1) / 2
        x = torch.exp(-torch.square(x) / (2 * sigma * sigma))
        x *= 1 / x.sum()
        return x

    def _apply_filter(self, x):
        x = nn.functional.conv2d(x, self._filter_w, groups=self.channels)
        x = nn.functional.conv2d(x, self._filter_h, groups=self.channels)
        return x

    def _ssim(self, x1, x2, *, calc_ssim):
        mu1 = self._apply_filter(x1)
        mu2 = self._apply_filter(x2)
        mu1_sq = torch.square(mu1)
        mu2_sq = torch.square(mu2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self._apply_filter(torch.square(x1)) - mu1_sq
        sigma2_sq = self._apply_filter(torch.square(x2)) - mu2_sq
        sigma12 = self._apply_filter(x1 * x2) - mu1_mu2

        C2 = self._C[1]
        cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

        if not calc_ssim:
            return cs

        C1 = self._C[0]
        ssim = cs * (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        return ssim

    @staticmethod
    def _downsample(x: torch.Tensor):
        h, w = x.shape[-2:]
        if (w & 1) != 0:
            x = torch.cat((x, x[..., -1:]), dim=-1)
        if (h & 1) != 0:
            x = torch.cat((x, x[..., -1:, :]), dim=-2)
        return nn.functional.avg_pool2d(x, 2)

    @staticmethod
    def _mean(x):
        return x.flatten(-2).mean(dim=-1)

    def forward(self, x1, x2):
        assert x1.shape == x2.shape
        w, h = x1.shape[2:]

        if w >= 16 * self._FILTER_SIZE and h >= 16 * self._FILTER_SIZE:
            weight = self._weight5
        else:
            assert w >= 8 * self._FILTER_SIZE and h >= 8 * self._FILTER_SIZE
            weight = self._weight4

        level = weight.shape[0]
        value_list = list()
        for i in range(level - 1):
            cs = self._ssim(x1, x2, calc_ssim=False)
            value_list.append(self._mean(cs))

            x1 = self._downsample(x1)
            x2 = self._downsample(x2)

        ssim = self._ssim(x1, x2, calc_ssim=True)
        value_list.append(self._mean(ssim))

        value = torch.stack(value_list, dim=-1)
        value = weight * value.clamp_min(1.0e-12).log()
        msssim = torch.exp(torch.sum(value, dim=-1))
        return msssim.mean(dim=-1)
