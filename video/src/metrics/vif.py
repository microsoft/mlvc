# Copyright (c) Photosynthesis Team
# Modifications Copyright (c) Microsoft Corporation
#
# This file is adapted from the PIQ library:
# https://github.com/photosynthesis-team/piq/blob/master/piq/vif.py
# Licensed under the Apache License, Version 2.0.
#
# Original VIF reference implementation in MATLAB:
# https://live.ece.utexas.edu/research/Quality/VIF.htm
#
# Modifications:
# - Converted functional API to nn.Module class
# - Added per-scale outputs
# - Made number of scales configurable

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_filter(
    kernel_size: int, sigma: float, device: Optional[str] = None, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=dtype, device=device)
    coords -= (kernel_size - 1) / 2.0

    g = coords**2
    g = (-(g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma**2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)


class VIF(nn.Module):
    EPSILON = 1e-8

    def __init__(self, data_range: Union[int, float] = 1.0, sigma_n_sq: float = 2.0, num_scales: int = 2):
        super().__init__()

        self.data_range = data_range
        self.sigma_n_sq = sigma_n_sq
        self.num_scales = num_scales
        self.eps = self.EPSILON
        self._init_filters()

    def _init_filters(self):
        for scale in range(self.num_scales):
            kernel_size = 2 ** (4 - scale) + 1
            kernel = gaussian_filter(kernel_size, sigma=kernel_size / 5, dtype=torch.float32)
            kernel = kernel.view(1, 1, kernel_size, kernel_size)
            self.register_buffer(f"_kernel_{scale}", kernel, False)

    def _get_luminance(self, x: torch.Tensor) -> torch.Tensor:
        num_channels = x.shape[1]
        if num_channels == 1:
            return x
        else:
            # reference VIF is based on ITU-R BT.601 and not BT.709
            # however, both options correlate similarly with human ratings
            return 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        x = pred / float(self.data_range) * 255
        y = target / float(self.data_range) * 255

        x = self._get_luminance(x)
        y = self._get_luminance(y)

        x_vif_total, y_vif_total = 0, 0
        per_scale_x_sums, per_scale_y_sums = [], []
        for scale in range(self.num_scales):
            kernel = getattr(self, f"_kernel_{scale}")
            if scale > 0:
                x = F.conv2d(x, kernel)[:, :, ::2, ::2]
                y = F.conv2d(y, kernel)[:, :, ::2, ::2]

            mu_x, mu_y = F.conv2d(x, kernel), F.conv2d(y, kernel)
            mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

            sigma_x_sq = F.conv2d(x**2, kernel) - mu_x_sq
            sigma_y_sq = F.conv2d(y**2, kernel) - mu_y_sq
            sigma_xy = F.conv2d(x * y, kernel) - mu_xy

            sigma_x_sq = torch.relu(sigma_x_sq)
            sigma_y_sq = torch.relu(sigma_y_sq)

            g = sigma_xy / (sigma_y_sq + self.eps)
            sigma_v_sq = sigma_x_sq - g * sigma_xy

            mask = sigma_y_sq >= self.eps
            g = torch.where(mask, g, torch.zeros_like(g))
            sigma_v_sq = torch.where(mask, sigma_v_sq, sigma_x_sq)
            sigma_y_sq = torch.where(mask, sigma_y_sq, torch.zeros_like(sigma_y_sq))

            mask = sigma_x_sq >= self.eps
            g = torch.where(mask, g, torch.zeros_like(g))
            sigma_v_sq = torch.where(mask, sigma_v_sq, torch.zeros_like(sigma_v_sq))

            sigma_v_sq = torch.where(g >= 0, sigma_v_sq, sigma_x_sq)
            g = torch.relu(g)

            sigma_v_sq = torch.where(sigma_v_sq > self.eps, sigma_v_sq, torch.ones_like(sigma_v_sq) * self.eps)

            x_term = torch.log10(1.0 + (g**2.0) * sigma_y_sq / (sigma_v_sq + self.sigma_n_sq))
            y_term = torch.log10(1.0 + sigma_y_sq / self.sigma_n_sq)

            # Sum over channels and spatial dimensions
            x_sum = torch.sum(x_term, dim=[1, 2, 3])
            y_sum = torch.sum(y_term, dim=[1, 2, 3])
            per_scale_x_sums.append(x_sum)
            per_scale_y_sums.append(y_sum)

            x_vif_total = x_vif_total + x_sum
            y_vif_total = y_vif_total + y_sum

        overall_score = x_vif_total / (y_vif_total + self.eps)
        per_scale_scores = [xs / (ys + self.eps) for xs, ys in zip(per_scale_x_sums, per_scale_y_sums)]

        return overall_score, per_scale_scores
