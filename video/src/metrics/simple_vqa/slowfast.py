# Copyright 2019, Facebook, Inc.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Adapted from PyTorchVideo 0.1.5 (Apache-2.0); see LICENSE.pytorchvideo.
# Modified to retain only the fixed SlowFast-R50 (8x8) inference architecture.

import torch
from torch import nn


class _Stem(nn.Module):
    def __init__(self, channels: int, temporal_kernel: int):
        super().__init__()
        self.conv = nn.Conv3d(3, channels, (temporal_kernel, 7, 7), (1, 2, 2), (temporal_kernel // 2, 3, 3), bias=False)
        self.norm = nn.BatchNorm3d(channels)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.activation(self.norm(self.conv(x))))


class _Bottleneck(nn.Module):
    def __init__(self, dim_in: int, dim_inner: int, dim_out: int, temporal_kernel: int, stride: int):
        super().__init__()
        self.conv_a = nn.Conv3d(
            dim_in, dim_inner, (temporal_kernel, 1, 1), padding=(temporal_kernel // 2, 0, 0), bias=False
        )
        self.norm_a = nn.BatchNorm3d(dim_inner)
        self.act_a = nn.ReLU()
        self.conv_b = nn.Conv3d(dim_inner, dim_inner, (1, 3, 3), (1, stride, stride), (0, 1, 1), bias=False)
        self.norm_b = nn.BatchNorm3d(dim_inner)
        self.act_b = nn.ReLU()
        self.conv_c = nn.Conv3d(dim_inner, dim_out, 1, bias=False)
        self.norm_c = nn.BatchNorm3d(dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_a(self.norm_a(self.conv_a(x)))
        x = self.act_b(self.norm_b(self.conv_b(x)))
        return self.norm_c(self.conv_c(x))


class _ResBlock(nn.Module):
    def __init__(self, dim_in: int, dim_inner: int, dim_out: int, temporal_kernel: int, stride: int):
        super().__init__()
        projection = dim_in != dim_out or stride != 1
        self.branch1_conv = nn.Conv3d(dim_in, dim_out, 1, (1, stride, stride), bias=False) if projection else None
        self.branch1_norm = nn.BatchNorm3d(dim_out) if projection else None
        self.branch2 = _Bottleneck(dim_in, dim_inner, dim_out, temporal_kernel, stride)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        if self.branch1_conv is not None:
            shortcut = self.branch1_conv(shortcut)
        if self.branch1_norm is not None:
            shortcut = self.branch1_norm(shortcut)
        return self.activation(shortcut + self.branch2(x))


class _ResStage(nn.Module):
    def __init__(self, depth: int, dim_in: int, dim_inner: int, dim_out: int, temporal_kernel: int, stride: int):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                _ResBlock(
                    dim_in if index == 0 else dim_out,
                    dim_inner,
                    dim_out,
                    temporal_kernel,
                    stride if index == 0 else 1,
                )
                for index in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x)
        return x


class _FuseFastToSlow(nn.Module):
    def __init__(self, dim_in: int):
        super().__init__()
        self.conv_fast_to_slow = nn.Conv3d(dim_in, dim_in * 2, (7, 1, 1), (4, 1, 1), (3, 0, 0), bias=False)
        self.norm = nn.BatchNorm3d(dim_in * 2)
        self.activation = nn.ReLU()

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        fuse = self.activation(self.norm(self.conv_fast_to_slow(x[1])))
        return [torch.cat([x[0], fuse], 1), x[1]]


class _MultiPathway(nn.Module):
    def __init__(self, slow: nn.Module, fast: nn.Module, fusion: nn.Module):
        super().__init__()
        self.multipathway_blocks = nn.ModuleList([slow, fast])
        self.multipathway_fusion = fusion

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        for index, block in enumerate(self.multipathway_blocks):
            x[index] = block(x[index])
        return self.multipathway_fusion(x)


class _PoolConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.ModuleList([nn.AvgPool3d((8, 7, 7), 1), nn.AvgPool3d((32, 7, 7), 1)])

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        for index, pool in enumerate(self.pool):
            x[index] = pool(x[index])
        return torch.cat(x, 1)


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.proj = nn.Linear(2304, 400)
        self.output_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x).permute(0, 2, 3, 4, 1)
        x = self.proj(x).permute(0, 4, 1, 2, 3)
        x = self.output_pool(x)
        return x.view(x.shape[0], -1)


class SlowFast(nn.Module):
    def __init__(self):
        super().__init__()
        blocks: list[nn.Module] = [_MultiPathway(_Stem(64, 1), _Stem(8, 5), _FuseFastToSlow(8))]
        dim_in = 64
        for index, depth in enumerate((3, 4, 6, 3)):
            dim_out = 256 * 2**index
            stride = 1 if index == 0 else 2
            slow = _ResStage(depth, dim_in + dim_in // 4, dim_out // 4, dim_out, 1 if index < 2 else 3, stride)
            fast = _ResStage(depth, dim_in // 8, dim_out // 32, dim_out // 8, 3, stride)
            fusion = _FuseFastToSlow(dim_out // 8) if index < 3 else nn.Identity()
            blocks.append(_MultiPathway(slow, fast, fusion))
            dim_in = dim_out
        # Retain the unused classifier so the official checkpoint loads strictly.
        blocks.extend([_PoolConcat(), _Head()])
        self.blocks = nn.ModuleList(blocks)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, _Bottleneck) and module.norm_c.weight is not None:
                nn.init.zeros_(module.norm_c.weight)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        for block in self.blocks[:5]:
            x = block(x)
        return self.blocks[6](self.blocks[5](x))


def slowfast_r50() -> SlowFast:
    return SlowFast()
