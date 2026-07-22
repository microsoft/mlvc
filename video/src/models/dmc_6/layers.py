# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn
import torch.nn.functional as F
from enum import Enum
from ..utils import CkptModule


class NetworkMode(str, Enum):
    FP32 = "fp32"


class DMCSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, *, q_index=None):
        for module in self:
            input = module(input, q_index=q_index)
        return input


class DMCCommonParams:
    def __init__(
        self,
        *args,
        qp_num: int = 64,
        network_mode: NetworkMode = NetworkMode.FP32,
        override_fake_quant_q_index=None,
        conv_cls=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.qp_num = qp_num
        self.network_mode = network_mode
        self.override_fake_quant_q_index = override_fake_quant_q_index
        self.conv_cls = conv_cls


class DMCCkptModule(DMCCommonParams, CkptModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DMCModule(DMCCommonParams, nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DMCConv2d(DMCCommonParams, nn.Conv2d):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, x, q_index=None):
        return super().forward(x)


class Activation(DMCModule):
    def __init__(self, *, activation: str, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation

    def forward(self, x, q_index=None):
        if self.activation == "WSiLU":
            return torch.sigmoid(4.0 * x) * x
        elif self.activation == "SiLU":
            return F.silu(x)
        elif self.activation == "ReLU":
            return F.relu(x)
        elif self.activation == "LeakyReLU":
            return F.leaky_relu(x)
        elif self.activation == "ReLU1":
            return torch.clamp(x, min=0.0, max=1.0)
        elif self.activation == "sigmoid":
            return F.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")


class ChunkMode(str, Enum):
    SPLIT = "split"
    INTERLEAVE = "interleave"
    GATED = "gated"  # GLU-type activation


class ActivationChunkAdd(DMCModule):
    def __init__(
        self,
        activation: str,
        num_chunks: int,
        chunk_mode: ChunkMode = ChunkMode.SPLIT,
        gate_activation: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.act = Activation(activation=activation, **kwargs)
        self.num_chunks = num_chunks
        self.chunk_mode = chunk_mode
        self.gate_activation = (
            Activation(activation=gate_activation, **kwargs)
            if gate_activation is not None and chunk_mode == ChunkMode.GATED
            else None
        )

    def forward(self, x, q_index=None):
        if self.chunk_mode == ChunkMode.GATED:
            assert self.num_chunks == 2, "Gated activation only supports num_chunks=2"

            gates, values = x.chunk(2, dim=1)
            assert self.gate_activation is not None
            gates = self.gate_activation(gates)
            gated = gates * values

            return gated

        out = self.act(x, q_index=q_index)

        if self.chunk_mode == ChunkMode.INTERLEAVE:
            result = out[:, 0 :: self.num_chunks, :, :]
            for i in range(1, self.num_chunks):
                result = result + out[:, i :: self.num_chunks, :, :]
        elif self.chunk_mode == ChunkMode.SPLIT:
            chunks = out.chunk(self.num_chunks, dim=1)
            result = sum(chunks)
        else:
            raise ValueError(f"Unsupported chunk mode: {self.chunk_mode}")

        return result


class DMCPixelShuffle(nn.PixelShuffle):
    def __init__(self, upscale_factor):
        super().__init__(upscale_factor)

    def forward(self, x, q_index=None):
        return super().forward(x)


class SubpelConv2x(DMCModule):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        *,
        padding=0,
        bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        conv_cls = self.conv_cls or DMCConv2d
        self.conv = DMCSequential(
            conv_cls(in_ch, out_ch * 4, kernel_size=kernel_size, padding=padding, bias=bias, **kwargs),
            DMCPixelShuffle(2),
        )
        self.padding = padding

    def forward(self, x, q_index=None):
        return self.conv(x, q_index=q_index)


class SubpelConv4x(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, *, padding=0, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch * 16,  # 16 = 4**2 for PixelShuffle(4)
            kernel_size=kernel_size,
            padding=padding,
            **kwargs,
        )
        self.pixel_shuffle = nn.PixelShuffle(4)
        self.padding = padding

    def forward(self, x):
        x = self.conv(x)
        return self.pixel_shuffle(x)


class SubpelConvNx(nn.Module):
    def __init__(self, in_ch, out_ch, factor, kernel_size, *, padding=0, **kwargs):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv2d(in_ch, out_ch * factor * factor, kernel_size=kernel_size, padding=padding, **kwargs)
        self.pixel_shuffle = nn.PixelShuffle(factor)
        self.padding = padding

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))


class DepthConvBlock(DMCModule):
    def __init__(
        self,
        in_ch,
        out_ch,
        *,
        activation: str,
        enable_chunk_add: bool = True,
        num_chunks: int = 2,
        chunk_mode: ChunkMode = ChunkMode.SPLIT,
        shortcut=False,
        force_adaptor=False,
        zero_init_residual=False,
        dc_channel_multiplier=1.0,
        ffn_channel_multiplier=4.0,
        ffn_gate_activation=None,
        bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        conv_cls = self.conv_cls or DMCConv2d
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = conv_cls(in_ch, out_ch, 1, bias=bias, **kwargs)
        self.shortcut = shortcut

        mid_ch = int(dc_channel_multiplier * out_ch)
        self.dc = DMCSequential(
            conv_cls(out_ch, mid_ch, 1, bias=bias, **kwargs),
            Activation(activation=activation, **kwargs),
            conv_cls(
                mid_ch,
                mid_ch,
                3,
                padding=1,
                groups=mid_ch,
                bias=bias,
                **kwargs,
            ),
            conv_cls(mid_ch, out_ch, 1, bias=bias, **kwargs),
        )

        mid_ch = int(ffn_channel_multiplier * out_ch)
        if not enable_chunk_add:
            self.ffn = DMCSequential(
                conv_cls(out_ch, mid_ch, 1, bias=bias, **kwargs),
                Activation(activation=activation, **kwargs),
                conv_cls(mid_ch, out_ch, 1, bias=bias, **kwargs),
            )
        else:
            chunk_channels = mid_ch // num_chunks
            self.ffn = DMCSequential(
                conv_cls(out_ch, mid_ch, 1, bias=bias, **kwargs),
                ActivationChunkAdd(
                    activation=activation,
                    num_chunks=num_chunks,
                    chunk_mode=chunk_mode,
                    gate_activation=ffn_gate_activation,
                    **kwargs,
                ),
                conv_cls(chunk_channels, out_ch, 1, bias=bias, **kwargs),
            )

        self.alpha1 = nn.Parameter(torch.zeros(1)) if zero_init_residual else None
        self.alpha2 = nn.Parameter(torch.zeros(1)) if zero_init_residual else None

    def forward(self, x, q_index=None):
        if self.adaptor is not None:
            x = self.adaptor(x, q_index=q_index)

        dc_out = self.dc(x, q_index=q_index)
        if self.alpha1 is not None:
            dc_out = dc_out * self.alpha1
        dc_out = dc_out + x

        ffn_out = self.ffn(dc_out, q_index=q_index)
        if self.alpha2 is not None:
            ffn_out = ffn_out * self.alpha2
        out = ffn_out + dc_out

        if self.shortcut:
            out = out + x

        return out


class DownsampleMode(str, Enum):
    STRIDED_CONV = "strided_conv"
    PIXEL_UNSHUFFLE = "pixel_unshuffle"


class ResidualBlockWithStride2(DMCModule):
    def __init__(
        self,
        in_ch,
        out_ch,
        *,
        activation=None,  # For backward compatibility, use depth_conv_block_params instead
        downsample_mode: DownsampleMode = DownsampleMode.STRIDED_CONV,
        depth_conv_block_params=None,
        shortcut=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        depth_conv_block_kwargs = {}
        if depth_conv_block_params is not None:
            depth_conv_block_kwargs.update(depth_conv_block_params)
        if activation is not None:
            depth_conv_block_kwargs["activation"] = activation

        self.downsample_mode = downsample_mode
        conv_cls = self.conv_cls or DMCConv2d
        if downsample_mode == DownsampleMode.PIXEL_UNSHUFFLE:
            self.down = conv_cls(in_ch * 4, out_ch, 1, **kwargs)
        else:
            self.down = conv_cls(in_ch, out_ch, 2, stride=2, **kwargs)

        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=shortcut, **depth_conv_block_kwargs, **kwargs)

    def forward(self, x, q_index=None):
        if self.downsample_mode == DownsampleMode.PIXEL_UNSHUFFLE:
            x = F.pixel_unshuffle(x, 2)
        x = self.down(x, q_index=q_index)
        out = self.conv(x, q_index=q_index)
        return out


class ResidualBlockUpsample(DMCModule):
    def __init__(
        self,
        in_ch,
        out_ch,
        *,
        activation=None,  # For backward compatibility, use depth_conv_block_params instead
        depth_conv_block_params=None,
        shortcut=True,
        up_bias=True,  # Whether the internal SubpelConv2x 1x1 conv has a bias.
        **kwargs,
    ):
        super().__init__(**kwargs)
        depth_conv_block_kwargs = {}
        if depth_conv_block_params is not None:
            depth_conv_block_kwargs.update(depth_conv_block_params)
        if activation is not None:
            depth_conv_block_kwargs["activation"] = activation

        self.up = SubpelConv2x(in_ch, out_ch, 1, bias=up_bias, **kwargs)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=shortcut, **depth_conv_block_kwargs, **kwargs)

    def forward(self, x, q_index=None):
        out = self.up(x, q_index=q_index)
        out = self.conv(out, q_index=q_index)
        return out
