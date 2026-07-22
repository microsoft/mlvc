# Copyright (c) OpenMMLab. All rights reserved.
# Modifications Copyright (c) Microsoft Corporation
#
# This file is adapted from:
#   - https://github.com/ckkelvinchan/BasicVSR_PlusPlus
#   - https://github.com/open-mmlab/mmagic
# Licensed under the Apache License, Version 2.0.
#
# Reference: Chan et al., "BasicVSR++: Improving Video Super-Resolution
# with Enhanced Propagation and Alignment", CVPR 2022, arXiv:2104.13371.
#
# Modifications:
# - Standalone inference module with no mmcv/mmedit dependency
# - Uses torchvision.ops.deform_conv2d instead of mmcv.ops.ModulatedDeformConv2d

"""Standalone BasicVSR++ inference module (no mmcv/mmedit dependency)."""

from __future__ import annotations

import logging
import math
import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

logger = logging.getLogger(__name__)


def _revise_state_dict_keys(state_dict, revise_keys):
    """Apply mmengine-style ``revise_keys`` to a state dict.

    Each ``(pattern, replacement)`` is ``re.sub``-applied over every key, in
    sequence. Mirrors ``mmengine.runner.checkpoint._load_checkpoint_to_model``
    so stacked prefixes (e.g. a DDP ``module.`` wrapper around an EMA submodel)
    are peeled cleanly.
    """
    for pattern, replacement in revise_keys:
        state_dict = OrderedDict((re.sub(pattern, replacement, k), v) for k, v in state_dict.items())
    return state_dict


def flow_warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    interpolation: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Warp an image/feature map with optical flow."""
    n, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, dtype=x.dtype, device=x.device),
        torch.arange(0, w, dtype=x.dtype, device=x.device),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).expand(n, -1, -1, -1)
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    return F.grid_sample(
        x,
        vgrid_scaled,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


class ResidualBlockNoBN(nn.Module):
    """Residual block without batch normalization."""

    def __init__(self, mid_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class ResidualBlocksWithInputConv(nn.Module):
    """Input conv + N residual blocks.

    Structure matches mmcv convention:
      main.0 = Conv2d, main.1 = LeakyReLU, main.2 = Sequential of ResBlocks
    """

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Sequential(*[ResidualBlockNoBN(out_channels) for _ in range(num_blocks)]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class PixelShufflePack(nn.Module):
    """Pixel shuffle upsampler."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int, upsample_kernel: int):
        super().__init__()
        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            upsample_kernel,
            1,
            upsample_kernel // 2,
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pixel_shuffle(self.upsample_conv(x))


class _ConvModule(nn.Module):
    """Conv + optional activation."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, act: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activate = nn.ReLU(inplace=True) if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class SPyNetBasicModule(nn.Module):
    """Single level of SPyNet."""

    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            _ConvModule(8, 32, 7, 1, 3),
            _ConvModule(32, 64, 7, 1, 3),
            _ConvModule(64, 32, 7, 1, 3),
            _ConvModule(32, 16, 7, 1, 3),
            _ConvModule(16, 2, 7, 1, 3, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.basic_module(x)


class SPyNet(nn.Module):
    """SPyNet for optical flow estimation (6-level pyramid)."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        n, _, h, w = ref.size()
        ref_pyr = [(ref - self.mean) / self.std]
        supp_pyr = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref_pyr.append(F.avg_pool2d(ref_pyr[-1], 2, 2, count_include_pad=False))
            supp_pyr.append(F.avg_pool2d(supp_pyr[-1], 2, 2, count_include_pad=False))
        ref_pyr = ref_pyr[::-1]
        supp_pyr = supp_pyr[::-1]
        flow = ref_pyr[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref_pyr)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            flow = flow_up + self.basic_module[level](
                torch.cat(
                    [
                        ref_pyr[level],
                        flow_warp(supp_pyr[level], flow_up.permute(0, 2, 3, 1), padding_mode="border"),
                        flow_up,
                    ],
                    dim=1,
                )
            )
        return flow

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        h, w = ref.shape[2:4]
        w_up = w if w % 32 == 0 else 32 * (w // 32 + 1)
        h_up = h if h % 32 == 0 else 32 * (h // 32 + 1)
        ref_up = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp_up = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.compute_flow(ref_up, supp_up), size=(h, w), mode="bilinear", align_corners=False)
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)
        return flow


class ModulatedDeformConv2d(nn.Module):
    """Modulated deformable conv2d using ``torchvision.ops.deform_conv2d``.

    Drop-in replacement for ``mmcv.ops.ModulatedDeformConv2d`` for inference.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.deform_groups = deform_groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


# Second-order deformable alignment
class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Predicts offset and mask from concatenated features (warped neighbours +
    current frame + optical flows), then applies modulated deformable conv.
    """

    def __init__(self, *args, max_residue_magnitude: int = 10, **kwargs):
        self.max_residue_magnitude = max_residue_magnitude
        super().__init__(*args, **kwargs)

        # 3 * out_channels (cond_n1 + feat_current + cond_n2) + 4 (flow_1 + flow_2)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        # initialize last conv to zero so initial offset/mask is neutral
        last_conv = self.conv_offset[-1]
        assert isinstance(last_conv, nn.Conv2d)
        nn.init.constant_(last_conv.weight, 0.0)
        assert last_conv.bias is not None
        nn.init.constant_(last_conv.bias, 0.0)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        extra_feat: torch.Tensor,
        flow_1: torch.Tensor,
        flow_2: torch.Tensor,
    ) -> torch.Tensor:
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset with residue clamping (Eq. 6 in paper)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        mask = torch.sigmoid(mask)

        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


class BasicVSRPlusPlusNet(nn.Module):
    """BasicVSR++ with second-order grid propagation and deformable alignment.

    Supports either 4× super-resolution (is_low_res_input=True) or
    same-resolution restoration (is_low_res_input=False, for denoising).

    Args:
        mid_channels: channel number for intermediate features.
        num_blocks: residual blocks per propagation branch.
        max_residue_magnitude: clamp magnitude for offset residue (Eq. 6).
        is_low_res_input: if True, 4× upsampling; if False, same-res output.
        cpu_cache_length: sequences longer than this cache features on CPU.
    """

    def __init__(
        self,
        mid_channels: int = 64,
        num_blocks: int = 7,
        max_residue_magnitude: int = 10,
        is_low_res_input: bool = True,
        cpu_cache_length: int = 100,
    ):
        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        self.spynet = SPyNet()

        # feature extraction
        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        else:
            # strided convs downsample 4× for internal processing
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5),
            )

        # 4 propagation branches with second-order deformable alignment
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ["backward_1", "forward_1", "backward_2", "forward_2"]
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=max_residue_magnitude,
            )
            self.backbone[module] = ResidualBlocksWithInputConv((2 + i) * mid_channels, mid_channels, num_blocks)

        # reconstruction + upsampling
        self.reconstruction = ResidualBlocksWithInputConv(5 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lqs: torch.Tensor):
        """Compute bidirectional optical flow using SPyNet."""
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(
        self,
        feats: dict[str, list[torch.Tensor]],
        flows: torch.Tensor,
        module_name: str,
    ) -> dict[str, list[torch.Tensor]]:
        """Propagate features through a single branch with deformable alignment."""
        n, t, _, h, w = flows.size()

        frame_idx = list(range(0, t + 1))
        flow_idx = list(range(-1, t))
        mapping_idx = list(range(0, len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]

        if "backward" in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()

            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()
                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)

            # concatenate features from all completed branches + current
            feat = [feat_current] + [feats[k][idx] for k in feats if k not in ("spatial", module_name)] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]
            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(
        self,
        lqs: torch.Tensor,
        feats: dict[str, list[torch.Tensor]],
    ) -> torch.Tensor:
        """Reconstruct output frames from propagated features."""
        outputs: list[torch.Tensor] = []
        num_outputs = len(feats["spatial"])
        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != "spatial"]
            hr.insert(0, feats["spatial"][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)

            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i])
            else:
                hr += lqs[:, i]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            lqs: (n, t, 3, h, w) input sequence in [0, 1].

        Returns:
            (n, t, 3, h', w') restored sequence.
            h'=4h, w'=4w if is_low_res_input, else h'=h, w'=w.
        """
        n, t, c, h, w = lqs.size()

        self.cpu_cache = t > self.cpu_cache_length and lqs.is_cuda

        # Pad to multiple of 4 to avoid spatial mismatch between
        # stride-2 feature extraction and bicubic-downsampled flow.
        mod = 4
        pad_h = (mod - h % mod) % mod
        pad_w = (mod - w % mod) % mod
        if pad_h > 0 or pad_w > 0:
            lqs = F.pad(
                lqs.view(-1, c, h, w),
                (0, pad_w, 0, pad_h),
                mode="reflect",
            ).view(n, t, c, h + pad_h, w + pad_w)
            _, _, _, h_pad, w_pad = lqs.size()
        else:
            h_pad, w_pad = h, w

        # downsample for optical flow computation
        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h_pad, w_pad),
                scale_factor=0.25,
                mode="bicubic",
            ).view(n, t, c, h_pad // 4, w_pad // 4)

        # spatial features
        feats: dict[str, list[torch.Tensor]] = {}
        if self.cpu_cache:
            feats["spatial"] = []
            for i in range(t):
                feat = self.feat_extract(lqs[:, i]).cpu()
                feats["spatial"].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h_pad, w_pad))
            h_f, w_f = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h_f, w_f)
            feats["spatial"] = [feats_[:, i] for i in range(t)]

        # optical flow
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # 4-branch propagation (backward_1 → forward_1 → backward_2 → forward_2)
        for iter_ in [1, 2]:
            for direction in ["backward", "forward"]:
                module = f"{direction}_{iter_}"
                feats[module] = []
                flows = flows_backward if direction == "backward" else flows_forward
                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        output = self.upsample(lqs, feats)

        # Crop back to original resolution
        if pad_h > 0 or pad_w > 0:
            if self.is_low_res_input:
                output = output[:, :, :, : h * 4, : w * 4]
            else:
                output = output[:, :, :, :h, :w]

        return output


def load_basicvsrpp(
    weights_path: str,
    map_location: str = "cpu",
    mid_channels: int = 64,
    num_blocks: int = 15,
    is_low_res_input: bool = False,
    max_residue_magnitude: int = 10,
) -> BasicVSRPlusPlusNet:
    """Load BasicVSR++ with pretrained weights.

    Handles various checkpoint formats from the official repo and mmcv
    (generator_ema, generator, state_dict prefixes).

    Args:
        weights_path: path to .pth checkpoint.
        map_location: device for torch.load.
        mid_channels: channel number (default 64 for all official models).
        num_blocks: residual blocks per branch (15 for denoising, 7 for SR).
        is_low_res_input: False for denoising (same-res), True for SR (4×).
        max_residue_magnitude: offset residue clamp (default 10).

    Returns:
        BasicVSRPlusPlusNet model with loaded weights, in eval mode.
    """
    model = BasicVSRPlusPlusNet(
        mid_channels=mid_channels,
        num_blocks=num_blocks,
        max_residue_magnitude=max_residue_magnitude,
        is_low_res_input=is_low_res_input,
    )

    ckpt = torch.load(weights_path, map_location=map_location, weights_only=False)

    # unwrap checkpoint container
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "params_ema" in ckpt:
            sd = ckpt["params_ema"]
        elif "params" in ckpt:
            sd = ckpt["params"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    # Strip the DDP 'module.' wrapper first (mmengine's default revise_keys), so
    # stacked prefixes such as 'module.generator_ema.' are handled correctly.
    sd = _revise_state_dict_keys(sd, [(r"^module\.", "")])

    # GAN training checkpoints bundle generator + generator_ema (and
    # discriminator/loss/step_counter). Select one submodel subset preferring EMA
    # which both picks the right weights and drops the non-model keys
    # required for a strict load.
    if any(k.startswith("generator_ema.") for k in sd):
        sd = OrderedDict((k[len("generator_ema.") :], v) for k, v in sd.items() if k.startswith("generator_ema."))
        logger.info("Using generator_ema weights (%d keys).", len(sd))
    elif any(k.startswith("generator.") for k in sd):
        sd = OrderedDict((k[len("generator.") :], v) for k, v in sd.items() if k.startswith("generator."))
        logger.info("Using generator weights (%d keys).", len(sd))

    # Peel any 'module.' wrapper that was nested inside the submodel prefix.
    sd = _revise_state_dict_keys(sd, [(r"^module\.", "")])

    model.load_state_dict(sd, strict=True)
    logger.info("BasicVSR++ loaded %d keys (strict).", len(sd))

    return model.eval()
