# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..common_model import CompressionModel
from .layers import (
    SubpelConv2x,
    DepthConvBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride2,
    DMCConv2d,
    NetworkMode,
)
from ..utils import apply_upper_lower_bound, CkptModule
from ...utils.stream_helper import (
    get_downsampled_shape,
    open_encoder_streams,
    flush_encoder_streams,
    open_decoder_streams,
    check_decoder_eof,
)


class FeatureExtractor(CkptModule):
    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        depth_conv_block_params: dict,
        network_mode: NetworkMode,
        *,
        num_conv1_layers: int = 2,
        num_conv2_layers: int = 4,
    ):
        super().__init__()

        block_args = {**depth_conv_block_params, "network_mode": network_mode}

        if num_conv1_layers < 1:
            raise ValueError("num_conv1_layers must be >= 1")
        if num_conv2_layers < 1:
            raise ValueError("num_conv2_layers must be >= 1")

        conv1_layers = [DepthConvBlock(in_ch, mid_ch, **block_args)]
        for _ in range(num_conv1_layers - 1):
            conv1_layers.append(DepthConvBlock(mid_ch, mid_ch, **block_args))
        self.conv1 = nn.Sequential(*conv1_layers)

        conv2_layers = []
        for _ in range(num_conv2_layers - 1):
            conv2_layers.append(DepthConvBlock(mid_ch, mid_ch, **block_args))
        conv2_layers.append(DepthConvBlock(mid_ch, out_ch, **block_args))
        self.conv2 = nn.Sequential(*conv2_layers)

    def internal_forward(self, x, quant):
        x1, ctx_t = self.forward_part1(x, quant)
        ctx = self.forward_part2(x1)
        return ctx, ctx_t

    def forward_part1(self, x, quant):
        x1 = self.conv1(x)
        ctx_t = x1 * quant
        return x1, ctx_t

    def forward_part2(self, x1):
        ctx = self.conv2(x1)
        return ctx


class FeatureAdaptorI(CkptModule):
    def __init__(self, in_ch: int, out_ch: int, depth_conv_block_params: dict, network_mode: NetworkMode):
        super().__init__()

        block_args = {**depth_conv_block_params, "network_mode": network_mode}
        self.conv = DepthConvBlock(in_ch, out_ch, **block_args)

    def internal_forward(self, x):
        return self.conv(x)


class FeatureAdaptorP(CkptModule):
    def __init__(self, in_ch: int, out_ch: int, network_mode: NetworkMode):
        super().__init__()

        self.conv = DMCConv2d(in_ch, out_ch, 1, network_mode=network_mode)

    def internal_forward(self, x):
        return self.conv(x)


class Encoder(CkptModule):
    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        depth_conv_block_params: dict,
        pixel_shuffle_factor: int,
        network_mode: NetworkMode,
    ):
        super().__init__()

        self.pixel_shuffle_factor = pixel_shuffle_factor

        conv_args = {"network_mode": network_mode}
        block_args = {**depth_conv_block_params, "network_mode": network_mode}

        self.conv1 = DMCConv2d(in_ch, mid_ch, 1, **conv_args)
        self.conv2 = nn.Sequential(
            DepthConvBlock(mid_ch * 2, mid_ch, **block_args),
            DepthConvBlock(mid_ch, mid_ch, **block_args),
        )
        self.conv3 = DepthConvBlock(mid_ch, mid_ch, **block_args)
        self.down = DMCConv2d(mid_ch, out_ch, 3, stride=2, padding=1, **conv_args)

    def internal_forward(self, x, ctx, quant_step):
        feature = F.pixel_unshuffle(x, self.pixel_shuffle_factor)
        feature = self.conv1(feature)
        feature = self.conv2(torch.cat((feature, ctx), dim=1))
        feature = self.conv3(feature)
        feature = feature * quant_step
        feature = self.down(feature)
        return feature


class Decoder(CkptModule):
    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        depth_conv_block_params: dict,
        *,
        memory_activation: str,
        gate_activation: str,
        network_mode: NetworkMode,
    ):
        super().__init__()

        block_args = {**depth_conv_block_params, "network_mode": network_mode}
        conv_args = {"network_mode": network_mode}

        self.up = SubpelConv2x(in_ch, mid_ch, 3, padding=1, network_mode=network_mode)
        self.conv1 = nn.Sequential(
            DepthConvBlock(mid_ch * 2, mid_ch, **block_args),
            DepthConvBlock(mid_ch, mid_ch, **block_args),
            DepthConvBlock(mid_ch, mid_ch, **block_args),
        )
        self.conv2 = DMCConv2d(mid_ch, 3 * out_ch, 1, **conv_args)

        self.memory_activation = {
            "tanh": torch.tanh,
            "hard_tanh": torch.nn.Hardtanh(),
            "approx_tanh": lambda x: F.relu(x + 1) - F.relu(x - 1) - 1,
            "identity": nn.Identity(),
        }[memory_activation]

        self.gate_activation = {
            "sigmoid": torch.sigmoid,
            "hard_sigmoid": torch.nn.Hardsigmoid(),
            "approx_sigmoid": lambda x: 0.2 * (F.relu(x + 2.5) - F.relu(x - 2.5)),
            "identity": nn.Identity(),
        }[gate_activation]

    def internal_forward(self, x, ctx, quant_step, memory):
        feature = self.up(x)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = self.conv2(feature)

        state, forget_gate, output_gate = torch.chunk(feature, 3, dim=1)
        forget_gate = self.gate_activation(forget_gate)
        output_gate = self.gate_activation(output_gate)
        state = self.memory_activation(state)

        memory = forget_gate * memory + (1 - forget_gate) * state
        feature = output_gate * self.memory_activation(memory)
        feature = feature * quant_step
        return feature, memory


class ReconGeneration(CkptModule):
    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        depth_conv_block_params: dict,
        pixel_shuffle_factor: int,
        network_mode: NetworkMode,
    ):
        super().__init__()

        self.pixel_shuffle_factor = pixel_shuffle_factor

        conv_args = {"network_mode": network_mode}
        block_args = {**depth_conv_block_params, "network_mode": network_mode}

        self.conv = nn.Sequential(
            DepthConvBlock(in_ch, mid_ch, **block_args),
            DepthConvBlock(mid_ch, mid_ch, **block_args),
            DepthConvBlock(mid_ch, mid_ch, **block_args),
            DepthConvBlock(mid_ch, mid_ch, **block_args),
        )
        self.head = DMCConv2d(mid_ch, out_ch, 1, **conv_args)

    def internal_forward(self, x, quant_step):
        out = self.conv(x)
        out = out * quant_step
        out = self.head(out)
        out = F.pixel_shuffle(out, self.pixel_shuffle_factor)
        return out


class HyperEncoder(CkptModule):
    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        depth_conv_block_params: dict,
        network_mode: NetworkMode,
        num_downsample_layers: int = 3,
        variant: str = "full",
    ):
        super().__init__()

        if num_downsample_layers < 1:
            raise ValueError("num_downsample_layers must be >= 1")

        if variant == "mini":
            layers = []
            for i in range(num_downsample_layers):
                in_c = in_ch if i == 0 else mid_ch
                out_c = out_ch if i == num_downsample_layers - 1 else mid_ch
                layers.append(DMCConv2d(in_c, out_c, 2, stride=2, network_mode=network_mode))
        else:
            block_args = {**depth_conv_block_params, "network_mode": network_mode}
            residual_block_args = {"depth_conv_block_params": depth_conv_block_params, "network_mode": network_mode}

            layers: list[nn.Module] = [DepthConvBlock(in_ch, mid_ch, **block_args)]
            for i in range(num_downsample_layers):
                out_c = out_ch if i == num_downsample_layers - 1 else mid_ch
                layers.append(ResidualBlockWithStride2(mid_ch, out_c, **residual_block_args))

        self.conv = nn.Sequential(*layers)

    def internal_forward(self, x):
        return self.conv(x)


class HyperDecoder(CkptModule):
    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        depth_conv_block_params: dict,
        network_mode: NetworkMode,
        num_upsample_layers: int = 3,
        variant: str = "full",
    ):
        super().__init__()

        if num_upsample_layers < 1:
            raise ValueError("num_upsample_layers must be >= 1")

        if variant == "mini":
            layers = []
            for i in range(num_upsample_layers):
                in_c = in_ch if i == 0 else mid_ch
                out_c = out_ch if i == num_upsample_layers - 1 else mid_ch
                layers.append(SubpelConv2x(in_c, out_c, 1, network_mode=network_mode))
        else:
            block_args = {**depth_conv_block_params, "network_mode": network_mode}
            residual_block_args = {"depth_conv_block_params": depth_conv_block_params, "network_mode": network_mode}

            layers = []
            for i in range(num_upsample_layers):
                in_c = in_ch if i == 0 else mid_ch
                layers.append(ResidualBlockUpsample(in_c, mid_ch, **residual_block_args))
            layers.append(DepthConvBlock(mid_ch, out_ch, **block_args))

        self.conv = nn.Sequential(*layers)

    def internal_forward(self, x):
        return self.conv(x)


class PriorFusion(CkptModule):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, depth_conv_block_params: dict, network_mode: NetworkMode):
        super().__init__()

        block_args = {**depth_conv_block_params, "network_mode": network_mode}
        conv_args = {"network_mode": network_mode}

        self.conv = nn.Sequential(
            DepthConvBlock(in_ch, mid_ch, **block_args),
            DepthConvBlock(mid_ch, mid_ch, **block_args),
            DepthConvBlock(mid_ch, mid_ch, **block_args),
            DMCConv2d(mid_ch, out_ch, 1, **conv_args),
        )

    def internal_forward(self, x):
        return self.conv(x)


class SpatialPrior(CkptModule):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, depth_conv_block_params: dict, network_mode: NetworkMode):
        super().__init__()

        block_args = {**depth_conv_block_params, "network_mode": network_mode}
        conv_args = {"network_mode": network_mode}

        self.conv = nn.Sequential(
            DepthConvBlock(in_ch, mid_ch, **block_args),
            DepthConvBlock(mid_ch, mid_ch, **block_args),
            DMCConv2d(mid_ch, out_ch, 1, **conv_args),
        )

    def internal_forward(self, x):
        return self.conv(x)


class DMC(CompressionModel):
    def __init__(
        self,
        *,
        depth_conv_block_params,
        override_shared_depth_conv_block_params=None,
        qp_num: int = 64,
        hidden_channels: int = 256,
        feature_channels: int = 256,
        recon_channels: int = 256,
        z_channels: int = 128,
        y_channels: int = 128,
        y_scale_repeat: int = 2,
        spatial_prior_channels: int = 384,
        prior_fusion_channels: int | None = None,
        pixel_shuffle_factor: int = 8,
        hyperprior_num_blocks: int = 3,
        hyperprior_variant: str = "full",
        feature_extractor_num_conv1_layers: int = 2,
        feature_extractor_num_conv2_layers: int = 4,
        input_offset: float | None = None,
        memory_activation: str = "tanh",
        gate_activation: str = "sigmoid",
        chain_feature_adaptors: bool = False,
        qp_shift: Sequence[int] = (0, 8, 4),
        network_mode: NetworkMode = NetworkMode.FP32,
        y_distribution_scale_step_in_log_space: bool = False,
        y_distribution_scale_input_in_index_space: bool = True,
    ):

        extra_qp = max(qp_shift)

        super().__init__(
            y_distribution="gaussian",
            qp_num=qp_num,
            extra_qp=extra_qp,
            z_channel=z_channels,
            bit_estimator_qp_split=True,
            y_distribution_scale_step_in_log_space=y_distribution_scale_step_in_log_space,
            y_distribution_scale_input_in_index_space=y_distribution_scale_input_in_index_space,
        )

        if override_shared_depth_conv_block_params is None:
            override_shared_depth_conv_block_params = {}

        shared_depth_conv_block_params = {
            **depth_conv_block_params,
            **override_shared_depth_conv_block_params,
        }

        self.input_offset = input_offset
        self.feature_channels = feature_channels
        self.y_channels = y_channels
        self.y_scale_repeat = y_scale_repeat
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.hyperprior_num_blocks = hyperprior_num_blocks
        self.hyperprior_variant = hyperprior_variant

        if self.y_channels % self.y_scale_repeat != 0:
            raise ValueError(
                f"y_channels must be divisible by y_scale_repeat, y_channels={self.y_channels}, \
                             y_scale_repeat={self.y_scale_repeat}"
            )

        input_channels = 3 * self.pixel_shuffle_factor * self.pixel_shuffle_factor

        self.chain_feature_adaptors = chain_feature_adaptors
        fai_depth_conv_block_params = shared_depth_conv_block_params
        if chain_feature_adaptors:
            fai_depth_conv_block_params = {
                **shared_depth_conv_block_params,
                "bias": False,
            }

        self.feature_adaptor_i = FeatureAdaptorI(
            in_ch=input_channels,
            out_ch=2 * feature_channels,
            depth_conv_block_params=fai_depth_conv_block_params,
            network_mode=network_mode,
        )

        self.feature_adaptor_p = FeatureAdaptorP(
            in_ch=feature_channels,
            out_ch=feature_channels,
            network_mode=network_mode,
        )

        self.feature_extractor = FeatureExtractor(
            in_ch=feature_channels,
            mid_ch=hidden_channels,
            out_ch=hidden_channels,
            depth_conv_block_params=shared_depth_conv_block_params,
            network_mode=network_mode,
            num_conv1_layers=feature_extractor_num_conv1_layers,
            num_conv2_layers=feature_extractor_num_conv2_layers,
        )

        self.encoder = Encoder(
            in_ch=input_channels,
            mid_ch=hidden_channels,
            out_ch=y_channels,
            depth_conv_block_params=depth_conv_block_params,
            pixel_shuffle_factor=pixel_shuffle_factor,
            network_mode=NetworkMode.FP32,  # Always use FP32
        )

        self.hyper_encoder = HyperEncoder(
            in_ch=y_channels,
            mid_ch=z_channels,
            out_ch=z_channels,
            depth_conv_block_params=depth_conv_block_params,
            network_mode=NetworkMode.FP32,  # Always use FP32
            num_downsample_layers=hyperprior_num_blocks,
            variant=hyperprior_variant,
        )

        self.hyper_decoder = HyperDecoder(
            in_ch=z_channels,
            mid_ch=z_channels,
            out_ch=y_channels,
            depth_conv_block_params=shared_depth_conv_block_params,
            network_mode=network_mode,
            num_upsample_layers=hyperprior_num_blocks,
            variant=hyperprior_variant,
        )

        self.temporal_prior_encoder = ResidualBlockWithStride2(
            hidden_channels,
            y_channels * 2,
            depth_conv_block_params=shared_depth_conv_block_params,
            network_mode=network_mode,
        )

        self.y_prior_fusion = PriorFusion(
            in_ch=y_channels * 3,
            mid_ch=prior_fusion_channels or y_channels * 3,
            out_ch=y_channels * 2,
            depth_conv_block_params=shared_depth_conv_block_params,
            network_mode=network_mode,
        )

        self.y_spatial_prior = SpatialPrior(
            in_ch=y_channels * 3,
            mid_ch=spatial_prior_channels,
            out_ch=y_channels,
            depth_conv_block_params=shared_depth_conv_block_params,
            network_mode=network_mode,
        )

        self.decoder = Decoder(
            in_ch=y_channels,
            mid_ch=hidden_channels,
            out_ch=feature_channels,
            memory_activation=memory_activation,
            gate_activation=gate_activation,
            depth_conv_block_params=shared_depth_conv_block_params,
            network_mode=network_mode,
        )

        self.recon_generation_net = ReconGeneration(
            in_ch=feature_channels,
            mid_ch=recon_channels,
            out_ch=input_channels,
            depth_conv_block_params=shared_depth_conv_block_params,
            pixel_shuffle_factor=pixel_shuffle_factor,
            network_mode=NetworkMode.FP32,
        )

        self.q_encoder = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, hidden_channels, 1, 1)))
        self.q_decoder = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, feature_channels, 1, 1)))
        self.q_feature = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, hidden_channels, 1, 1)))
        self.q_recon = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, recon_channels, 1, 1)))
        self._initialize_weights()

        self.qp_shift = tuple(qp_shift)
        self.frame_index_map = 0, 1, 0, 2, 0, 2, 0, 2

        self._register_load_state_dict_pre_hook(self._convert_legacy_state_dict)

    def _convert_legacy_state_dict(self, state_dict, *unused):
        rename_rules = {
            "feature_adaptor_i.adaptor.": "feature_adaptor_i.conv.adaptor.",
            "feature_adaptor_i.dc.": "feature_adaptor_i.conv.dc.",
            "feature_adaptor_i.ffn.": "feature_adaptor_i.conv.ffn.",
            "feature_adaptor_i.alpha1": "feature_adaptor_i.conv.alpha1",
            "feature_adaptor_i.alpha2": "feature_adaptor_i.conv.alpha2",
            "feature_adaptor_p.weight": "feature_adaptor_p.conv.weight",
            "feature_adaptor_p.bias": "feature_adaptor_p.conv.bias",
        }

        new_items = {}
        for k, v in list(state_dict.items()):
            for old, new in rename_rules.items():
                if k.startswith(old):
                    new_key = new + k[len(old) :]
                    new_items[new_key] = v
                    del state_dict[k]
                    break
        state_dict.update(new_items)

    @property
    def feature_downscale_factor(self):
        # relative to the input frame
        return self.pixel_shuffle_factor

    @property
    def y_downscale_factor(self):
        # relative to the input frame
        return self.feature_downscale_factor * 2

    @property
    def z_downscale_factor(self):
        # relative to the y latent
        return 2**self.hyperprior_num_blocks

    @property
    def padding_size(self):
        return self.y_downscale_factor

    def set_use_ckpt(self, use_ckpt=True, recon_module=False, hyper_module=False):
        self.feature_extractor.set_use_ckpt(use_ckpt)
        if recon_module or not use_ckpt:
            self.recon_generation_net.set_use_ckpt(use_ckpt)
        self.encoder.set_use_ckpt(use_ckpt)
        self.decoder.set_use_ckpt(use_ckpt)
        if hyper_module or not use_ckpt:
            self.hyper_encoder.set_use_ckpt(use_ckpt)
            self.hyper_decoder.set_use_ckpt(use_ckpt)
            self.y_prior_fusion.set_use_ckpt(use_ckpt)
            self.y_spatial_prior.set_use_ckpt(use_ckpt)

    def clear_grad_for_y_q(self):
        self.q_encoder.grad = None
        self.q_decoder.grad = None
        self.q_feature.grad = None
        self.q_recon.grad = None

    def shift_qp(self, qp, fa_idx):
        s = self.qp_shift[fa_idx]
        if isinstance(qp, list):
            qp = [item + s for item in qp]
        else:
            qp = qp + s
        return qp

    def shift_input(self, x):
        if not self.input_offset:
            return x
        return x + self.input_offset

    def unshift_output(self, x):
        if not self.input_offset:
            return x
        return x - self.input_offset

    def clamp_x_hat(self, x_hat):
        min_v, max_v = 0.0, 1.0
        if self.input_offset:
            min_v += self.input_offset
            max_v += self.input_offset
        return x_hat.clamp_(min_v, max_v)

    def calc_dual_prior2(self, y, common_params, y_spatial_prior, y_scales):
        quant_step, means = common_params.chunk(2, 1)
        quant_step = apply_upper_lower_bound(quant_step, lower=0.5)

        q_enc = 1.0 / quant_step
        q_dec = quant_step
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_dual(B, C, H, W, dtype, device)

        y = y * q_enc
        (y_raw_0, scales_0), y_hat_0, y_q_0 = self.process_with_mask(
            y, y_scales, means, mask=mask_0, chunks=2, return_y_q=True
        )
        means = y_spatial_prior(torch.cat((y_hat_0, common_params), dim=1))
        (y_raw_1, scales_1), y_hat_1, y_q_1 = self.process_with_mask(
            y, y_scales, means, mask=mask_1, chunks=2, return_y_q=True
        )

        y_hat = (y_hat_0 + y_hat_1) * q_dec

        y_raw = (y_raw_0, scales_0), (y_raw_1, scales_1)
        y_q = (y_q_0, scales_0), (y_q_1, scales_1)
        return y_raw, y_hat, y_q

    def decompress_dual_prior2(self, streams, common_params, y_scales):
        quant_step, means_0 = common_params.chunk(2, 1)
        quant_step = apply_upper_lower_bound(quant_step, lower=0.5)

        B, C, H, W = means_0.shape
        mask_0, mask_1 = self.get_mask_dual(B, C, H, W, means_0.dtype, means_0.device)

        y_hat_0 = self.decode_y_with_mask(streams, means_0, y_scales, mask=mask_0, chunks=2)
        means_1 = self.y_spatial_prior(torch.cat((y_hat_0, common_params), dim=1))
        y_hat_1 = self.decode_y_with_mask(streams, means_1, y_scales, mask=mask_1, chunks=2)

        y_hat = (y_hat_0 + y_hat_1) * quant_step
        return y_hat

    def get_y_scales(self, z_hat, slice_shape):
        repeat = self.y_scale_repeat
        base_channels = self.y_channels // repeat
        assert self.y_channels % repeat == 0, (
            f"y_channels={self.y_channels} must be divisible by y_scale_repeat={repeat}"
        )

        y_scales = z_hat[:, :base_channels, None, :, None, :, None]
        y_scales = torch.abs(y_scales)
        z_factor = self.z_downscale_factor
        y_scales = y_scales.expand(-1, -1, repeat, -1, z_factor, -1, z_factor)
        y_scales = y_scales.reshape(-1, self.y_channels, z_hat.shape[-2] * z_factor, z_hat.shape[-1] * z_factor)
        y_scales = self.slice_to_y(y_scales, slice_shape)
        return y_scales

    def apply_feature_adaptor(self, dpb):
        if dpb["ref_feature"] is None:
            ref_frame = self.shift_input(dpb["ref_frame"])
            factor = self.pixel_shuffle_factor
            feature = self.feature_adaptor_i(F.pixel_unshuffle(ref_frame, factor))
            ref_feature, ref_memory = feature.chunk(2, dim=1)
            if self.chain_feature_adaptors:
                ref_feature = self.feature_adaptor_p(ref_feature)
        else:
            ref_feature, ref_memory = dpb["ref_feature"].chunk(2, dim=1)
            ref_feature = self.feature_adaptor_p(ref_feature)
        return ref_feature, ref_memory

    def context_generation(self, dpb, quant):
        ref_feature, ref_memory = self.apply_feature_adaptor(dpb)
        ctx, ctx_t = self.feature_extractor(ref_feature, quant)
        return ctx, ctx_t, ref_memory

    def res_prior_param_decoder(self, z_hat, ctx_t, slice_shape):
        hierarchical_params = self.hyper_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(ctx_t)
        hierarchical_params = self.slice_to_y(hierarchical_params, slice_shape)
        params = self.y_prior_fusion(torch.cat((hierarchical_params, temporal_params), dim=1))
        return params

    def get_recon(self, feature, q_index):
        q_recon = self.q_recon[q_index]
        x_hat = self.recon_generation_net(feature, q_recon)
        if not self.training:
            x_hat = self.clamp_x_hat(x_hat)
        return x_hat

    def get_recon_and_feature(self, y_hat, ctx, q_decoder, memory, q_index, get_recon=True):
        feature, memory = self.decoder(y_hat, ctx, q_decoder, memory)
        x_hat = self.get_recon(feature, q_index) if get_recon else None
        return x_hat, feature, memory

    def compress_core(self, x, dpb, q_index, fa_idx=None, do_shift_qp=True, get_recon=True) -> dict[str, Any]:
        x = self.shift_input(x)

        if do_shift_qp:
            assert fa_idx is not None, "fa_idx must be provided when do_shift_qp is True"
            q_index = self.shift_qp(q_index, fa_idx)

        q_encoder = self.q_encoder[q_index]
        q_decoder = self.q_decoder[q_index]
        q_feature = self.q_feature[q_index]

        ctx, ctx_t, memory = self.context_generation(dpb, q_feature)

        y = self.encoder(x, ctx, q_encoder)

        hyper_inp, slice_shape = self.pad_for_y(y, z_factor=self.z_downscale_factor)

        z = self.hyper_encoder(hyper_inp)
        z_raw, z_hat = self.quantize(z)
        y_scales = self.get_y_scales(z_hat, slice_shape)
        params = self.res_prior_param_decoder(z_hat, ctx_t, slice_shape)
        y_raw, y_hat, y_q = self.calc_dual_prior2(y, params, self.y_spatial_prior, y_scales)

        x_hat, feature, memory = self.get_recon_and_feature(y_hat, ctx, q_decoder, memory, q_index, get_recon)
        if x_hat is not None:
            x_hat = self.unshift_output(x_hat)
        feature = torch.cat((feature, memory), dim=1)

        dpb = dict(
            ref_frame=x_hat,
            ref_feature=feature,
        )

        result = dict(
            dpb=dpb,
            q_index=q_index,
            z_raw=z_raw,
            z_hat=z_hat,
            y_raw=y_raw,
            y_hat=y_hat,
            y_q=y_q,
        )

        return result

    def decode_core(
        self,
        dpb,
        q_index,
        fa_idx=None,
        do_shift_qp=True,
        get_recon=True,
        *,
        z_hat,
        y_res_q,
    ):
        """Factored out of compress_core so the decode can be re-run independently, useful for drift loss."""

        if do_shift_qp:
            assert fa_idx is not None, "fa_idx must be provided when do_shift_qp is True"
            q_index = self.shift_qp(q_index, fa_idx)

        q_decoder = self.q_decoder[q_index]
        q_feature = self.q_feature[q_index]

        ctx, ctx_t, memory = self.context_generation(dpb, q_feature)

        height = dpb["ref_frame"].shape[-2]
        width = dpb["ref_frame"].shape[-1]
        y_height, y_width = get_downsampled_shape(height, width, self.y_downscale_factor)
        slice_shape = self.get_to_y_slice_shape(y_height, y_width, z_factor=self.z_downscale_factor)
        params = self.res_prior_param_decoder(z_hat, ctx_t, slice_shape)
        y_hat = self.reconstruct_y_hat(y_res_q, params)

        x_hat, feature, memory = self.get_recon_and_feature(y_hat, ctx, q_decoder, memory, q_index, get_recon)
        if x_hat is not None:
            x_hat = self.unshift_output(x_hat)
        feature = torch.cat((feature, memory), dim=1)

        out_dpb = dict(
            ref_frame=x_hat,
            ref_feature=feature,
        )

        return x_hat, feature, out_dpb, ctx, ctx_t

    def reconstruct_y_hat(self, y_res_q, common_params):
        """This mirrors calc_dual_prior2 but uses pre-quantized residuals instead of quantizing from the original y."""

        quant_step, means = common_params.chunk(2, 1)
        quant_step = apply_upper_lower_bound(quant_step, lower=0.5)

        q_dec = quant_step
        B, C, H, W = means.size()
        dtype = means.dtype
        device = means.device
        mask_0, mask_1 = self.get_mask_dual(B, C, H, W, dtype, device)

        (y_0, _scales_0), (y_1, _scales_1) = y_res_q

        y_hat_0 = self.unpack_with_mask(y_0, means, mask_0, chunks=2)
        means_1 = self.y_spatial_prior(torch.cat((y_hat_0, common_params), dim=1))
        y_hat_1 = self.unpack_with_mask(y_1, means_1, mask_1, chunks=2)

        y_hat = (y_hat_0 + y_hat_1) * q_dec
        return y_hat

    def compress(self, x, dpb, *, q_index, fa_idx, calc_bits_estimates=False):
        result = self.compress_core(x, dpb, q_index, fa_idx)

        dpb = result["dpb"]
        z_raw = result["z_raw"]
        y_raw = result["y_raw"]
        # unshifted q-index for z bit estimator
        q_index_z = q_index

        streams = open_encoder_streams(z_raw.size(0))

        self.encode_y(streams, y_raw)
        self.bit_estimator_z.encode_z(streams, z_raw, q_index_z)

        bit_stream = flush_encoder_streams(streams)

        result = dict(
            x_hat=dpb["ref_frame"],
            dpb=dpb,
            bit_stream=bit_stream,
        )

        if calc_bits_estimates:
            bits_y = self.get_y_bits_estimate(y_raw)

            bits_z = self.get_z_bits(z_raw, self.bit_estimator_z, q_index_z)
            bits_z = bits_z.flatten(1).sum(dim=-1)

            result.update(
                bits_estimate_y=bits_y,
                bits_estimate_z=bits_z,
                bits_estimate=bits_y + bits_z,
            )

        return result

    def decompress(self, bit_stream, dpb, *, q_index, fa_idx, height, width):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        # unshifted q-index for z bit estimator
        q_index_z = q_index
        q_index = self.shift_qp(q_index, fa_idx)

        q_decoder = self.q_decoder[q_index]
        q_feature = self.q_feature[q_index]

        ctx, ctx_t, memory = self.context_generation(dpb, q_feature)

        z_size = get_downsampled_shape(height, width, self.z_downscale_factor * self.y_downscale_factor)
        y_height, y_width = get_downsampled_shape(height, width, self.y_downscale_factor)

        slice_shape = self.get_to_y_slice_shape(y_height, y_width, z_factor=self.z_downscale_factor)

        streams = open_decoder_streams(bit_stream)

        z_hat = self.bit_estimator_z.decode_z(streams, z_size, q_index_z).to(dtype=dtype, device=device)
        y_scales = self.get_y_scales(z_hat, slice_shape)
        params_0 = self.res_prior_param_decoder(z_hat, ctx_t, slice_shape)

        y_hat = self.decompress_dual_prior2(streams, params_0, y_scales)

        check_decoder_eof(streams)
        del streams

        x_hat, feature, memory = self.get_recon_and_feature(y_hat, ctx, q_decoder, memory, q_index)
        x_hat = self.unshift_output(x_hat)
        feature = torch.cat((feature, memory), dim=1)

        dpb = dict(
            ref_frame=x_hat,
            ref_feature=feature,
        )

        return dict(x_hat=x_hat, dpb=dpb)

    def forward(self, x, dpb, *, q_index, fa_idx):
        result = self.compress_core(x, dpb, q_index, fa_idx)

        dpb = result["dpb"]

        z_hat = result["z_hat"]
        y_raw = result["y_raw"]
        y_q = result["y_q"]
        z_raw = result["z_raw"]
        # unshifted q-index for z bit estimator
        q_index_z = q_index

        bits_y = self.get_y_bits_estimate(y_raw)
        bits_z = self.get_z_bits(z_raw, self.bit_estimator_z, q_index_z)
        bits_z = bits_z.flatten(1).sum(dim=-1)
        bits = bits_y + bits_z

        _, _, H, W = x.size()
        pixel_num = H * W

        bpp_y = bits_y / pixel_num
        bpp_z = bits_z / pixel_num
        bpp = bits / pixel_num

        return dict(
            bpp_y=bpp_y,
            bpp_z=bpp_z,
            bpp=bpp,
            dpb=dpb,
            x_hat=dpb["ref_frame"],
            bits=bits,
            bits_y=bits_y,
            bits_z=bits_z,
            # Latents exposed for a drift decode
            z_hat=z_hat,
            y_raw=y_raw,
            y_q=y_q,
        )
