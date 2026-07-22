# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..common_model import CompressionModel
from .layers import SubpelConv2x, DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2
from ..utils import CkptModule
from ...utils.stream_helper import (
    get_downsampled_shape,
    open_encoder_streams,
    flush_encoder_streams,
    open_decoder_streams,
    check_decoder_eof,
)


g_ch_src_d = 3 * 8 * 8
g_ch_recon = 320
g_ch_y = 128
g_ch_z = 128
g_ch_d = 256

qp_shift = [0, 8, 4]
extra_qp = max(qp_shift)


class FeatureExtractor(CkptModule):
    def __init__(self, activation: str):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d, activation=activation),
            DepthConvBlock(g_ch_d, g_ch_d, activation=activation),
        )
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d, activation=activation),
            DepthConvBlock(g_ch_d, g_ch_d, activation=activation),
            DepthConvBlock(g_ch_d, g_ch_d, activation=activation),
            DepthConvBlock(g_ch_d, g_ch_d, activation=activation),
        )

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


class Encoder(CkptModule):
    def __init__(self, activation: str):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_src_d, g_ch_d, 1)
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d, activation=activation),
            DepthConvBlock(g_ch_d, g_ch_d, activation=activation),
        )
        self.conv3 = DepthConvBlock(g_ch_d, g_ch_d, activation=activation)
        self.down = nn.Conv2d(g_ch_d, g_ch_y, 3, stride=2, padding=1)

    def internal_forward(self, x, ctx, quant_step):
        feature = F.pixel_unshuffle(x, 8)
        feature = self.conv1(feature)
        feature = self.conv2(torch.cat((feature, ctx), dim=1))
        feature = self.conv3(feature)
        feature = feature * quant_step
        feature = self.down(feature)
        return feature


class Decoder(CkptModule):
    def __init__(self, activation: str):
        super().__init__()
        self.up = SubpelConv2x(g_ch_y, g_ch_d, 3, padding=1)
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d, activation=activation),
            DepthConvBlock(g_ch_d, g_ch_d, activation=activation),
            DepthConvBlock(g_ch_d, g_ch_d, activation=activation),
        )
        self.conv2 = nn.Conv2d(g_ch_d, g_ch_d, 1)

    def internal_forward(self, x, ctx, quant_step):
        feature = self.up(x)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = self.conv2(feature)
        feature = feature * quant_step
        return feature


class ReconGeneration(CkptModule):
    def __init__(self, activation: str):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_recon, activation=activation),
            DepthConvBlock(g_ch_recon, g_ch_recon, activation=activation),
            DepthConvBlock(g_ch_recon, g_ch_recon, activation=activation),
            DepthConvBlock(g_ch_recon, g_ch_recon, activation=activation),
        )
        self.head = nn.Conv2d(g_ch_recon, g_ch_src_d, 1)

    def internal_forward(self, x, quant_step):
        out = self.conv(x)
        out = out * quant_step
        out = self.head(out)
        out = F.pixel_shuffle(out, 8)
        if not self.training:
            out = torch.clamp(out, 0.0, 1.0)
        return out


class HyperEncoder(CkptModule):
    def __init__(self, activation: str):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y, g_ch_z, activation=activation),
            ResidualBlockWithStride2(g_ch_z, g_ch_z, activation=activation),
            ResidualBlockWithStride2(g_ch_z, g_ch_z, activation=activation),
        )

    def internal_forward(self, x):
        return self.conv(x)


class HyperDecoder(CkptModule):
    def __init__(self, activation: str):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockUpsample(g_ch_z, g_ch_z, activation=activation),
            ResidualBlockUpsample(g_ch_z, g_ch_z, activation=activation),
            DepthConvBlock(g_ch_z, g_ch_y, activation=activation),
        )

    def internal_forward(self, x):
        return self.conv(x)


class PriorFusion(CkptModule):
    def __init__(self, activation: str):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3, activation=activation),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3, activation=activation),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3, activation=activation),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 3, 1),
        )

    def internal_forward(self, x):
        return self.conv(x)


class SpatialPrior(CkptModule):
    def __init__(self, activation: str):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 4, g_ch_y * 3, activation=activation),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3, activation=activation),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 2, 1),
        )

    def internal_forward(self, x):
        return self.conv(x)


class DMC(CompressionModel):
    def __init__(self, activation: str, qp_num: int = 64):

        super().__init__(
            y_distribution="gaussian",
            qp_num=qp_num,
            extra_qp=extra_qp,
            z_channel=g_ch_z,
            bit_estimator_qp_split=True,
        )

        self.feature_adaptor_i = DepthConvBlock(g_ch_src_d, g_ch_d, activation=activation)
        self.feature_adaptor_p = nn.Conv2d(g_ch_d, g_ch_d, 1)
        self.feature_extractor = FeatureExtractor(activation=activation)

        self.encoder = Encoder(activation=activation)
        self.hyper_encoder = HyperEncoder(activation=activation)
        self.hyper_decoder = HyperDecoder(activation=activation)

        self.temporal_prior_encoder = ResidualBlockWithStride2(g_ch_d, g_ch_y * 2, activation=activation)

        self.y_prior_fusion = PriorFusion(activation=activation)
        self.y_spatial_prior = SpatialPrior(activation=activation)

        self.decoder = Decoder(activation=activation)
        self.recon_generation_net = ReconGeneration(activation=activation)

        self.q_encoder = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        self.q_decoder = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        self.q_feature = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        self.q_recon = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_recon, 1, 1)))
        self._initialize_weights()

        self.frame_index_map = 0, 1, 0, 2, 0, 2, 0, 2

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

    @staticmethod
    def shift_qp(qp, fa_idx):
        s = qp_shift[fa_idx]
        if isinstance(qp, list):
            qp = [item + s for item in qp]
        else:
            qp = qp + s
        return qp

    def apply_feature_adaptor(self, dpb):
        if dpb["ref_feature"] is None:
            feature = self.feature_adaptor_i(F.pixel_unshuffle(dpb["ref_frame"], 8))
        else:
            feature = self.feature_adaptor_p(dpb["ref_feature"])
        return feature

    def context_generation(self, dpb, quant):
        feature = self.apply_feature_adaptor(dpb)
        return self.feature_extractor(feature, quant)

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
            x_hat = x_hat.clamp_(0, 1)
        return x_hat

    def get_recon_and_feature(self, y_hat, ctx, q_decoder, q_index, get_recon=True):
        feature = self.decoder(y_hat, ctx, q_decoder)
        x_hat = self.get_recon(feature, q_index) if get_recon else None
        return x_hat, feature

    def compress_core(self, x, dpb, q_index, fa_idx=None, do_shift_qp=True, get_recon=True) -> dict[str, Any]:
        if do_shift_qp:
            assert fa_idx is not None, "fa_idx must be provided when do_shift_qp is True"
            q_index = self.shift_qp(q_index, fa_idx)

        q_encoder = self.q_encoder[q_index]
        q_decoder = self.q_decoder[q_index]
        q_feature = self.q_feature[q_index]

        ctx, ctx_t = self.context_generation(dpb, q_feature)

        y = self.encoder(x, ctx, q_encoder)

        hyper_inp, slice_shape = self.pad_for_y(y)

        z = self.hyper_encoder(hyper_inp)
        z_raw, z_hat = self.quantize(z)
        params = self.res_prior_param_decoder(z_hat, ctx_t, slice_shape)
        y_raw, y_hat = self.calc_dual_prior(y, params, self.y_spatial_prior)

        x_hat, feature = self.get_recon_and_feature(y_hat, ctx, q_decoder, q_index, get_recon)

        dpb = dict(
            ref_frame=x_hat,
            ref_feature=feature,
        )

        result = dict(
            dpb=dpb,
            q_index=q_index,
            z_raw=z_raw,
            y_raw=y_raw,
        )

        return result

    def compress(self, x, dpb, *, q_index, fa_idx, calc_bits_estimates=False):
        result = self.compress_core(x, dpb, q_index, fa_idx)

        dpb = result["dpb"]
        z_raw = result["z_raw"]
        y_raw = result["y_raw"]
        q_index = result["q_index"]

        streams = open_encoder_streams(z_raw.size(0))

        self.encode_y(streams, y_raw)
        self.bit_estimator_z.encode_z(streams, z_raw, q_index)

        bit_stream = flush_encoder_streams(streams)

        result = dict(
            x_hat=dpb["ref_frame"],
            dpb=dpb,
            bit_stream=bit_stream,
        )

        if calc_bits_estimates:
            bits_y = self.get_y_bits_estimate(y_raw)

            bits_z = self.get_z_bits(z_raw, self.bit_estimator_z, q_index)
            bits_z = bits_z.flatten(1).sum(dim=-1)

            result.update(
                bits_estimate_y=bits_y,
                bits_estimate_z=bits_z,
                bits_estimate=bits_y + bits_z,
            )

        return result

    def decode_z_hat(self, decoder_stream, q_index, height, width):
        """CPU: Decode z_hat (without prior)"""

        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        z_size = get_downsampled_shape(height, width, 64)
        y_height, y_width = get_downsampled_shape(height, width, 16)

        z_hat = self.bit_estimator_z.decode_z(decoder_stream, z_size, q_index).to(dtype=dtype, device=device)

        slice_shape = self.get_to_y_slice_shape(y_height, y_width)

        return {
            "z_hat": z_hat,
            "y_height": y_height,
            "y_width": y_width,
            "slice_shape": slice_shape,
        }

    def estimate_params0(self, part1, dpb, q_index):
        """GPU/NPU: Estimate first set of params (from z_hat)"""

        q_feature = self.q_feature[q_index]

        ctx, ctx_t = self.context_generation(dpb, q_feature)
        params_0 = self.res_prior_param_decoder(part1["z_hat"], ctx_t, part1["slice_shape"])

        # Prepare for dual prior
        quant_step, scales_0, means_0 = self.separate_prior(params_0, is_video=True)

        return {
            "scales_0": scales_0,
            "quant_step": quant_step,
            "means_0": means_0,
            "params_0": params_0,
            "ctx": ctx,
        }

    def decompress(self, bit_stream, dpb, *, q_index, fa_idx, height, width):
        streams = open_decoder_streams(bit_stream)

        q_index = self.shift_qp(q_index, fa_idx)

        q_decoder = self.q_decoder[q_index]

        part1 = self.decode_z_hat(streams, q_index, height, width)
        part2 = self.estimate_params0(part1, dpb, q_index)
        y_hat = self.decompress_dual_prior(
            streams, part2["params_0"], part2["scales_0"], part2["means_0"], part2["quant_step"]
        )

        check_decoder_eof(streams)
        del streams

        x_hat, feature = self.get_recon_and_feature(y_hat, part2["ctx"], q_decoder, q_index)

        dpb = dict(
            ref_frame=x_hat,
            ref_feature=feature,
        )

        return dict(x_hat=x_hat, dpb=dpb)

    def forward(self, x, dpb, *, q_index, fa_idx):
        result = self.compress_core(x, dpb, q_index, fa_idx)

        dpb = result["dpb"]

        y_raw = result["y_raw"]
        z_raw = result["z_raw"]
        q_index = result["q_index"]

        bits_y = self.get_y_bits_estimate(y_raw)
        bits_z = self.get_z_bits(z_raw, self.bit_estimator_z, q_index)
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
        )
