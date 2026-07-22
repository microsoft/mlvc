# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from numbers import Integral

import torch
import torch.nn.functional as F
from torch import nn

from ..common_model import CompressionModel
from .layers import DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2
from ...utils.stream_helper import (
    get_downsampled_shape,
    open_encoder_streams,
    flush_encoder_streams,
    open_decoder_streams,
    check_decoder_eof,
)

g_ch_src = 3 * 8 * 8
g_ch_enc_dec = 368


class IntraEncoder(nn.Module):
    def __init__(self, N, activation: str):
        super().__init__()

        self.enc_1 = DepthConvBlock(g_ch_src, g_ch_enc_dec, activation=activation)
        self.enc_2 = nn.Sequential(
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            nn.Conv2d(g_ch_enc_dec, N, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):
        out = F.pixel_unshuffle(x, 8)
        out = self.enc_1(out)
        out = out * quant_step
        return self.enc_2(out)


class IntraDecoder(nn.Module):
    def __init__(self, N, activation: str):
        super().__init__()

        self.dec_1 = nn.Sequential(
            ResidualBlockUpsample(N, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec, activation=activation),
        )
        self.dec_2 = DepthConvBlock(g_ch_enc_dec, g_ch_src, activation=activation)

    def forward(self, x, quant_step):
        out = self.dec_1(x)
        out = out * quant_step
        out = self.dec_2(out)
        out = F.pixel_shuffle(out, 8)
        return out


class DMCI(CompressionModel):
    def __init__(self, activation: str, N=256, z_channel=128):
        # super().__init__(z_channel=z_channel)
        super().__init__(y_distribution="gaussian", z_channel=z_channel, bit_estimator_qp_split=True)

        self.enc = IntraEncoder(N, activation=activation)

        self.hyper_enc = nn.Sequential(
            DepthConvBlock(N, z_channel, activation=activation),
            ResidualBlockWithStride2(z_channel, z_channel, activation=activation),
            ResidualBlockWithStride2(z_channel, z_channel, activation=activation),
        )

        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(z_channel, z_channel, activation=activation),
            ResidualBlockUpsample(z_channel, z_channel, activation=activation),
            DepthConvBlock(z_channel, N, activation=activation),
        )

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(N, N * 2, activation=activation),
            DepthConvBlock(N * 2, N * 2, activation=activation),
            DepthConvBlock(N * 2, N * 2, activation=activation),
            nn.Conv2d(N * 2, N * 2 + 2, 1),
        )

        self.y_spatial_prior_reduction = nn.Conv2d(N * 2 + 2, N * 1, 1)
        self.y_spatial_prior_adaptor_1 = DepthConvBlock(N * 2, N * 2, force_adaptor=True, activation=activation)
        self.y_spatial_prior_adaptor_2 = DepthConvBlock(N * 2, N * 2, force_adaptor=True, activation=activation)
        self.y_spatial_prior_adaptor_3 = DepthConvBlock(N * 2, N * 2, force_adaptor=True, activation=activation)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(N * 2, N * 2, activation=activation),
            DepthConvBlock(N * 2, N * 2, activation=activation),
            DepthConvBlock(N * 2, N * 2, activation=activation),
            nn.Conv2d(N * 2, N * 2, 1),
        )

        self.dec = IntraDecoder(N, activation=activation)

        self.q_scale_enc = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))
        self._initialize_weights()

    @staticmethod
    def get_q_scale(q_scale, index):
        if isinstance(index, Integral):
            # to restore indexed dimension
            index = index, None

        return q_scale[index]

    def forward(self, x, *, q_index=None, recon_only=False):
        _, _, H, W = x.size()
        curr_q_enc = self.get_q_scale(self.q_scale_enc, q_index)
        curr_q_dec = self.get_q_scale(self.q_scale_dec, q_index)

        y = self.enc(x, curr_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_raw, z_hat = self.quantize(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        y_raw, y_hat = self.calc_four_part_prior(
            y,
            params,
            self.y_spatial_prior_adaptor_1,
            self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3,
            self.y_spatial_prior,
            y_spatial_prior_reduction=self.y_spatial_prior_reduction,
        )

        x_hat = self.dec(y_hat, curr_q_dec)
        if not self.training:
            x_hat = x_hat.clamp_(0.0, 1.0)
        if recon_only:
            return x_hat

        bits_y = self.get_y_bits_estimate(y_raw)

        z_index = q_index if self.bit_estimator_qp_split else None
        bits_z = self.get_z_bits(z_raw, self.bit_estimator_z, z_index)
        bits_z = bits_z.flatten(1).sum(dim=-1)
        bits = bits_y + bits_z

        pixel_num = H * W
        bpp_y = bits_y / pixel_num
        bpp_z = bits_z / pixel_num
        bpp = bits / pixel_num

        return {
            "x_hat": x_hat,
            "bits": bits,
            "bpp": bpp,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
        }

    def compress(self, x, q_index, *, calc_bits_estimates=False):
        curr_q_enc = self.get_q_scale(self.q_scale_enc, q_index)
        curr_q_dec = self.get_q_scale(self.q_scale_dec, q_index)

        y = self.enc(x, curr_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_q = torch.clamp(torch.round(z), -128.0, 127.0)
        z_hat = z_q

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        y_raw, y_hat = self.calc_four_part_prior(
            y,
            params,
            self.y_spatial_prior_adaptor_1,
            self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3,
            self.y_spatial_prior,
            y_spatial_prior_reduction=self.y_spatial_prior_reduction,
        )

        x_hat = self.dec(y_hat, curr_q_dec).clamp_(0, 1)

        streams = open_encoder_streams(x_hat.size(0))

        z_index = q_index
        self.encode_y(streams, y_raw)
        self.bit_estimator_z.encode_z(streams, z_hat, z_index)

        bit_stream = flush_encoder_streams(streams)

        result = dict(
            x_hat=x_hat,
            bit_stream=bit_stream,
        )

        if calc_bits_estimates:
            bits_y = self.get_y_bits_estimate(y_raw)

            bits_z = self.get_z_bits(z_hat, self.bit_estimator_z, z_index)
            bits_z = bits_z.flatten(1).sum(dim=-1)

            result.update(
                bits_estimate_y=bits_y,
                bits_estimate_z=bits_z,
                bits_estimate=bits_y + bits_z,
            )

        return result

    def decompress(self, bit_stream, q_index, *, width, height):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        curr_q_dec = self.get_q_scale(self.q_scale_dec, q_index)

        z_size = get_downsampled_shape(height, width, 64)
        y_height, y_width = get_downsampled_shape(height, width, 16)
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)

        streams = open_decoder_streams(bit_stream)
        z_hat = self.bit_estimator_z.decode_z(streams, z_size, q_index).to(dtype=dtype, device=device)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        y_hat = self.decompress_four_part_prior(
            streams,
            params,
            self.y_spatial_prior_adaptor_1,
            self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3,
            self.y_spatial_prior,
            y_spatial_prior_reduction=self.y_spatial_prior_reduction,
        )
        check_decoder_eof(streams)
        del streams

        x_hat = self.dec(y_hat, curr_q_dec).clamp_(0, 1)
        return dict(x_hat=x_hat)
