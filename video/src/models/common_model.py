# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Literal, Optional, overload

import torch
from torch import nn

from .entropy_models import BitEstimator, GaussianEncoder
from .utils import apply_upper_lower_bound
from ..utils.onnx_compat import replicate_pad
from ..utils.stream_helper import get_padding_size


class CompressionModel(nn.Module):
    q_log_base: torch.Tensor
    q_log_scale: torch.Tensor
    bit_estimator_z: BitEstimator

    def __init__(
        self,
        *,
        qp_num: int = 64,
        extra_qp: int = 0,
        y_distribution,
        z_channel,
        mv_z_channel=None,
        bit_estimator_qp_split=False,
        y_distribution_scale_step_in_log_space=True,
        y_distribution_scale_input_in_index_space=False,
        quant_min: Optional[float] = None,
        quant_max: Optional[float] = None,
    ):
        super().__init__()

        self._qp_num = qp_num
        self._extra_qp = extra_qp
        self.y_distribution = y_distribution
        self.z_channel = z_channel
        self.mv_z_channel = mv_z_channel
        self.force_zero_thres = None

        self.bit_estimator_qp_split = bit_estimator_qp_split
        bit_estimator_qp_number = self.get_qp_num() + extra_qp if bit_estimator_qp_split else 1
        if z_channel > 0:
            self.bit_estimator_z = BitEstimator(bit_estimator_qp_number, z_channel)
        else:
            self.bit_estimator_z = None  # type: ignore[assignment]

        if mv_z_channel is not None:
            self.bit_estimator_z_mv = BitEstimator(bit_estimator_qp_number, mv_z_channel)
        else:
            self.bit_estimator_z_mv = None

        if quant_min is None:
            if quant_max is None:
                quant_max = 32
            quant_min = 1 / quant_max
        elif quant_max is None:
            quant_max = 1 / quant_min
        q_log_min = torch.log(torch.tensor(quant_min))
        q_log_max = torch.log(torch.tensor(quant_max))
        self.register_buffer("q_log_base", (q_log_max + q_log_min) / 2, False)
        self.register_buffer("q_log_scale", (q_log_max - q_log_min) / 2, False)

        self.gaussian_encoder = GaussianEncoder(
            distribution=y_distribution,
            scale_step_in_log_space=y_distribution_scale_step_in_log_space,
            scale_input_in_index_space=y_distribution_scale_input_in_index_space,
        )
        self.force_zero_thres = None
        self.noise_level = 0.5

        self.masks = {}
        self.force_generate_mask = False

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, 1.0)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

    def set_force_zero_thres(self, scale_thres):
        self.force_zero_thres = scale_thres

    def set_force_generate_mask(self, force):
        self.force_generate_mask = force

    def get_force_generate_mask(self):
        return self.force_generate_mask

    def get_z_pmf(self):
        assert self.bit_estimator_z is not None
        return self.bit_estimator_z.get_pmf()

    def get_mv_z_pmf(self):
        assert self.bit_estimator_z_mv is not None
        return self.bit_estimator_z_mv.get_pmf()

    def get_y_pmf(self):
        return self.gaussian_encoder.get_pmf()

    def quantize(self, x):
        """
        "Quantize" tensor according to the current settings.
        In eval mode entropy coder inputs and outputs are true quantized values.

        Returns:
              (inputs, outputs) - entropy coder inputs (for bits estimates) and outputs (for video decoding)
        """

        x_q = torch.round(x)
        if not self.training:
            return x_q, x_q

        x_raw = self.add_noise(x)
        x_hat = x + (x_q - x).detach()
        return x_raw, x_hat

    def get_one_q_scale(self, q_gauge, q_index):
        q_scale = self.q_log_base + self.q_log_scale * torch.tanh(q_gauge)
        min_q = q_scale[0]
        max_q = q_scale[1]
        step = (max_q - min_q) / (self.get_qp_num() - 1)
        q = torch.exp(min_q + step * q_index)
        if q.ndim == 0:
            return q[None, None, None, None]
        else:
            return q[:, None, None, None]

    def get_curr_q(self, q_guage, q_basic, q_index):
        q_step = self.get_one_q_scale(q_guage, q_index)
        if q_basic is None:
            return q_step
        return q_step * q_basic

    def get_qp_num(self):
        return self._qp_num

    def prepare_for_eval(self, config: dict | None = None):
        """Return a model optimized for eval. May return a copy."""
        return self

    def set_noise_level(self, noise_level):
        self.noise_level = noise_level

    def set_use_ckpt(self, use_ckpt=True, recon_module=False, hyper_module=False):
        pass

    def clear_grad_for_y_q(self):
        pass

    def get_noise_level(self):
        return self.noise_level

    def add_noise(self, x):
        noise = torch.nn.init.uniform_(torch.empty_like(x), -self.noise_level, self.noise_level)
        return x + noise.detach()

    @staticmethod
    def add_specified_noise(x, n):
        noise = torch.nn.init.uniform_(torch.empty_like(x), -n, n)
        return x + noise.detach()

    @staticmethod
    def probs_to_bits(probs):
        factor = -1.0 / math.log(2.0)
        bits = torch.log(probs + 1e-5) * factor
        bits = apply_upper_lower_bound(bits, lower=0.0)
        return bits

    def get_z_bits(self, z, bit_estimator, index):
        if self.training:
            probs = bit_estimator.get_prob(z, index)
        else:
            probs = bit_estimator.get_cdf(z + 0.5, index) - bit_estimator.get_cdf(z - 0.5, index)
            probs = probs.to(torch.float32)
        return CompressionModel.probs_to_bits(probs)

    def build_encoder_pmf(self):
        self.gaussian_encoder.build_pmf()
        assert self.bit_estimator_z is not None
        self.bit_estimator_z.build_pmf()
        if self.bit_estimator_z_mv is not None:
            self.bit_estimator_z_mv.build_pmf()

    def reset_encoder_pmf(self):
        self.gaussian_encoder.reset_pmf()
        assert self.bit_estimator_z is not None
        self.bit_estimator_z.reset_pmf()
        if self.bit_estimator_z_mv is not None:
            self.bit_estimator_z_mv.reset_pmf()

    def pad_for_y(self, y, *, z_factor=4):
        if self.training:
            return y, None

        _, _, H, W = y.size()
        padding = get_padding_size(H, W, z_factor)
        y = replicate_pad(y, pad=padding)

        return y, tuple(-x for x in padding)

    @staticmethod
    def get_to_y_slice_shape(height, width, *, z_factor=4):
        padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, z_factor)
        return -padding_l, -padding_r, -padding_t, -padding_b

    def slice_to_y(self, param, slice_shape):
        if self.training:
            return param

        return replicate_pad(param, pad=slice_shape)

    @overload
    @staticmethod
    def separate_prior(
        params: torch.Tensor, *, is_video: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    @staticmethod
    def separate_prior(
        params: torch.Tensor, *, is_video: Literal[False] = ...
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @staticmethod
    def separate_prior(params: torch.Tensor, *, is_video: bool = False):
        if is_video:
            quant_step, scales, means = params.chunk(3, 1)
            quant_step = apply_upper_lower_bound(quant_step, lower=0.5)
            return quant_step, scales, means
        else:
            q = params[:, :2, :, :]
            q_enc, q_dec = (torch.sigmoid(q) * 1.5 + 0.5).chunk(2, 1)
            scales, means = params[:, 2:, :, :].chunk(2, 1)
        return q_enc, q_dec, scales, means

    def get_mask(self, height, width, dtype, device):
        curr_mask_str = f"{width}x{height}"
        if curr_mask_str not in self.masks or self.force_generate_mask:
            micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
            mask_0 = micro_mask.repeat((height + 1) // 2, (width + 1) // 2)
            mask_0 = mask_0[:height, :width]
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_1 = torch.ones_like(mask_0) - mask_0
            self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    @overload
    def process_with_mask(
        self, y, scales, means, *, mask, chunks: int, return_y_q: Literal[False] = ...
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]: ...

    @overload
    def process_with_mask(
        self, y, scales, means, *, mask, chunks: int, return_y_q: Literal[True]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]: ...

    def process_with_mask(self, y, scales, means, *, mask, chunks: int, return_y_q: bool = False):
        y_res = self.pack_with_mask(y - means, mask, chunks)
        scales = self.pack_with_mask(scales, mask, chunks)

        y_raw, y_q = self.quantize(y_res)

        if not self.training and self.force_zero_thres is not None:
            cond = torch.less(scales, self.force_zero_thres)
            y_raw[cond] = 0
            scales[cond] = 0.01

        y_hat = self.unpack_with_mask(y_q, means, mask, chunks)

        if return_y_q:
            return (y_raw, scales), y_hat, y_q
        return (y_raw, scales), y_hat

    def encode_y(self, stream, y_raw):
        for y, scales in reversed(y_raw):
            self.gaussian_encoder.encode_y(stream, y, scales, skip_thres=self.force_zero_thres)

    def get_y_bits_estimate(self, y_raw) -> torch.Tensor:
        bits_y = None
        for y, scales in reversed(y_raw):
            bits = self.probs_to_bits(self.gaussian_encoder.get_probs(y, scales))
            bits = torch.sum(bits, dim=(1, 2, 3))
            if bits_y is None:
                bits_y = bits
            else:
                bits_y += bits

        assert bits_y is not None
        return bits_y

    def decode_y_with_mask(self, stream, means, scales, *, mask, chunks):
        scales = self.pack_with_mask(scales, mask, chunks)
        x = self.gaussian_encoder.decode_y(stream, scales, self.force_zero_thres)
        x = x.to(dtype=means.dtype, device=means.device)
        x = self.unpack_with_mask(x, means, mask, chunks)
        return x

    @staticmethod
    def pack_with_mask(x, mask, chunks: int):
        assert x.shape[1] % chunks == 0

        x = x * mask
        while chunks > 1:
            assert chunks % 2 == 0
            chunks //= 2

            x1, x2 = x.chunk(2, dim=1)
            x = x1 + x2

        return x

    @staticmethod
    def unpack_with_mask(x, means, mask, chunks: int):
        x = torch.cat([x] * chunks, dim=1)
        x = (x + means) * mask
        return x

    @staticmethod
    def get_one_channel_four_parts_mask(height, width, dtype, device):
        micro_mask_0 = torch.tensor(((1, 0), (0, 0)), dtype=dtype, device=device)
        mask_0 = micro_mask_0.repeat((height + 1) // 2, (width + 1) // 2)
        mask_0 = mask_0[:height, :width]
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)

        micro_mask_1 = torch.tensor(((0, 1), (0, 0)), dtype=dtype, device=device)
        mask_1 = micro_mask_1.repeat((height + 1) // 2, (width + 1) // 2)
        mask_1 = mask_1[:height, :width]
        mask_1 = torch.unsqueeze(mask_1, 0)
        mask_1 = torch.unsqueeze(mask_1, 0)

        micro_mask_2 = torch.tensor(((0, 0), (1, 0)), dtype=dtype, device=device)
        mask_2 = micro_mask_2.repeat((height + 1) // 2, (width + 1) // 2)
        mask_2 = mask_2[:height, :width]
        mask_2 = torch.unsqueeze(mask_2, 0)
        mask_2 = torch.unsqueeze(mask_2, 0)

        micro_mask_3 = torch.tensor(((0, 0), (0, 1)), dtype=dtype, device=device)
        mask_3 = micro_mask_3.repeat((height + 1) // 2, (width + 1) // 2)
        mask_3 = mask_3[:height, :width]
        mask_3 = torch.unsqueeze(mask_3, 0)
        mask_3 = torch.unsqueeze(mask_3, 0)

        return mask_0, mask_1, mask_2, mask_3

    def get_mask_four_parts(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.masks or self.force_generate_mask:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0, m1, m2, m3 = self.get_one_channel_four_parts_mask(height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    def calc_four_part_prior(
        self,
        y,
        common_params,
        y_spatial_prior_adaptor_1,
        y_spatial_prior_adaptor_2,
        y_spatial_prior_adaptor_3,
        y_spatial_prior,
        *,
        y_spatial_prior_reduction=None,
        q_enc=None,
        q_dec=None,
        q_spatial_adaptor_1=None,
        q_spatial_adaptor_2=None,
        q_spatial_adaptor_3=None,
        **kwargs,
    ):
        """
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        """
        if q_enc is None or q_dec is None:
            if y_spatial_prior_reduction is None:
                quant_step, scales, means = self.separate_prior(common_params, is_video=True)
                q_enc = 1.0 / quant_step
                q_dec = quant_step
            else:
                q_enc, q_dec, scales, means = self.separate_prior(common_params, is_video=False)
        else:
            scales, means = common_params.chunk(2, 1)

        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params, **kwargs)

        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        y = y * q_enc

        y_raw_0, y_hat_0 = self.process_with_mask(y, scales, means, mask=mask_0, chunks=4)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        if q_spatial_adaptor_1 is not None:
            params = params * q_spatial_adaptor_1
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params, **kwargs), **kwargs).chunk(2, 1)
        y_raw_1, y_hat_1 = self.process_with_mask(y, scales, means, mask=mask_1, chunks=4)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        if q_spatial_adaptor_2 is not None:
            params = params * q_spatial_adaptor_2
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params, **kwargs), **kwargs).chunk(2, 1)
        y_raw_2, y_hat_2 = self.process_with_mask(y, scales, means, mask=mask_2, chunks=4)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        if q_spatial_adaptor_3 is not None:
            params = params * q_spatial_adaptor_3
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params, **kwargs), **kwargs).chunk(2, 1)
        y_raw_3, y_hat_3 = self.process_with_mask(y, scales, means, mask=mask_3, chunks=4)

        y_hat = y_hat_so_far + y_hat_3
        y_hat = y_hat * q_dec

        y_raw = y_raw_0, y_raw_1, y_raw_2, y_raw_3
        return y_raw, y_hat

    def decompress_four_part_prior(
        self,
        stream,
        common_params,
        y_spatial_prior_adaptor_1,
        y_spatial_prior_adaptor_2,
        y_spatial_prior_adaptor_3,
        y_spatial_prior,
        *,
        y_spatial_prior_reduction=None,
        q_dec=None,
        q_spatial_adaptor_1=None,
        q_spatial_adaptor_2=None,
        q_spatial_adaptor_3=None,
        **kwargs,
    ):
        if q_dec is None:
            if y_spatial_prior_reduction is None:
                q_dec, scales, means = self.separate_prior(common_params, is_video=True)
            else:
                _, q_dec, scales, means = self.separate_prior(common_params, is_video=False)
        else:
            scales, means = common_params.chunk(2, 1)

        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params, **kwargs)

        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        y_hat_curr_step = self.decode_y_with_mask(stream, means, scales, mask=mask_0, chunks=4)
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        if q_spatial_adaptor_1 is not None:
            params = params * q_spatial_adaptor_1
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params, **kwargs), **kwargs).chunk(2, 1)
        y_hat_curr_step = self.decode_y_with_mask(stream, means, scales, mask=mask_1, chunks=4)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        if q_spatial_adaptor_2 is not None:
            params = params * q_spatial_adaptor_2
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params, **kwargs), **kwargs).chunk(2, 1)
        y_hat_curr_step = self.decode_y_with_mask(stream, means, scales, mask=mask_2, chunks=4)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        if q_spatial_adaptor_3 is not None:
            params = params * q_spatial_adaptor_3
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params, **kwargs), **kwargs).chunk(2, 1)
        y_hat_curr_step = self.decode_y_with_mask(stream, means, scales, mask=mask_3, chunks=4)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        y_hat = y_hat_so_far * q_dec

        return y_hat

    @staticmethod
    def get_one_channel_dual_mask(height, width, dtype, device):
        micro_mask_0 = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
        mask_0 = micro_mask_0.repeat((height + 1) // 2, (width + 1) // 2)
        mask_0 = mask_0[:height, :width]
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)

        micro_mask_1 = torch.tensor(((0, 1), (1, 0)), dtype=dtype, device=device)
        mask_1 = micro_mask_1.repeat((height + 1) // 2, (width + 1) // 2)
        mask_1 = mask_1[:height, :width]
        mask_1 = torch.unsqueeze(mask_1, 0)
        mask_1 = torch.unsqueeze(mask_1, 0)

        return mask_0, mask_1

    def get_mask_dual(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}x{str(dtype)}x{str(device)}"
        with torch.no_grad():
            if curr_mask_str not in self.masks or self.force_generate_mask:
                assert channel % 2 == 0
                m = torch.ones((batch, channel // 2, height, width), dtype=dtype, device=device)
                m0, m1 = self.get_one_channel_dual_mask(height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1), dim=1)
                mask_1 = torch.cat((m * m1, m * m0), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    def calc_dual_prior(self, y, common_params, y_spatial_prior, update_scales=True, **kwargs):
        quant_step, scales, means = self.separate_prior(common_params, is_video=True)
        q_enc = 1.0 / quant_step
        q_dec = quant_step
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_dual(B, C, H, W, dtype, device)

        y = y * q_enc
        y_raw_0, y_hat_0 = self.process_with_mask(y, scales, means, mask=mask_0, chunks=2)
        if update_scales:
            scales, means = y_spatial_prior(torch.cat((y_hat_0, common_params), dim=1), **kwargs).chunk(2, 1)
        else:
            means = y_spatial_prior(torch.cat((y_hat_0, common_params), dim=1), **kwargs)
        y_raw_1, y_hat_1 = self.process_with_mask(y, scales, means, mask=mask_1, chunks=2)

        y_hat = y_hat_0 + y_hat_1
        y_hat = y_hat * q_dec

        y_raw = y_raw_0, y_raw_1
        return y_raw, y_hat

    def decompress_dual_prior_torch(self, y_hat_0, params_0, update_scales=True, **kwargs):
        """NPU: Estimate second part of params (from first part of y_hat)"""
        params_1 = torch.cat((y_hat_0, params_0), dim=1)
        if update_scales:
            scales_1, means_1 = self.y_spatial_prior(params_1, **kwargs).chunk(2, 1)  # type: ignore[attr-defined]
        else:
            means_1 = self.y_spatial_prior(params_1, **kwargs)  # type: ignore[attr-defined]
            scales_1 = None

        return {
            "scales_1": scales_1,
            "means_1": means_1,
        }

    def decompress_dual_prior(
        self, stream, common_params, scales_r_0, means_0, quant_step, update_scales=True, **kwargs
    ):
        B, C, H, W = means_0.shape
        mask_0, mask_1 = self.get_mask_dual(B, C, H, W, means_0.dtype, means_0.device)
        params_0 = common_params

        y_hat_0 = self.decode_y_with_mask(stream, means_0, scales_r_0, mask=mask_0, chunks=2)
        part1 = self.decompress_dual_prior_torch(y_hat_0, params_0, update_scales=update_scales, **kwargs)
        if not update_scales:
            assert part1["scales_1"] is None
            part1["scales_1"] = scales_r_0
        y_hat_1 = self.decode_y_with_mask(stream, part1["means_1"], part1["scales_1"], mask=mask_1, chunks=2)

        y_hat = (y_hat_0 + y_hat_1) * quant_step
        return y_hat

    @property
    def total_qp_num(self) -> int:
        return self.get_qp_num() + self._extra_qp
