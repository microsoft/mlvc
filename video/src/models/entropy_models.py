# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import itertools
import math
from numbers import Integral
from typing import Optional, Sequence

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from src.models.utils import apply_upper_lower_bound

__all__ = ["quantize_pmf", "BitEstimator", "GaussianEncoder"]


def quantize_pmf(pmf, scale_bits: int):
    """
    Convert probability mass function to quantized cumulative distribution function
    1 frequency unit is left for out of distribution values

    :param pmf: probability mass function as 1-d array
    :param scale_bits: bits used by CDF function
    """
    pmf = np.asarray(pmf, dtype=np.float64)
    assert np.all(pmf >= 0)

    scale = 1 << scale_bits
    assert len(pmf) < scale

    # cumulative distribution function
    cdf = np.cumsum(pmf)
    # normalized to 1
    cdf /= cdf[-1]
    # quantized to scale
    cdf = np.round(cdf * scale).astype(np.int32)

    # back to quantized pmf
    quantized = cdf - np.pad(cdf[:-1], (1, 0))

    # count zeros after quantization
    zeros = np.equal(quantized, 0)
    n_zeros = sum(zeros)

    if n_zeros > 0:
        # set all 0 to 1
        quantized[zeros] = 1

        #  and borrow frequency from low frequency, further from center elements
        while n_zeros > 0:
            # can borrow from those
            candidates = np.flatnonzero(quantized > 1)
            n_candidates = len(candidates)
            assert n_candidates > 0
            if n_zeros >= n_candidates:
                d = n_zeros // n_candidates
                if d > 1:
                    d = min(d, quantized[candidates].min() - 1)

                quantized[candidates] -= d
                # repeat if there was not enough candidates
                n_zeros -= d * n_candidates
            else:
                # weights are current mass, original pmf and minus distance from center (favoring left) in that order
                weights = -np.abs(candidates - len(quantized) / 2 + 0.75), pmf[candidates], quantized[candidates]
                # sort by weight and take best candidates
                indices = np.lexsort(weights)[:n_zeros]
                quantized[candidates[indices]] -= 1
                # done
                break

    return quantized


def quantize_pmf_set(pmf, tail_mass, pmf_length):
    entropy_coder_precision = 16

    pmf_list = list()
    for p, t, length in itertools.zip_longest(pmf, tail_mass, pmf_length):
        p = np.concatenate((p[:length], t), axis=0)
        q = quantize_pmf(p, entropy_coder_precision)
        pmf_list.append(q)

    return pmf_length + 1, np.concatenate(pmf_list)


class Bitparm(nn.Module):
    def __init__(self, qp_num, channel, final=False):
        super().__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty([qp_num, channel, 1, 1]), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty([qp_num, channel, 1, 1]), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty([qp_num, channel, 1, 1]), 0, 0.01))
        else:
            self.a = None

    def forward(self, x, index):
        h = self.h
        b = self.b

        if index is not None:
            if isinstance(index, Integral):
                # to restore indexed dimension
                index = index, None

            h = h[index]
            b = b[index]

        x = x * F.softplus(h) + b
        if self.final:
            return x

        a = self.a
        assert a is not None
        if index is not None:
            a = a[index]
        return x + torch.tanh(x) * torch.tanh(a)


class AEHelper(abc.ABC):
    def __init__(self):
        super().__init__()
        self._pmf_offsets = None
        self._pmf_lengths = None
        self._quantized_pmf = None
        self._entropy_encoder = None
        self._entropy_decoder = None

    @abc.abstractmethod
    def build_pmf(self):
        raise NotImplementedError()

    def set_pmf(self, pmf_lengths, pmf_offsets, quantized_pmf):
        self._pmf_lengths = pmf_lengths
        self._quantized_pmf = quantized_pmf
        self._pmf_offsets = pmf_offsets
        self._entropy_encoder = self._entropy_decoder = None

    def reset_pmf(self):
        self.set_pmf(None, None, None)

    def get_pmf(self):
        return self._pmf_lengths, self._pmf_offsets, self._quantized_pmf

    @property
    def entropy_encoder(self):
        encoder = self._entropy_encoder
        if encoder is None:
            if self._quantized_pmf is None:
                self.build_pmf()
                if self._quantized_pmf is None:
                    raise ValueError("PMF tables are not initialized")

            from msrtc.rans import EntropyEncoder

            self._entropy_encoder = encoder = EntropyEncoder(
                pmfLengths=self._pmf_lengths,
                pmfOffsets=self._pmf_offsets,
                pmfTable=self._quantized_pmf,
                symbolBits=16,
                bypassBits=2,
            )
        return encoder

    def _encode(self, stream, indices: torch.Tensor, values: torch.Tensor):
        if not isinstance(stream, Sequence):
            self.entropy_encoder.push(stream, indices.flatten().numpy(), values.flatten().numpy())
        else:
            for s, i, v in itertools.zip_longest(stream, indices, values):
                self.entropy_encoder.push(s, i.flatten().numpy(), v.flatten().numpy())

    @property
    def entropy_decoder(self):
        decoder = self._entropy_decoder
        if decoder is None:
            if self._quantized_pmf is None:
                self.build_pmf()
                if self._quantized_pmf is None:
                    raise ValueError("PMF tables are not initialized")

            from msrtc.rans import EntropyDecoder

            self._entropy_decoder = decoder = EntropyDecoder(
                pmfLengths=self._pmf_lengths,
                pmfOffsets=self._pmf_offsets,
                pmfTable=self._quantized_pmf,
                symbolBits=16,
                bypassBits=2,
            )
        return decoder

    def _decode(self, values: torch.Tensor, indices: torch.Tensor, stream):
        if not isinstance(stream, Sequence):
            self.entropy_decoder.decode(values.view(-1).numpy(), indices.view(-1).numpy(), stream)
        else:
            for v, i, s in itertools.zip_longest(values, indices, stream):
                self.entropy_decoder.decode(v.view(-1).numpy(), i.view(-1).numpy(), s)


class BitEstimator(AEHelper, nn.Module):
    def __init__(self, qp_num, channel):
        super().__init__()
        self.f1 = Bitparm(qp_num, channel)
        self.f2 = Bitparm(qp_num, channel)
        self.f3 = Bitparm(qp_num, channel)
        self.f4 = Bitparm(qp_num, channel, True)
        self.qp_num = qp_num
        self.channel = channel

    def forward(self, x, index):
        return self.get_cdf(x, index)

    def get_logits_cdf(self, x, index):
        x = self.f1(x, index)
        x = self.f2(x, index)
        x = self.f3(x, index)
        x = self.f4(x, index)
        return x

    def get_cdf(self, x, index):
        return torch.sigmoid(self.get_logits_cdf(x, index))

    def get_prob(self, x, index):
        lower = self.get_cdf(x - 0.5, index)
        upper = self.get_cdf(x + 0.5, index)
        prob = upper - lower
        prob = apply_upper_lower_bound(prob, lower=1e-9)
        return prob

    def build_pmf(self):
        with torch.no_grad():
            device = next(self.parameters()).device
            medians = torch.zeros((self.qp_num, self.channel, 1, 1), device=device)
            index = torch.arange(self.qp_num, device=device, dtype=torch.int32)

            minima = medians + 8
            for i in range(8, 1, -1):
                samples = torch.zeros_like(medians) - i
                probs = self.forward(samples, index)
                minima = torch.where(torch.less(probs, 0.0001), i, minima)

            maxima = medians + 8
            for i in range(8, 1, -1):
                samples = torch.zeros_like(medians) + i
                probs = self.forward(samples, index)
                maxima = torch.where(torch.greater(probs, 0.9999), i, maxima)

            minima = minima.int()
            maxima = maxima.int()

            pmf_offsets = minima

            pmf_start = medians - minima
            pmf_length = maxima + minima + 1

            max_length = pmf_length.max()
            device = pmf_start.device
            samples = torch.arange(max_length.item(), device=device)

            samples = samples[None, None, None, :] + pmf_start

            half = float(0.5)

            lower = self.forward(samples - half, index)
            upper = self.forward(samples + half, index)
            pmf = upper - lower

            pmf = pmf[:, :, 0, :]
            upper = self.forward(maxima.to(torch.float32), index)
            tail_mass = lower[:, :, 0, :1] + (1.0 - upper[:, :, 0, -1:])

            pmf = pmf.flatten(0, -2).cpu().numpy()
            tail_mass = tail_mass.flatten(0, -2).cpu().numpy()
            pmf_length = pmf_length.flatten().cpu().numpy()
            pmf_offsets = pmf_offsets.flatten().cpu().numpy()
            pmf_length, pmf = quantize_pmf_set(pmf, tail_mass, pmf_length)
            self.set_pmf(pmf_length, pmf_offsets, pmf)

    def build_indices(self, size, qp):
        B, C, H, W = size
        if isinstance(qp, torch.Tensor):
            qp = qp.cpu().to(torch.int32)
            if qp.ndim > 0:
                qp = qp[:, None, None, None]
        indices = torch.arange(C, dtype=torch.int32).view(1, -1, 1, 1) + qp * self.channel
        if indices.size(0) > 1:
            B = 1
        return indices.repeat(B, 1, H, W)

    def encode_z(self, stream, x: torch.Tensor, qp):
        indices = self.build_indices(x.shape, qp)
        values = x.to(torch.int32).cpu()
        self._encode(stream, indices, values)

    def decode_z(self, stream, size, qp):
        B = len(stream) if isinstance(stream, Sequence) else 1
        size = B, self.channel, *size

        indices = self.build_indices(size, qp)
        values = torch.empty_like(indices)
        self._decode(values, indices, stream)
        return values


class GaussianEncoder(AEHelper):
    def __init__(self, distribution, *, scale_step_in_log_space=True, scale_input_in_index_space=False):
        super().__init__()
        assert distribution in ["laplace", "gaussian"]
        self.distribution = distribution
        if distribution == "laplace":
            self.cdf_distribution = torch.distributions.laplace.Laplace
            self.scale_min = 0.01
            self.scale_max = 64.0
            self.scale_level = 256
        elif distribution == "gaussian":
            self.cdf_distribution = torch.distributions.normal.Normal
            self.scale_min = 0.11
            self.scale_max = 16.0
            self.scale_level = 128  # <= 256
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        self.scale_step_in_log_space = scale_step_in_log_space
        self.scale_table = self.get_scale_table(
            self.scale_min, self.scale_max, self.scale_level, self.scale_step_in_log_space
        )
        self.scale_step = (self.scale_max - self.scale_min) / (self.scale_level - 1)

        self.log_scale_min = math.log(self.scale_min)
        self.log_scale_max = math.log(self.scale_max)
        self.log_scale_step = (self.log_scale_max - self.log_scale_min) / (self.scale_level - 1)

        self.scale_input_in_index_space = scale_input_in_index_space

    @staticmethod
    def get_scale_table(min_val, max_val, levels, log_steps):
        if log_steps:
            return torch.exp(torch.linspace(math.log(min_val), math.log(max_val), levels))
        else:
            return torch.linspace(min_val, max_val, levels)

    def get_probs(self, values, scales):
        if self.scale_input_in_index_space:
            scales = self.convert_indices_to_scales(scales)

        scales = apply_upper_lower_bound(scales, lower=self.scale_min, upper=self.scale_max)
        if self.distribution == "laplace":
            return self.get_laplace_prob(values, scales)
        else:
            return self.get_gaussian_prob(values, scales)

    @staticmethod
    def get_gaussian_prob(values, scales):
        # noinspection PyUnusedLocal
        def _standardized_cumulative(inputs):
            half = float(0.5)
            const = float(-(2**-0.5))
            # Using the complementary error function maximizes numerical precision.
            return half * torch.erfc(const * inputs)

        def _cdf2(inputs):
            const = float(-(2**-0.5))
            return torch.erfc(const * inputs)

        values = torch.abs(values)
        upper = _cdf2((0.5 - values) / scales)
        lower = _cdf2((-0.5 - values) / scales)
        prob = upper - lower
        prob = apply_upper_lower_bound(0.5 * prob, lower=1e-9)
        return prob

    @staticmethod
    def get_laplace_prob(values, scales):
        # noinspection PyUnusedLocal
        def _cdf(inputs):
            # this is the original function of cdf, but we only care diffence of cdf
            return 0.5 + 0.5 * torch.sign(inputs) * (1.0 - torch.exp(-torch.abs(inputs)))

        def _cdf2(inputs):
            return torch.sign(inputs) * (1.0 - torch.exp(-torch.abs(inputs)))

        upper = _cdf2((values + 0.5) / scales)
        lower = _cdf2((values - 0.5) / scales)
        prob = upper - lower
        prob = apply_upper_lower_bound(0.5 * prob, lower=1e-9)
        return prob

    def build_pmf(self):
        pmf_center = torch.zeros_like(self.scale_table) + 8
        scales = torch.zeros_like(pmf_center) + self.scale_table
        cdf_distribution = self.cdf_distribution(0.0, scales)
        for i in range(8, 1, -1):
            samples = torch.zeros_like(pmf_center) + i
            probs = cdf_distribution.cdf(samples)
            probs = torch.squeeze(probs)
            pmf_center = torch.where(torch.greater(probs, 0.9999), i, pmf_center)

        pmf_center = pmf_center.int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.arange(max_length, device=device) - pmf_center[:, None]
        samples = samples.float()

        scales = torch.zeros_like(samples) + self.scale_table[:, None]
        cdf_distribution = self.cdf_distribution(0.0, scales)

        upper = cdf_distribution.cdf(samples + 0.5)
        lower = cdf_distribution.cdf(samples - 0.5)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        pmf_center = pmf_center.numpy()
        pmf_length, quantized_pmf = quantize_pmf_set(pmf.numpy(), tail_mass.numpy(), pmf_length.numpy())

        self.set_pmf(pmf_length, pmf_center, quantized_pmf)

    def convert_scales_to_indices(self, scales, clamp=True):
        if clamp:
            scales = apply_upper_lower_bound(scales, lower=self.scale_min, upper=self.scale_max)

        if self.scale_step_in_log_space:
            indices = (torch.log(scales) - self.log_scale_min) / self.log_scale_step
        else:
            indices = (scales - self.scale_min) / self.scale_step

        return indices

    def convert_indices_to_scales(self, indices):
        if self.scale_step_in_log_space:
            return torch.exp(self.log_scale_min + self.log_scale_step * indices)
        else:
            return self.scale_min + self.scale_step * indices

    def build_indices(self, scales, skip_thres=None, calibration_indices: Optional[torch.Tensor] = None):
        if not self.scale_input_in_index_space:
            indices = self.convert_scales_to_indices(scales, clamp=False)
        else:
            if calibration_indices is not None:
                raise ValueError("calibration indices are not supported with scales passed in index space")
            if skip_thres is not None:
                raise ValueError("skip threshold is not supported with scales passed in index space")
            indices = scales

        if calibration_indices is not None:
            indices_ = indices.clamp_(0, self.scale_level - 1)
            indices = indices_.to(torch.int32)
            indices[calibration_indices] = torch.round(indices_[calibration_indices]).to(torch.int32)
        else:
            indices = indices.to(torch.int32).clamp_(0, self.scale_level - 1)

        if skip_thres is not None:
            indices[scales > skip_thres] = -1
        return indices

    def encode_y(self, stream, x, scales, skip_thres=None, indices=None):
        symbols = x.to(torch.int32).cpu()
        if indices is None:
            indices = self.build_indices(scales, skip_thres).cpu()

        self._encode(stream, indices, symbols)

    def decode_y(self, stream, scales, skip_thres=None, indices=None):
        if indices is None:
            indices = self.build_indices(scales, skip_thres).cpu()

        values = torch.empty_like(indices)
        self._decode(values, indices, stream)
        return values
