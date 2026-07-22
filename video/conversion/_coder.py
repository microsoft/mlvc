# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import numpy as np
from .types import BasePmf, GaussianCoderPmf, BitEstimatorPmf
from msrtc.rans import EntropyEncoder, EntropyDecoder, RansEncoderStream, RansDecoderStream


class _CoderBase:
    def __init__(self, pmf: BasePmf) -> None:
        self._pmf_lengths = np.array(pmf.pmf_lengths, dtype=np.int32)
        self._pmf_offsets = np.array(pmf.pmf_offsets, dtype=np.int32)
        self._pmf_table = np.array(pmf.pmf_table, dtype=np.int32)
        self._entropy_encoder = None
        self._entropy_decoder = None

    @property
    def entropy_encoder(self) -> EntropyEncoder:
        if self._entropy_encoder is None:
            self._entropy_encoder = EntropyEncoder(
                pmfLengths=self._pmf_lengths,
                pmfOffsets=self._pmf_offsets,
                pmfTable=self._pmf_table,
                symbolBits=16,
                bypassBits=2,
            )
        return self._entropy_encoder

    @property
    def entropy_decoder(self) -> EntropyDecoder:
        if self._entropy_decoder is None:
            self._entropy_decoder = EntropyDecoder(
                pmfLengths=self._pmf_lengths,
                pmfOffsets=self._pmf_offsets,
                pmfTable=self._pmf_table,
                symbolBits=16,
                bypassBits=2,
            )
        return self._entropy_decoder


class GaussianEncoder(_CoderBase):
    def __init__(self, pmf: GaussianCoderPmf) -> None:
        super().__init__(pmf)
        self._scale_min = pmf.scale_min
        self._scale_max = pmf.scale_max
        self._scale_levels = pmf.scale_levels
        self._index_space = pmf.index_space
        self._log_scale_min = math.log(self._scale_min)
        self._log_scale_max = math.log(self._scale_max)
        self._log_scale_step = (self._log_scale_max - self._log_scale_min) / (self._scale_levels - 1)

    def _build_indices(self, scales: np.ndarray) -> np.ndarray:
        scales = scales.reshape(-1)
        indices = (np.log(scales) - self._log_scale_min) / self._log_scale_step
        indices = indices.astype(np.int32).clip(0, self._scale_levels - 1)
        return indices

    def encode_y(self, stream: RansEncoderStream, x: np.ndarray, scales: np.ndarray) -> None:
        if self._index_space:
            indices = scales.astype(np.int32).reshape(-1)
        else:
            indices = self._build_indices(scales)
        symbols = x.reshape(-1).astype(np.int32)
        self.entropy_encoder.push(stream, indices, symbols)

    def decode_y(self, stream: RansDecoderStream, scales: np.ndarray) -> np.ndarray:
        if self._index_space:
            indices = scales.astype(np.int32).reshape(-1)
        else:
            indices = self._build_indices(scales)
        values = np.zeros_like(indices)
        self.entropy_decoder.decode(values, indices, stream)
        return values.reshape(scales.shape)


class BitEstimator(_CoderBase):
    def __init__(self, pmf: BitEstimatorPmf) -> None:
        super().__init__(pmf)
        self._qp_num = pmf.qp_num
        self._channel = pmf.channels
        if self._qp_num * self._channel != len(self._pmf_lengths):
            raise ValueError(f"Invalid pmf data size {self._qp_num=}, {self._channel=}, {len(self._pmf_lengths)=}")

    def _build_indices(self, size: tuple[int, int, int, int], qp: int) -> np.ndarray:
        if qp < 0 or qp >= self._qp_num:
            raise ValueError(f"qp must be in [0, {self._qp_num})")
        B, C, H, W = size
        indices = np.arange(C, dtype=np.int32).reshape((1, -1, 1, 1)) + qp * self._channel
        return np.tile(indices, (B, 1, H, W))

    def encode_z(self, stream: RansEncoderStream, x: np.ndarray, qp: int) -> None:
        indices = self._build_indices(x.shape, qp).flatten()
        values = x.flatten().astype(np.int32)
        return self.entropy_encoder.push(stream, indices, values)

    def decode_z(self, stream: RansDecoderStream, size: tuple[int, int], qp: int) -> np.ndarray:
        out_size = 1, self._channel, *size
        indices = self._build_indices(out_size, qp).flatten()
        values = np.zeros_like(indices)
        self.entropy_decoder.decode(values, indices, stream)
        return values.reshape(out_size)
