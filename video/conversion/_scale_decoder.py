# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

from .types import ScaleDecoderType


SCALE_DECODER_FILENAMES: dict[str, str] = {}


class BaseScaleDecoder(ABC):
    def __init__(
        self,
        *,
        type: str,
        y_shape: tuple[int, int, int],
        index_space: bool,
        scale_max_idx: int,
    ) -> None:
        self._type = type
        self._y_shape = y_shape
        self._index_space = index_space
        self._scale_max_idx = scale_max_idx

    @property
    def type(self) -> str:
        return self._type

    @abstractmethod
    def extract_scales(self, z_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass

    def _normalize_and_pack(self, y_scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, y_height, y_width = self._y_shape
        y_scales = y_scales[:, :, :y_height, :y_width]

        if self._index_space:
            y_scales = np.clip(y_scales, 0, self._scale_max_idx)

        mask_0, mask_1 = _get_mask_dual(self._y_shape)
        x1, x2 = np.split(mask_0 * y_scales, 2, axis=1)
        y1, y2 = np.split(mask_1 * y_scales, 2, axis=1)
        scales_0 = x1 + x2
        scales_1 = y1 + y2
        return scales_0, scales_1


class UpsampleScaleDecoder(BaseScaleDecoder):
    def __init__(self, *, channel_repeat: int, spatial_repeat: int, **kwargs) -> None:
        super().__init__(type=ScaleDecoderType.UPSAMPLE, **kwargs)
        if not self._index_space:
            raise ValueError("UpsampleScaleDecoder only supports index_space=True")
        latent_channels = self._y_shape[0]
        assert latent_channels % channel_repeat == 0
        self._channel_repeat = channel_repeat
        self._spatial_repeat = spatial_repeat
        self._base_channels = latent_channels // channel_repeat

    def extract_scales(self, z_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        latent_channels = self._y_shape[0]
        spatial_repeat = self._spatial_repeat

        # Extract and upsample scales
        y_scales = z_raw[:, : self._base_channels, :, :].astype(np.float32, copy=False)
        y_scales = np.abs(y_scales)
        y_scales = np.expand_dims(y_scales, axis=(2, 4, 6))
        y_scales = np.repeat(y_scales, self._channel_repeat, axis=2)
        y_scales = np.repeat(y_scales, spatial_repeat, axis=4)
        y_scales = np.repeat(y_scales, spatial_repeat, axis=6)
        y_scales = y_scales.reshape(
            z_raw.shape[0],
            latent_channels,
            z_raw.shape[-2] * spatial_repeat,
            z_raw.shape[-1] * spatial_repeat,
        )

        # Normalize and pack
        return self._normalize_and_pack(y_scales)


def scale_decoder_factory(
    type: str,
    model_path: Path | str,
    filename: str | None = None,
    *,
    y_shape: tuple[int, int, int],
    index_space: bool,
    scale_max_idx: int,
    channel_repeat: int | None = None,
    spatial_repeat: int | None = None,
) -> BaseScaleDecoder:
    if type == ScaleDecoderType.UPSAMPLE:
        assert channel_repeat is not None and spatial_repeat is not None
        return UpsampleScaleDecoder(
            y_shape=y_shape,
            index_space=index_space,
            scale_max_idx=scale_max_idx,
            channel_repeat=channel_repeat,
            spatial_repeat=spatial_repeat,
        )

    try:
        from ._scale_decoder_ext import scale_decoder_factory_ext
    except ImportError as e:
        raise ValueError(f"Unsupported scale decoder type: {type}") from e

    return scale_decoder_factory_ext(
        type,
        model_path,
        filename,
        y_shape=y_shape,
        index_space=index_space,
        scale_max_idx=scale_max_idx,
    )


_MASK_CACHE = {}


def _get_mask_dual(size, dtype=np.float32):
    def _get_one_channel_dual_mask(height, width, dtype):
        micro_mask_0 = np.array([[1, 0], [0, 1]], dtype=dtype)
        mask_0 = np.tile(micro_mask_0, ((height + 1) // 2, (width + 1) // 2))[:height, :width]
        mask_0 = np.expand_dims(mask_0, axis=(0, 1))

        micro_mask_1 = np.array([[0, 1], [1, 0]], dtype=dtype)
        mask_1 = np.tile(micro_mask_1, ((height + 1) // 2, (width + 1) // 2))[:height, :width]
        mask_1 = np.expand_dims(mask_1, axis=(0, 1))
        return mask_0, mask_1

    if (size, dtype) in _MASK_CACHE:
        return _MASK_CACHE[(size, dtype)]

    batch = 1
    channel, height, width = size
    assert channel % 2 == 0

    m = np.ones((batch, channel // 2, height, width), dtype=dtype)
    m0, m1 = _get_one_channel_dual_mask(height, width, dtype)
    mask_0 = np.concatenate([m * m0, m * m1], axis=1)
    mask_1 = np.concatenate([m * m1, m * m0], axis=1)
    _MASK_CACHE[(size, dtype)] = mask_0, mask_1
    return mask_0, mask_1
