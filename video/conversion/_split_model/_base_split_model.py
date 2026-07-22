# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod
from typing import Any
from .._full_model import BaseFullModel
from ..types import (
    ModelPartId,
    ModelParams,
    EncoderOutput,
    DecoderOutput,
    SplitModelParams,
    RuntimeParams,
    ModelPartMetadata,
    ModelPrecision,
    ModelType,
    ConversionMetadata,
    PaddingMode,
    PaddingDirection,
)
from .._model_wrapper import ModelWrapper
from .._scale_decoder import BaseScaleDecoder
from .._coder import GaussianEncoder, BitEstimator
from ..utils import yuv_444_to_420, yuv_420_to_444
from src.utils.stream_helper import get_padding_size, get_downsampled_shape
from msrtc.rans import RansEncoderStream, RansDecoderStream


def prepare_model_parts(torch_model_parts, runtime_params):
    model_parts = {}
    for model_part_id in torch_model_parts.keys():
        model = torch_model_parts[model_part_id]
        model.eval()

        model_part_metadata = ModelPartMetadata(
            precision=ModelPrecision.FP32,
            input_fields=list(model.InputType._fields),
            output_fields=list(model.OutputType._fields),
        )

        model_parts[model_part_id] = ModelWrapper(
            model_type=ModelType.TORCH,
            model=model,
            metadata=model_part_metadata,
            runtime_params=runtime_params,
        )
    return model_parts


class DynamicDimension:
    def __init__(self, height, width, downsample_latent: int, downsample_hyperprior: int):
        self.y_h, self.y_w = height // downsample_latent, width // downsample_latent
        self.y_slice_shape = tuple(-x for x in get_padding_size(self.y_h, self.y_w, downsample_hyperprior))


class BaseModelPart(ABC, torch.nn.Module):
    model: Any
    height: int
    width: int
    dims: DynamicDimension

    def __init__(
        self,
        model: torch.nn.Module,
        height: int,
        width: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.height = height
        self.width = width
        self.dims = DynamicDimension(
            height,
            width,
            self.model.model_params.downsample_latent,
            self.model.model_params.downsample_hyperprior,
        )


class BaseEncoderPart(BaseModelPart):
    pass


class BaseDecoderPart(BaseModelPart):
    mask_0: torch.Tensor
    mask_1: torch.Tensor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        p = next(iter(self.model.parameters()))
        mask = self.model.get_mask_dual(
            1,
            self.model.model_params.latent_channels,
            self.dims.y_h,
            self.dims.y_w,
            p.dtype,
            p.device,
        )
        self.mask_0 = mask[0].clone()
        self.mask_1 = mask[1].clone()

    def half(self):
        self.mask_0 = self.mask_0.half()
        self.mask_1 = self.mask_1.half()
        return super().half()


@dataclass(kw_only=True)
class BaseReferenceData:
    ltr_frame: bool = False
    ref_feature: np.ndarray
    ref_exists: bool = False


class BaseReferenceManager(ABC):
    def __init__(
        self,
        model_width: int,
        model_height: int,
        feature_channels: int,
        downsample_feature: int,
        use_decoder: bool = True,
        dtype: str = "float32",
    ) -> None:
        self._model_width = model_width
        self._model_height = model_height
        self._feature_channels = feature_channels
        self._downsample_feature = downsample_feature
        self._use_decoder = use_decoder
        self._dtype = dtype

        self._buffer: dict[int, BaseReferenceData] = {}

    def clear(self) -> None:
        self._buffer.clear()

    def save(self, frame_idx: int, output: EncoderOutput | DecoderOutput, mark_as_ltr: bool = False) -> None:
        self._save(frame_idx, output)
        self._buffer[frame_idx].ltr_frame = mark_as_ltr
        self._prune()

    def load(self, ref_frame_idx: int | None, feature_reset: bool = False) -> BaseReferenceData:
        if ref_frame_idx is None:
            # I-frame
            return self._make_dummy_reference()
        else:
            ref_data = self._buffer[ref_frame_idx]
            if feature_reset:
                return self._feature_reset(ref_data)
            return ref_data

    def _make_dummy_reference(self) -> BaseReferenceData:
        return BaseReferenceData(
            ref_feature=np.zeros(
                (
                    1,
                    self._feature_channels,
                    self._model_height // self._downsample_feature,
                    self._model_width // self._downsample_feature,
                ),
                dtype=getattr(np, self._dtype),
            ),
            ref_exists=False,
        )

    def _feature_reset(self, ref_data: BaseReferenceData) -> BaseReferenceData:
        return replace(
            ref_data,
            ref_feature=np.zeros_like(ref_data.ref_feature),
            ref_exists=False,
        )

    @abstractmethod
    def _save(
        self,
        frame_idx: int,
        output: EncoderOutput | DecoderOutput,
    ) -> None:
        pass

    def _prune(self) -> None:
        str_frames = [k for k, v in self._buffer.items() if not v.ltr_frame]
        ltr_frames = [k for k, v in self._buffer.items() if v.ltr_frame]

        # Remove excess STR frames
        if len(str_frames) > 1:
            for k in sorted(str_frames)[:-1]:
                del self._buffer[k]

        # Remove excess LTR frames (3 is an arbitrary limit)
        if len(ltr_frames) > 3:
            for k in sorted(ltr_frames)[:-3]:
                del self._buffer[k]


@dataclass(kw_only=True)
class BaseReferenceDataWithFrame(BaseReferenceData):
    ref_frame: np.ndarray


class BaseReferenceManagerGrayFrame(BaseReferenceManager):
    _buffer: dict[int, BaseReferenceDataWithFrame]

    def _make_dummy_reference(self) -> BaseReferenceDataWithFrame:
        ref_feature = super()._make_dummy_reference().ref_feature
        gray_frame = 0.5 * np.ones(
            (1, 3, self._model_height, self._model_width),
            dtype=getattr(np, self._dtype),
        )
        return BaseReferenceDataWithFrame(
            ref_feature=ref_feature,
            ref_frame=gray_frame,
        )


class BaseSplitModel(ABC):
    def __init__(
        self,
        *,
        split_type: str,
        full_model: BaseFullModel,
        model_parts: dict[ModelPartId, ModelWrapper],
        runtime_params: RuntimeParams,
        model_width: int,
        model_height: int,
        scale_decoder: BaseScaleDecoder | None = None,
        conversion_metadata: ConversionMetadata | None = None,
    ):
        self._split_model_params = SplitModelParams(
            split_type=split_type,
            model_width=model_width,
            model_height=model_height,
            extra_params={},
        )
        self._full_model = full_model
        self._model_parts = model_parts
        self._scale_decoder = scale_decoder
        self._runtime_params = runtime_params
        self._conversion_metadata = conversion_metadata

        self._gaussian_encoder = GaussianEncoder(full_model.gaussian_coder_pmf)
        self._bit_estimator_z = BitEstimator(full_model.bit_estimator_pmf)

    def __getitem__(self, key: ModelPartId) -> ModelWrapper:
        return self._model_parts[key]

    def set_input_padding(self, padding_mode: PaddingMode, padding_direction: PaddingDirection) -> None:
        self._padding_mode = padding_mode
        self._padding_direction = padding_direction

    @abstractmethod
    def make_ref_manager(self, **kwargs) -> BaseReferenceManager:
        pass

    @abstractmethod
    def encode(
        self,
        frame_idx: int,
        yuv420: tuple[np.ndarray, np.ndarray],
        q_index: int,
        ref_data: BaseReferenceData,
        padding_mode: PaddingMode,
        padding_direction: PaddingDirection,
    ) -> EncoderOutput:
        pass

    @abstractmethod
    def decode(
        self,
        frame_idx: int,
        bitstream: bytes,
        padding: tuple[int, int, int, int, bool],
        q_index: int,
        ref_data: BaseReferenceData,
    ) -> DecoderOutput:
        pass

    def _encode_bitstream(
        self,
        y_raw_1: np.ndarray,
        scales_1: np.ndarray,
        y_raw_0: np.ndarray,
        scales_0: np.ndarray,
        z_raw: np.ndarray,
        q_index: int,
    ) -> bytes:
        stream = RansEncoderStream()
        self._gaussian_encoder.encode_y(
            stream,
            y_raw_1.astype(np.float32, copy=False),
            scales_1.astype(np.float32, copy=False),
        )
        self._gaussian_encoder.encode_y(
            stream,
            y_raw_0.astype(np.float32, copy=False),
            scales_0.astype(np.float32, copy=False),
        )
        self._bit_estimator_z.encode_z(stream, z_raw.astype(np.float32, copy=False), q_index)
        return bytes(stream.flush())

    def _decode_z_raw(self, stream: RansDecoderStream, q_index: int, downsample: int = 64) -> np.ndarray:
        z_size = get_downsampled_shape(
            self._split_model_params.model_height,
            self._split_model_params.model_width,
            downsample,
        )
        z_raw = self._bit_estimator_z.decode_z(stream, z_size, q_index).astype(np.float32, copy=False)
        return z_raw

    def _decode_y_raw(self, stream: RansDecoderStream, scales: np.ndarray, eof: bool = False) -> np.ndarray:
        y_raw = self._gaussian_encoder.decode_y(stream, scales.astype(np.float32, copy=False))
        if eof:
            stream.decodeEOF()
        return y_raw.astype(np.float32, copy=False)

    def _prepare_input_frame(
        self,
        yuv420: tuple[np.ndarray, np.ndarray],
        padding_mode: PaddingMode,
        padding_direction: PaddingDirection,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], tuple[int, int, int, int, bool]]:
        model_width = self._split_model_params.model_width
        model_height = self._split_model_params.model_height
        frame_height, frame_width = yuv420[0].shape[-2:]

        # Check if image needs to be rotated to the side
        fits_natively = frame_width <= model_width and frame_height <= model_height
        fits_rotated = frame_height <= model_width and frame_width <= model_height
        if not fits_natively and not fits_rotated:
            raise ValueError(
                f"Frame size ({frame_width}x{frame_height}) is larger than the model input "
                f"size ({model_width}x{model_height})"
            )

        # Convert to yuv444
        x = yuv_420_to_444(yuv420).astype(np.float32, copy=False)

        # Rotate to the side
        if not fits_natively:
            x = np.transpose(x, (0, 1, 3, 2))

        # Padding
        pad_w = model_width - x.shape[-1]
        pad_h = model_height - x.shape[-2]
        if padding_direction == PaddingDirection.BOTTOM_RIGHT:
            padding = (0, pad_w, 0, pad_h, not fits_natively)
        elif padding_direction == PaddingDirection.UPPER_LEFT:
            padding = (pad_w, 0, pad_h, 0, not fits_natively)
        elif padding_direction == PaddingDirection.BOTH:
            start_w = pad_w // 2
            start_h = pad_h // 2
            padding = (start_w, pad_w - start_w, start_h, pad_h - start_h, not fits_natively)
        else:
            raise NotImplementedError(f"Unsupported padding direction: {self._padding_direction}")

        if padding_mode == PaddingMode.HYBRID:
            edge_padding = 8
            x = np.pad(
                x,
                (
                    (0, 0),
                    (0, 0),
                    (min(padding[2], edge_padding), min(padding[3], edge_padding)),
                    (min(padding[0], edge_padding), min(padding[1], edge_padding)),
                ),
                mode="edge",
            )
            x = np.pad(
                x,
                (
                    (0, 0),
                    (0, 0),
                    (max(padding[2] - edge_padding, 0), max(padding[3] - edge_padding, 0)),
                    (max(padding[0] - edge_padding, 0), max(padding[1] - edge_padding, 0)),
                ),
                mode="constant",
                constant_values=0.5,
            )
        else:
            if padding_mode == PaddingMode.CONSTANT_GRAY:
                padding_kwargs = dict(mode="constant", constant_values=0.5)
            elif padding_mode == PaddingMode.CONSTANT_BLACK:
                padding_kwargs = dict(mode="constant", constant_values=0.0)
            elif padding_mode == PaddingMode.CONSTANT_WHITE:
                padding_kwargs = dict(mode="constant", constant_values=1.0)
            else:
                padding_kwargs = dict(mode=padding_mode.value)

            x = np.pad(
                x,
                (
                    (0, 0),
                    (0, 0),
                    (padding[2], padding[3]),
                    (padding[0], padding[1]),
                ),
                **padding_kwargs,  # type: ignore[arg-type]
            )

        assert x.shape[1] == 3
        assert x.shape[2] == model_height
        assert x.shape[3] == model_width

        # Scale to pixel range
        x *= self._full_model.model_params.pixel_range

        yuv420_output = (
            yuv420[0].astype(np.float32, copy=False)[np.newaxis, ...],
            yuv420[1][0:1].astype(np.float32, copy=False)[np.newaxis, ...],
            yuv420[1][1:2].astype(np.float32, copy=False)[np.newaxis, ...],
        )
        return x, yuv420_output, padding

    def _prepare_output_frame(
        self, x_hat: np.ndarray, padding: tuple[int, int, int, int, bool]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Remove padding
        pad_left, pad_right, pad_top, pad_bottom, rotated = padding
        height = x_hat.shape[-2] - pad_top - pad_bottom
        width = x_hat.shape[-1] - pad_left - pad_right
        yuv444 = x_hat[..., pad_top : pad_top + height, pad_left : pad_left + width]

        # Rotate
        if rotated:
            yuv444 = np.transpose(yuv444, (0, 1, 3, 2))

        # Convert to 0-1 range
        yuv444 /= self._full_model.model_params.pixel_range

        yuv420 = yuv_444_to_420(yuv444)
        return tuple(t.astype(np.float32, copy=False) for t in yuv420)  # type: ignore[return-value]

    def _get_fa_idx(self, frame_index: int) -> int:
        model_params = self._full_model.model_params
        frame_index_map = model_params.frame_index_map
        fa_idx = frame_index_map[(frame_index + 1) % len(frame_index_map)]
        return fa_idx

    def _get_q_index_shift(self, frame_index: int) -> int:
        model_params = self._full_model.model_params
        qp_shift = model_params.qp_shift
        return qp_shift[self._get_fa_idx(frame_index)]

    def _extract_scales(self, z_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._scale_decoder is None:
            raise ValueError("Scale decoder is not available for this split model")
        return self._scale_decoder.extract_scales(z_raw)

    @property
    def model_params(self) -> ModelParams:
        return self._full_model.model_params

    @property
    def split_model_params(self) -> SplitModelParams:
        return self._split_model_params

    @property
    def full_model(self) -> BaseFullModel:
        return self._full_model

    @property
    def model_parts(self) -> dict[ModelPartId, ModelWrapper]:
        return self._model_parts

    @property
    def scale_decoder(self) -> BaseScaleDecoder | None:
        return self._scale_decoder

    @property
    def runtime_params(self) -> RuntimeParams:
        return self._runtime_params

    @property
    def conversion_metadata(self) -> ConversionMetadata | None:
        return self._conversion_metadata

    @property
    def model_type(self) -> ModelType:
        return next(iter(self._model_parts.values())).model_type

    @property
    def downsample_factor(self) -> int:
        return self.model_params.downsample_hyperprior * self.model_params.downsample_latent
