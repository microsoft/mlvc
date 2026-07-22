# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
from collections import namedtuple
from ..types import (
    ModelPartId,
    RuntimeParams,
    EncoderOutput,
    DecoderOutput,
    ModelData,
    PaddingMode,
    PaddingDirection,
)
from .._full_model import BaseFullModel
from .._model_wrapper import ModelWrapper
from ._base_split_model import prepare_model_parts
from ._dmc61sbr_e1d1 import Encoder as Dmc61sbrEncoder
from ._dmc61sbr_e1d1 import Decoder as Dmc61sbrDecoder
from ._dmc61sr_e1d1 import ReferenceData as Dmc61sbrReferenceData
from ._dmc61sbr_e1d1 import ReferenceManager as Dmc61sbrReferenceManager
from ._dmc61sbr_e1d1 import SplitModel as Dmc61srSplitModel
from msrtc.rans import RansDecoderStream


class Encoder(Dmc61sbrEncoder):
    InputType = namedtuple("MLVCEncoderPart1Input", ["x_luma", "x_chroma", "ref_feature", "q_index_shifted"])
    OutputType = namedtuple(
        "MLVCEncoderPart1Output",
        ["feature", "z_raw", "y_raw_0", "y_raw_1"],
    )

    def __init__(self, *args, upsample_method: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._upsample_method = upsample_method

    def forward(
        self,
        x_luma: torch.Tensor,
        x_chroma: torch.Tensor,
        ref_feature: torch.Tensor,
        q_index_shifted: torch.Tensor,
    ) -> OutputType:
        x = torch_yuv420_to_yuv444(x_luma, x_chroma, method=self._upsample_method)
        return super().forward(x, ref_feature, q_index_shifted)  # type: ignore[misc]


class Decoder(Dmc61sbrDecoder):
    InputType = namedtuple(
        "MLVCDecoderInput",
        [
            "z_raw",
            "y_raw_0",
            "y_raw_1",
            "ref_feature",
            "q_index_shifted",
        ],
    )
    OutputType = namedtuple("MLVCDecoderOutput", ["x_hat_luma", "x_hat_chroma", "feature"])

    def __init__(self, *args, downsample_method: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._downsample_method = downsample_method

    def forward(self, z_raw, y_raw_0, y_raw_1, ref_feature, q_index_shifted) -> OutputType:
        res = super().forward(z_raw, y_raw_0, y_raw_1, ref_feature, q_index_shifted)
        x_hat_luma, x_hat_chroma = torch_yuv444_to_yuv420(res.x_hat, method=self._downsample_method)
        return self.OutputType(
            x_hat_luma=x_hat_luma,
            x_hat_chroma=x_hat_chroma,
            feature=res.feature,
        )


class ReferenceData(Dmc61sbrReferenceData):
    pass


class ReferenceManager(Dmc61sbrReferenceManager):
    pass


class SplitModel(Dmc61srSplitModel):
    def __init__(
        self,
        full_model: BaseFullModel,
        model_parts: dict[ModelPartId, ModelWrapper] | None,
        runtime_params: RuntimeParams,
        model_height: int,
        model_width: int,
        use_encoder: bool,
        use_decoder: bool,
        split_type: str = "dmc61sbr_e1d1_i420",
        upsample_method: str = "conv_transposed",
        downsample_method: str = "conv",
        **kwargs,
    ):

        if model_parts is None:
            torch_model_parts = {}
            if use_encoder:
                torch_model_parts.update(
                    {
                        ModelPartId.ENCODER: Encoder(
                            full_model,
                            model_height,
                            model_width,
                            upsample_method=upsample_method,
                        ),
                    }
                )
            if use_decoder:
                torch_model_parts.update(
                    {
                        ModelPartId.DECODER: Decoder(
                            full_model,
                            model_height,
                            model_width,
                            downsample_method=downsample_method,
                        ),
                    }
                )
            model_parts = prepare_model_parts(torch_model_parts, runtime_params)

        super().__init__(
            split_type=split_type,
            full_model=full_model,
            model_parts=model_parts,
            runtime_params=runtime_params,
            model_height=model_height,
            model_width=model_width,
            use_encoder=use_encoder,
            use_decoder=use_decoder,
            **kwargs,
        )

        self._split_model_params.extra_params.update(
            {
                "upsample_method": upsample_method,
                "downsample_method": downsample_method,
            }
        )

    def encode(
        self,
        frame_idx: int,
        yuv420: tuple[np.ndarray, np.ndarray],
        q_index: int,
        ref_data: ReferenceData,
        padding_mode: PaddingMode,
        padding_direction: PaddingDirection,
    ) -> EncoderOutput:
        x, original_frame, padding = self._prepare_input_frame(yuv420, padding_mode, padding_direction)
        q_index_shifted = q_index + self._get_q_index_shift(frame_idx)

        x_luma, x_chroma = numpy_yuv444_to_yuv420(x)

        # Part 1
        model = self.model_parts[ModelPartId.ENCODER]
        inputs = Encoder.InputType(
            x_chroma=x_chroma,
            x_luma=x_luma,
            ref_feature=ref_data.ref_feature,
            q_index_shifted=np.array([q_index_shifted], dtype=np.int32),
        )
        outputs: Encoder.OutputType = model.predict(inputs)

        # Bitstream
        scales_0, scales_1 = self._extract_scales(outputs.z_raw)
        bitstream = self._encode_bitstream(outputs.y_raw_1, scales_1, outputs.y_raw_0, scales_0, outputs.z_raw, q_index)

        return EncoderOutput(
            model_data={
                ModelPartId.ENCODER: ModelData(inputs=inputs, outputs=outputs),
            },
            padding=padding,
            original_frame=original_frame,
            reconstructed_frame=None,
            bitstream=bitstream,
            timers={
                ModelPartId.ENCODER.value: model.inference_time,
            },
        )

    def decode(
        self,
        frame_idx: int,
        bitstream: bytes,
        padding: tuple[int, int, int, int, bool],
        q_index: int,
        ref_data: ReferenceData,
    ) -> DecoderOutput:
        stream = RansDecoderStream(bitstream)
        z_raw = self._decode_z_raw(stream, q_index, downsample=128)

        scales_0, scales_1 = self._extract_scales(z_raw)
        y_raw_0 = self._decode_y_raw(stream, scales_0)
        y_raw_1 = self._decode_y_raw(stream, scales_1, eof=True)
        q_index_shifted = q_index + self._get_q_index_shift(frame_idx)

        model = self.model_parts[ModelPartId.DECODER]
        inputs = Decoder.InputType(
            z_raw=z_raw,
            y_raw_0=y_raw_0,
            y_raw_1=y_raw_1,
            ref_feature=ref_data.ref_feature,
            q_index_shifted=np.array([q_index_shifted], dtype=np.int32),
        )
        outputs: Decoder.OutputType = model.predict(inputs)

        x_hat = numpy_yuv420_to_yuv444(outputs.x_hat_luma, outputs.x_hat_chroma)
        return DecoderOutput(
            model_data={ModelPartId.DECODER: ModelData(inputs=inputs, outputs=outputs)},
            reconstructed_frame=self._prepare_output_frame(x_hat, padding=padding),
            timers={ModelPartId.DECODER.value: model.inference_time},
        )


def torch_yuv420_to_yuv444(y: torch.Tensor, uv: torch.Tensor, method: str) -> torch.Tensor:
    if method == "repeat":
        uv_upsampled = uv.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    elif method == "interpolate":
        h, w = y.shape[2], y.shape[3]
        uv_upsampled = torch.nn.functional.interpolate(uv, size=(h, w), mode="nearest")
    elif method == "conv_transposed":
        weight = torch.ones((2, 1, 2, 2), device=uv.device, dtype=uv.dtype)
        uv_upsampled = torch.nn.functional.conv_transpose2d(
            uv,
            weight,
            bias=None,
            stride=2,
            padding=0,
            groups=2,
        )
    else:
        raise NotImplementedError(f"Unknown upsample method: {method}")

    yuv444 = torch.cat([y, uv_upsampled], dim=1)
    return yuv444


def torch_yuv444_to_yuv420(yuv444: torch.Tensor, method: str) -> tuple[torch.Tensor, torch.Tensor]:
    y = yuv444[:, 0:1, :, :]

    if method == "avg_pool":
        uv = torch.nn.functional.avg_pool2d(yuv444[:, 1:3, :, :], kernel_size=2, stride=2)
    elif method == "naive":
        uv = yuv444[:, 1:3, ::2, ::2]
    elif method == "conv":
        uv_full = yuv444[:, 1:3, :, :]
        kernel_2d = torch.ones((2, 2), device=uv_full.device, dtype=uv_full.dtype) / 4.0
        weight = kernel_2d.view(1, 1, 2, 2).repeat(2, 1, 1, 1)
        uv = torch.nn.functional.conv2d(uv_full, weight, bias=None, stride=2, padding=0, groups=2)
    else:
        raise NotImplementedError(f"Unknown downsample method: {method}")

    return y, uv


def numpy_yuv420_to_yuv444(y: np.ndarray, uv: np.ndarray) -> np.ndarray:
    u = uv[:, 0:1, :, :]
    v = uv[:, 1:2, :, :]
    u_upsampled = np.repeat(np.repeat(u, 2, axis=2), 2, axis=3)
    v_upsampled = np.repeat(np.repeat(v, 2, axis=2), 2, axis=3)
    yuv444 = np.concatenate([y, u_upsampled, v_upsampled], axis=1)
    return yuv444


def numpy_yuv444_to_yuv420(yuv444: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = yuv444[:, 0:1, :, :]
    u = yuv444[:, 1:2, :, :]
    v = yuv444[:, 2:3, :, :]

    def avg_pool2d_numpy(x, kernel_size=2):
        B, C, H, W = x.shape
        H_out = H // kernel_size
        W_out = W // kernel_size
        x_reshaped = x.reshape(B, C, H_out, kernel_size, W_out, kernel_size)
        return x_reshaped.mean(axis=(3, 5))

    u_downsampled = avg_pool2d_numpy(u)
    v_downsampled = avg_pool2d_numpy(v)
    uv = np.concatenate([u_downsampled, v_downsampled], axis=1)
    return y, uv
