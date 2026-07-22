# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
from collections import namedtuple
from dataclasses import dataclass
from ..types import (
    ModelPartId,
    ModelData,
    EncoderOutput,
    DecoderOutput,
    RuntimeParams,
    PaddingMode,
    PaddingDirection,
)
from .._full_model import BaseFullModel
from .._model_wrapper import ModelWrapper
from .._scale_decoder import BaseScaleDecoder, UpsampleScaleDecoder
from ._base_split_model import (
    BaseDecoderPart,
    BaseReferenceData,
    BaseReferenceManager,
    BaseEncoderPart,
    BaseSplitModel,
    prepare_model_parts,
)
from src.models.utils import apply_upper_lower_bound
from src.utils.stream_helper import get_downsampled_shape
from msrtc.rans import RansDecoderStream


class Encoder(BaseEncoderPart):
    InputType = namedtuple("MLVCEncoderPart1Input", ["x", "ref_feature", "q_index_shifted"])
    OutputType = namedtuple(
        "MLVCEncoderPart1Output",
        ["feature", "z_raw", "y_raw_0", "y_raw_1"],
    )

    def forward(
        self,
        x: torch.Tensor,
        ref_feature: torch.Tensor,
        q_index_shifted: torch.Tensor,
    ):
        dpb = {
            "ref_feature": ref_feature,
        }

        results = self.model.compress_core(x=x, dpb=dpb, q_index=q_index_shifted, do_shift_qp=False, get_recon=False)

        feature = results["dpb"]["ref_feature"]
        (y_raw_0, scales_0), (y_raw_1, scales_1) = results["y_raw"]
        z_raw = results["z_raw"]

        return self.OutputType(
            feature=feature,
            z_raw=z_raw,
            y_raw_0=y_raw_0,
            y_raw_1=y_raw_1,
        )


class Decoder(BaseDecoderPart):
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
    OutputType = namedtuple("MLVCDecoderOutput", ["x_hat", "feature"])

    def forward(self, z_raw, y_raw_0, y_raw_1, ref_feature, q_index_shifted):
        dpb = {
            "ref_feature": ref_feature,
        }

        q_feature = self.model.q_feature[q_index_shifted]
        q_decoder = self.model.q_decoder[q_index_shifted]

        # Part 1
        ctx, ctx_t = self.model.context_generation(dpb, q_feature)
        params = self.model.res_prior_param_decoder(z_raw, ctx_t, self.dims.y_slice_shape)
        quant_step, means_0 = params.chunk(2, 1)
        quant_step = apply_upper_lower_bound(quant_step, lower=0.5)

        # Part 2
        y_hat_0 = self.model.unpack_with_mask(y_raw_0, means_0, self.mask_0, chunks=2)
        means_1 = self.model.y_spatial_prior(torch.cat((y_hat_0, params), dim=1))

        # Part 3
        y_hat_1 = self.model.unpack_with_mask(y_raw_1, means_1, self.mask_1, 2)
        y_hat = (y_hat_0 + y_hat_1) * quant_step
        x_hat, feature = self.model.get_recon_and_feature(y_hat, ctx, q_decoder, q_index_shifted)

        return self.OutputType(x_hat=x_hat, feature=feature)


@dataclass(kw_only=True)
class ReferenceData(BaseReferenceData):
    pass


class ReferenceManager(BaseReferenceManager):
    _buffer: dict[int, ReferenceData]

    def _save(self, frame_idx: int, output: EncoderOutput | DecoderOutput) -> None:
        model_outputs: Decoder.OutputType | Encoder.OutputType
        if self._use_decoder:
            model_outputs = output.model_data[ModelPartId.DECODER].outputs
        else:
            model_outputs = output.model_data[ModelPartId.ENCODER].outputs
        self._buffer[frame_idx] = ReferenceData(ref_feature=model_outputs.feature.copy())

    def load(self, ref_frame_idx: int | None, feature_reset: bool = False) -> ReferenceData:
        return super().load(ref_frame_idx, feature_reset=feature_reset)  # type: ignore[return-value]


class SplitModel(BaseSplitModel):
    def __init__(
        self,
        *,
        full_model: BaseFullModel,
        model_parts: dict[ModelPartId, ModelWrapper] | None,
        runtime_params: RuntimeParams,
        model_height: int,
        model_width: int,
        use_encoder: bool,
        use_decoder: bool,
        split_type: str = "dmc61sr_e1d1",
        scale_decoder: BaseScaleDecoder | None = None,
        **kwargs,
    ):

        if model_parts is None:
            torch_model_parts = {}
            if use_encoder:
                torch_model_parts.update(
                    {
                        ModelPartId.ENCODER: Encoder(full_model, model_height, model_width),
                    }
                )
            if use_decoder:
                torch_model_parts.update(
                    {
                        ModelPartId.DECODER: Decoder(full_model, model_height, model_width),
                    }
                )
            model_parts = prepare_model_parts(torch_model_parts, runtime_params)

        if scale_decoder is None:
            assert full_model.model_params.y_scale_repeat is not None
            y_height, y_width = get_downsampled_shape(
                model_height, model_width, full_model.model_params.downsample_latent
            )
            scale_decoder = UpsampleScaleDecoder(
                y_shape=(full_model.model_params.latent_channels, y_height, y_width),
                index_space=full_model.gaussian_coder_pmf.index_space,
                scale_max_idx=full_model.gaussian_coder_pmf.scale_levels - 1,
                channel_repeat=full_model.model_params.y_scale_repeat,
                spatial_repeat=full_model.model_params.downsample_hyperprior,
            )

        super().__init__(
            split_type=split_type,
            full_model=full_model,
            model_parts=model_parts,
            scale_decoder=scale_decoder,
            runtime_params=runtime_params,
            model_height=model_height,
            model_width=model_width,
            **kwargs,
        )

    def make_ref_manager(self, **kwargs) -> ReferenceManager:
        return ReferenceManager(
            model_width=self._split_model_params.model_width,
            model_height=self._split_model_params.model_height,
            feature_channels=self.model_params.feature_channels,
            downsample_feature=self.model_params.downsample_feature,
            **kwargs,
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

        # Part 1
        model = self.model_parts[ModelPartId.ENCODER]
        inputs = Encoder.InputType(
            x=x,
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
        z_raw = self._decode_z_raw(stream, q_index, downsample=self.downsample_factor)

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

        return DecoderOutput(
            model_data={ModelPartId.DECODER: ModelData(inputs=inputs, outputs=outputs)},
            reconstructed_frame=self._prepare_output_frame(outputs.x_hat, padding=padding),
            timers={ModelPartId.DECODER.value: model.inference_time},
        )
