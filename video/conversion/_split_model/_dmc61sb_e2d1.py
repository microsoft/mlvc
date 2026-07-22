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
    BaseReferenceDataWithFrame,
    BaseReferenceManagerGrayFrame,
    BaseSplitModel,
    prepare_model_parts,
)
from ._dmc61s_e2d1 import EncoderPart1 as Dmc61sEncoderPart1
from ._dmc61s_e2d1 import EncoderPart2 as Dmc61sEncoderPart2
from src.models.utils import apply_upper_lower_bound
from src.utils.stream_helper import get_downsampled_shape
from msrtc.rans import RansDecoderStream


class EncoderPart1(Dmc61sEncoderPart1):
    pass


class EncoderPart2(Dmc61sEncoderPart2):
    def forward(self, feature, q_index_shifted):
        feature, _ = feature.chunk(2, dim=1)
        output = super().forward(feature, q_index_shifted)
        x_hat = self.model.unshift_output(output.x_hat)
        return self.OutputType(x_hat=x_hat)


class Decoder(BaseDecoderPart):
    InputType = namedtuple(
        "MLVCDecoderInput",
        [
            "z_raw",
            "y_raw_0",
            "y_raw_1",
            "ref_frame",
            "ref_feature",
            "ref_exists",
            "q_index_shifted",
        ],
    )
    OutputType = namedtuple("MLVCDecoderOutput", ["x_hat", "feature"])

    def forward(self, z_raw, y_raw_0, y_raw_1, ref_frame, ref_feature, ref_exists, q_index_shifted):
        dpb = {
            "ref_frame": ref_frame,
            "ref_feature": ref_feature,
            "ref_exists": ref_exists,
        }

        q_feature = self.model.q_feature[q_index_shifted]
        q_decoder = self.model.q_decoder[q_index_shifted]

        # Part 1
        ctx, ctx_t, memory = self.model.context_generation(dpb, q_feature)
        params = self.model.res_prior_param_decoder(z_raw, ctx_t, self.dims.y_slice_shape)
        quant_step, means_0 = params.chunk(2, 1)
        quant_step = apply_upper_lower_bound(quant_step, lower=0.5)

        # Part 2
        y_hat_0 = self.model.unpack_with_mask(y_raw_0, means_0, self.mask_0, chunks=2)
        means_1 = self.model.y_spatial_prior(torch.cat((y_hat_0, params), dim=1))

        # Part 3
        y_hat_1 = self.model.unpack_with_mask(y_raw_1, means_1, self.mask_1, 2)
        y_hat = (y_hat_0 + y_hat_1) * quant_step
        x_hat, feature, memory = self.model.get_recon_and_feature(y_hat, ctx, q_decoder, memory, q_index_shifted)
        x_hat = self.model.unshift_output(x_hat)
        feature = torch.cat((feature, memory), dim=1)

        return self.OutputType(x_hat=x_hat, feature=feature)


@dataclass(kw_only=True)
class ReferenceData(BaseReferenceDataWithFrame):
    pass


class ReferenceManager(BaseReferenceManagerGrayFrame):
    _buffer: dict[int, ReferenceData]

    def _save(
        self,
        frame_idx: int,
        output: EncoderOutput | DecoderOutput,
    ) -> None:
        model_outputs1: EncoderPart1.OutputType | Decoder.OutputType
        model_outputs2: EncoderPart2.OutputType | Decoder.OutputType
        if self._use_decoder:
            model_outputs1 = output.model_data[ModelPartId.DECODER].outputs
            model_outputs2 = output.model_data[ModelPartId.DECODER].outputs
        else:
            model_outputs1 = output.model_data[ModelPartId.ENCODER_PART1].outputs
            model_outputs2 = output.model_data[ModelPartId.ENCODER_PART2].outputs

        self._buffer[frame_idx] = ReferenceData(
            ref_feature=model_outputs1.feature.copy(),
            ref_frame=model_outputs2.x_hat.copy(),
            ref_exists=True,
        )


class SplitModel(BaseSplitModel):
    def __init__(
        self,
        full_model: BaseFullModel,
        model_parts: dict[ModelPartId, ModelWrapper] | None,
        runtime_params: RuntimeParams,
        model_height: int,
        model_width: int,
        use_encoder: bool,
        use_decoder: bool,
        split_type: str = "dmc61sb_e2d1",
        scale_decoder: BaseScaleDecoder | None = None,
        **kwargs,
    ):

        if model_parts is None:
            torch_model_parts = {}
            if use_encoder:
                torch_model_parts.update(
                    {
                        ModelPartId.ENCODER_PART1: EncoderPart1(full_model, model_height, model_width),
                        ModelPartId.ENCODER_PART2: EncoderPart2(full_model, model_height, model_width),
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
        model1 = self.model_parts[ModelPartId.ENCODER_PART1]
        inputs1 = EncoderPart1.InputType(
            x=x,
            ref_frame=ref_data.ref_frame,
            ref_feature=ref_data.ref_feature,
            q_index_shifted=np.array([q_index_shifted], dtype=np.int32),
            ref_exists=np.array([ref_data.ref_exists], dtype=np.float32),
        )
        outputs1: EncoderPart1.OutputType = model1.predict(inputs1)

        # Part 2
        model2 = self.model_parts[ModelPartId.ENCODER_PART2]
        inputs2 = EncoderPart2.InputType(
            feature=outputs1.feature,
            q_index_shifted=np.array([q_index_shifted], dtype=np.int32),
        )
        outputs2: EncoderPart2.OutputType = model2.predict(inputs2)

        # Bitstream
        scales_0, scales_1 = self._extract_scales(outputs1.z_raw)
        bitstream = self._encode_bitstream(
            outputs1.y_raw_1, scales_1, outputs1.y_raw_0, scales_0, outputs1.z_raw, q_index
        )

        return EncoderOutput(
            model_data={
                ModelPartId.ENCODER_PART1: ModelData(inputs=inputs1, outputs=outputs1),
                ModelPartId.ENCODER_PART2: ModelData(inputs=inputs2, outputs=outputs2),
            },
            padding=padding,
            original_frame=original_frame,
            reconstructed_frame=self._prepare_output_frame(outputs2.x_hat, padding),
            bitstream=bitstream,
            timers={
                ModelPartId.ENCODER_PART1.value: model1.inference_time,
                ModelPartId.ENCODER_PART2.value: model2.inference_time,
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
            ref_frame=ref_data.ref_frame,
            ref_feature=ref_data.ref_feature,
            ref_exists=np.array([ref_data.ref_exists], dtype=np.float32),
            q_index_shifted=np.array([q_index_shifted], dtype=np.int32),
        )
        outputs: Decoder.OutputType = model.predict(inputs)

        return DecoderOutput(
            model_data={ModelPartId.DECODER: ModelData(inputs=inputs, outputs=outputs)},
            reconstructed_frame=self._prepare_output_frame(outputs.x_hat, padding=padding),
            timers={ModelPartId.DECODER.value: model.inference_time},
        )
