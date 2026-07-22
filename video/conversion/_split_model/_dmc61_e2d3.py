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
from ._base_split_model import (
    BaseEncoderPart,
    BaseDecoderPart,
    BaseReferenceDataWithFrame,
    BaseReferenceManagerGrayFrame,
    BaseSplitModel,
    prepare_model_parts,
)
from msrtc.rans import RansDecoderStream


class EncoderPart1(BaseEncoderPart):
    InputType = namedtuple("MLVCEncoderPart1Input", ["x", "ref_frame", "ref_feature", "q_index_shifted", "ref_exists"])
    OutputType = namedtuple(
        "MLVCEncoderPart1Output",
        ["feature", "z_raw", "y_raw_0", "scales_0", "y_raw_1", "scales_1"],
    )

    def forward(
        self,
        x: torch.Tensor,
        ref_frame: torch.Tensor,
        ref_feature: torch.Tensor,
        q_index_shifted: torch.Tensor,
        ref_exists: torch.Tensor,
    ):
        dpb = {
            "ref_frame": ref_frame,
            "ref_feature": ref_feature,
            "ref_exists": ref_exists,
        }

        results = self.model.compress_core(x=x, dpb=dpb, q_index=q_index_shifted, do_shift_qp=False, get_recon=False)

        feature = results["dpb"]["ref_feature"]
        (y_raw_0, scales_0), (y_raw_1, scales_1) = results["y_raw"]
        z_raw = results["z_raw"]

        return self.OutputType(
            feature=feature,
            z_raw=z_raw,
            y_raw_0=y_raw_0,
            scales_0=scales_0,
            y_raw_1=y_raw_1,
            scales_1=scales_1,
        )


class EncoderPart2(BaseEncoderPart):
    InputType = namedtuple("MLVCEncoderPart2Input", ["feature", "q_index_shifted"])
    OutputType = namedtuple(
        "MLVCEncoderPart2Output",
        ["x_hat"],
    )

    def forward(self, feature, q_index_shifted):
        x_hat = self.model.get_recon(feature, q_index_shifted)
        return self.OutputType(x_hat=x_hat)


class DecoderPart1(BaseDecoderPart):
    InputType = namedtuple(
        "MLVCDecoderPart1Input",
        ["z_raw", "ref_frame", "ref_feature", "ref_exists", "q_index_shifted"],
    )
    OutputType = namedtuple(
        "MLVCDecoderPart1Output",
        ["scales_0", "quant_step", "means_0", "params_0", "ctx"],
    )

    def forward(self, z_raw, ref_frame, ref_feature, ref_exists, q_index_shifted):
        dpb = {
            "ref_frame": ref_frame,
            "ref_feature": ref_feature,
            "ref_exists": ref_exists,
        }
        part1 = {"z_hat": z_raw, "slice_shape": self.dims.y_slice_shape}
        result = self.model.estimate_params0(part1=part1, dpb=dpb, q_index=q_index_shifted)
        result["scales_0"] = self.model.pack_with_mask(result["scales_0"], self.mask_0, chunks=2)
        return self.OutputType(**result)


class DecoderPart2(BaseDecoderPart):
    InputType = namedtuple("MLVCDecoderPart2Input", ["y_raw_0", "means_0", "params_0"])
    OutputType = namedtuple("MLVCDecoderPart2Output", ["scales_1", "means_1", "y_hat_0"])

    def forward(self, y_raw_0, means_0, params_0):
        y_hat_0 = self.model.unpack_with_mask(y_raw_0, means_0, self.mask_0, chunks=2)
        result = self.model.decompress_dual_prior_torch(y_hat_0=y_hat_0, params_0=params_0)
        scales_1 = result["scales_1"]
        scales_1 = self.model.pack_with_mask(scales_1, self.mask_1, 2)
        means_1 = result["means_1"]
        return self.OutputType(scales_1=scales_1, means_1=means_1, y_hat_0=y_hat_0)


class DecoderPart3(BaseDecoderPart):
    InputType = namedtuple(
        "MLVCDecoderPart3Input",
        ["means_1", "y_raw_1", "y_hat_0", "ctx", "q_index_shifted", "quant_step"],
    )
    OutputType = namedtuple("MLVCDecoderPart3Output", ["x_hat", "feature"])

    def forward(self, means_1, y_raw_1, y_hat_0, ctx, q_index_shifted, quant_step):
        q_decoder = self.model.q_decoder[q_index_shifted]
        y_hat_1 = self.model.unpack_with_mask(y_raw_1, means_1, self.mask_1, 2)
        y_hat = (y_hat_0 + y_hat_1) * quant_step
        x_hat, feature = self.model.get_recon_and_feature(y_hat, ctx, q_decoder, q_index_shifted)
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
        model_outputs1: EncoderPart1.OutputType | DecoderPart3.OutputType
        model_outputs2: EncoderPart2.OutputType | DecoderPart3.OutputType
        if self._use_decoder:
            model_outputs1 = output.model_data[ModelPartId.DECODER_PART3].outputs
            model_outputs2 = output.model_data[ModelPartId.DECODER_PART3].outputs
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
                        ModelPartId.DECODER_PART1: DecoderPart1(full_model, model_height, model_width),
                        ModelPartId.DECODER_PART2: DecoderPart2(full_model, model_height, model_width),
                        ModelPartId.DECODER_PART3: DecoderPart3(full_model, model_height, model_width),
                    }
                )
            model_parts = prepare_model_parts(torch_model_parts, runtime_params)

        super().__init__(
            split_type="dmc61_e2d3",
            full_model=full_model,
            model_parts=model_parts,
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
        bitstream = self._encode_bitstream(
            outputs1.y_raw_1,
            outputs1.scales_1,
            outputs1.y_raw_0,
            outputs1.scales_0,
            outputs1.z_raw,
            q_index_shifted,
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
        q_index_shifted = q_index + self._get_q_index_shift(frame_idx)

        # Part 1
        stream = RansDecoderStream(bitstream)
        z_raw = self._decode_z_raw(stream, q_index_shifted)
        model1 = self.model_parts[ModelPartId.DECODER_PART1]
        inputs1 = DecoderPart1.InputType(
            z_raw=z_raw,
            ref_frame=ref_data.ref_frame,
            ref_feature=ref_data.ref_feature,
            ref_exists=np.array([ref_data.ref_exists], dtype=np.float32),
            q_index_shifted=np.array([q_index_shifted], dtype=np.int32),
        )
        outputs1: DecoderPart1.OutputType = model1.predict(inputs1)

        # Part 2
        scales_0 = outputs1.scales_0
        y_raw_0 = self._decode_y_raw(stream, scales_0)
        model2 = self.model_parts[ModelPartId.DECODER_PART2]
        inputs2 = DecoderPart2.InputType(
            y_raw_0=y_raw_0,
            means_0=outputs1.means_0,
            params_0=outputs1.params_0,
        )
        outputs2: DecoderPart2.OutputType = model2.predict(inputs2)

        # Part 3
        scales_1 = outputs2.scales_1
        y_raw_1 = self._decode_y_raw(stream, scales_1, eof=True)
        model3 = self.model_parts[ModelPartId.DECODER_PART3]
        inputs3 = DecoderPart3.InputType(
            means_1=outputs2.means_1,
            y_raw_1=y_raw_1,
            y_hat_0=outputs2.y_hat_0,
            ctx=outputs1.ctx,
            q_index_shifted=np.array([q_index_shifted], dtype=np.int32),
            quant_step=outputs1.quant_step,
        )
        outputs3: DecoderPart3.OutputType = model3.predict(inputs3)

        return DecoderOutput(
            model_data={
                ModelPartId.DECODER_PART1: ModelData(inputs=inputs1, outputs=outputs1),
                ModelPartId.DECODER_PART2: ModelData(inputs=inputs2, outputs=outputs2),
                ModelPartId.DECODER_PART3: ModelData(inputs=inputs3, outputs=outputs3),
            },
            reconstructed_frame=self._prepare_output_frame(outputs3.x_hat, padding=padding),
            timers={
                ModelPartId.DECODER_PART1.value: model1.inference_time,
                ModelPartId.DECODER_PART2.value: model2.inference_time,
                ModelPartId.DECODER_PART3.value: model3.inference_time,
            },
        )
