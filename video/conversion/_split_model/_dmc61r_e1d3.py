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
    BaseReferenceData,
    BaseReferenceManager,
    BaseSplitModel,
    prepare_model_parts,
)
from ._dmc61_e2d3 import DecoderPart2 as Dmc61DecoderPart2
from ._dmc61_e2d3 import DecoderPart3 as Dmc61DecoderPart3
from msrtc.rans import RansDecoderStream


class Encoder(BaseEncoderPart):
    InputType = namedtuple("MLVCEncoderPart1Input", ["x", "ref_feature", "q_index_shifted"])
    OutputType = namedtuple(
        "MLVCEncoderPart1Output",
        ["feature", "z_raw", "y_raw_0", "scales_0", "y_raw_1", "scales_1"],
    )

    def forward(
        self,
        x: torch.Tensor,
        ref_feature: torch.Tensor,
        q_index_shifted: torch.Tensor,
    ):
        dpb = {"ref_feature": ref_feature}
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


class DecoderPart1(BaseDecoderPart):
    InputType = namedtuple(
        "MLVCDecoderPart1Input",
        ["z_raw", "ref_feature", "q_index_shifted"],
    )
    OutputType = namedtuple(
        "MLVCDecoderPart1Output",
        ["scales_0", "quant_step", "means_0", "params_0", "ctx"],
    )

    def forward(self, z_raw, ref_feature, q_index_shifted):
        dpb = {"ref_feature": ref_feature}
        part1 = {"z_hat": z_raw, "slice_shape": self.dims.y_slice_shape}
        result = self.model.estimate_params0(part1=part1, dpb=dpb, q_index=q_index_shifted)
        result["scales_0"] = self.model.pack_with_mask(result["scales_0"], self.mask_0, chunks=2)
        return self.OutputType(**result)


class DecoderPart2(Dmc61DecoderPart2):
    pass


class DecoderPart3(Dmc61DecoderPart3):
    pass


@dataclass(kw_only=True)
class ReferenceData(BaseReferenceData):
    pass


class ReferenceManager(BaseReferenceManager):
    _buffer: dict[int, ReferenceData]

    def _save(
        self,
        frame_idx: int,
        output: EncoderOutput | DecoderOutput,
    ) -> None:
        model_outputs: Encoder.OutputType | DecoderPart3.OutputType
        if self._use_decoder:
            model_outputs = output.model_data[ModelPartId.DECODER_PART3].outputs
        else:
            model_outputs = output.model_data[ModelPartId.ENCODER].outputs

        self._buffer[frame_idx] = ReferenceData(ref_feature=model_outputs.feature.copy())

    def load(self, ref_frame_idx: int | None, feature_reset: bool = False) -> ReferenceData:
        return super().load(ref_frame_idx, feature_reset=feature_reset)  # type: ignore[return-value]


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
                        ModelPartId.ENCODER: Encoder(full_model, model_height, model_width),
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
            split_type="dmc61r_e1d3",
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
        model = self.model_parts[ModelPartId.ENCODER]
        inputs = Encoder.InputType(
            x=x,
            ref_feature=ref_data.ref_feature,
            q_index_shifted=np.array([q_index_shifted], dtype=np.int32),
        )
        outputs: Encoder.OutputType = model.predict(inputs)

        # Bitstream
        bitstream = self._encode_bitstream(
            outputs.y_raw_1,
            outputs.scales_1,
            outputs.y_raw_0,
            outputs.scales_0,
            outputs.z_raw,
            q_index_shifted,
        )

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
        q_index_shifted = q_index + self._get_q_index_shift(frame_idx)

        # Part 1
        stream = RansDecoderStream(bitstream)
        z_raw = self._decode_z_raw(stream, q_index_shifted)
        model1 = self.model_parts[ModelPartId.DECODER_PART1]
        inputs1 = DecoderPart1.InputType(
            z_raw=z_raw,
            ref_feature=ref_data.ref_feature,
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
