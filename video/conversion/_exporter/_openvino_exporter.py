# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

import openvino as ov
import torch

from ..types import ModelData, ModelPartMetadata, ModelPrecision, ModelType, TensorLayout
from ._onnx_exporter import OnnxExporter


class OpenvinoExporter(OnnxExporter):
    def __init__(self, **kwargs) -> None:
        super().__init__(model_type=ModelType.OPENVINO, **kwargs)
        if TensorLayout.NHWC in (self._image_layout, self._feature_layout):
            raise ValueError("OpenVINO exporter only supports NCHW layouts")

    @torch.inference_mode()
    def _export(
        self,
        model_name: str,
        model: torch.nn.Module,
        example_model_data: list[ModelData],
        output_path: Path,
        fake_quantized: bool = False,
    ) -> ModelPartMetadata:
        metadata = super()._export(
            model_name=model_name,
            model=model,
            example_model_data=example_model_data,
            output_path=output_path,
            fake_quantized=fake_quantized,
        )
        onnx_model_path = str(output_path / f"{model_name}.onnx")
        ov_model = ov.convert_model(onnx_model_path)
        model_output_path = str(output_path / f"{model_name}.xml")
        ov.save_model(
            ov_model,
            model_output_path,
            compress_to_fp16=self._precision == ModelPrecision.FP16,
        )
        return metadata
