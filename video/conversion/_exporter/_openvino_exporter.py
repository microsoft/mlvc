# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import openvino as ov  # type: ignore[import-not-found]
from pathlib import Path
from ._onnx_exporter import OnnxExporter
from ..types import ModelType, ModelPrecision, ModelData


class OpenvinoExporter(OnnxExporter):
    def __init__(self, **kwargs) -> None:
        super().__init__(model_type=ModelType.OPENVINO, **kwargs)

    @torch.inference_mode()
    def _export(
        self,
        model_name: str,
        model: torch.nn.Module,
        example_model_data: list[ModelData],
        output_path: Path,
        fake_quantized: bool = False,
    ) -> None:
        super()._export(
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
