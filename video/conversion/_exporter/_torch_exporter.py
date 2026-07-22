# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from pathlib import Path
from ._base_exporter import BaseExporter
from ..types import ModelType, ModelPrecision, ModelData


class TorchExporter(BaseExporter):
    def __init__(self, **kwargs) -> None:
        super().__init__(model_type=ModelType.TORCH, **kwargs)

    @torch.inference_mode()
    def _export(
        self,
        model_name: str,
        model: torch.nn.Module,
        example_model_data: list[ModelData],
        output_path: Path,
        fake_quantized: bool = False,
    ) -> None:
        file_name = output_path / f"{model_name}.torch"
        if self._precision == ModelPrecision.FP16:
            print("Warning: Running Torch model in FP16 precision can be slow on CPU")
            torch.save(model.half(), file_name)
        elif self._precision == ModelPrecision.FP32:
            torch.save(model, file_name)
        else:
            raise ValueError(f"Unsupported precision {self._precision}")
        print(f"Saved Torch model to {file_name}")
