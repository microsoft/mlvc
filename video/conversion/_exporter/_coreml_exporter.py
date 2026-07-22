# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
import coremltools as ct
from pathlib import Path
from coremltools.optimize.torch.quantization import (
    LinearQuantizer,
    LinearQuantizerConfig,
    ModuleLinearQuantizerConfig,
)

from . import _coreml_utils  # noqa: F401
from ..utils import iter_namedtuple, to_torch_namedtuple
from ._base_exporter import BaseExporter
from ..types import ModelType, ModelPrecision, ConversionMetadata, ModelData


def _quantize(model: torch.nn.Module, example_model_data: list[ModelData]) -> torch.nn.Module:
    # Only for benchmarking purposes, highly degrades model accuracy

    config = LinearQuantizerConfig(
        global_config=ModuleLinearQuantizerConfig(
            quantization_scheme="symmetric",
            milestones=[0, 1000, 1000, 0],
            # weight_per_channel=False
        )
    )
    quantizer = LinearQuantizer(model, config)
    example_inputs = example_model_data[-1].inputs
    model_prepared = quantizer.prepare(example_inputs=to_torch_namedtuple(example_inputs))
    model_prepared.eval()
    quantizer.step()

    # For time being, use example data to gather quantization parameters
    with torch.no_grad():
        model_prepared.eval()
        print(f"Running model to gather quantization parameters... ({len(example_model_data)} steps)")
        for example_data in example_model_data:
            model_prepared(*example_data.inputs)

    quantized_model = quantizer.finalize()
    return quantized_model


class CoreMLExporter(BaseExporter):
    def __init__(
        self,
        quantize_int8: bool = False,
        minimum_deployment_target: str | None = None,
        coreml_prepend_pass_pipelines=[],
        coreml_append_pass_pipelines: list[str] = [
            "mlvc::fast_prediction_workaround",
            "mlvc::pixel_shuffle_workaround",
            "mlvc::split_gated_conv",
            "mlvc::gather_first",
        ],
        **kwargs,
    ):
        super().__init__(model_type=ModelType.COREML, **kwargs)
        self._quantize_int8 = quantize_int8
        self._minimum_deployment_target = minimum_deployment_target
        self._coreml_prepend_pass_pipelines = coreml_prepend_pass_pipelines
        self._coreml_append_pass_pipelines = coreml_append_pass_pipelines

    @torch.inference_mode()
    def _export(
        self,
        model_name: str,
        model: torch.nn.Module,
        example_model_data: list[ModelData],
        output_path: Path,
        fake_quantized: bool = False,
    ) -> None:
        if self._quantize_int8:
            print("Quantizing model...")
            model = _quantize(model, example_model_data)

        example_input = example_model_data[-1].inputs
        example_output = example_model_data[-1].outputs
        traced_model = torch.jit.trace(model, example_inputs=to_torch_namedtuple(example_input))
        print(f"{model_name} traced successfully")

        def _dtype_mapping(dtype):
            if np.issubdtype(dtype, np.floating):
                return np.float32 if self._precision == ModelPrecision.FP32 else np.float16
            return dtype

        inputs = [
            ct.TensorType(
                shape=value.shape,
                dtype=_dtype_mapping(value.dtype),
                name=field,
            )
            for field, value in iter_namedtuple(example_input)
        ]
        outputs = [
            ct.TensorType(dtype=_dtype_mapping(value.dtype), name=field)
            for field, value in iter_namedtuple(example_output)
        ]

        pipeline = ct.PassPipeline.DEFAULT
        disabled_passes = [] if not fake_quantized else ["mlvc::pixel_shuffle_workaround"]
        for i, pass_name in enumerate(self._coreml_prepend_pass_pipelines):
            if pass_name not in disabled_passes:
                pipeline.insert_pass(i, pass_name)  # type: ignore[attr-defined]
        for pass_name in self._coreml_append_pass_pipelines:
            if pass_name not in disabled_passes:
                pipeline.append_pass(pass_name)  # type: ignore[attr-defined]

        compute_precision = ct.precision.FLOAT32 if self._precision == ModelPrecision.FP32 else ct.precision.FLOAT16

        if self._minimum_deployment_target is not None:
            deployment_target = getattr(ct.target, self._minimum_deployment_target)
        elif fake_quantized or self._quantize_int8:
            deployment_target = ct.target.macOS14
        else:
            deployment_target = ct.target.macOS13

        converted = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=outputs,
            minimum_deployment_target=deployment_target,
            convert_to="mlprogram",
            compute_precision=compute_precision,
            pass_pipeline=pipeline,  # type: ignore[arg-type]
            skip_model_load=True,
        )

        model_output_path = str(output_path / f"{model_name}.mlpackage")
        converted.save(model_output_path)  # type: ignore[attr-defined]
        print(f"Saved CoreML model to {model_output_path}")

    def _compose_metadata(self, *args, **kwargs) -> ConversionMetadata:
        res = super()._compose_metadata(*args, **kwargs)
        res.params.exporter_params.extra_params.update(
            {
                "quantize_int8": self._quantize_int8,
                "minimum_deployment_target": self._minimum_deployment_target,
                "coreml_prepend_pass_pipelines": self._coreml_prepend_pass_pipelines,
                "coreml_append_pass_pipelines": self._coreml_append_pass_pipelines,
            }
        )
        return res
