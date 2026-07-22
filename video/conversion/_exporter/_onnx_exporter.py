# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from pathlib import Path
from ._base_exporter import BaseExporter
from ..types import ModelType, TargetDevice, ModelPrecision, ModelData, ConversionMetadata
from ..utils import get_namedtuple_fields, to_torch_namedtuple
from ._onnx_utils import DEFAULT_ONNX_PASSES, OnnxOptimizer


class OnnxExporter(BaseExporter):
    def __init__(
        self,
        *,
        target_device: TargetDevice,
        model_type: ModelType = ModelType.ONNX,
        opset_version: int = 18,
        optimization_passes: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_type=model_type,
            target_device=target_device,
            **kwargs,
        )
        self._opset_version = opset_version

        if optimization_passes is not None:
            self._optimization_passes = optimization_passes
        else:
            if target_device == TargetDevice.INTEL:
                self._optimization_passes = [
                    "onnxscript_optimizations",
                    "depth_to_space_crd_to_dcr",
                ]
            elif target_device == TargetDevice.QUALCOMM:
                self._optimization_passes = [
                    "onnxscript_optimizations",
                    "use_space_to_depth",
                    "depth_to_space_crd_to_dcr",
                    "replace_reciprocal_op",
                    "qc_workaround_floor_after_round",
                    "replace_slice_with_split",
                    "split_gated_conv:2",
                    "qc_workaround_squeeze_gather_4d",
                    "qc_workaround_clip_min_only",
                ]
            else:
                self._optimization_passes = DEFAULT_ONNX_PASSES

    @torch.inference_mode()
    def _export(
        self,
        model_name: str,
        model: torch.nn.Module,
        example_model_data: list[ModelData],
        output_path: Path,
        fake_quantized: bool = False,
    ) -> None:
        example_input = example_model_data[-1].inputs
        example_output = example_model_data[-1].outputs
        input_names = get_namedtuple_fields(example_input)
        output_names = get_namedtuple_fields(example_output)

        model_output_path = str(output_path / f"{model_name}.onnx")
        torch.onnx.export(
            model=model,
            args=to_torch_namedtuple(example_input),
            f=model_output_path,
            export_params=True,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=self._opset_version,
            dynamo=False,
        )

        if self._precision == ModelPrecision.FP16:
            import onnx
            import warnings
            from onnxconverter_common import float16

            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="onnxconverter_common.float16",
            )

            model_fp16 = float16.convert_float_to_float16(onnx.load(model_output_path))
            onnx.save(model_fp16, model_output_path)

        if len(self._optimization_passes) > 0:
            print(f"Optimizing ONNX model with passes: {self._optimization_passes}")
            opt_model = onnx.load(model_output_path)
            optimizer = OnnxOptimizer(opt_model)
            opt_model = optimizer.optimize(passes=self._optimization_passes)
            onnx.save(opt_model, model_output_path)

        print(f"Saved ONNX model to {model_output_path}")

    def _compose_metadata(self, *args, **kwargs) -> ConversionMetadata:
        res = super()._compose_metadata(*args, **kwargs)
        res.params.exporter_params.extra_params.update(
            {
                "optimization_passes": self._optimization_passes,
            }
        )
        return res
