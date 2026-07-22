# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import torch
import platform
import tempfile
import tarfile
import shutil
import numpy as np
from pathlib import Path
from typing import Any, NamedTuple
from collections import namedtuple
from . import _windowsml
from .types import (
    ModelType,
    ModelOpProfile,
    ModelPartMetadata,
    ModelPrecision,
    CoremlComputeUnits,
    OnnxExecutionProvider,
    RuntimeParams,
)


class ModelWrapper:
    def __init__(
        self,
        model_type: ModelType,
        model: Any,
        metadata: ModelPartMetadata,
        runtime_params: RuntimeParams,
    ) -> None:
        self._model_type = model_type
        self._model = model
        self._metadata = metadata
        self._runtime_params = runtime_params
        self._input_type = namedtuple("ModelWrapperInput", metadata.input_fields)
        self._output_type = namedtuple("ModelWrapperOutput", metadata.output_fields)
        self._inference_time = 0.0

    @classmethod
    def load_converted_model(
        cls,
        model_type: ModelType,
        model_path: Path | str,
        metadata: ModelPartMetadata,
        runtime_params: RuntimeParams,
        function_name: str | None = None,
    ):

        if function_name is not None and model_type != ModelType.COREML:
            raise ValueError("Function name is only supported for CoreML models")

        if model_type == ModelType.ONNX:
            model = _load_onnx_model(model_path, runtime_params)
        elif model_type == ModelType.COREML:
            model = _load_coreml_model(model_path, runtime_params, function_name)
        elif model_type == ModelType.OPENVINO:
            model = _load_openvino_model(model_path, runtime_params)
        elif model_type == ModelType.TORCH:
            model = _load_torch_model(model_path, runtime_params)
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")

        wrapper = cls(
            model_type=model_type,
            model=model,
            metadata=metadata,
            runtime_params=runtime_params,
        )
        return wrapper

    @torch.inference_mode()
    def predict(self, inputs: NamedTuple) -> Any:
        # Prepare input
        model_inputs = self._input_type(**{k: self._transform_input(v) for k, v in inputs._asdict().items()})

        # Predict
        predict_start = time.perf_counter()
        if self._model_type == ModelType.TORCH:
            model_outputs = self._model(**model_inputs._asdict())._asdict()
        elif self._model_type == ModelType.ONNX:
            model_outputs = self._model.run(None, model_inputs._asdict())
            model_outputs = self._output_type(*model_outputs)._asdict()
        elif self._model_type == ModelType.COREML:
            model_outputs = self._model.predict(model_inputs._asdict())
        elif self._model_type == ModelType.OPENVINO:
            model_outputs = self._model.infer(
                model_inputs._asdict(),
                share_inputs=True,
                share_outputs=True,
            )
            model_outputs = self._output_type(*[v for v in model_outputs.values()])._asdict()
        else:
            raise NotImplementedError(f"Model type {self._model_type} not implemented")
        self._inference_time = time.perf_counter() - predict_start

        # Prepare output
        model_outputs = self._output_type(**{k: self._transform_output(v) for k, v in model_outputs.items()})
        return model_outputs

    def _transform_input(self, value: np.ndarray) -> torch.Tensor | np.ndarray:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Input must be a np.ndarray, got {type(value)}")

        if self._model_type in ModelType.COREML:
            # For CoreML it is more efficient to always use float32 inputs
            if value.dtype == np.float16:
                # Use torch for faster conversion
                value = torch.from_numpy(value).to(torch.float32).numpy()
        else:
            # Convert input to model precision
            dtype = ModelPrecision.to_numpy_dtype(self._metadata.precision)
            if value.dtype in [np.float16, np.float32] and value.dtype != dtype:
                value = value.astype(dtype)

        if self._model_type == ModelType.TORCH:
            return torch.from_numpy(value)
        return value

    def _transform_output(self, value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.numpy()

        dtype = ModelPrecision.to_numpy_dtype(self._metadata.precision)
        torch_dtype = ModelPrecision.to_torch_dtype(self._metadata.precision)
        if value.dtype in [np.float16, np.float32] and value.dtype != dtype:
            # Use torch for faster conversion
            value = torch.from_numpy(value).to(dtype=torch_dtype).numpy()
        return value

    @property
    def metadata(self) -> ModelPartMetadata:
        return self._metadata

    @property
    def torch_model(self) -> torch.nn.Module:
        if self._model_type != ModelType.TORCH:
            raise ValueError("Model is not a torch model")
        return self._model

    @property
    def model_type(self) -> ModelType:
        return self._model_type

    @property
    def model(self) -> Any:
        return self._model

    @property
    def inference_time(self) -> float:
        return self._inference_time

    @property
    def op_profile(self) -> list[ModelOpProfile]:
        res: list[ModelOpProfile] = []
        if self._model_type == ModelType.OPENVINO:
            for p in self._model.profiling_info:
                preferred_compute_device = self._runtime_params.openvino_device.value
                res.append(
                    ModelOpProfile(
                        operator_type=p.node_type,
                        cost=1e3 * p.real_time.total_seconds(),
                        preferred_compute_device=preferred_compute_device,
                        operator_name=p.node_name,
                        status=p.status.name,
                        exec_type=p.exec_type,
                        real_time=p.real_time.total_seconds(),
                        cpu_time=p.cpu_time.total_seconds(),
                    )
                )
        return res


def _load_onnx_model(model_path: Path | str, runtime_params: RuntimeParams):
    print(f"Loading ONNX model from {model_path} with execution provider {runtime_params.onnx_execution_provider}")
    _windowsml.ensure_initialized()
    import onnxruntime as ort  # type: ignore[import-not-found]

    sess_options = ort.SessionOptions()
    # sess_options.intra_op_num_threads = 1  # To match roottools results
    # sess_options.log_severity_level = 0

    provider_name = "CPUExecutionProvider"
    provider_options: dict[str, str] = {}
    if runtime_params.onnx_execution_provider == OnnxExecutionProvider.CPU:
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    elif runtime_params.onnx_execution_provider == OnnxExecutionProvider.CUDA:
        provider_name = "CUDAExecutionProvider"
    elif runtime_params.onnx_execution_provider == OnnxExecutionProvider.DIRECTML:
        # Need to disable all optimizations to have valid results
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_mem_pattern = False
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        provider_name = "DmlExecutionProvider"
        provider_options = {
            "disable_metacommands": "true",
            "performance_preference": "high_performance",
            "device_filter": "gpu",
        }
    elif runtime_params.onnx_execution_provider == OnnxExecutionProvider.OPENVINO:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        if platform.system() == "Windows":
            import onnxruntime.tools.add_openvino_win_libs as utils  # type: ignore[import-not-found]

            utils.add_openvino_libs_to_path()

        provider_name = "OpenVINOExecutionProvider"
        provider_options = {
            "device_type": "NPU",
            "precision": "FP16",
        }
    elif runtime_params.onnx_execution_provider == OnnxExecutionProvider.QNN:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

        provider_name = "QNNExecutionProvider"
        provider_options = {
            "backend_path": "QnnHtp.dll",
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
        }
    elif runtime_params.onnx_execution_provider == OnnxExecutionProvider.COREML:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        provider_name = "CoreMLExecutionProvider"
        provider_options = {
            "ModelFormat": "MLProgram",
            "MLCoremlComputeUnits": "CPUAndNeuralEngine",
            "RequireStaticInputShapes": "1",
            "SpecializationStrategy": "FastPrediction",
        }
    else:
        raise NotImplementedError(f"Unsupported onnx backend: {runtime_params.onnx_execution_provider}")

    provider_options.update(runtime_params.onnx_provider_options)

    # When the plugin EP system is available (onnxruntime-windowsml or onnxruntime >= 1.24),
    # prefer add_provider_for_devices() over the legacy providers list.
    # Fall back to the legacy path if the EP is available as a provider but not as a device
    # (e.g. onnxruntime-windowsml is not installed but onnxruntime has get_ep_devices).
    providers: list | None = None
    if hasattr(ort, "get_ep_devices"):
        ep_device_type = {
            "CPUExecutionProvider": ort.OrtHardwareDeviceType.CPU,
            "CUDAExecutionProvider": ort.OrtHardwareDeviceType.GPU,
            "DmlExecutionProvider": ort.OrtHardwareDeviceType.GPU,
            "OpenVINOExecutionProvider": ort.OrtHardwareDeviceType.NPU,
            "QNNExecutionProvider": ort.OrtHardwareDeviceType.NPU,
            "CoreMLExecutionProvider": ort.OrtHardwareDeviceType.NPU,
        }[provider_name]
        ep_devices = [d for d in ort.get_ep_devices() if d.ep_name == provider_name and d.device.type == ep_device_type]
        if ep_devices:
            plugin_options = {k: v for k, v in provider_options.items() if k not in ["backend_path"]}
            sess_options.add_provider_for_devices(ep_devices, plugin_options)
        elif provider_name in ort.get_available_providers():
            providers = [(provider_name, provider_options)]
        else:
            raise RuntimeError(f"Requested EP {provider_name} not found in available EP devices or providers")
    else:
        if provider_name not in ort.get_available_providers():
            raise RuntimeError(f"{provider_name} is not available")
        providers = [(provider_name, provider_options)]

    model = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=providers,
    )
    print(f"Loading ONNX model from {model_path} with execution provider {runtime_params.onnx_execution_provider} done")
    return model


def _load_coreml_model(
    model_path: Path | str,
    runtime_params: RuntimeParams,
    function_name: str | None,
):
    import coremltools as ct

    compute_units = {
        CoremlComputeUnits.CPU: ct.ComputeUnit.CPU_ONLY,
        CoremlComputeUnits.GPU: ct.ComputeUnit.CPU_AND_GPU,
        CoremlComputeUnits.NPU: ct.ComputeUnit.CPU_AND_NE,
        CoremlComputeUnits.ALL: ct.ComputeUnit.ALL,
    }[runtime_params.coreml_compute_units]
    optimization_hints = {}
    if runtime_params.coreml_fast_prediction:
        optimization_hints["specializationStrategy"] = ct.SpecializationStrategy.FastPrediction

    try:
        tmp_model_path = None
        if Path(model_path).suffix == ".tar":
            tmp_model_path = tempfile.mkdtemp(suffix=".mlpackage")
            with tarfile.open(model_path, "r") as tar:
                tar.extractall(path=tmp_model_path)

        model = ct.models.MLModel(
            str(model_path) if tmp_model_path is None else str(tmp_model_path),
            compute_units=compute_units,
            optimization_hints=optimization_hints,
            function_name=function_name,
        )
    finally:
        if tmp_model_path is not None:
            shutil.rmtree(tmp_model_path, ignore_errors=True)

    return model


def _load_openvino_model(model_path: Path | str, runtime_params: RuntimeParams):
    import openvino as ov  # type: ignore[import-not-found]

    config = {}
    if runtime_params.openvino_enable_profiling:
        config[ov.properties.enable_profiling] = True

    load_model_path = (
        Path(model_path) if not runtime_params.openvino_use_onnx else Path(model_path).with_suffix(".onnx")
    )

    core = ov.Core()
    compiled_model = core.compile_model(
        model=load_model_path,
        device_name=runtime_params.openvino_device.value.upper(),
        config=config,
    )
    infer_request = compiled_model.create_infer_request()
    return infer_request


def _load_torch_model(model_path: Path | str, runtime_params: RuntimeParams):
    model = torch.load(str(model_path), weights_only=False)
    model = model.to(torch.device(runtime_params.torch_device.value))
    model.eval()
