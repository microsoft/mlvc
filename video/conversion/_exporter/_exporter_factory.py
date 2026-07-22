# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ..types import TargetDevice, ModelType
from ..const import get_default_target_device
from ._base_exporter import BaseExporter


def exporter_factory(model_type: ModelType, target_device: TargetDevice | None = None, **kwargs) -> BaseExporter:

    if target_device is None:
        target_device = get_default_target_device(model_type)

    if model_type == ModelType.COREML:
        from ._coreml_exporter import CoreMLExporter

        return CoreMLExporter(target_device=target_device, **kwargs)
    elif model_type == ModelType.ONNX:
        from ._onnx_exporter import OnnxExporter

        return OnnxExporter(target_device=target_device, **kwargs)
    elif model_type == ModelType.OPENVINO:
        from ._openvino_exporter import OpenvinoExporter

        return OpenvinoExporter(target_device=target_device, **kwargs)
    elif model_type == ModelType.TORCH:
        from ._torch_exporter import TorchExporter

        return TorchExporter(target_device=target_device, **kwargs)

    raise ValueError(f"Unknown model type: {model_type}")
