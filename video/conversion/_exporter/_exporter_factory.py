# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ..const import get_default_target_device
from ..types import ModelPrecision, ModelType, TargetDevice, TensorLayout
from ._base_exporter import BaseExporter


def exporter_factory(
    model_type: ModelType,
    target_device: TargetDevice | None = None,
    *,
    image_layout: TensorLayout | None = None,
    feature_layout: TensorLayout | None = None,
    **kwargs,
) -> BaseExporter:

    if target_device is None:
        target_device = get_default_target_device(model_type)

    if "precision" in kwargs:
        kwargs["precision"] = ModelPrecision(kwargs["precision"])

    default_layout = (
        TensorLayout.NHWC
        if model_type == ModelType.ONNX and target_device == TargetDevice.QUALCOMM
        else TensorLayout.NCHW
    )
    image_layout = default_layout if image_layout is None else TensorLayout(image_layout)
    feature_layout = default_layout if feature_layout is None else TensorLayout(feature_layout)

    if model_type == ModelType.COREML:
        from ._coreml_exporter import CoreMLExporter

        return CoreMLExporter(
            target_device=target_device,
            image_layout=image_layout,
            feature_layout=feature_layout,
            **kwargs,
        )
    elif model_type == ModelType.ONNX:
        from ._onnx_exporter import OnnxExporter

        return OnnxExporter(
            target_device=target_device,
            image_layout=image_layout,
            feature_layout=feature_layout,
            **kwargs,
        )
    elif model_type == ModelType.OPENVINO:
        from ._openvino_exporter import OpenvinoExporter

        return OpenvinoExporter(
            target_device=target_device,
            image_layout=image_layout,
            feature_layout=feature_layout,
            **kwargs,
        )
    elif model_type == ModelType.TORCH:
        from ._torch_exporter import TorchExporter

        return TorchExporter(
            target_device=target_device,
            image_layout=image_layout,
            feature_layout=feature_layout,
            **kwargs,
        )

    raise ValueError(f"Unknown model type: {model_type}")
