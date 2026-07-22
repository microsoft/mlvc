# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import platform
from pathlib import Path
from .types import ModelType, TargetDevice


DEFAULT_JOB_OUTPUTS_DIR = os.environ.get("VIDEO_JOB_OUTPUTS_DIR", "~/datasets/job-outputs/")
DEFAULT_TEST_DATA_DIR = os.environ.get("VIDEO_TEST_DATA_DIR", "~/datasets/test-set/")
DEFAULT_EXPORT_DIR = str((Path(__file__).parent.parent / "output" / "models").resolve())
DEFAULT_BUNDLE_DIR = str((Path(__file__).parent.parent / "output" / "model_bundles").resolve())

DEFAULT_MODEL_WIDTH = 640
DEFAULT_MODEL_HEIGHT = 368

DEFAULT_CONVERT_FRAME_COUNT = 48
DEFAULT_BENCHMARK_FRAME_COUNT = 300
DEFAULT_INCLUDE_BITSTREAM_OVERHEAD = False
DEFAULT_METRICS_BIT_DEPTH = 32

DEFAULT_TEST_CONFIG_PATH = "yuv/960x540_30fps/VCD-960x540_30fps.json"
DEFAULT_NUM_CLIPS_LIMIT = 5
DEFAULT_TEST_Q_INDEX_LIST = [0, 21, 42, 63]
DEFAULT_ANCHOR_PATH = "benchmark_test/anchor/VCD_960x540_30fps/intel_hw_hevc_lp.json"

DEFAULT_MODEL_TYPE = ModelType.ONNX
if sys.platform == "darwin":
    # Apple
    DEFAULT_MODEL_TYPE = ModelType.COREML
elif sys.platform == "win32" and platform.machine() == "AMD64":
    # Intel
    DEFAULT_MODEL_TYPE = ModelType.OPENVINO
elif sys.platform == "win32" and platform.machine() == "ARM64":
    # Qualcomm
    DEFAULT_MODEL_TYPE = ModelType.ONNX


def get_default_target_device(model_type: ModelType) -> TargetDevice:
    if model_type == ModelType.COREML:
        return TargetDevice.APPLE
    elif model_type == ModelType.ONNX:
        return TargetDevice.QUALCOMM
    elif model_type == ModelType.OPENVINO:
        return TargetDevice.INTEL
    elif model_type == ModelType.TORCH:
        return TargetDevice.GENERIC
    raise ValueError(f"Unknown model type: {model_type}")


def get_default_test_video(model_width: int, model_height: int) -> tuple[Path, int, int]:
    test_video_resolutions = [
        (1920, 1080, "30"),
        (1280, 720, "30fps"),
        (960, 540, "30fps"),
        (640, 360, "30fps"),
        (426, 240, "30fps"),
        (320, 180, "30fps"),
        (160, 90, "30.0fps"),
    ]

    if model_width >= model_height:
        # Landscape orientation
        for width, height, suffix in test_video_resolutions:
            if model_height >= height and model_width >= width:
                return (
                    Path(
                        f"yuv/{width}x{height}_30fps/s1/0380a333cb0fef69001fb4260e2a705f_{width}x{height}_{suffix}.yuv"
                    ),
                    width,
                    height,
                )
    else:
        # Portrait orientation
        for height, width, suffix in test_video_resolutions:
            if model_width >= width and model_height >= height:
                return (
                    Path(
                        f"yuv/{height}x{width}_30fps/s4/3f7df79cd3338701dfce79b3ec82531c_{width}x{height}_{suffix}.yuv"
                    ),
                    width,
                    height,
                )

    raise ValueError(f"Could not get default test video for resolution: {model_width}x{model_height}")
