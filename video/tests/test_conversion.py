# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import pytest
import platform
from pathlib import Path
from conversion.types import ModelType
from conversion import (
    load_split_model,
    full_model_factory,
    split_full_model,
    exporter_factory,
    aggregate_frame_loop_results,
    FrameLoop,
)

_TEST_DATA = []

if sys.platform == "darwin":
    # Apple
    _TEST_DATA.extend(
        [
            # DMC61
            ("coreml", "dmc61", 42, 43.1735, 0.002274),
            ("coreml", "dmc61", 63, 46.0871, 0.009965),
            ("coreml", "dmc61_silu", 42, 43.7401, 0.003524),
            ("coreml", "dmc61_silu", 63, 46.5296, 0.014740),
            ("coreml", "dmc61r_silu", 42, 43.2337, 0.003038),
            ("coreml", "dmc61r_silu", 63, 46.4472, 0.014670),
            # DMC61S
            ("coreml", "dmc61s", 42, 42.7607, 0.004653),
            ("coreml", "dmc61s", 63, 45.3373, 0.014826),
            ("coreml", "dmc61s_silu", 42, 42.8402, 0.005399),
            ("coreml", "dmc61s_silu", 63, 45.5020, 0.019219),
            ("coreml", "dmc61s_lrelu", 42, 42.7718, 0.005191),
            ("coreml", "dmc61s_lrelu", 63, 45.3752, 0.016510),
            ("coreml", "dmc61sr_lrelu", 42, 42.4134, 0.005538),
            ("coreml", "dmc61sr_lrelu", 63, 45.2609, 0.018576),
            # DMC61SB
            ("coreml", "dmc61sb", 42, 43.3025, 0.004028),
            ("coreml", "dmc61sb", 63, 45.6791, 0.013194),
            ("coreml", "dmc61sb_silu", 42, 41.9917, 0.004549),
            ("coreml", "dmc61sb_silu", 63, 44.1281, 0.016111),
            ("coreml", "dmc61sb_lrelu", 42, 42.9394, 0.004010),
            ("coreml", "dmc61sb_lrelu", 63, 45.2115, 0.014896),
            ("coreml", "dmc61sbr_lrelu", 42, 43.0709, 0.005226),
            ("coreml", "dmc61sbr_lrelu", 63, 45.5799, 0.016319),
            ("coreml", "dmc61sbr_reglu", 42, 43.5419, 0.005122),
            ("coreml", "dmc61sbr_reglu", 63, 46.2123, 0.017778),
            # DMC61SB-mini
            ("coreml", "dmc61sbr_mini_reglu", 42, 43.0705, 0.014219),
            ("coreml", "dmc61sbr_mini_reglu", 63, 45.0526, 0.042674),
        ]
    )
elif sys.platform == "win32" and platform.machine() == "AMD64":
    # Intel
    _TEST_DATA.extend(
        [
            ("openvino", "dmc61s_lrelu", 42, 42.7326, 0.004983),
            ("openvino", "dmc61s_lrelu", 63, 45.3849, 0.016597),
            ("openvino", "dmc61sbr_lrelu", 42, 43.0871, 0.004826),
            ("openvino", "dmc61sbr_lrelu", 63, 45.5938, 0.016667),
        ]
    )
elif sys.platform == "win32" and platform.machine() == "ARM64":
    # Qualcomm
    _TEST_DATA.extend(
        [
            ("onnx", "dmc61s_lrelu", 42, 42.7361, 0.004983),
            ("onnx", "dmc61s_lrelu", 63, 45.3582, 0.016476),
            ("onnx", "dmc61sbr_lrelu", 42, 43.1019, 0.004913),
            ("onnx", "dmc61sbr_lrelu", 63, 45.5713, 0.015503),
        ]
    )


@pytest.fixture(scope="session")
def persistent_tmpdir(tmp_path_factory) -> Path:
    temp_dir = tmp_path_factory.mktemp("persistent_temp")
    return temp_dir


def convert_test_model(model_version: str, model_type: ModelType, output_path: Path, **extra_params) -> Path:
    full_model = full_model_factory(model_version=model_version)
    split_model = split_full_model(full_model=full_model)
    exporter = exporter_factory(
        split_model=split_model,
        model_type=model_type,
        skip_if_exists=True,
        output_path=output_path,
        frame_count=1,  # To speed up tests
        q_index_list=[63],  # To speed up tests
        **extra_params,
    )
    model_path = exporter.run()
    print(f"Model exported to {model_path}")
    return model_path


@pytest.mark.parametrize("model_type,model_version,q_index,expected_psnr,expected_bpp", _TEST_DATA)
def test_conversion(
    persistent_tmpdir: Path,
    model_type: str,
    model_version: str,
    q_index: int,
    expected_psnr: float,
    expected_bpp: float,
):
    # Convert
    model_path = convert_test_model(model_version, ModelType(model_type), persistent_tmpdir)

    # Validate
    split_model = load_split_model(model_path)
    loop = FrameLoop(
        split_model=split_model,
        video_path="yuv/640x360_30fps/s1/0380a333cb0fef69001fb4260e2a705f_640x360_30fps.yuv",
        image_width=640,
        image_height=360,
        frame_count=128,
        use_decoder=True,
    )
    loop_results = loop.run(q_index=q_index, reset_period=32)
    loop_summary = aggregate_frame_loop_results(loop_results)

    # Assert
    psnr = loop_summary.psnr.median
    bpp = loop_summary.bpp.median
    print(f'Expected: ("{model_type}", "{model_version}", {q_index}, {psnr:.4f}, {bpp:.6f}),')
    assert pytest.approx(psnr, abs=1e-4) == expected_psnr
    assert pytest.approx(bpp, abs=1e-5) == expected_bpp
