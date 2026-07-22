# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import json
import argparse
import dataclasses
from pathlib import Path
from enum import Enum
from typing import Any
from conversion.const import (
    DEFAULT_JOB_OUTPUTS_DIR,
    DEFAULT_TEST_DATA_DIR,
    DEFAULT_EXPORT_DIR,
    DEFAULT_MODEL_TYPE,
    DEFAULT_MODEL_WIDTH,
    DEFAULT_MODEL_HEIGHT,
    DEFAULT_TEST_CONFIG_PATH,
    DEFAULT_NUM_CLIPS_LIMIT,
    DEFAULT_TEST_Q_INDEX_LIST,
    DEFAULT_ANCHOR_PATH,
    DEFAULT_CONVERT_FRAME_COUNT,
    DEFAULT_INCLUDE_BITSTREAM_OVERHEAD,
    DEFAULT_METRICS_BIT_DEPTH,
)
from conversion.types import (
    TargetDevice,
    ModelType,
    ModelPrecision,
    ScaleDecoderType,
    RuntimeParams,
    CoremlComputeUnits,
    OnnxExecutionProvider,
    TorchDevice,
    OpenvinoDevice,
    ConversionMetadata,
    MlvcVersion,
    PaddingMode,
    PaddingDirection,
)
from conversion import (
    ModelTester,
    full_model_factory,
    split_full_model,
    load_split_model,
    exporter_factory,
    model_bundler_factory,
    print_runtime_params,
    print_validate_conversion_results,
    print_profile_results,
    print_benchmark_results,
    print_validation_test_results,
    get_available_models,
    get_available_split_types,
)


class Command(str, Enum):
    EXPORT = "export"
    TEST = "test"
    BUNDLE = "bundle"


class TestCommand(str, Enum):
    VALIDATE_CONVERSION = "validate_conversion"
    PROFILE = "profile"
    BENCHMARK = "benchmark"
    RUN_VALIDATION_TEST = "run_validation_test"


# -------------------------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------------------------


def _add_common_arguments(parser: argparse.ArgumentParser):
    data_dirs_group = parser.add_argument_group("Data directories")
    data_dirs_group.add_argument(
        "--job-outputs-dir",
        type=str,
        default=DEFAULT_JOB_OUTPUTS_DIR,
        help="Path to the job outputs directory (env: VIDEO_JOB_OUTPUTS_DIR)",
    )

    data_dirs_group.add_argument(
        "--test-data-dir",
        type=str,
        default=DEFAULT_TEST_DATA_DIR,
        help="Path to the test data directory (env: VIDEO_TEST_DATA_DIR)",
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser | argparse._ArgumentGroup) -> None:
    default_value = RuntimeParams()
    parser.add_argument(
        "--coreml-compute-units",
        type=CoremlComputeUnits,
        choices=[v.value for v in CoremlComputeUnits],
        default=default_value.coreml_compute_units.value,
        help="Compute units to use for model inference",
    )
    parser.add_argument(
        "--coreml-fast-prediction",
        action=argparse.BooleanOptionalAction,
        default=default_value.coreml_fast_prediction,
        help="Enable fast prediction for CoreML models",
    )
    parser.add_argument(
        "--onnx-execution-provider",
        type=OnnxExecutionProvider,
        choices=[x.value for x in OnnxExecutionProvider],
        default=default_value.onnx_execution_provider.value,
        help="Execution provider for ONNX models",
    )
    parser.add_argument(
        "--torch-device",
        type=TorchDevice,
        choices=[v.value for v in TorchDevice],
        default=default_value.torch_device.value,
        help="Device to use for PyTorch models",
    )
    parser.add_argument(
        "--openvino-device",
        type=OpenvinoDevice,
        choices=[x.value for x in OpenvinoDevice],
        default=default_value.openvino_device.value,
        help="Device to use for OpenVINO models",
    )
    parser.add_argument(
        "--openvino-enable-profiling",
        action=argparse.BooleanOptionalAction,
        default=default_value.openvino_enable_profiling,
        help="Enable profiling for OpenVINO models",
    )
    parser.add_argument(
        "--openvino-use-onnx",
        action=argparse.BooleanOptionalAction,
        default=default_value.openvino_use_onnx,
        help="Use ONNX model files",
    )


def _add_exporter_arguments(parser: argparse.ArgumentParser | argparse._ArgumentGroup):
    parser.add_argument(
        "--model-type",
        "-t",
        type=ModelType,
        default=DEFAULT_MODEL_TYPE.value,
        choices=[x.value for x in ModelType],
        help="Type of the model conversion",
    )
    parser.add_argument(
        "--target-device",
        "-d",
        type=TargetDevice,
        default=None,
        choices=[x.value for x in TargetDevice],
        help="Target device for the model",
    )

    parser.add_argument(
        "--precision",
        "-p",
        type=ModelPrecision,
        choices=[x.value for x in ModelPrecision],
        default=ModelPrecision.FP16.value,
        help="Precision of the model",
    )

    parser.add_argument(
        "--scale-decoder-type",
        type=ScaleDecoderType,
        default=None,
        choices=[x.value for x in ScaleDecoderType],
        help="Scale decoder export format (default: auto)",
    )

    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default=DEFAULT_EXPORT_DIR,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--output-name",
        "-n",
        type=str,
        default=None,
        help="Name of the converted model",
    )

    parser.add_argument(
        "--skip-if-exists",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip conversion if the model already exists",
    )

    parser.add_argument(
        "--frame-count",
        type=int,
        default=DEFAULT_CONVERT_FRAME_COUNT,
        help="Number of frames to process in conversion loop",
    )

    parser.add_argument(
        "--exporter-params-json",
        type=str,
        default="{}",
        help="JSON string with additional exporter parameters",
    )


def _add_run_validation_test_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--test-config",
        type=str,
        default=DEFAULT_TEST_CONFIG_PATH,
        help="Path to the test config file",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="./output/validation/VCD_960x540_30fps",
        help="Path to the output file",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="test",
        help="Name of the output file",
    )

    parser.add_argument(
        "--anchor-path",
        type=str,
        default=DEFAULT_ANCHOR_PATH,
        help="Path to the anchor file",
    )

    parser.add_argument(
        "--num-clips-limit",
        type=int,
        default=DEFAULT_NUM_CLIPS_LIMIT,
        help="Number of clips to process",
    )

    parser.add_argument(
        "--q-index-list",
        type=int,
        nargs="+",
        help=(
            "List of quantization indices to use. "
            f"If neither q-index nor bitrate values are set, defaults to {DEFAULT_TEST_Q_INDEX_LIST}."
        ),
        default=None,
    )

    parser.add_argument(
        "--bitrate-list",
        type=float,
        nargs="+",
        help="List of constant bitrate values to use (bits per second), e.g. --bitrate-list 100e3 150e3 225e3 337e3",
        default=None,
    )

    parser.add_argument(
        "--scenarios-list",
        type=str,
        nargs="+",
        help="List of test scenarios to run",
        default=None,
    )

    parser.add_argument(
        "--use-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use encoder in the test",
    )

    parser.add_argument(
        "--use-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use decoder in the test",
    )

    parser.add_argument(
        "--iframe-period",
        type=int,
        default=None,
        help="Override I-frame period",
    )
    parser.add_argument(
        "--reset-period",
        type=int,
        default=None,
        help="Override reset period",
    )
    parser.add_argument(
        "--ltr-start-idx",
        type=int,
        default=None,
        help="Override frame index of the first LTR frame",
    )
    parser.add_argument(
        "--ltr-period",
        type=int,
        default=None,
        help="Override period between LTR frames",
    )

    parser.add_argument(
        "--padding-mode",
        type=PaddingMode,
        default=None,
        choices=[x.value for x in PaddingMode],
        help="Override frame padding fill strategy for validation tests",
    )

    parser.add_argument(
        "--padding-direction",
        type=PaddingDirection,
        default=None,
        choices=[x.value for x in PaddingDirection],
        help="Override where padding is placed around the source frame",
    )

    parser.add_argument(
        "--include-bitstream-overhead",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_INCLUDE_BITSTREAM_OVERHEAD,
        help="Estimate bitstream overhead for validation-test bit accounting",
    )

    parser.add_argument(
        "--metrics-bit-depth",
        default=DEFAULT_METRICS_BIT_DEPTH,
        type=int,
        help="Limit bit-depth of reconstructed frame before metrics calculation",
    )

    parser.add_argument(
        "--save-output-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save output data during validation test",
    )

    parser.add_argument(
        "--save-debug-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save additional model debug data during validation test",
    )

    parser.add_argument(
        "--save-yuv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save output YUV video during validation test",
    )


def _add_benchmark_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--collect-powermetrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect power metrics during benchmark",
    )

    group_test_video = parser.add_argument_group("Test video")
    group_test_video.add_argument(
        "--test-video-path",
        type=str,
        default=None,
        help="Path to the test video",
    )
    group_test_video.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Width of the test video",
    )
    group_test_video.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Height of the test video",
    )
    group_test_video.add_argument(
        "--frame-count",
        type=int,
        default=600,
        help="Number of frames to process",
    )


def _extract_runtime_params(args: dict[str, Any]) -> RuntimeParams:
    kwargs = {}
    for k in [
        "coreml_compute_units",
        "coreml_fast_prediction",
        "onnx_execution_provider",
        "torch_device",
        "openvino_device",
        "openvino_enable_profiling",
        "openvino_use_onnx",
    ]:
        if k in args:
            kwargs[k] = args.pop(k)
    return RuntimeParams.from_dict(kwargs)


def _add_export_arguments(parser: argparse.ArgumentParser):
    # Full model
    group_model = parser.add_argument_group("Full model")
    group_model.add_argument(
        "--model-version",
        "-m",
        required=True,
        type=str,
        choices=get_available_models(),
        help="Version of the model to convert",
    )
    group_model.add_argument(
        "--weights-version",
        type=str,
        default=None,
        help="Version of the model weights",
    )
    group_model.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="Path to the model weights",
    )
    parser.add_argument(
        "--iframe-period",
        type=int,
        default=None,
        help="Override I-frame period",
    )
    parser.add_argument(
        "--reset-period",
        type=int,
        default=None,
        help="Override reset period",
    )
    parser.add_argument(
        "--ltr-start-idx",
        type=int,
        default=None,
        help="Override frame index of the first LTR frame",
    )
    parser.add_argument(
        "--ltr-period",
        type=int,
        default=None,
        help="Override period between LTR frames",
    )
    group_model.add_argument(
        "--model-params-json",
        type=str,
        default="{}",
        help="JSON string with model parameters",
    )

    # Split model
    group_split = parser.add_argument_group("Split model")
    group_split.add_argument(
        "--split-type",
        type=str,
        choices=get_available_split_types(),
        default=None,
        help="Type of the model split",
    )

    group_split.add_argument(
        "--model-width",
        "-mw",
        type=int,
        default=DEFAULT_MODEL_WIDTH,
        help="Width of the model input",
    )
    group_split.add_argument(
        "--model-height",
        "-mh",
        type=int,
        default=DEFAULT_MODEL_HEIGHT,
        help="Height of the model input",
    )

    # Runtime
    group_runtime = parser.add_argument_group("Runtime params")
    _add_runtime_arguments(group_runtime)

    # Exporter
    group_exporter = parser.add_argument_group("Exporter")
    _add_exporter_arguments(group_exporter)

    # Testing
    group_testing = parser.add_argument_group("Testing")
    group_testing.add_argument(
        "--validate-conversion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate the model after conversion",
    )

    group_testing.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Profile the model after conversion",
    )

    group_testing.add_argument(
        "--benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Benchmark the model after conversion",
    )


def _add_test_model_arguments(parser: argparse.ArgumentParser):
    group_model_arguments = parser.add_argument_group("Model")
    group_model_arguments.add_argument(
        "--model-path",
        "-m",
        required=True,
        type=str,
        help="Path to the model",
    )
    group_model_arguments.add_argument(
        "--model-id",
        "-mid",
        type=str,
        default=None,
        help="ID of the model in the bundle, e.g. 640x368",
    )

    group_runtime_arguments = parser.add_argument_group("Runtime parameters")
    _add_runtime_arguments(group_runtime_arguments)

    subparsers = parser.add_subparsers(dest="test_command", required=True)

    # Validate conversion command
    subparsers.add_parser(
        TestCommand.VALIDATE_CONVERSION.value,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Validate conversion of the model",
    )

    # Profile command
    profile_parser = subparsers.add_parser(
        TestCommand.PROFILE.value,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Profile the model",
    )
    profile_parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to the output .json file",
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        TestCommand.BENCHMARK.value,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Benchmark the model",
    )
    _add_benchmark_arguments(benchmark_parser)

    # Run validation test command
    run_validation_test_parser = subparsers.add_parser(
        TestCommand.RUN_VALIDATION_TEST.value,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Run validation test",
    )
    _add_run_validation_test_arguments(run_validation_test_parser)


def _add_bundle_arguments(parser: argparse.ArgumentParser):
    def _parse_size(s: str) -> tuple[int, int]:
        try:
            w, h = s.lower().replace(",", "x").split("x")
            return int(w), int(h)
        except Exception:
            raise argparse.ArgumentTypeError(f"Invalid size '{s}'. Expected format WxH, e.g. 640x360")

    def _parse_mlvc_version(s: str) -> MlvcVersion:
        m = re.fullmatch(r"v(\d+)\.(\d+)", s)
        if m is None:
            raise argparse.ArgumentTypeError(
                f"Invalid bundle version '{s}'. Expected format X.Y where X and Y are numbers (e.g. 1.0, 0.1)."
            )
        return MlvcVersion(major=int(m[1]), minor=int(m[2]))

    group_bundle_arguments = parser.add_argument_group("Bundler")
    group_bundle_arguments.add_argument(
        "--model-type",
        "-t",
        type=ModelType,
        default=DEFAULT_MODEL_TYPE.value,
        choices=[x.value for x in ModelType],
        help="Type of the model",
    )
    group_bundle_arguments.add_argument(
        "--target-device",
        "-d",
        type=TargetDevice,
        default=None,
        choices=[x.value for x in TargetDevice],
        help="Target device for the model",
    )

    group_bundle_arguments.add_argument(
        "--model-version",
        "-m",
        required=True,
        type=str,
        choices=get_available_models(),
        help="Version of the model to convert",
    )

    group_bundle_arguments.add_argument(
        "--model-input-sizes",
        "-s",
        type=_parse_size,
        nargs="+",
        metavar="WxH",
        default=[
            (960, 544),
            (640, 368),
            (432, 240),
            (320, 192),
        ],
        help="List of input sizes as WxH (e.g. 640x360 368x208)",
    )

    group_bundle_arguments.add_argument(
        "--mlvc-version",
        type=_parse_mlvc_version,
        default=MlvcVersion(major=0, minor=0),
        help="Version of the model bundle (format: X.Y)",
    )

    group_bundle_arguments.add_argument(
        "--skip-if-exists",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip conversion if the model already exists",
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Model conversion script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Export command
    export_parser = subparsers.add_parser(
        Command.EXPORT.value,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Export the model",
    )
    _add_export_arguments(export_parser)

    # Test command
    test_parser = subparsers.add_parser(
        Command.TEST.value,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Test the model",
    )
    _add_test_model_arguments(test_parser)

    # Prepare release command
    bundle_parser = subparsers.add_parser(
        Command.BUNDLE.value,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Prepare model release",
    )
    _add_bundle_arguments(bundle_parser)

    return parser.parse_args()


def _save_json(data, file_path: Path | str):
    print(f"Saving data to {file_path}")
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f)


# -------------------------------------------------------------------------------------
# Commands
# -------------------------------------------------------------------------------------


def _export_model(
    *,
    model_type: ModelType,
    target_device: TargetDevice | None,
    model_version: str,
    weights_version: str | None,
    weights_path: str | None,
    iframe_period: int | None,
    reset_period: int | None,
    ltr_start_idx: int | None,
    ltr_period: int | None,
    model_params_json: str,
    split_type: str | None,
    runtime_params: RuntimeParams,
    model_width: int,
    model_height: int,
    exporter_params_json: str,
    job_outputs_dir: str = DEFAULT_JOB_OUTPUTS_DIR,
    **kwargs,
) -> Path:
    model_params = json.loads(model_params_json)
    exporter_params = json.loads(exporter_params_json)

    full_model = full_model_factory(
        model_version=model_version,
        weights_version=weights_version,
        weights_path=weights_path,
        iframe_period=iframe_period,
        reset_period=reset_period,
        ltr_start_idx=ltr_start_idx,
        ltr_period=ltr_period,
        job_outputs_path=job_outputs_dir,
        **model_params,
    )

    split_model = split_full_model(
        split_type=split_type,
        full_model=full_model,
        runtime_params=runtime_params,
        model_width=model_width,
        model_height=model_height,
    )

    exporter = exporter_factory(
        split_model=split_model,
        model_type=model_type,
        target_device=target_device,
        **exporter_params,
        **kwargs,
    )
    model_path = exporter.run()
    print(f"Model exported to {model_path}")
    return model_path


def _test_model(
    model_path: Path | str,
    model_id: str | None,
    job_outputs_dir: Path | str,
    test_data_dir: Path | str,
    **kwargs,
):
    test_command = TestCommand(kwargs.pop("test_command"))
    runtime_params = _extract_runtime_params(kwargs)

    split_model = load_split_model(
        model_path=model_path,
        model_id=model_id,
        runtime_params=runtime_params,
    )
    model_tester = ModelTester(
        split_model=split_model,
        job_outputs_dir=job_outputs_dir,
        test_data_dir=test_data_dir,
    )
    if test_command == TestCommand.VALIDATE_CONVERSION:
        res = model_tester.validate_conversion(**kwargs)
        print_runtime_params(runtime_params)
        print_validate_conversion_results(res)
    elif test_command == TestCommand.PROFILE:
        output_path = kwargs.pop("output_path")
        res = model_tester.profile(**kwargs)
        print_runtime_params(runtime_params)
        print_profile_results(res)
        if output_path is not None:
            _save_json(dataclasses.asdict(res), output_path)
    elif test_command == TestCommand.BENCHMARK:
        res = model_tester.benchmark(**kwargs)
        print_runtime_params(runtime_params)
        print_benchmark_results(res)
    elif test_command == TestCommand.RUN_VALIDATION_TEST:
        anchor_path = Path(kwargs.pop("anchor_path"))
        output_path = Path(kwargs.pop("output_path"))
        output_name = kwargs.pop("output_name")
        save_output_data = kwargs.pop("save_output_data", False)

        output_data_dir = None
        if save_output_data:
            output_data_dir = output_path / output_name
            output_data_dir.mkdir(parents=True, exist_ok=True)

        res = model_tester.run_validation_test(output_data_dir=output_data_dir, **kwargs)
        print_runtime_params(runtime_params)
        print_validation_test_results(res, anchor_path, job_outputs_dir)
        _save_json(dataclasses.asdict(res), output_path / f"{output_name}.json")
    else:
        raise NotImplementedError(f"Test command {test_command} not implemented")


def _bundle(
    mlvc_version: MlvcVersion,
    model_version: str,
    model_type: ModelType,
    target_device: TargetDevice | None,
    model_input_sizes: list[tuple[int, int]],
    job_outputs_dir: Path | str,
    skip_if_exists: bool = False,
    **kwargs,
):

    # Full model
    full_model = full_model_factory(model_version=model_version, job_outputs_path=job_outputs_dir)

    # Export models for all input sizes
    model_paths: list[Path | str] = []
    for model_width, model_height in model_input_sizes:
        print(f"Converting model for input size {model_width}x{model_height}...")

        split_model = split_full_model(full_model=full_model, model_width=model_width, model_height=model_height)

        exporter = exporter_factory(
            split_model=split_model,
            model_type=model_type,
            target_device=target_device,
            skip_if_exists=skip_if_exists,
        )
        model_path = exporter.run()
        model_paths.append(model_path)

    # Bundle models
    names = [ConversionMetadata.load(model_path).name for model_path in model_paths]
    bundle_name = names[0] if len(set(names)) == 1 else "mixed_models"
    bundler = model_bundler_factory(
        bundle_name=bundle_name,
        mlvc_version=mlvc_version,
        model_type=model_type,
        target_device=target_device,
        **kwargs,
    )
    bundler.bundle(model_paths=model_paths)


if __name__ == "__main__":
    args = _parse_args()
    args = vars(args)

    cmd = Command(args.pop("command"))
    job_outputs_dir = args.pop("job_outputs_dir")
    test_data_dir = args.pop("test_data_dir")

    if cmd == Command.EXPORT:
        run_validate_conversion: bool = args.pop("validate_conversion")
        run_profile: bool = args.pop("profile")
        run_benchmark: bool = args.pop("benchmark")
        runtime_params = _extract_runtime_params(args)

        model_path = _export_model(
            runtime_params=runtime_params,
            job_outputs_dir=job_outputs_dir,
            test_data_dir=test_data_dir,
            **args,
        )

        if run_profile or run_validate_conversion or run_benchmark:
            split_model = load_split_model(model_path, runtime_params)
            model_tester = ModelTester(
                split_model=split_model,
                job_outputs_dir=job_outputs_dir,
                test_data_dir=test_data_dir,
            )

            if run_validate_conversion:
                print("Validating conversion...")
                res = model_tester.validate_conversion()
                print_runtime_params(runtime_params)
                print_validate_conversion_results(res)
                _save_json(dataclasses.asdict(res), model_path / "validate_conversion.json")

            if run_profile:
                print("Profiling...")
                res = model_tester.profile()
                print_runtime_params(runtime_params)
                print_profile_results(res)
                _save_json(dataclasses.asdict(res), model_path / "profile.json")

            if run_benchmark:
                print("Benchmarking...")
                res = model_tester.benchmark()
                print_runtime_params(runtime_params)
                print_benchmark_results(res)
                _save_json(dataclasses.asdict(res), model_path / "benchmark.json")

    elif cmd == Command.TEST:
        _test_model(
            job_outputs_dir=job_outputs_dir,
            test_data_dir=test_data_dir,
            **args,
        )
    elif cmd == Command.BUNDLE:
        _bundle(job_outputs_dir=job_outputs_dir, **args)
    else:
        raise NotImplementedError(f"Command {cmd} not implemented")
