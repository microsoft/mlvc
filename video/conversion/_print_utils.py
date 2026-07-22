# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import numpy as np
from pathlib import Path
from tabulate import tabulate
from .types import (
    RuntimeParams,
    FrameLoopParams,
    FrameLoopSummary,
    AggregatedMetric,
    ProfileResults,
    BenchmarkResults,
    ValidateConversionResults,
    ValidationTestResults,
)
from .const import DEFAULT_JOB_OUTPUTS_DIR
from .utils import parse_metrics, download_job_outputs, transform_validation_test_results
from src.metrics.bjontegaard_metric import calculate_bd_rate


def print_runtime_params(params: RuntimeParams) -> None:
    runtime_params_table = [
        ["CoreML Compute Units", params.coreml_compute_units.value],
        ["CoreML Fast Prediction", params.coreml_fast_prediction],
        ["ONNX Execution Provider", params.onnx_execution_provider.value],
        ["Torch Device", params.torch_device.value],
    ]
    print()
    print("Runtime parameters:")
    print(tabulate(runtime_params_table, headers=["Common", "Value"], tablefmt="pretty"))


def _compose_timers_table(timers: dict[str, AggregatedMetric]) -> str:
    def append_timer(name: str, timer: AggregatedMetric) -> None:
        val = timer.median
        fps = 1.0 / val if val > 0 else np.nan
        timer_table_data.append(
            [
                name,
                f"{1e3 * val:.1f}",
                f"{fps:.1f}",
                f"{1e2 * val / total_time:.1f} %",
            ]
        )

    total_time = max([t.median for t in timers.values()])

    # Compose timers table
    timer_table_headers = ["Timer name", "Median (ms)", "FPS", "Share (%)"]
    timer_table_data = []
    for timer_name, timer_value in timers.items():
        append_timer(timer_name, timer_value)

    return tabulate(
        timer_table_data,
        headers=timer_table_headers,
        tablefmt="pretty",
        colalign=("left",),
    )


def print_frame_loop_comparison(
    params: FrameLoopParams,
    azureml_res: FrameLoopSummary | None,
    conversion_res: FrameLoopSummary | None,
    validation_res: FrameLoopSummary,
) -> None:
    def append_row(
        name: str,
        val0: float,
        val1: float,
        val2: float,
        format_str="{:.2f}",
        unit: str = "",
    ):
        table_data.append(
            [
                name,
                unit,
                format_str.format(val0),
                format_str.format(val1),
                format_str.format(val2),
                format_str.format(val2 - val1),
            ]
        )

    def append_metric(
        name: str,
        val0: AggregatedMetric,
        val1: AggregatedMetric,
        val2: AggregatedMetric,
        compact=False,
        **kwargs,
    ):
        if not compact:
            append_row(f"{name} (min)", val0.min, val1.min, val2.min, **kwargs)
            append_row(f"{name} (max)", val0.max, val1.max, val2.max, **kwargs)
            append_row(f"{name} (median)", val0.median, val1.median, val2.median, **kwargs)
        append_row(f"{name} (mean)", val0.mean, val1.mean, val2.mean, **kwargs)

    # Compose table
    table_headers = ["Metric", "Unit", "AzureML", "Conversion", "Validation", "Delta"]
    table_data = []

    append_metric(
        "PSNR",
        azureml_res.psnr if azureml_res is not None else AggregatedMetric.from_array([]),
        conversion_res.psnr if conversion_res is not None else AggregatedMetric.from_array([]),
        validation_res.psnr if validation_res is not None else AggregatedMetric.from_array([]),
        unit="dB",
    )

    append_metric(
        "PSNR Y",
        azureml_res.psnr_y if azureml_res is not None else AggregatedMetric.from_array([]),
        conversion_res.psnr_y if conversion_res is not None else AggregatedMetric.from_array([]),
        validation_res.psnr_y if validation_res is not None else AggregatedMetric.from_array([]),
        unit="dB",
        compact=True,
    )

    append_metric(
        "PSNR U",
        azureml_res.psnr_u if azureml_res is not None else AggregatedMetric.from_array([]),
        conversion_res.psnr_u if conversion_res is not None else AggregatedMetric.from_array([]),
        validation_res.psnr_u if validation_res is not None else AggregatedMetric.from_array([]),
        unit="dB",
        compact=True,
    )

    append_metric(
        "PSNR V",
        azureml_res.psnr_v if azureml_res is not None else AggregatedMetric.from_array([]),
        conversion_res.psnr_v if conversion_res is not None else AggregatedMetric.from_array([]),
        validation_res.psnr_v if validation_res is not None else AggregatedMetric.from_array([]),
        unit="dB",
        compact=True,
    )

    append_metric(
        "BPP",
        azureml_res.bpp if azureml_res is not None else AggregatedMetric.from_array([]),
        conversion_res.bpp if conversion_res is not None else AggregatedMetric.from_array([]),
        validation_res.bpp if validation_res is not None else AggregatedMetric.from_array([]),
        unit="bpp",
        compact=True,
        format_str="{:.4f}",
    )

    # Print table
    print()
    print(f"Validation results (QP={params.q_index}):")
    print(f"  - video={params.video_path}")
    print(f"  - size={params.image_width}x{params.image_height}, frames={params.frame_count}")

    results_table = tabulate(table_data, headers=table_headers, tablefmt="pretty", colalign=("left",))
    timers_table = _compose_timers_table(validation_res.timers)
    _print_tables_side_by_side(results_table, timers_table)


def _print_tables_side_by_side(table1: str, table2: str) -> None:
    lines1 = table1.split("\n")
    lines2 = table2.split("\n")
    max_lines = max(len(lines1), len(lines2))
    table1_width = max(len(line) for line in lines1)
    for i in range(max_lines):
        line1 = lines1[i] if i < len(lines1) else ""
        line2 = lines2[i] if i < len(lines2) else ""
        format_str = "{line1:<%d}  {line2}" % (table1_width + 2)
        print(format_str.format(line1=line1, line2=line2))


def print_validate_conversion_results(results: ValidateConversionResults):
    for i, (params, validation) in enumerate(zip(results.params, results.validation_results)):
        azureml = results.azureml_results[i] if i < len(results.azureml_results) else None
        conversion = results.conversion_results[i] if i < len(results.conversion_results) else None
        print_frame_loop_comparison(params, azureml, conversion, validation)


def print_profile_results(results: ProfileResults) -> None:
    npu_table_header = [
        "Model",
        "Op",
        "Output Name",
        "Cost",
        "Preferred Device",
        "Supported Devices",
    ]
    npu_table_data = []

    stats_table_header = [
        "Model",
        "CPU ops",
        "CPU cost",
        "NPU ops",
        "NPU cost",
        "GPU",
        "GPU cost",
        "Total ops",
        "Total cost",
    ]
    stats_table_data = []

    for model_part_id, profile in results.profiles.items():
        for op in profile.operations:
            if op.supported_compute_devices is not None and "NPU" not in op.supported_compute_devices:
                npu_table_data.append(
                    (
                        model_part_id.value,
                        op.operator_name,
                        op.output_name,
                        f"{op.cost or float('nan'):.2f}",
                        op.preferred_compute_device,
                        ", ".join(op.supported_compute_devices),
                    )
                )

        stats_table_data.append(
            (
                model_part_id.value,
                f"{profile.op_stats.num_ops_cpu}",
                f"{profile.op_stats.cost_cpu:.2f}",
                f"{profile.op_stats.num_ops_npu}",
                f"{profile.op_stats.cost_npu:.2f}",
                f"{profile.op_stats.num_ops_gpu}",
                f"{profile.op_stats.cost_gpu:.2f}",
                f"{profile.op_stats.total_ops}",
                f"{profile.op_stats.total_cost:.2f}",
            )
        )

    if len(stats_table_data) > 0:
        print()
        print("Compute device support:")
        print(
            tabulate(
                stats_table_data,
                headers=stats_table_header,
                tablefmt="pretty",
                colalign=("left",),
            )
        )

    if len(npu_table_data) > 0:
        print()
        print("Operations not supported on NPU:")
        print(
            tabulate(
                npu_table_data,
                headers=npu_table_header,
                tablefmt="pretty",
                colalign=("left",),
            )
        )


def print_benchmark_results(results: BenchmarkResults) -> None:

    def append_powermetric(name: str, unit: str, value: AggregatedMetric) -> None:
        powermetrics_table_data.append(
            [
                name,
                unit,
                f"{value.min:.1f}",
                f"{value.max:.1f}",
                f"{value.mean:.1f}",
                f"{value.median:.1f}",
                f"{value.count}",
            ]
        )

    # Compose powermetrics table
    powermetrics_table_headers = ["Metric", "Unit", "Min", "Max", "Mean", "Median", "Samples"]
    powermetrics_table_data = []
    if "frame_loop" in results.powermetrics_data.stats:
        stats = results.powermetrics_data.stats["frame_loop"]
        append_powermetric("NPU", "W", stats.npu_power)
        append_powermetric("GPU", "W", stats.gpu_power)
        append_powermetric("CPU", "W", stats.cpu_power)
        append_powermetric("Combined", "W", stats.combined_power)
        powermetrics_table = tabulate(
            powermetrics_table_data,
            headers=powermetrics_table_headers,
            tablefmt="pretty",
            colalign=("left",),
        )
    else:
        powermetrics_table = ""

    timers_table = _compose_timers_table(results.frame_loop_summary.timers)

    # Print table
    print()
    print("Benchmark results:")
    _print_tables_side_by_side(timers_table, powermetrics_table)


def print_validation_test_results(
    test_data: ValidationTestResults,
    anchor_path: Path | str,
    job_outputs_dir: Path | str = DEFAULT_JOB_OUTPUTS_DIR,
):
    anchor_file_path = download_job_outputs(anchor_path, job_outputs_dir)
    with open(anchor_file_path, "r") as f:
        anchor_data = json.load(f)

    def _aggregate_metrics(df):
        return (
            df.groupby(["test_class", "q_name"])
            .agg(
                {
                    "kbps": "mean",
                    "bpp": "mean",
                    "psnr": "mean",
                    "psnr_y": "mean",
                    "psnr_u": "mean",
                    "psnr_v": "mean",
                    "sequence_name": "count",
                }
            )
            .rename(columns={"sequence_name": "sequence_count"})
        )

    data = parse_metrics(
        {
            "test": transform_validation_test_results(test_data),
            "anchor": anchor_data,
        },
        only_common_clips=True,
    )
    test_classes = data["test"]["test_class"].unique()

    table_header = [
        "Scenario",
        "Num videos",
        "Num points",
        "BD-rate Y",
        "BD-rate U",
        "BD-rate V",
        "BD-rate",
    ]
    table_data = []
    anchor = _aggregate_metrics(data["anchor"])
    test = _aggregate_metrics(data["test"])
    for test_class in test_classes:
        df1 = anchor.loc[test_class]
        df2 = test.loc[test_class]

        video_count = df2.sequence_count.max()
        point_count = len(df2)
        bd_rate_y = calculate_bd_rate(df1["kbps"], df1["psnr_y"], df2["kbps"], df2["psnr_y"])
        bd_rate_u = calculate_bd_rate(df1["kbps"], df1["psnr_u"], df2["kbps"], df2["psnr_u"])
        bd_rate_v = calculate_bd_rate(df1["kbps"], df1["psnr_v"], df2["kbps"], df2["psnr_v"])
        bd_rate = calculate_bd_rate(df1["kbps"], df1["psnr"], df2["kbps"], df2["psnr"])
        table_data.append(
            (
                test_class,
                f"{video_count}",
                f"{point_count}",
                f"{bd_rate_y:.1f}",
                f"{bd_rate_u:.1f}",
                f"{bd_rate_v:.1f}",
                f"{bd_rate:.1f}",
            )
        )

    print()
    print("Validation test results:")
    print(f"  - Anchor: {anchor_path}")
    print(tabulate(table_data, headers=table_header, tablefmt="pretty", colalign=("left",)))
