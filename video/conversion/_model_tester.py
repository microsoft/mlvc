# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import json
import time
from tqdm import tqdm
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from azure.core.exceptions import ResourceNotFoundError

from .const import (
    DEFAULT_JOB_OUTPUTS_DIR,
    DEFAULT_TEST_DATA_DIR,
    DEFAULT_BENCHMARK_FRAME_COUNT,
    DEFAULT_TEST_CONFIG_PATH,
    DEFAULT_NUM_CLIPS_LIMIT,
    DEFAULT_TEST_Q_INDEX_LIST,
    get_default_test_video,
)
from .utils import download_test_data, download_job_outputs
from .types import (
    ModelType,
    ModelPartId,
    ModelProfile,
    ConversionMetadata,
    FrameLoopSummary,
    FrameLoopParams,
    ValidateConversionResults,
    ProfileResults,
    BenchmarkResults,
    ValidationTestResults,
    AggregatedMetric,
)
from ._split_model import BaseSplitModel
from ._powermetrics import PowerMetricsCollector
from ._frame_loop import FrameLoop, aggregate_frame_loop_results
from ._model_profile import profile_coreml_model, aggregate_op_profile


class ModelTester:
    def __init__(
        self,
        split_model: BaseSplitModel,
        job_outputs_dir: Path | str = DEFAULT_JOB_OUTPUTS_DIR,
        test_data_dir: Path | str = DEFAULT_TEST_DATA_DIR,
    ) -> None:
        self._split_model = split_model
        self._job_outputs_dir = Path(job_outputs_dir)
        self._test_data_dir = Path(test_data_dir)

    def validate_conversion(self) -> ValidateConversionResults:
        conversion_metadata = self._split_model.conversion_metadata
        if conversion_metadata is None:
            raise ValueError("Conversion metadata not found!")
        azureml_results = self._load_azureml_results(conversion_metadata)
        validation_results = self._run_frame_loop_params_sweep(conversion_metadata.conversion_loop_params)

        return ValidateConversionResults(
            params=conversion_metadata.conversion_loop_params,
            azureml_results=azureml_results,
            conversion_results=conversion_metadata.conversion_loop_results,
            validation_results=validation_results,
        )

    def profile(self) -> ProfileResults:
        runtime_params = self._split_model.runtime_params
        model_type = next(iter(self._split_model.model_parts.values())).model_type

        profiles: dict[ModelPartId, ModelProfile] = {}
        if model_type == ModelType.COREML:
            # CoreML profiling does not require running the model
            for model_part_id, model_part in self._split_model.model_parts.items():
                profiles[model_part_id] = profile_coreml_model(
                    model_part.model,
                    coreml_compute_units=runtime_params.coreml_compute_units,
                )
        elif model_type == ModelType.OPENVINO:
            # Run OpenVINO model to collect profiling data
            test_video_path, image_width, image_height = get_default_test_video(
                self._split_model.split_model_params.model_width,
                self._split_model.split_model_params.model_height,
            )
            loop = FrameLoop(
                split_model=self._split_model,
                video_path=test_video_path,
                image_width=image_width,
                image_height=image_height,
                frame_count=48,
                test_data_dir=self._test_data_dir,
            )
            loop_results = loop.run(q_index=42, progress_bar=False, collect_profiling_info=True)

            frame_results = loop_results.frames[-1]  # TODO: aggregate?
            for model_part_id, op_profile in frame_results.op_profiles.items():
                if len(op_profile) == 0:
                    continue
                profiles[model_part_id] = ModelProfile(
                    operations=op_profile,
                    op_stats=aggregate_op_profile(op_profile),
                )
        else:
            print(f"Warn: Model type {model_type} not supported for profiling.")
        return ProfileResults(profiles=profiles)

    def benchmark(
        self,
        test_video_path: Path | str | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
        frame_count: int = DEFAULT_BENCHMARK_FRAME_COUNT,
        use_decoder: bool = True,
        q_index: int = 42,
        collect_powermetrics: bool = True,
        save_powermetrics_raw_data: bool = False,
    ) -> BenchmarkResults:
        powermetrics = PowerMetricsCollector(save_raw_data=save_powermetrics_raw_data)
        if collect_powermetrics:
            print("Starting powermetrics collection ...")
            powermetrics.start()

        if test_video_path is None:
            test_video_path, image_width, image_height = get_default_test_video(
                self._split_model.split_model_params.model_width,
                self._split_model.split_model_params.model_height,
            )
        else:
            if image_width is None or image_height is None:
                raise ValueError("Image width and height must be specified if test video path is provided")

        loop = FrameLoop(
            split_model=self._split_model,
            video_path=test_video_path,
            image_width=image_width,
            image_height=image_height,
            frame_count=frame_count,
            use_decoder=use_decoder,
            test_data_dir=self._test_data_dir,
        )

        with powermetrics.set_context("wait_before_loop"):
            print("Waiting 5 seconds before starting the frame loop ...")
            time.sleep(5)
        with powermetrics.set_context("frame_loop"):
            print("Running frame loop ...")
            loop_results = loop.run(q_index=q_index, progress_bar=False)
        with powermetrics.set_context("wait_after_loop"):
            print("Waiting 2 seconds after the frame loop ...")
            time.sleep(2)
        powermetrics.stop()
        print("Powermetrics collection finished.")

        return BenchmarkResults(
            frame_loop_params=loop_results.params,
            frame_loop_summary=aggregate_frame_loop_results(loop_results),
            powermetrics_data=powermetrics.get_data(),
        )

    def run_validation_test(
        self,
        test_config: Path | str = DEFAULT_TEST_CONFIG_PATH,
        *,
        scenarios_list: list[str] | None = None,
        num_clips_limit: int = DEFAULT_NUM_CLIPS_LIMIT,
        q_index_list: list[int] | None = None,
        bitrate_list: list[float] | None = None,
        clip_bitrate_list_overrides: dict[tuple[str, str], list[float]] = {},  # [scenario, video_name] -> [bitrates]
        use_encoder: bool = True,
        use_decoder: bool = True,
        save_values: bool = True,
        encoded_data_dir: Path | str | None = None,
        output_data_dir: Path | str | None = None,
        **kwargs,
    ) -> ValidationTestResults:

        test_clips = _load_test_config(
            test_config=test_config,
            scenarios_list=scenarios_list,
            num_clips_limit=num_clips_limit,
            test_data_dir=self._test_data_dir,
        )

        # Build params queue
        if q_index_list is None and bitrate_list is None:
            q_index_list = DEFAULT_TEST_Q_INDEX_LIST
        params_queue: list[tuple[TestClipMetadata, dict[str, Any]]] = []
        for clip in test_clips:
            for q_index in q_index_list or []:
                params_queue.append((clip, {"q_index": q_index}))

            clip_bitrate_list = clip_bitrate_list_overrides.get((clip.scenario, clip.video_name), bitrate_list)
            for bitrate in clip_bitrate_list or []:
                params_queue.append((clip, {"bitrate": bitrate}))

        # Sweep over clips and rates
        def _get_clip_path(path: Path | str | None, video_path: Path) -> Path | None:
            if path is None:
                return None
            return Path(path) / f"{video_path.parent.name}_{video_path.stem}"

        def _params_to_str(params: dict[str, Any]) -> str:
            res = []
            for k, v in params.items():
                if v is not None:
                    if k == "bitrate":
                        res.append(f"bitrate_kbps={v / 1000:.0f}")
                    else:
                        res.append(f"{k}={v}")
            return ", ".join(res)

        res_params: list[FrameLoopParams] = []
        res_summaries: list[FrameLoopSummary] = []
        progress = tqdm(params_queue, desc="Validation")
        for clip, params in progress:
            progress.set_postfix_str(f"{clip.scenario}/{clip.video_name}, {_params_to_str(params)}")
            loop = FrameLoop(
                split_model=self._split_model,
                video_path=clip.video_path,
                image_width=clip.image_width,
                image_height=clip.image_height,
                frame_count=clip.frame_count,
                fps=clip.fps,
                use_encoder=use_encoder,
                use_decoder=use_decoder,
                encoded_data_dir=_get_clip_path(encoded_data_dir, clip.video_path),
                output_data_dir=_get_clip_path(output_data_dir, clip.video_path),
                test_data_dir=self._test_data_dir,
            )

            loop_results = loop.run(**params, progress_bar=False, **kwargs)
            summary = aggregate_frame_loop_results(loop_results, save_values=save_values)

            res_params.append(loop_results.params)
            res_summaries.append(summary)
        return ValidationTestResults(params=res_params, results=res_summaries)

    def _run_frame_loop_params_sweep(
        self,
        params_queue: list[FrameLoopParams],
        save_values: bool = False,
        encoded_data_dir: Path | str | None = None,
        output_data_dir: Path | str | None = None,
        **kwargs,
    ) -> list[FrameLoopSummary]:

        def _get_clip_path(path: Path | str | None, params: FrameLoopParams) -> Path | None:
            if path is None:
                return None
            video_path = Path(params.video_path)
            return Path(path) / f"{video_path.parent.name}_{video_path.stem}"

        def _get_clip_label(params: FrameLoopParams) -> str:
            video_path = Path(params.video_path)
            return f"{video_path.parent.name}/{video_path.stem}"

        def _get_rate_label(params: FrameLoopParams) -> str:
            if params.q_index is not None:
                return f"q_index={params.q_index}"
            if params.bitrate is not None:
                return f"bitrate_kbps={params.bitrate / 1000:.0f}"
            return "rate=unknown"

        results: list[FrameLoopSummary] = []
        progress = tqdm(params_queue, desc="Validation")
        for params in progress:
            progress.set_postfix_str(f"{_get_clip_label(params)}, {_get_rate_label(params)}")
            loop = FrameLoop(
                split_model=self._split_model,
                video_path=params.video_path,
                image_width=params.image_width,
                image_height=params.image_height,
                frame_count=params.frame_count,
                fps=params.fps,
                use_encoder=params.use_encoder,
                use_decoder=params.use_decoder,
                encoded_data_dir=_get_clip_path(encoded_data_dir, params),
                output_data_dir=_get_clip_path(output_data_dir, params),
                test_data_dir=self._test_data_dir,
            )

            loop_results = loop.run(
                q_index=params.q_index,
                bitrate=params.bitrate,
                iframe_period=params.iframe_period,
                reset_period=params.reset_period,
                ltr_start_idx=params.ltr_start_idx,
                ltr_period=params.ltr_period,
                proactive_ltr_recovery=params.proactive_ltr_recovery,
                padding_mode=params.padding_mode,
                padding_direction=params.padding_direction,
                include_bitstream_overhead=params.include_bitstream_overhead,
                metrics_bit_depth=params.metrics_bit_depth,
                progress_bar=False,
                **kwargs,
            )
            summary = aggregate_frame_loop_results(loop_results, save_values=save_values)
            results.append(summary)
        return results

    def _load_azureml_results(self, conversion_metadata: ConversionMetadata) -> list[FrameLoopSummary]:

        weights_path = conversion_metadata.params.full_model_params.weights_path
        if weights_path is None:
            return []
        weights_path = Path(weights_path)

        # Load AzureML metrics (not ideal, can be improved)
        metrics_path = None
        try:
            match = re.search(r"epo_?(\d+)", weights_path.stem)
            if match:
                epoch = int(match.group(1))
                metrics_path = download_job_outputs(
                    weights_path.parent / f"metrics_VCD_640x360_30fps_40s48f_epo_{epoch}.json",
                    self._job_outputs_dir,
                )
            else:
                print("Epoch not found in weights path")
        except ResourceNotFoundError:
            print("Warning: Metrics file not found!")

        azureml_results: list[FrameLoopSummary] = []
        if metrics_path is not None:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            for params in conversion_metadata.conversion_loop_params:
                scenario = Path(params.video_path).parent.name
                video_name = Path(params.video_path).name
                q_index = params.q_index
                assert q_index is not None, (
                    "Q-index must be specified in conversion loop params to load AzureML results"
                )
                q_name = f"{q_index}"

                if (
                    scenario not in metrics
                    or video_name not in metrics[scenario]
                    or q_name not in metrics[scenario][video_name]
                ):
                    loop_summary = FrameLoopSummary(
                        q_index=AggregatedMetric.from_array([]),
                        psnr=AggregatedMetric.from_array([]),
                        psnr_y=AggregatedMetric.from_array([]),
                        psnr_u=AggregatedMetric.from_array([]),
                        psnr_v=AggregatedMetric.from_array([]),
                        bpp=AggregatedMetric.from_array([]),
                        timers={},
                    )
                else:
                    clip_metrics = metrics[scenario][video_name][q_name]
                    loop_summary = FrameLoopSummary(
                        q_index=AggregatedMetric.from_array([q_index] if q_index is not None else []),
                        psnr=AggregatedMetric.from_array([clip_metrics["ave_all_frame_psnr"]]),
                        psnr_y=AggregatedMetric.from_array([clip_metrics["ave_all_frame_psnr_y"]]),
                        psnr_u=AggregatedMetric.from_array([clip_metrics["ave_all_frame_psnr_u"]]),
                        psnr_v=AggregatedMetric.from_array([clip_metrics["ave_all_frame_psnr_v"]]),
                        bpp=AggregatedMetric.from_array([clip_metrics["ave_all_frame_bpp"]]),
                        timers={},
                    )

                azureml_results.append(loop_summary)

        return azureml_results


@dataclass(frozen=True)
class TestClipMetadata:
    scenario: str
    video_path: Path
    image_width: int
    image_height: int
    frame_count: int
    fps: float

    @property
    def video_name(self) -> str:
        return self.video_path.name


def _load_test_config(
    *,
    test_config: Path | str,
    scenarios_list: list[str] | None = None,
    num_clips_limit: int | None = None,
    test_data_dir: Path | str = DEFAULT_TEST_DATA_DIR,
) -> list[TestClipMetadata]:
    with open(download_test_data(test_config, test_data_dir), "r") as f:
        config = json.load(f)
    res: list[TestClipMetadata] = []
    for scenario, test_class_data in config["test_classes"].items():
        if scenarios_list is not None and scenario not in scenarios_list:
            continue

        for filename, metadata in test_class_data["sequences"].items():
            if num_clips_limit is not None and len(res) >= num_clips_limit:
                break

            video_path = Path(test_config).parent / test_class_data["base_path"] / filename
            res.append(
                TestClipMetadata(
                    scenario=scenario,
                    video_path=video_path,
                    image_width=metadata["width"],
                    image_height=metadata["height"],
                    frame_count=metadata["frames"],
                    fps=metadata.get("fps", 30.0),
                )
            )

    if len(res) == 0:
        raise ValueError("No valid test clips found")

    return res
