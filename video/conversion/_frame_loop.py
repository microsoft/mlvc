# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from .const import (
    DEFAULT_TEST_DATA_DIR,
    DEFAULT_INCLUDE_BITSTREAM_OVERHEAD,
    DEFAULT_METRICS_BIT_DEPTH,
)
from ._split_model import BaseSplitModel
from .types import (
    FrameMetrics,
    FrameType,
    FrameLoopParams,
    FrameLoopFrameResult,
    FrameLoopResults,
    FrameLoopSummary,
    AggregatedMetric,
    ModelPartId,
    ModelData,
    ModelOpProfile,
    PaddingMode,
    PaddingDirection,
)
from ._rate_controller import RateController
from .utils import (
    calc_psnr,
    read_video_frames,
    MlvcFrameHeader,
    read_mlvc_bitstreams,
    save_mlvc_bitstreams,
    download_test_data,
)

Q_INDEX_DROP_SENTINEL = -1


class FrameLoop:
    def __init__(
        self,
        split_model: BaseSplitModel,
        video_path: Path | str,
        image_width: int,
        image_height: int,
        frame_count: int,
        fps: float = 30.0,
        use_encoder: bool = True,
        use_decoder: bool = True,
        encoded_data_dir: Path | str | None = None,
        output_data_dir: Path | str | None = None,
        test_data_dir: Path | str = DEFAULT_TEST_DATA_DIR,
    ):
        if not use_encoder and not use_decoder:
            raise ValueError("At least one of use_encoder or use_decoder must be True")
        if not use_encoder and encoded_data_dir is None:
            raise ValueError("encoded_data_dir is required when use_encoder is False")

        self._split_model = split_model

        self._video_path = Path(video_path)
        self._image_width = image_width
        self._image_height = image_height
        self._frame_count = frame_count
        self._fps = fps

        self._use_encoder = use_encoder
        self._use_decoder = use_decoder
        self._encoded_data_dir = encoded_data_dir
        self._output_data_dir = output_data_dir

        self._test_data_dir = test_data_dir

    def run(
        self,
        q_index: int | None = None,
        bitrate: float | None = None,
        iframe_period: int | None = None,
        reset_period: int | None = None,
        ltr_start_idx: int | None = None,
        ltr_period: int | None = None,
        proactive_ltr_recovery: bool | None = None,
        q_index_overrides: dict[int, int | None] = {},  # frame_id -> q_index (sticky)
        padding_mode: PaddingMode | None = None,
        padding_direction: PaddingDirection | None = None,
        include_bitstream_overhead: bool = DEFAULT_INCLUDE_BITSTREAM_OVERHEAD,
        metrics_bit_depth: int = DEFAULT_METRICS_BIT_DEPTH,
        output_model_data: bool = False,
        collect_profiling_info: bool = False,
        progress_bar: bool = True,
        save_debug_data: bool = False,
        save_yuv: bool = False,
    ) -> FrameLoopResults:

        # Validate rate control params
        if q_index is not None and bitrate is not None:
            raise ValueError("Only one of q_index or bitrate can be set")
        if q_index is None and bitrate is None:
            raise ValueError("One of q_index or bitrate must be set")

        # Determine name for bitrate settings
        if q_index is not None:
            q_name = f"q_index={q_index}"
        elif bitrate is not None:
            q_name = f"bitrate_kbps={int(bitrate / 1000)}"
        else:
            assert False, "One of q_index or bitrate must be set"

        # Load YUV420 frames
        yuv420_frames = read_video_frames(
            video_path=download_test_data(self._video_path, self._test_data_dir),
            image_height=self._image_height,
            image_width=self._image_width,
            frame_count=self._frame_count,
        )

        # Load encoded bitstreams if not using encoder
        if not self._use_encoder:
            assert self._encoded_data_dir is not None
            encoded_bitstreams = read_mlvc_bitstreams(Path(self._encoded_data_dir) / q_name / "output.mlvc")
        else:
            encoded_bitstreams = None

        # Prepare output data dir
        if self._output_data_dir is not None:
            output_data_dir = Path(self._output_data_dir) / q_name
            if output_data_dir.exists():
                # print(f"Removing existing output data directory: {output_data_dir}")
                shutil.rmtree(output_data_dir)
            output_data_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_data_dir = None

        # Resolve loop params (use model defaults if not provided)
        iframe_period = iframe_period if iframe_period is not None else self._split_model.model_params.iframe_period
        reset_period = reset_period if reset_period is not None else self._split_model.model_params.reset_period
        ltr_start_idx = ltr_start_idx if ltr_start_idx is not None else self._split_model.model_params.ltr_start_idx
        ltr_period = ltr_period if ltr_period is not None else self._split_model.model_params.ltr_period
        proactive_ltr_recovery = proactive_ltr_recovery if proactive_ltr_recovery is not None else True
        padding_mode = padding_mode if padding_mode is not None else PaddingMode.EDGE
        padding_direction = padding_direction if padding_direction is not None else PaddingDirection.BOTTOM_RIGHT

        # Rate controller
        rate_controller: RateController | None = None
        if self._use_encoder and bitrate is not None:
            rate_controller: RateController | None = RateController(
                bitrate=bitrate,
                fps=self._fps,
                image_width=self._image_width,
                image_height=self._image_height,
            )

        ref_manager_encoder = self._split_model.make_ref_manager(use_decoder=False)
        ref_manager_decoder = self._split_model.make_ref_manager(use_decoder=True)
        frame_results: list[FrameLoopFrameResult] = []

        cur_frame_idx: int = 0
        presentation_time: float = 0.0
        latest_ltr_frame_idx: int | None = None
        last_reconstructed_frame: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        q_index_override: int | None = None
        for i, yuv420 in tqdm(
            enumerate(yuv420_frames),
            total=len(yuv420_frames),
            disable=not progress_bar,
        ):
            start_time = time.perf_counter()
            presentation_time += 1.0 / self._fps

            # IDR
            frame_type = FrameType.P_FRAME
            if i == 0 or (iframe_period is not None and cur_frame_idx % iframe_period == 0):
                frame_type = FrameType.I_FRAME
                cur_frame_idx = 0
                latest_ltr_frame_idx = None
                ref_manager_encoder.clear()
                ref_manager_decoder.clear()

            # Feature reset, mark as LTR
            feature_reset = reset_period is not None and (cur_frame_idx + 1) % reset_period == 0
            mark_as_ltr = (
                ltr_period is not None
                and ltr_period > 0
                and (
                    cur_frame_idx == ltr_start_idx
                    or (cur_frame_idx > ltr_start_idx and cur_frame_idx % ltr_period == 0)
                )
            )

            # Proactive LTR recovery
            if (
                frame_type != FrameType.I_FRAME
                and proactive_ltr_recovery
                and mark_as_ltr
                and latest_ltr_frame_idx is not None
            ):
                frame_type = FrameType.LTR_RECOVERY

            # Determine reference frame index
            if frame_type == FrameType.I_FRAME:
                ref_frame_idx = None
            elif frame_type == FrameType.P_FRAME:
                ref_frame_idx = cur_frame_idx - 1
            elif frame_type == FrameType.LTR_RECOVERY:
                ref_frame_idx = latest_ltr_frame_idx
            else:
                raise NotImplementedError(f"Unsupported frame type: {frame_type}")
            bitstream_overhead_bits = (
                8 * _estimate_bitstream_overhead_bytes(frame_type) if include_bitstream_overhead else 0
            )

            # Determine frame_q_index
            if i in q_index_overrides:
                q_index_override = q_index_overrides[i]

            frame_q_index: int | None = None
            if self._use_encoder:
                if q_index_override is not None:
                    # Frame q-index override
                    frame_q_index = q_index_override
                elif rate_controller is not None:
                    # Constant bitrate mode
                    frame_q_index = rate_controller.solve_q_index(
                        presentation_time,
                        frame_type,
                        reserved_overhead_bits=bitstream_overhead_bits,
                    )
                elif q_index is not None:
                    # Constant QP mode
                    frame_q_index = q_index
                else:
                    assert False, "Unable to determine frame_q_index"
            else:
                assert encoded_bitstreams is not None
                header, _ = encoded_bitstreams[i]
                frame_q_index = header.q_index

            # Handle frame dropping
            if frame_q_index is None or frame_q_index == Q_INDEX_DROP_SENTINEL:
                if frame_type == FrameType.P_FRAME and not mark_as_ltr and not feature_reset:
                    assert last_reconstructed_frame is not None
                    y, uv = yuv420
                    original_frame = (
                        y.astype(np.float32, copy=False)[np.newaxis, ...],
                        uv[0:1].astype(np.float32, copy=False)[np.newaxis, ...],
                        uv[1:2].astype(np.float32, copy=False)[np.newaxis, ...],
                    )
                    metrics = self._calc_frame_metrics(
                        original_frame=original_frame,
                        reconstructed_frame=last_reconstructed_frame,
                        bitstream_size_bits=0,
                        metrics_bit_depth=metrics_bit_depth,
                    )
                    frame_results.append(_build_dropped_frame_result(presentation_time, frame_type, metrics))
                    continue
                else:
                    # non-droppable (I / LTR-recovery / LTR-marked / reset): fall back to lowest Q-index
                    frame_q_index = 0

            # Encode
            reconstructed_frame = None
            if self._use_encoder:
                ref_data = ref_manager_encoder.load(ref_frame_idx, feature_reset=feature_reset)
                encoder_output = self._split_model.encode(
                    cur_frame_idx,
                    yuv420,
                    q_index=frame_q_index,
                    ref_data=ref_data,
                    padding_mode=padding_mode,
                    padding_direction=padding_direction,
                )
                ref_manager_encoder.save(cur_frame_idx, encoder_output, mark_as_ltr=mark_as_ltr)
                original_frame = encoder_output.original_frame
                reconstructed_frame = encoder_output.reconstructed_frame
                bitstream = encoder_output.bitstream
                padding = encoder_output.padding
            else:
                assert encoded_bitstreams is not None
                encoder_output = None
                _, original_frame, padding = self._split_model._prepare_input_frame(
                    yuv420, padding_mode, padding_direction
                )
                _, bitstream = encoded_bitstreams[i]

            # Update rate controller
            rate_control_info = None
            if rate_controller is not None:
                rate_control_info = rate_controller.update(
                    header_bits=bitstream_overhead_bits,
                    payload_bits=8 * len(bitstream),
                )

            # Decode
            if self._use_decoder:
                ref_data = ref_manager_decoder.load(ref_frame_idx, feature_reset=feature_reset)
                decoder_output = self._split_model.decode(
                    cur_frame_idx,
                    bitstream,
                    padding,
                    q_index=frame_q_index,
                    ref_data=ref_data,
                )
                ref_manager_decoder.save(cur_frame_idx, decoder_output, mark_as_ltr=mark_as_ltr)
                reconstructed_frame = decoder_output.reconstructed_frame
            else:
                decoder_output = None

            # Collect profiling info
            op_profiles: dict[ModelPartId, list[ModelOpProfile]] = {}
            if collect_profiling_info:
                for model_part_id, model_part in self._split_model.model_parts.items():
                    op_profiles[model_part_id] = model_part.op_profile

            # Calculate metrics and save frame results
            assert reconstructed_frame is not None
            metrics = self._calc_frame_metrics(
                original_frame=original_frame,
                reconstructed_frame=reconstructed_frame,
                bitstream_size_bits=bitstream_overhead_bits + 8 * len(bitstream),
                metrics_bit_depth=metrics_bit_depth,
            )
            loop_time = time.perf_counter() - start_time
            frame_result = FrameLoopFrameResult(
                presentation_time=presentation_time,
                frame_type=frame_type,
                frame_idx=cur_frame_idx,
                ref_frame_idx=ref_frame_idx,
                q_index=frame_q_index,
                feature_reset=feature_reset,
                marked_as_ltr=mark_as_ltr,
                encoder=encoder_output,
                decoder=decoder_output,
                rate_control_info=rate_control_info,
                metrics=metrics,
                timers={"FrameLoopTotal": loop_time},
                op_profiles=op_profiles,
            )

            # Save debug data
            if save_debug_data and output_data_dir is not None:
                _save_model_data(frame_result.all_model_data, output_data_dir / f"model_data_{i}.npz")

            # Release model data if not requested
            if not output_model_data:
                if frame_result.encoder is not None:
                    frame_result.encoder.model_data.clear()
                if frame_result.decoder is not None:
                    frame_result.decoder.model_data.clear()
            frame_results.append(frame_result)

            # Update current frame index
            if mark_as_ltr:
                latest_ltr_frame_idx = cur_frame_idx
            cur_frame_idx += 1
            last_reconstructed_frame = reconstructed_frame

        # Save output data
        if output_data_dir is not None:
            if self._use_encoder:
                save_mlvc_bitstreams(
                    output_data_dir / "output.mlvc",
                    [
                        (MlvcFrameHeader(q_index=r.q_index), r.encoder.bitstream if r.encoder is not None else b"")
                        for r in frame_results
                    ],
                )

            if self._use_decoder and save_yuv:
                output_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
                last_output_frame = None
                for frame_res in frame_results:
                    if frame_res.decoder is not None:
                        last_output_frame = frame_res.decoder.reconstructed_frame
                    elif last_output_frame is None:
                        raise ValueError("Cannot save dropped frame before any reconstructed frame is available")
                    output_frames.append(last_output_frame)
                _save_yuv_output(output_frames, output_data_dir / "output.yuv")

        return FrameLoopResults(
            params=FrameLoopParams(
                video_path=self._video_path.as_posix(),
                image_width=self._image_width,
                image_height=self._image_height,
                frame_count=self._frame_count,
                fps=self._fps,
                q_index=q_index,
                bitrate=bitrate,
                q_index_overrides=dict(q_index_overrides),
                iframe_period=iframe_period,
                reset_period=reset_period,
                ltr_start_idx=ltr_start_idx,
                ltr_period=ltr_period,
                proactive_ltr_recovery=proactive_ltr_recovery,
                padding_mode=padding_mode,
                padding_direction=padding_direction,
                include_bitstream_overhead=include_bitstream_overhead,
                metrics_bit_depth=metrics_bit_depth,
                use_encoder=self._use_encoder,
                use_decoder=self._use_decoder,
            ),
            frames=frame_results,
        )

    def _calc_frame_metrics(
        self,
        original_frame: tuple[np.ndarray, np.ndarray, np.ndarray],  # YUV420
        reconstructed_frame: tuple[np.ndarray, np.ndarray, np.ndarray],  # YUV420
        bitstream_size_bits: int,
        metrics_bit_depth: int = 32,
    ) -> FrameMetrics:

        def limit_bit_depth(x: np.ndarray) -> np.ndarray:
            if metrics_bit_depth >= 32:
                return x
            max_val = 2**metrics_bit_depth - 1
            return (np.clip(np.round(max_val * x.astype(np.float32)), 0.0, max_val) / max_val).astype(np.float32)

        y, u, v = original_frame
        y_rec, u_rec, v_rec = tuple(limit_bit_depth(t) for t in reconstructed_frame)

        psnr_y = calc_psnr(y, y_rec)
        psnr_u = calc_psnr(u, u_rec)
        psnr_v = calc_psnr(v, v_rec)
        psnr = (6 * psnr_y + psnr_u + psnr_v) / 8
        bpp = bitstream_size_bits / (self._image_width * self._image_height)

        return FrameMetrics(
            psnr=psnr,
            psnr_y=psnr_y,
            psnr_u=psnr_u,
            psnr_v=psnr_v,
            bpp=bpp,
        )


def aggregate_frame_loop_results(
    results: FrameLoopResults,
    save_values: bool = False,
) -> FrameLoopSummary:
    q_index = np.array([r.q_index for r in results.frames], dtype=int)
    psnr = np.array([r.metrics.psnr for r in results.frames])
    psnr_y = np.array([r.metrics.psnr_y for r in results.frames])
    psnr_u = np.array([r.metrics.psnr_u for r in results.frames])
    psnr_v = np.array([r.metrics.psnr_v for r in results.frames])
    bpp = np.array([r.metrics.bpp for r in results.frames])

    timers: dict[str, list[float]] = {
        "MLVCEncoderTotal": [],
        "MLVCDecoderTotal": [],
    }
    for r in results.frames:
        encoder_models_total = 0.0
        decoder_models_total = 0.0

        for timer_name, timer_value in r.all_timers.items():
            if timer_name not in timers:
                timers[timer_name] = []
            timers[timer_name].append(timer_value)
            if "MLVCEncoder" in timer_name:
                encoder_models_total += timer_value
            if "MLVCDecoder" in timer_name:
                decoder_models_total += timer_value
        timers["MLVCEncoderTotal"].append(encoder_models_total)
        timers["MLVCDecoderTotal"].append(decoder_models_total)

    return FrameLoopSummary(
        q_index=AggregatedMetric.from_array(q_index, digits=1, save_values=save_values),
        psnr=AggregatedMetric.from_array(psnr, digits=4, save_values=save_values),
        psnr_y=AggregatedMetric.from_array(psnr_y, digits=4, save_values=save_values),
        psnr_u=AggregatedMetric.from_array(psnr_u, digits=4, save_values=save_values),
        psnr_v=AggregatedMetric.from_array(psnr_v, digits=4, save_values=save_values),
        bpp=AggregatedMetric.from_array(bpp, digits=6, save_values=save_values),
        timers={
            name: AggregatedMetric.from_array(times, digits=5, save_values=save_values)
            for name, times in timers.items()
        },
    )


def _estimate_bitstream_overhead_bytes(frame_type: FrameType) -> int:
    if frame_type == FrameType.I_FRAME:
        return 28
    elif frame_type == FrameType.P_FRAME:
        return 16
    elif frame_type == FrameType.LTR_RECOVERY:
        return 16
    else:
        raise ValueError(f"Unsupported frame type: {frame_type}")


def _build_dropped_frame_result(
    presentation_time: float, frame_type: FrameType, metrics: FrameMetrics
) -> FrameLoopFrameResult:
    return FrameLoopFrameResult(
        presentation_time=presentation_time,
        frame_type=frame_type,
        frame_idx=-1,
        ref_frame_idx=None,
        q_index=Q_INDEX_DROP_SENTINEL,
        feature_reset=False,
        marked_as_ltr=False,
        encoder=None,
        decoder=None,
        rate_control_info=None,
        metrics=metrics,
        timers={},
        op_profiles={},
    )


def _save_model_data(model_data_dict: dict[ModelPartId, ModelData], output_path: Path) -> None:
    data = {}
    for model_part_id, model_data in model_data_dict.items():
        for name, value in model_data.inputs._asdict().items():
            data[f"{model_part_id.value}_input_{name}"] = value
        for name, value in model_data.outputs._asdict().items():
            data[f"{model_part_id.value}_output_{name}"] = value
    np.savez(output_path, **data)


def _save_yuv_output(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    filename: str | Path,
) -> None:
    with open(filename, "wb") as f:
        for frame in frames:
            y, u, v = frame
            f.write(np.round(255 * y).clip(0, 255).astype(np.uint8).tobytes())
            f.write(np.round(255 * u).clip(0, 255).astype(np.uint8).tobytes())
            f.write(np.round(255 * v).clip(0, 255).astype(np.uint8).tobytes())
