# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import re
import dataclasses
import functools
import json
import logging
import numbers
import os
import time
from typing import Any, Optional, Sequence, Callable, List, Union, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from PIL import Image

from .common import dump_json, downsample_mask, AuxiliaryModels
from .distributed.operations import run_distributed
from .rate_controller import FrameType as RateControllerFrameType, LeakyBucket, RateController
from .stream_helper import prepare_frame, is_bytes_like, PaddingMode, PaddingAlignment
from .video_reader import PNGReader, YUVReader
from .video_writer import PNGWriter, YUVWriter, FFMpegWriter
from ..metrics.msssim import MS_SSIM
from ..metrics.vif import VIF
from ..models.segment import FaRLModel
from ..transforms.functional import yuv_444_to_420, ycbcr2rgb, rgb2ycbcr, yuv_420_to_444

__all__ = [
    "EncoderTestParams",
    "run_encoder_test",
    "EncoderTestTask",
    "EncoderTestDataset",
    "EncoderTester",
    "EncoderTestTaskProcessor",
    "identity_collate_fn",
    "create_frame_reader",
]


@dataclasses.dataclass
class EncoderTestParams:
    is_yuv420: bool
    i_frame_q_index_list: Optional[Sequence[int] | Dict[str, Sequence[int]]] = None
    p_frame_q_index_list: Optional[Sequence[int] | Dict[str, Sequence[int]]] = None
    bitrate_list: Optional[Sequence[float] | Dict[str, Sequence[float]]] = None
    float16: bool = False
    frame_step: int = 1
    max_n_frames: Optional[int] = None
    intra_period: Optional[int] = None
    reset_period: Optional[int] = None
    padding_mode: PaddingMode = PaddingMode.REPLICATE
    padding_alignment: PaddingAlignment = PaddingAlignment.BOTTOM_RIGHT
    padding_resolution: Optional[str] = None  # WxH
    calc_ssim: bool = False
    calc_ssim_rgb: bool = False
    calc_vif: bool = False
    calc_lpips: bool = False
    calc_lrwer_rgb: bool = False
    calc_deqa_score_rgb: bool = False
    calc_hrd_metrics: bool = False
    calc_psnr_roi: bool = False
    calc_psnr: bool = True
    calc_psnr_rgb: bool = False
    encode_bit_stream: bool = False
    use_decoder: bool = False
    use_i_frame_model: bool = True
    i_frame_pass_count: int = 1
    i_frame_qp_shift: Optional[int] = None
    ltr_period: Optional[int] = None
    ltr_start_idx: Optional[int] = None
    ltr_qp_shift: Optional[int] = None
    calc_bits_estimates: bool = False
    stream_folder_path: Optional[str] = None
    frame_rate: Optional[numbers.Number] = None
    decoder_folder_path: Optional[str] = None
    decoder_filename_format: Optional[str] = None
    decoder_format: Optional[str] = None
    decoder_compression_options: Optional[str] = None
    lr_model_data_path: Optional[str] = None
    max_batch_size: int = 1
    verbose: int = 0
    verbose_json: bool = False
    precomputed_masks_path: Optional[str] = None
    calc_ane_divergence: bool = False
    ane_exact_conv: bool = False


@dataclasses.dataclass
class EncoderTestTaskQPoint:
    """
    Encoder test task q-point related parameters
    """

    qp_desc: str
    i_frame_q_index: Optional[int] = None
    p_frame_q_index: Optional[int] = None
    bitrate: Optional[float] = None

    decoded_path: Optional[str] = None
    stream_path: Optional[str] = None


@dataclasses.dataclass
class EncoderTestTask:
    params: EncoderTestParams
    dataset_name: str
    seq_name: str

    src_type: str
    src_path: str
    src_width: int
    src_height: int
    n_frames: int
    frame_rate: float

    intra_period: int

    q_points: Sequence[EncoderTestTaskQPoint]
    decoder_format: str

    lr_model_data_filepath: Optional[dict] = None
    mask_path_root: Optional[str] = None


def create_frame_reader(task):
    if task.src_type == "png":
        reader = PNGReader(task.src_path, task.src_width, task.src_height)
    elif task.src_type == "yuv420":
        reader = YUVReader(task.src_path, task.src_width, task.src_height)
    else:
        raise ValueError("Unknown source format")

    return functools.partial(reader.read_one_frame, dst_format="420" if task.params.is_yuv420 else "rgb")


def identity_collate_fn(x):
    return x


def should_use_ltr_features(frame_idx, ltr_start_idx, ltr_period):
    if ltr_period is None or ltr_period <= 0:
        return False
    return (frame_idx > ltr_start_idx) and (frame_idx % ltr_period == 0)


def should_save_ltr_features(frame_idx, ltr_start_idx, ltr_period):
    if ltr_period is None or ltr_period <= 0:
        return False
    return (frame_idx == ltr_start_idx) or should_use_ltr_features(frame_idx, ltr_start_idx, ltr_period)


def use_ltr_features(dpb, params, intra_idx, qp_shift):
    # 1. use_i_frame=1, use_decoder=0: dpb is None for first frame, then dict
    # 2. use_i_frame=0, use_decoder=0: dpb is dict always
    # 3. use_i_frame=1, use_decoder=1: dpb is None for first frame, then tuple
    # 4. use_i_frame=0, use_decoder=1: dpb is dict for first frame, then tuple

    if params.use_decoder:
        if dpb is None:
            c_dpb = d_dpb = None
        else:
            c_dpb, d_dpb = dpb if isinstance(dpb, tuple) else (dpb, dpb)

        prev_c_ltr = c_dpb["ltr_feature"] if c_dpb is not None else None
        prev_d_ltr = d_dpb["ltr_feature"] if d_dpb is not None else None
        if should_use_ltr_features(intra_idx, params.ltr_start_idx, params.ltr_period):
            assert c_dpb is not None
            assert d_dpb is not None
            c_dpb["ref_feature"] = c_dpb["ltr_feature"]
            d_dpb["ref_feature"] = d_dpb["ltr_feature"]
            qp_shift = params.ltr_qp_shift

        dpb = (c_dpb, d_dpb)
        prev_feature = (prev_c_ltr, prev_d_ltr)
    else:
        prev_feature = dpb["ltr_feature"] if dpb is not None else None
        if should_use_ltr_features(intra_idx, params.ltr_start_idx, params.ltr_period):
            dpb["ref_feature"] = dpb["ltr_feature"]
            qp_shift = params.ltr_qp_shift

    return dpb, qp_shift, prev_feature


def save_ltr_features(dpb, params, intra_idx, prev_feature):
    if params.use_decoder:
        c_dpb, d_dpb = dpb
        c_prev_ltr, d_prev_ltr = prev_feature
        if should_save_ltr_features(intra_idx, params.ltr_start_idx, params.ltr_period):
            c_dpb["ltr_feature"] = c_dpb["ref_feature"]
            d_dpb["ltr_feature"] = d_dpb["ref_feature"]
        else:
            c_dpb["ltr_feature"] = c_prev_ltr
            d_dpb["ltr_feature"] = d_prev_ltr
        dpb = (c_dpb, d_dpb)
    else:
        prev_ltr_feature = prev_feature
        if should_save_ltr_features(intra_idx, params.ltr_start_idx, params.ltr_period):
            dpb["ltr_feature"] = dpb["ref_feature"]
        else:
            dpb["ltr_feature"] = prev_ltr_feature

    return dpb


class _CachedMaskReader:
    """Loads precomputed binary face/ROI masks from disk.

    Cache layout: {mask_path_root}/{frame_idx:06d}_mask.png - uint8 0/255, single channel,
    at native frame resolution.
    """

    def __init__(self, mask_path_root: str, expected_count: int, width: int, height: int, device: torch.device):
        self._root = mask_path_root
        self._expected_count = expected_count
        self._width = width
        self._height = height
        self._device = device

        if not os.path.isdir(mask_path_root):
            raise FileNotFoundError(f"Precomputed mask directory not found: {mask_path_root}")
        present = sum(1 for name in os.listdir(mask_path_root) if name.endswith("_mask.png"))
        if present < expected_count:
            raise FileNotFoundError(f"Mask cache incomplete at {mask_path_root}: {present}/{expected_count} files")

    def __call__(self, frame_idx: int) -> torch.Tensor:
        path = os.path.join(self._root, f"{frame_idx:06d}_mask.png")
        with Image.open(path) as im:
            mask = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        if mask.shape != (self._height, self._width):
            raise ValueError(f"Mask {path} shape {mask.shape} does not match frame ({self._height}, {self._width})")
        return torch.from_numpy(mask).to(self._device, non_blocking=True).unsqueeze(0).unsqueeze(0)


class EncoderTestTaskProcessor:
    def __init__(
        self,
        task: EncoderTestTask,
        *,
        i_frame_model: Optional[nn.Module],
        p_frame_model: Optional[nn.Module],
        auxiliary_models: Optional[AuxiliaryModels] = None,
        read_one_frame: Optional[Callable] = None,
    ):
        self.task = task
        self.i_frame_model = i_frame_model
        self.p_frame_model = p_frame_model

        model = i_frame_model if i_frame_model is not None else p_frame_model
        assert model is not None
        self.device = next(iter(model.parameters())).device

        if read_one_frame is None:
            read_one_frame = create_frame_reader(task)

        self.read_one_frame = read_one_frame

        def get_padding_size(m, default=16):
            return getattr(m, "padding_size", default) if m is not None else default

        self.padding_size = max(get_padding_size(self.i_frame_model), get_padding_size(self.p_frame_model))

        self.target_width = None
        self.target_height = None
        if task.params.padding_resolution:
            try:
                w_str, h_str = task.params.padding_resolution.lower().split("x")
                self.target_width = int(w_str)
                self.target_height = int(h_str)
            except ValueError:
                raise ValueError(
                    f"Invalid padding_resolution format: {task.params.padding_resolution}. "
                    f"Expected 'WxH' format, e.g., '640x368'"
                )

        for qp in task.q_points:
            if qp.stream_path:
                os.makedirs(qp.stream_path, exist_ok=True)

        def make_frame_writer(task_qp: EncoderTestTaskQPoint):
            if not task_qp.decoded_path:
                return None

            if task.decoder_format == "png":
                os.makedirs(task_qp.decoded_path, exist_ok=True)
                return PNGWriter(task_qp.decoded_path, task.src_width, task.src_height)
            if task.decoder_format == "yuv420":
                os.makedirs(os.path.dirname(task_qp.decoded_path), exist_ok=True)
                return YUVWriter(task_qp.decoded_path, task.src_width, task.src_height)
            if task.decoder_format == "mp4":
                compression_options = task.params.decoder_compression_options
                if compression_options:
                    import shlex

                    compression_options = shlex.split(compression_options, comments=False, posix=True)
                else:
                    compression_options = None
                os.makedirs(os.path.dirname(task_qp.decoded_path), exist_ok=True)
                return FFMpegWriter(
                    task_qp.decoded_path,
                    width=task.src_width,
                    height=task.src_height,
                    fps=task.frame_rate,
                    ffmpeg_options=compression_options,
                )
            else:
                raise ValueError("Unknown source format")

        if any(qp.decoded_path for qp in task.q_points):
            self.rec_frame_writer_list = tuple(make_frame_writer(qp) for qp in task.q_points)
        else:
            self.rec_frame_writer_list = None

        self._rate_controllers: Optional[list[RateController]] = None
        if task.params.bitrate_list is not None:
            self._rate_controllers = []
            for task_qp in task.q_points:
                assert task_qp.bitrate is not None
                self._rate_controllers.append(
                    RateController(
                        image_width=task.src_width,
                        image_height=task.src_height,
                        bitrate=float(task_qp.bitrate),
                        fps=task.frame_rate,
                    )
                )

        if task.params.calc_ssim:
            self._msssim = MS_SSIM(channels=1, data_range=1)
            self._msssim.to(self.device)
        else:
            self._msssim = None

        if task.params.calc_ssim_rgb:
            self._msssim_rgb = MS_SSIM(channels=3, data_range=1)
            self._msssim_rgb.to(self.device)
        else:
            self._msssim_rgb = None

        if task.params.calc_vif:
            self._vif = VIF()
            self._vif.to(self.device)
        else:
            self._vif = None

        if task.params.calc_lpips:
            assert auxiliary_models is not None
            lpips_model = auxiliary_models.perceptual_model_lpips_reference
            assert lpips_model is not None
            self._lpips = lpips_model
        else:
            self._lpips = None

        self._segmentation_model = None
        self._mask_reader: Optional[_CachedMaskReader] = None
        self._farl_nondeterministic_logged = False
        if task.params.calc_psnr_roi:
            if task.mask_path_root:
                self._mask_reader = _CachedMaskReader(
                    task.mask_path_root,
                    expected_count=task.n_frames,
                    width=task.src_width,
                    height=task.src_height,
                    device=self.device,
                )
            else:
                assert auxiliary_models is not None
                segmentation_model = auxiliary_models.segmentation_model
                assert segmentation_model is not None
                self._segmentation_model = segmentation_model

        if task.params.calc_lrwer_rgb:
            assert auxiliary_models is not None
            lr_model = auxiliary_models.lip_reading_model
            assert lr_model is not None
            self._lr_model = lr_model
            assert task.lr_model_data_filepath is not None
            self.lr_model_data = {
                "transcript_data": self._load_transcript_data(task.lr_model_data_filepath["transcript"]),
                "landmarks_data": self._load_landmark_data(task.lr_model_data_filepath["landmarks"]),
            }
        else:
            self._lr_model = None
            self.lr_model_data = None

        if task.params.calc_deqa_score_rgb:
            assert auxiliary_models is not None
            deqa_model = auxiliary_models.deqa_score_model
            assert deqa_model is not None
            self._deqa_model = deqa_model
            self._deqa_max_num_frames = auxiliary_models._deqa_max_num_frames
        else:
            self._deqa_model = None
            self._deqa_max_num_frames = None

        self.start_time = time.time()

        self.frame_pixel_num = 0
        self.frame_types = list()
        self.frame_stats = dict()
        self.rgb_frame_buffer = [[] for _ in range(len(task.q_points))]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(suppress_exceptions=exc_type is not None)

    @torch.inference_mode()
    def run(self):
        task = self.task
        params = task.params

        n_frames = task.n_frames
        frame_step = params.frame_step
        if frame_step > 1:
            n_frames = (n_frames + frame_step - 1) // frame_step

        dpb = None
        for frame_idx in range(n_frames):
            dpb = self._process_frame(frame_idx, dpb)

        if task.params.calc_lrwer_rgb:
            # Temporarily disable deterministic algorithm
            original_torch_deterministic_flag = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(False)
            # Log switch to non-deterministic mode
            if original_torch_deterministic_flag:
                logging.info("Forcing non-deterministic algorithms for LRWER calculation")

            logging.info(
                f"Calculating LRWER for {task.seq_name} for {[qp.qp_desc for qp in self.task.q_points]} q-points"
            )
            lrwer_rgb = [None] * len(self.task.q_points)
            assert self._lr_model is not None
            assert self.lr_model_data is not None
            for idx in range(len(self.task.q_points)):
                try:
                    rgb_rec_frames = self.rgb_frame_buffer[idx]
                    lrwer_rgb[idx] = self._lr_model(
                        frames_list=rgb_rec_frames,
                        landmarks_list=self.lr_model_data["landmarks_data"],
                        gt_transcript=self.lr_model_data["transcript_data"],
                    )
                except ValueError as e:
                    logging.error(
                        f"Error calculating LRWER for {task.seq_name} qp {task.q_points[idx].qp_desc}: {e}",
                        exc_info=True,
                    )

            # Restore original deterministic algorithm setting
            torch.use_deterministic_algorithms(original_torch_deterministic_flag)
            if original_torch_deterministic_flag:
                logging.info(f"Restored torch deterministic flag to: {original_torch_deterministic_flag}")

        if task.params.calc_deqa_score_rgb:
            # Temporarily disable deterministic algorithm
            original_torch_deterministic_flag = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(False)

            deqa_scores: list[Optional[list[float]]] = [None] * len(self.task.q_points)
            assert self._deqa_model is not None
            assert self._deqa_max_num_frames is not None
            for idx in range(len(self.task.q_points)):
                try:
                    rgb_rec_frames = self.rgb_frame_buffer[idx]
                    total_frames = len(rgb_rec_frames)
                    deqa_num_frames = min(total_frames, self._deqa_max_num_frames)

                    deqa_selected_indices = np.linspace(0, total_frames - 1, deqa_num_frames, dtype=int)
                    deqa_selected_frames = [rgb_rec_frames[i] for i in deqa_selected_indices]

                    rgb_rec_frames_pil = [Image.fromarray(frame) for frame in deqa_selected_frames]
                    deqa_scores_per_frame = [float(ds.cpu()) for ds in self._deqa_model(rgb_rec_frames_pil)]
                    deqa_scores[idx] = deqa_scores_per_frame
                except Exception as e:
                    logging.error(
                        f"Error calculating DeQA-Score for {task.seq_name} qp {task.q_points[idx].qp_desc}: {e}",
                        exc_info=True,
                    )
                    deqa_scores[idx] = None
            # Restore original deterministic algorithm setting
            torch.use_deterministic_algorithms(original_torch_deterministic_flag)

        if task.params.calc_hrd_metrics:
            self._add_hrd_metrics()

        test_time = time.time() - self.start_time
        results_dict = [self._generate_result(test_time, idx) for idx in range(len(self.task.q_points))]
        if task.params.calc_lrwer_rgb:
            for idx, result in enumerate(results_dict):
                result["ave_lrwer_rgb"] = lrwer_rgb[idx]

        if task.params.calc_deqa_score_rgb:
            for idx, result in enumerate(results_dict):
                scores = deqa_scores[idx]
                result["ave_all_frame_deqa_score_rgb"] = sum(scores) / len(scores) if scores is not None else None
                if task.params.verbose_json:
                    result["frame_deqa_score_rgb"] = deqa_scores[idx]
                    result["frame_type_deqa_score_rgb"] = [
                        self.frame_types[deqa_f_idx] for deqa_f_idx in deqa_selected_indices
                    ]

        if task.params.calc_ane_divergence and self.p_frame_model is not None:
            for idx, result in enumerate(results_dict):
                ane_results = self._run_ane_divergence(task, idx)
                if ane_results is not None:
                    result.update(ane_results)

        return results_dict

    def close(self, *, suppress_exceptions=False):
        try:
            writer_list = self.rec_frame_writer_list
            if writer_list is not None:
                for writer in writer_list:
                    if writer is None:
                        continue
                    if not suppress_exceptions:
                        writer.close()
                    else:
                        # noinspection PyBroadException
                        try:
                            writer.close()
                        except:  # noqa: E722
                            logging.error("Exception while closing decoded clip wtiter", exc_info=True)

                self.rec_frame_writer_list = None

            # done, no need to call self again
            suppress_exceptions = True
        finally:
            if not suppress_exceptions:
                self.close(suppress_exceptions=True)

    def _to_tensor(self, x):
        return torch.from_numpy(x).float().to(self.device, non_blocking=True).unsqueeze(0)

    @torch.inference_mode()
    def _run_ane_divergence(self, task: "EncoderTestTask", q_point_idx: int = 0):
        """Run ANE divergence validation: encode once, decode twice (AMP FP16 GPU + ANE-simulated)."""
        from .ane_mode import ane_simulation

        p_model = self.p_frame_model
        if p_model is None:
            return None

        logging.info(f"ANE divergence validation: {task.seq_name}")
        start = time.time()

        read_one_frame = create_frame_reader(task)
        params = task.params
        index_map = getattr(p_model, "frame_index_map", None)
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        q_index = task.q_points[q_point_idx].p_frame_q_index

        n_frames = task.n_frames
        frame_step = params.frame_step
        if frame_step > 1:
            n_frames = (n_frames + frame_step - 1) // frame_step

        # Per-frame metrics, collected over all (intra + p) frames.
        psnr_gpu_vs_ane = []
        psnr_ref_vs_gpu = []
        psnr_ref_vs_ane = []
        feature_mean_abs_diff, feature_max_abs_diff = [], []

        dpb_gpu = dpb_ane = None

        for frame_idx in range(n_frames):
            if frame_idx > 0 and frame_step > 1:
                for _ in range(frame_step - 1):
                    read_one_frame()

            image = read_one_frame()
            assert image is not None
            x, image_tensors, padding = prepare_frame(
                image,  # type: ignore[reportArgumentType]
                is_yuv420=params.is_yuv420,
                precision="fp32",
                device=self.device,
                padding_size=self.padding_size,
            )

            intra_idx = frame_idx % task.intra_period if task.intra_period > 0 else frame_idx
            if intra_idx == 0:
                # Gray I-frame: both chains restart from the same reference.
                dpb_gpu = dpb_ane = dict(
                    ref_frame=torch.ones_like(x) * 0.5,
                    ref_feature=None,
                    ref_mv_feature=None,
                    ref_y=None,
                    ref_mv_y=None,
                    ltr_feature=None,
                )
            fa_idx = index_map[(intra_idx + 1) % len(index_map)] if index_map is not None else intra_idx + 1

            # Encode + GPU decode in AMP fp16, then ANE-decode the same latent with the ANE DPB.
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                c_result = p_model.compress_core(x, dpb_gpu, q_index, fa_idx)  # type: ignore[union-attr]
            dpb_gpu = c_result["dpb"]

            with ane_simulation(exact_conv=params.ane_exact_conv):
                x_hat_ane, _, dpb_ane, _, _ = p_model.decode_core(  # type: ignore[union-attr]
                    dpb_ane, q_index, fa_idx, z_hat=c_result["z_hat"], y_res_q=c_result["y_raw"]
                )

            x_hat_gpu_out = self._unpad_clamp(dpb_gpu["ref_frame"], padding)
            x_hat_ane_out = self._unpad_clamp(x_hat_ane, padding)
            psnr_gpu_vs_ane.append(self._calc_yuv_psnr(x_hat_gpu_out, x_hat_ane_out, params))
            psnr_ref_vs_gpu.append(self._calc_yuv_psnr(image_tensors, x_hat_gpu_out, params))
            psnr_ref_vs_ane.append(self._calc_yuv_psnr(image_tensors, x_hat_ane_out, params))

            feat_gpu, feat_ane = dpb_gpu.get("ref_feature"), dpb_ane.get("ref_feature")
            if feat_gpu is not None and feat_ane is not None:
                diff = (feat_gpu.float() - feat_ane.float()).abs()
                feature_mean_abs_diff.append(float(diff.mean()))
                feature_max_abs_diff.append(float(diff.max()))
            else:
                feature_mean_abs_diff.append(0.0)
                feature_max_abs_diff.append(0.0)

        if not psnr_gpu_vs_ane:
            logging.warning("ANE divergence: %s -- no frames processed, skipping", task.seq_name)
            return None

        logging.info(
            "ANE divergence: %s -- avg GPU-ANE PSNR=%.2f dB, %.1fs",
            task.seq_name,
            np.mean(psnr_gpu_vs_ane),
            time.time() - start,
        )

        # (metric name, per-frame values, aggregator)
        specs = [
            ("divergence_psnr", psnr_gpu_vs_ane, np.mean),
            ("psnr_gpu", psnr_ref_vs_gpu, np.mean),
            ("psnr_ane", psnr_ref_vs_ane, np.mean),
            ("feature_mean_abs_diff", feature_mean_abs_diff, np.mean),
            ("feature_max_abs_diff", feature_max_abs_diff, np.max),
        ]
        result = {}
        for name, vals, agg in specs:
            result[f"ave_p_frame_ane_{name}"] = float(agg(vals))
            if params.verbose_json:
                result[f"frame_ane_{name}"] = vals
        return result

    @staticmethod
    def _unpad_clamp(x_hat, padding):
        """Remove spatial padding and clamp to [0, 1]."""
        x_hat = nn.functional.pad(x_hat, tuple(-p for p in padding))
        x_hat = torch.nan_to_num_(x_hat, nan=0.0, posinf=1.0, neginf=0.0)
        x_hat = torch.clamp_(x_hat, min=0.0, max=1.0)
        return x_hat

    @staticmethod
    def _calc_yuv_psnr(ref, x_hat, params):
        def psnr(a, b):
            mse = (a.float() - b.float()).square().mean()
            return 100.0 if mse < 1e-10 else float(-10.0 * torch.log10(mse))

        if params.is_yuv420:
            y, u, v = ref if isinstance(ref, tuple) else yuv_444_to_420(ref)
            y_rec, u_rec, v_rec = yuv_444_to_420(x_hat)
            return (6 * psnr(y, y_rec) + psnr(u, u_rec) + psnr(v, v_rec)) / 8
        return psnr(ref, x_hat)

    def _process_frame(self, frame_idx, dpb):
        frame_start_time = time.time()

        task = self.task
        params = task.params
        if frame_idx > 0 and params.frame_step > 1:
            for _ in range(params.frame_step - 1):
                self.read_one_frame()

        image = self.read_one_frame()
        assert image is not None
        x, image, padding = prepare_frame(
            image,  # type: ignore[reportArgumentType]
            is_yuv420=params.is_yuv420,
            precision="fp16" if params.float16 else "fp32",
            device=self.device,
            padding_size=self.padding_size,
            padding_mode=params.padding_mode,
            padding_alignment=params.padding_alignment,
            target_height=self.target_height,
            target_width=self.target_width,
        )

        if self._mask_reader is not None:
            mask = self._mask_reader(frame_idx * params.frame_step)
        elif self._segmentation_model is not None:
            if isinstance(self._segmentation_model.model, FaRLModel):
                # FaRL/facer warp-grid path reaches a CuBLAS-backed torch.bmm which fails in deterministic mode
                original_torch_deterministic_flag = torch.are_deterministic_algorithms_enabled()
                torch.use_deterministic_algorithms(False)
                if original_torch_deterministic_flag and not self._farl_nondeterministic_logged:
                    logging.info("Forcing non-deterministic algorithms for FaRL segmentation mask calculation")
                    self._farl_nondeterministic_logged = True
                try:
                    mask = self._segmentation_model(x, is_yuv420=params.is_yuv420)
                except Exception:
                    logging.exception(f"FaRL segmentation failed for {self.task.seq_name} frame {frame_idx}")
                    raise
                finally:
                    torch.use_deterministic_algorithms(original_torch_deterministic_flag)
            else:
                mask = self._segmentation_model(x, is_yuv420=params.is_yuv420)
        else:
            mask = None

        batch_size = len(task.q_points)
        if batch_size > 1:
            x = x.expand(batch_size, -1, -1, -1)
            if mask is not None:
                mask = mask.expand(batch_size, -1, -1, -1)

        height = image[0].shape[2]
        width = image[0].shape[3]

        frame_pixel_num = width * height
        if self.frame_pixel_num == 0:
            self.frame_pixel_num = frame_pixel_num
        elif self.frame_pixel_num != frame_pixel_num:
            raise ValueError("Video frame size changed in the middle of clip")

        intra_idx = frame_idx % task.intra_period if task.intra_period > 0 else frame_idx
        pass_count = 1
        qp_shift = None

        if intra_idx == 0:
            if params.use_i_frame_model:
                dpb = None
            else:
                # Since we only use p-frames, we need to start with a dummy i-frame
                dpb = self._make_dpb(torch.ones_like(x) * 0.5)
                pass_count = max(params.i_frame_pass_count, 1)
                qp_shift = params.i_frame_qp_shift

        def acc(a, b):
            return b if a is None else a + b

        dpb, qp_shift, prev_feature = use_ltr_features(dpb, params, intra_idx, qp_shift)

        q_index_override: Optional[int | list[int]] = None
        if self._rate_controllers is not None:
            q_index_override = self._compute_rate_controller_q_index(frame_idx, intra_idx)

        bits = None
        bits_estimate = None
        try:
            result = None

            for i in range(pass_count):
                result = self._run_model(
                    x,
                    dpb,
                    frame_idx=frame_idx,
                    intra_idx=intra_idx,
                    iteration_idx=i,
                    q_index_override=q_index_override,
                    qp_shift=qp_shift,
                )

                dpb = result["dpb"]

                bits = acc(bits, result["bits"])
                if params.calc_bits_estimates:
                    bits_estimate = acc(bits_estimate, result["bits_estimate"])

            dpb = save_ltr_features(dpb, params, intra_idx, prev_feature)

            assert result is not None
            x_hat = result["x_hat"]
        except ValueError:
            logging.exception(
                f"frame inference failed for {self.task.src_path},"
                f" q-point {','.join(qp.qp_desc for qp in task.q_points)}, frame {frame_idx}"
            )

            x_hat = torch.zeros_like(x)
            dpb = self._make_dpb(x_hat)

            bits = torch.zeros(x_hat.size(0))
            bits_estimate = bits

        if self._rate_controllers is not None and bits is not None:
            self._update_rate_controllers(bits)

        self.frame_types.append(0 if intra_idx == 0 and params.use_i_frame_model else 1)

        x_hat = nn.functional.pad(x_hat, tuple(-p for p in padding))
        if self._segmentation_model is not None:
            # FaRL produces mask at the padded resolution; crop back to native.
            # Cached masks are already at native resolution and need no crop.
            assert mask is not None
            mask = nn.functional.pad(mask, tuple(-p for p in padding))

        # normalize output for metrics
        x_hat = torch.nan_to_num_(x_hat, nan=0.0, posinf=1.0, neginf=0.0)
        x_hat = torch.clamp_(x_hat, min=0.0, max=1.0)

        assert bits is not None
        self._add_frame_stat("bpp", bits / (width * height))
        if bits_estimate is not None:
            if not isinstance(bits_estimate, torch.Tensor):
                # backward compatibility
                bits_estimate = torch.tensor((bits_estimate,), dtype=torch.float)

            self._add_frame_stat("bpp_estimate", bits_estimate / (width * height))

        extra_stats = self._process_rec(x_hat, image, mask)
        frame_end_time = time.time()

        if params.verbose >= 2:
            metrics = ", ".join(f"{n}: {self._format_tensor(v, '{:.4f}')}" for n, v in extra_stats.items())
            logging.info(
                f"frame {frame_idx}, {frame_end_time - frame_start_time:.3f} seconds, "
                f"bits: {self._format_tensor(bits, '{:.0f}')}, {metrics}"
            )

        return dpb

    def _compute_rate_controller_q_index(self, frame_idx: int, intra_idx: int) -> int | list[int]:
        assert self._rate_controllers is not None
        assert len(self._rate_controllers) == len(self.task.q_points)

        # Time & frame type
        presentation_time = (frame_idx + 1) / self.task.frame_rate
        if intra_idx == 0:
            frame_type = RateControllerFrameType.I_FRAME
        elif should_use_ltr_features(intra_idx, self.task.params.ltr_start_idx, self.task.params.ltr_period):
            frame_type = RateControllerFrameType.LTR_RECOVERY
        else:
            frame_type = RateControllerFrameType.P_FRAME

        # Solve for q-index for each bitrate point
        q_index_list: list[int] = list()
        for task_qp, rate_controller in zip(self.task.q_points, self._rate_controllers):
            q_index = rate_controller.solve_q_index(presentation_time, frame_type)
            if q_index is None:
                # Frame dropping is not implemented in encoder_tester, force lowest q_index
                q_index = 0
            q_index_list.append(q_index)

        if len(q_index_list) == 1:
            return q_index_list[0]

        return q_index_list

    def _update_rate_controllers(self, bits: torch.Tensor):
        assert self._rate_controllers is not None
        assert len(self._rate_controllers) == len(self.task.q_points)

        for rate_idx, rate_controller in enumerate(self._rate_controllers):
            rate_controller.update(
                header_bits=0,
                payload_bits=int(bits[rate_idx].item()),
            )

    @staticmethod
    def _format_tensor(x, fmt):
        return ", ".join(fmt.format(v) for v in x.cpu().flatten())

    def _run_model(
        self,
        x,
        dpb,
        *,
        frame_idx: int,
        intra_idx: int,
        iteration_idx: int,
        q_index_override: Optional[int | list[int]] = None,
        qp_shift: Optional[int] = None,
    ):
        params = self.task.params
        is_i_frame_model = params.use_i_frame_model and intra_idx == 0
        use_decoder = params.use_decoder
        use_encoder = params.encode_bit_stream or params.calc_bits_estimates or use_decoder

        model = self.i_frame_model if is_i_frame_model else self.p_frame_model
        assert model is not None
        if not use_encoder:
            f_args = (
                self._make_i_frame_args(q_index_override=q_index_override)
                if is_i_frame_model
                else self._make_p_frame_args(
                    model,
                    dpb,
                    intra_idx,
                    iteration_idx,
                    q_index_override=q_index_override,
                    qp_shift=qp_shift,
                )
            )

            result = model(x, **f_args)
            if is_i_frame_model:
                result["dpb"] = self._make_dpb(result["x_hat"])
        else:
            c_dpb, d_dpb = dpb if isinstance(dpb, tuple) else (dpb, dpb)

            # run encoder
            c_args = (
                self._make_i_frame_args(q_index_override=q_index_override)
                if is_i_frame_model
                else self._make_p_frame_args(
                    model,
                    c_dpb,
                    intra_idx,
                    iteration_idx,
                    q_index_override=q_index_override,
                    qp_shift=qp_shift,
                )
            )

            result = model.compress(x, **c_args, calc_bits_estimates=params.calc_bits_estimates)  # type: ignore[union-attr]

            bit_stream_list = result.pop("bit_stream")

            # for backward compatibility checking for a single bit stream
            can_batch_decode = True
            if is_bytes_like(bit_stream_list):
                bit_stream_list = (bit_stream_list,)
                can_batch_decode = False

            if len(bit_stream_list) != len(self.task.q_points):
                raise ValueError(
                    f"Invalid number of bit streams: {len(bit_stream_list)}, expected {len(self.task.q_points)}"
                )

            result["bits"] = torch.tensor([8 * len(stream) for stream in bit_stream_list], dtype=torch.float)
            if is_i_frame_model:
                result["dpb"] = self._make_dpb(result["x_hat"])

            for task_qp, stream in zip(self.task.q_points, bit_stream_list):
                stream_path = task_qp.stream_path
                if stream_path is None:
                    continue
                # save bit stream
                stream_path = os.path.join(stream_path, f"{frame_idx}.bin")
                with open(stream_path, "wb") as f:
                    f.write(stream)
                    f.close()

            if use_decoder:
                if not can_batch_decode and len(bit_stream_list) != 1:
                    raise ValueError("Batch decoding is not supported by the model")

                # run decoder
                d_args = (
                    c_args
                    if is_i_frame_model
                    else self._make_p_frame_args(
                        model,
                        d_dpb,
                        intra_idx,
                        iteration_idx,
                        q_index_override=q_index_override,
                    )
                )
                d_result = model.decompress(  # type: ignore[union-attr]
                    bit_stream_list if can_batch_decode else bit_stream_list[0],
                    height=x.shape[-2],
                    width=x.shape[-1],
                    **d_args,
                )

                # override output
                result["x_hat"] = d_result["x_hat"]

                # extend dpb
                d_dpb = self._make_dpb(result["x_hat"]) if is_i_frame_model else d_result["dpb"]
                result["dpb"] = result["dpb"], d_dpb

        return result

    def _make_i_frame_args(self, q_index_override: Optional[int | list[int]] = None):
        return dict(
            q_index=self._make_qindex_arg("i_frame_q_index", q_index_override=q_index_override),
        )

    def _make_p_frame_args(
        self,
        model,
        dpb,
        intra_idx: int,
        iteration_idx: int,
        q_index_override: Optional[int | list[int]] = None,
        qp_shift: Optional[int] = None,
    ):
        params = self.task.params
        if not params.use_i_frame_model:
            # take into account gray i-frame
            intra_idx += 1
        reset_period = params.reset_period
        if reset_period and reset_period > 1:
            intra_idx = (intra_idx - 1) % reset_period + 1

            if intra_idx == 1 and iteration_idx == 0:
                if "ref_frame" in dpb and "ref_frame_from_reset_head" in dpb and dpb["ref_feature"] is not None:
                    dpb["ref_frame"] = dpb.pop("ref_frame_from_reset_head")

                for name in "ref_feature", "ref_mv_feature":
                    if dpb.get(name) is not None:
                        dpb[name] = None

        index_map = getattr(model, "frame_index_map")
        if index_map is not None:
            fa_idx = index_map[intra_idx % len(index_map)]
        else:
            fa_idx = intra_idx

        q_index = self._make_qindex_arg("p_frame_q_index", q_index_override=q_index_override)

        if q_index_override is not None and (qp_shift is not None and qp_shift != 0):
            raise ValueError(f"qp_shift={qp_shift} is not supported when q_index_override is specified")

        if qp_shift is not None:
            max_qp_index = model.get_qp_num() - 1
            if isinstance(q_index, torch.Tensor):
                q_index = torch.clamp(q_index + qp_shift, min=0, max=max_qp_index)
            else:
                q_index = max(0, min(q_index + qp_shift, max_qp_index))

        args = dict(
            dpb=dpb,
            q_index=q_index,
            fa_idx=fa_idx,
        )
        if hasattr(model, "expert_schedule"):
            args["curr_poc"] = intra_idx - 1

        return args

    def _make_qindex_arg(self, name: str, q_index_override: Optional[int | list[int]] = None):
        q_points = self.task.q_points
        if len(q_points) == 1:
            if q_index_override is not None:
                assert not isinstance(q_index_override, list)
                return q_index_override
            return getattr(q_points[0], name)
        else:
            if q_index_override is not None:
                assert isinstance(q_index_override, list)
                qp = q_index_override
            else:
                qp = list(getattr(q, name) for q in q_points)
            assert len(qp) == len(q_points)
            return torch.tensor(qp, dtype=torch.long, device=self.device)

    @staticmethod
    def _make_dpb(x_hat):
        return dict(
            ref_frame=x_hat,
            ref_feature=None,
            ref_mv_feature=None,
            ref_y=None,
            ref_mv_y=None,
            ltr_feature=None,
        )

    def _process_rec(self, x_hat, image, mask=None):
        params = self.task.params
        rgb_required = (
            params.calc_psnr_rgb
            or params.calc_ssim_rgb
            or params.calc_lpips
            or params.calc_lrwer_rgb
            or params.calc_deqa_score_rgb
        )

        if params.is_yuv420:
            y, u, v = image
            y_rec, u_rec, v_rec = yuv_444_to_420(x_hat)
            if rgb_required:
                rgb = ycbcr2rgb(yuv_420_to_444(image))
                rgb_rec = ycbcr2rgb(x_hat)
            else:
                rgb = rgb_rec = None

            batch_size = y_rec.size(0)
        else:
            rgb = image
            rgb_rec = x_hat

            if params.calc_psnr or params.calc_ssim or params.calc_vif:
                y, u, v = yuv_444_to_420(rgb2ycbcr(rgb))
                y_rec, u_rec, v_rec = yuv_444_to_420(rgb2ycbcr(rgb_rec))
            else:
                y, u, v = y_rec, u_rec, v_rec = None, None, None

            batch_size = rgb_rec.size(0)

        if batch_size > 1:
            if y is not None:
                y = y.expand(batch_size, -1, -1, -1)
                assert u is not None and v is not None
                u = u.expand(batch_size, -1, -1, -1)
                v = v.expand(batch_size, -1, -1, -1)
            if rgb is not None:
                rgb = rgb.expand(batch_size, -1, -1, -1)

        if params.calc_lrwer_rgb or params.calc_deqa_score_rgb:
            assert rgb_rec is not None
            for batch_idx in range(batch_size):
                frame_to_store = rgb_rec[batch_idx : batch_idx + 1].squeeze(0).permute(1, 2, 0).cpu().numpy()
                frame_to_store = (frame_to_store * 255).astype(np.uint8)
                self.rgb_frame_buffer[batch_idx].append(frame_to_store)

        extra_stats = dict()

        if params.calc_psnr:
            psnr_y = self._calc_psnr(y, y_rec)
            self._add_frame_stat("psnr_y", psnr_y)
            psnr_u = self._calc_psnr(u, u_rec)
            self._add_frame_stat("psnr_u", psnr_u)
            psnr_v = self._calc_psnr(v, v_rec)
            self._add_frame_stat("psnr_v", psnr_v)
            psnr = (6 * psnr_y + psnr_u + psnr_v) / 8
            self._add_frame_stat("psnr", psnr)
            extra_stats["PSNR"] = psnr

        if params.calc_psnr_rgb:
            psnr_rgb = self._calc_psnr(rgb, rgb_rec)
            self._add_frame_stat("psnr_rgb", psnr_rgb)
            extra_stats["PSNR(RGB)"] = psnr_rgb

        if params.calc_psnr_roi:
            assert mask is not None
            mask_2x = downsample_mask(mask, factor=2)

            psnr_y = self._calc_psnr(y, y_rec, mask)
            self._add_frame_stat("psnr_y_roi", psnr_y)
            psnr_u = self._calc_psnr(u, u_rec, mask_2x)
            self._add_frame_stat("psnr_u_roi", psnr_u)
            psnr_v = self._calc_psnr(v, v_rec, mask_2x)
            self._add_frame_stat("psnr_v_roi", psnr_v)

            psnr = (6 * psnr_y + psnr_u + psnr_v) / 8
            self._add_frame_stat("psnr_roi", psnr)
            extra_stats["PSNR(ROI)"] = psnr

            inv_mask = 1 - mask
            inv_mask_2x = downsample_mask(inv_mask, factor=2)

            psnr_y_inv = self._calc_psnr(y, y_rec, inv_mask)
            psnr_u_inv = self._calc_psnr(u, u_rec, inv_mask_2x)
            psnr_v_inv = self._calc_psnr(v, v_rec, inv_mask_2x)

            psnr_inv = (6 * psnr_y_inv + psnr_u_inv + psnr_v_inv) / 8
            self._add_frame_stat("psnr_inv_roi", psnr_inv)
            extra_stats["PSNR(Inv ROI)"] = psnr_inv

        if params.calc_ssim:
            assert self._msssim is not None
            msssim_y = self._msssim(y, y_rec)
            self._add_frame_stat("msssim_y", msssim_y)
            msssim_u = self._msssim(u, u_rec)
            self._add_frame_stat("msssim_u", msssim_u)
            msssim_v = self._msssim(v, v_rec)
            self._add_frame_stat("msssim_v", msssim_v)
            msssim = (6 * msssim_y + msssim_u + msssim_v) / 8
            self._add_frame_stat("msssim", msssim)
            extra_stats["MS-SSIM"] = msssim

        if params.calc_ssim_rgb:
            assert self._msssim_rgb is not None
            msssim_rgb = self._msssim_rgb(rgb, rgb_rec)
            self._add_frame_stat("msssim_rgb", msssim_rgb)
            extra_stats["MS-SSIM(RGB)"] = msssim_rgb

        if params.calc_vif:
            assert self._vif is not None
            vif, vif_per_scale = self._vif(y_rec, y)
            self._add_frame_stat("vif", vif)
            for scale_idx, scale_vif in enumerate(vif_per_scale):
                self._add_frame_stat(f"vif_scale_{scale_idx}", scale_vif)
            extra_stats["VIF"] = vif
            for scale_idx, scale_vif in enumerate(vif_per_scale):
                extra_stats[f"VIF_scale_{scale_idx}"] = scale_vif

        if params.calc_lpips:
            assert self._lpips is not None
            lpips = self._lpips(rgb_rec, rgb)
            self._add_frame_stat("lpips", lpips)
            extra_stats["LPIPS"] = lpips

        if self.rec_frame_writer_list is not None:
            if params.is_yuv420:
                for idx, writer in enumerate(self.rec_frame_writer_list):
                    assert y_rec is not None and u_rec is not None and v_rec is not None
                    y_rec_i = y_rec[idx].cpu().numpy()
                    uv_rec_i = np.concatenate((u_rec[idx].cpu().numpy(), v_rec[idx].cpu().numpy()))
                    assert writer is not None
                    writer.write_one_frame(y=y_rec_i, uv=uv_rec_i, src_format="420")
            else:
                for idx, writer in enumerate(self.rec_frame_writer_list):
                    assert writer is not None
                    writer.write_one_frame(rgb=x_hat[idx].cpu().numpy(), src_format="rgb")

        return extra_stats

    @staticmethod
    def _calc_psnr(x1, x2, mask=None):
        mse = torch.square(x1 - x2)

        if mask is not None:
            mse = mse * mask
            mask_pixel_num = mask.sum(dim=(1, 2, 3))
            mse = mse.sum(dim=(1, 2, 3)) / mask_pixel_num.clamp(min=1e-10)
        else:
            mse = mse.sum(dim=(1, 2, 3)) / (mse.size(2) * mse.size(3))

        return -10 * torch.log10(torch.clamp_min_(mse, 1e-10))

    def _add_frame_stat(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        else:
            value = np.asarray(value)

        if len(value) != len(self.task.q_points):
            raise ValueError(f"Invalid stat value size: {len(value)}, expected {len(self.task.q_points)}")

        stat_list = self.frame_stats.get(name)
        if stat_list is None:
            self.frame_stats[name] = stat_list = list()

        stat_list.append(value)

    def _add_hrd_metrics(self):
        bpp_stats = self.frame_stats.get("bpp")
        if bpp_stats is None:
            return

        bpp_array = np.asarray(bpp_stats)
        hrd_level_array = np.zeros_like(bpp_array)
        for index in range(bpp_array.shape[1]):
            frame_bits = np.round(self.frame_pixel_num * bpp_array[:, index]).astype(int)
            target_bitrate = self.task.q_points[index].bitrate
            if target_bitrate is None:
                target_bitrate = np.mean(frame_bits) * self.task.frame_rate

            bucket = LeakyBucket(bitrate=float(target_bitrate), fps=self.task.frame_rate, bucket_size=1.0)
            for frame_idx, bits in enumerate(frame_bits):
                presentation_time = (frame_idx + 1) / self.task.frame_rate
                bucket.update(presentation_time, bits)
                hrd_level_array[frame_idx, index] = bucket.level

        for frame_idx in range(hrd_level_array.shape[0]):
            self._add_frame_stat("hrd_level", hrd_level_array[frame_idx])

    @staticmethod
    def _load_transcript_data(file_path: str):
        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            logging.error(f"Failed to load transcript {file_path}: {e}")
            return None

    @staticmethod
    def _load_landmark_data(file_path: str):
        try:
            loaded_np_arrays = np.load(file_path)
            landmarks_list = [loaded_np_arrays[f"arr_{i}"] for i in range(len(loaded_np_arrays.files))]
            return landmarks_list
        except Exception as e:
            logging.error(f"Failed to load landmarks {file_path}: {e}")
            return None

    def _generate_result(self, test_time, index):
        task = self.task
        result: dict[str, Any] = dict(
            ds_name=task.dataset_name,
            video_path=task.seq_name,
            frame_pixel_num=self.frame_pixel_num,
        )
        for type_index, type_name in enumerate(("i_frame", "p_frame")):
            n_frames = sum(1 for t in self.frame_types if t == type_index)
            result[f"{type_name}_num"] = n_frames
            for name, stat_list in self.frame_stats.items():
                value = float(sum(x[index] for t, x in zip(self.frame_types, stat_list) if t == type_index))
                value /= max(n_frames, 1)

                result[f"ave_{type_name}_{name}"] = value

        n_frames = len(self.frame_types)
        for name, stat_list in self.frame_stats.items():
            value = float(sum(x[index] for x in stat_list))
            value /= max(n_frames, 1)

            result[f"ave_all_frame_{name}"] = value

        for name in ["hrd_level"]:
            if name not in self.frame_stats:
                continue
            stat_list = self.frame_stats[name]
            values = np.asarray([x[index] for x in stat_list], dtype=float)
            result[f"p05_all_frame_{name}"] = float(np.percentile(values, 5))
            result[f"p95_all_frame_{name}"] = float(np.percentile(values, 95))

        if task.params.verbose_json:
            result["frame_type"] = self.frame_types.copy()
            for name, stat_list in self.frame_stats.items():
                result[f"frame_{name}"] = list(float(x[index]) for x in stat_list)

        task_qp = task.q_points[index]
        result["i_frame_q_index"] = task_qp.i_frame_q_index
        result["p_frame_q_index"] = task_qp.p_frame_q_index
        if task_qp.decoded_path:
            result["decoded_seq_name"] = os.path.basename(task_qp.decoded_path)
        result["test_time"] = test_time

        return result


class EncoderTestDataset(torch.utils.data.IterableDataset):
    def __init__(self, task_list: List[EncoderTestTask]):
        self.task_list = task_list

    def __iter__(self):
        return iter(self.get_frames())

    def get_frames(self):
        for task in self.task_list:
            reader = create_frame_reader(task)
            for _ in range(task.n_frames):
                yield reader()


class EncoderTester:
    def __init__(
        self,
        rank: int,
        params: EncoderTestParams,
        *,
        i_frame_model: Union[nn.Module, Callable[[EncoderTestTask], nn.Module], None],
        p_frame_model: Union[nn.Module, Callable[[EncoderTestTask], nn.Module], None],
        auxiliary_models: Optional[AuxiliaryModels] = None,
    ):
        self.rank = rank
        params = copy.copy(params)
        self.params = params

        self.n_qpoints = None

        if params.intra_period is not None and params.intra_period < 0:
            raise ValueError(f"Invalid intra period {params.intra_period}")

        if not params.use_i_frame_model:
            i_frame_model = None
        elif i_frame_model is None:
            raise ValueError("no i-frame model specified")
        elif params.intra_period == 1:
            p_frame_model = None
        elif p_frame_model is None:
            raise ValueError(f"i-frame period is {params.intra_period} and no p-frame model specified")

        self.i_frame_model = i_frame_model
        self.p_frame_model = p_frame_model
        self.auxiliary_models = auxiliary_models

        if params.bitrate_list is None:
            self._init_q_index_lists(params)
        elif params.i_frame_q_index_list is not None or params.p_frame_q_index_list is not None:
            raise ValueError("bitrate_list can not be used with i_frame_q_index_list or p_frame_q_index_list")

    def _init_q_index_lists(self, params: EncoderTestParams):
        i_list, p_list = params.i_frame_q_index_list, params.p_frame_q_index_list

        # each list defaults to the other
        if i_list is None:
            i_list = p_list
        elif p_list is None:
            p_list = i_list

        # drop unused lists
        if self.i_frame_model is None:
            i_list = p_list
        elif self.p_frame_model is None:
            p_list = i_list

        # retain per-video lists
        if not (isinstance(i_list, dict) and isinstance(p_list, dict)):
            # convert to tuples
            if i_list is not p_list:
                assert i_list is not None and p_list is not None
                i_list = tuple(i_list)
                p_list = tuple(p_list)
            elif i_list is not None:
                i_list = p_list = tuple(i_list)

        if i_list is None or len(i_list) == 0:
            raise ValueError("No q-indices to test")

        assert p_list is not None
        if len(i_list) != len(p_list):
            raise ValueError("Can not use different number of i-frame and p-frame q-indices")

        params.i_frame_q_index_list, params.p_frame_q_index_list = i_list, p_list  # type: ignore[assignment]

    def _get_rate_list(self, rate_list, seq_name):

        if isinstance(rate_list, dict):
            seq_base_name = re.sub(r"_(\d+)x(\d+)_", "_", seq_name)
            rate_list = rate_list[seq_base_name]
        return rate_list

    def _build_task_list(self, testset_desc_filename: str, *, testset_root: Optional[str] = None):
        with open(testset_desc_filename, "rt", encoding="utf-8") as f:
            testset_desc = json.load(f)

        testset_root = testset_root or testset_desc.get("root_path", None)
        testset_root = testset_root or os.path.dirname(testset_desc_filename)

        params = self.params
        max_batch_size = max(int(params.max_batch_size), 1)

        if params.bitrate_list is not None:
            if isinstance(params.bitrate_list, dict):
                key = list(params.bitrate_list.keys())[0]
                n_qpoints = len(params.bitrate_list[key])
            else:
                n_qpoints = len(params.bitrate_list)
        else:
            assert params.i_frame_q_index_list is not None
            if isinstance(params.i_frame_q_index_list, dict):
                key = list(params.i_frame_q_index_list.keys())[0]
                n_qpoints = len(params.i_frame_q_index_list[key])
            else:
                n_qpoints = len(params.i_frame_q_index_list)

        self.n_qpoints = n_qpoints
        for dataset_name, dataset_desc in testset_desc["test_classes"].items():
            if not dataset_desc.get("test", True):
                continue
            src_type = dataset_desc["src_type"]
            if params.decoder_format:
                decoder_format = params.decoder_format
            else:
                decoder_format = src_type

            decoder_ext = ""
            if params.decoder_folder_path:
                if decoder_format == "png":
                    decoder_ext = ""
                elif decoder_format == "yuv420":
                    decoder_ext = ".yuv"
                elif decoder_format == "mp4":
                    decoder_ext = ".mp4"
                else:
                    raise ValueError(f"Unsuppoted decoder_format {decoder_format}")

            for seq_name, seq_desc in dataset_desc["sequences"].items():
                intra_period = params.intra_period
                if intra_period is None:
                    intra_period = seq_desc.get("gop") or 1

                if intra_period < 0:
                    raise ValueError(f"Invalid intra period: {intra_period}")

                seq_base_name = seq_name
                if src_type != "png":
                    seq_base_name = os.path.splitext(seq_base_name)[0]

                n_frames = seq_desc["frames"]
                if params.max_n_frames is not None and params.max_n_frames > 0:
                    n_frames = min(n_frames, params.max_n_frames)

                if params.bitrate_list is not None:
                    i_frame_q_index_list = None
                    p_frame_q_index_list = None
                    bitrate_list = self._get_rate_list(params.bitrate_list, seq_name)
                    assert bitrate_list is not None
                else:
                    i_frame_q_index_list = self._get_rate_list(params.i_frame_q_index_list, seq_name)
                    p_frame_q_index_list = self._get_rate_list(params.p_frame_q_index_list, seq_name)
                    bitrate_list = None
                    assert i_frame_q_index_list is not None and p_frame_q_index_list is not None

                for rate_base_idx in range(0, n_qpoints, max_batch_size):
                    qp_list = list()
                    rate_end_idx = min(n_qpoints, rate_base_idx + max_batch_size)
                    for rate_idx in range(rate_base_idx, rate_end_idx):
                        if bitrate_list is not None:
                            task_qp = EncoderTestTaskQPoint(
                                qp_desc=f"bitrate_kbps={1e-3 * bitrate_list[rate_idx]:.1f}",
                                bitrate=bitrate_list[rate_idx],
                            )
                        else:
                            assert i_frame_q_index_list is not None and p_frame_q_index_list is not None
                            qp_desc = str(p_frame_q_index_list[rate_idx])
                            # in case of different q-indices for i-frames and p-frames
                            if p_frame_q_index_list[rate_idx] != i_frame_q_index_list[rate_idx]:
                                qp_desc = f"{i_frame_q_index_list[rate_idx]}i-{qp_desc}p"

                            task_qp = EncoderTestTaskQPoint(
                                i_frame_q_index=i_frame_q_index_list[rate_idx],
                                p_frame_q_index=p_frame_q_index_list[rate_idx],
                                qp_desc=qp_desc,
                            )
                        if params.encode_bit_stream and params.stream_folder_path:
                            task_qp.stream_path = os.path.join(
                                params.stream_folder_path, dataset_name, f"{seq_base_name}_qp{task_qp.qp_desc}"
                            )
                        if params.decoder_folder_path:
                            decoder_filename_format = params.decoder_filename_format or "{sequence_id}_qp{qp}{ext}"
                            decoder_filename = decoder_filename_format.format(
                                sequence_id=seq_base_name, qp=task_qp.qp_desc, ext=decoder_ext
                            )
                            task_qp.decoded_path = os.path.join(
                                params.decoder_folder_path, dataset_name, decoder_filename
                            )

                        qp_list.append(task_qp)

                    lr_model_data_filepath: dict[str, Optional[str]] = {
                        "transcript": None,
                        "landmarks": None,
                    }
                    if params.lr_model_data_path:
                        transcript_filepath = os.path.normpath(
                            os.path.join(params.lr_model_data_path, dataset_name, f"{seq_base_name}_transcript.txt")
                        )
                        if not os.path.exists(transcript_filepath):
                            logging.warning(f"Transcript file not found: {transcript_filepath}")
                            transcript_filepath = None
                        landmarks_filepath = os.path.normpath(
                            os.path.join(params.lr_model_data_path, dataset_name, f"{seq_base_name}_landmarks.npz")
                        )
                        if not os.path.exists(landmarks_filepath):
                            logging.warning(f"Landmark file not found: {landmarks_filepath}")
                            landmarks_filepath = None
                        lr_model_data_filepath["transcript"] = transcript_filepath
                        lr_model_data_filepath["landmarks"] = landmarks_filepath

                    mask_path_root: Optional[str] = None
                    if params.calc_psnr_roi and params.precomputed_masks_path:
                        mask_path_root = os.path.normpath(
                            os.path.join(
                                params.precomputed_masks_path,
                                dataset_desc["base_path"],
                                seq_base_name,
                            )
                        )

                    task = EncoderTestTask(
                        params,
                        dataset_name=dataset_name,
                        seq_name=seq_name,
                        src_type=src_type,
                        src_path=os.path.join(testset_root, dataset_desc["base_path"], seq_name),
                        src_width=seq_desc["width"],
                        src_height=seq_desc["height"],
                        n_frames=n_frames,
                        frame_rate=float(seq_desc.get("frame_rate", params.frame_rate or 30)),
                        intra_period=intra_period,
                        q_points=qp_list,
                        decoder_format=decoder_format,
                        lr_model_data_filepath=lr_model_data_filepath,
                        mask_path_root=mask_path_root,
                    )

                    yield task

    def _process_task_list(self, task_list: List[EncoderTestTask]):
        dataset = EncoderTestDataset(task_list)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            collate_fn=identity_collate_fn,
            num_workers=1,
            prefetch_factor=10,
        )
        frame_iterator = iter(data_loader)
        for task in task_list:
            i_frame_model = self._get_task_model(task, self.i_frame_model)
            if self.i_frame_model is self.p_frame_model:
                p_frame_model = i_frame_model
            else:
                p_frame_model = self._get_task_model(task, self.p_frame_model)
            with EncoderTestTaskProcessor(
                task,
                i_frame_model=i_frame_model,
                p_frame_model=p_frame_model,
                auxiliary_models=self.auxiliary_models,
                read_one_frame=lambda: next(frame_iterator),
            ) as processor:
                result = processor.run()

            yield task.dataset_name, task.seq_name, {qp.qp_desc: r for qp, r in zip(task.q_points, result)}

    @staticmethod
    def _get_task_model(task: EncoderTestTask, model: Union[nn.Module, Callable[[EncoderTestTask], nn.Module], None]):
        if model is not None and not isinstance(model, nn.Module):
            model = model(task)
        return model

    def _log_progress(self, n_processed, n_tasks, elapsed):
        rank_prefix = ""
        if self.rank >= 0:
            rank_prefix = f"rank {self.rank}: "

        logging.info(f"{rank_prefix}sequences: {n_processed}/{n_tasks} in {elapsed:.0f}s")

    @staticmethod
    def _build_results(result_list):
        output = dict()

        seen_qp = {}
        for dataset_name, seq_name, qp_map in result_list:
            for qp, result in qp_map.items():
                dataset_node = output.get(dataset_name)
                if dataset_node is None:
                    output[dataset_name] = dataset_node = dict()

                seq_node = dataset_node.get(seq_name)
                if seq_node is None:
                    dataset_node[seq_name] = seq_node = dict()

                # Deduplicate q-point entries to maintain number of points
                if qp in seq_node:
                    seen_qp.setdefault((dataset_name, seq_name, qp), 0)
                    seen_qp[(dataset_name, seq_name, qp)] += 1
                    qp_out = f"{qp}_{seen_qp[(dataset_name, seq_name, qp)]}"
                else:
                    qp_out = qp

                seq_node[qp_out] = result

        return output

    def run(self, testset_name, testset_desc_filename: str, *, testset_root: Optional[str] = None):
        task_list = list(self._build_task_list(testset_desc_filename, testset_root=testset_root))
        if len(task_list) == 0:
            raise ValueError(f"No test sequences found in testset {testset_name}")

        n_frames = sum(task.n_frames for task in task_list)
        n_videos = len(set((task.dataset_name, task.seq_name) for task in task_list))
        n_qpoints = self.n_qpoints

        def task_weight(task):
            return task.src_width * task.src_height * task.n_frames

        rate_unit = "bitrates" if self.params.bitrate_list is not None else "q-indices"
        logging.info(
            f"{testset_name}: Processing {n_frames} frames in {len(task_list)} sequences"
            f" from {n_videos} videos with {n_qpoints} {rate_unit}"
        )

        start_time = time.time()
        results = run_distributed(
            self.rank,
            task_list,
            list_processor=self._process_task_list,
            progress_callback=self._log_progress,
            weight_func=task_weight,
        )

        results = self._build_results(results)
        end_time = time.time()

        logging.info(f"{testset_name}: Total elapsed time: {(end_time - start_time) / 60:.1f} min")
        return results


def run_encoder_test(
    params: EncoderTestParams,
    *,
    testset_name: str,
    testset_desc_filename: str,
    rank=-1,
    i_frame_model: Union[nn.Module, Callable[[EncoderTestTask], nn.Module], None],
    p_frame_model: Union[nn.Module, Callable[[EncoderTestTask], nn.Module], None],
    auxiliary_models: Optional[AuxiliaryModels] = None,
    testset_root: Optional[str] = None,
    output_path: Optional[str] = None,
):
    tester = EncoderTester(
        rank, params, i_frame_model=i_frame_model, p_frame_model=p_frame_model, auxiliary_models=auxiliary_models
    )
    result = tester.run(testset_name, testset_desc_filename, testset_root=testset_root)

    if output_path is not None and rank <= 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wt", encoding="utf-8") as f:
            dump_json(result, f, indent=2, float_digits=6)

    return result
