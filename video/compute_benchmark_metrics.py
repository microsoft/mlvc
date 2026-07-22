# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
from typing import Dict, Any

import torch

from src.utils.app import BaseApp
from src.utils.encoder_tester import EncoderTestParams, run_encoder_test
from src.utils.frame_reader_model_adapter import FrameReaderModelFactory
from src.utils.common import AuxiliaryModels


class ComputeBenchmarkMetricApp(BaseApp):
    def __init__(self):
        super().__init__()
        self._device = None
        self._auxiliary_models = None

    @property
    def device(self):
        device = self._device
        if device is None:
            is_cuda_available = torch.cuda.is_available()
            logging.info(f"cuda: is_available() = {is_cuda_available}, device_count = {torch.cuda.device_count()}")
            self._device = device = torch.device("cuda" if is_cuda_available else "cpu")
        return device

    @property
    def auxiliary_models(self):
        if self._auxiliary_models is None:
            auxiliary = self.get_config_by_path("model.auxiliary", default=None)
            if auxiliary is not None:
                pretrained_path = self.resolve_path(auxiliary["path"], default_source="checkpoints_mount")
                auxiliary_model_config = auxiliary.get("config", {})
            else:
                pretrained_path = None
                auxiliary_model_config = {}
            self._auxiliary_models = AuxiliaryModels(pretrained_path, self.device, auxiliary_model_config)
        return self._auxiliary_models

    def run(self):
        config = self.get_config_by_path("benchmark_metrics", expected_type=dict)
        for testset_name, testset_config in config.items():
            self.collect_metrics_on_testset(testset_name, testset_config)

    def collect_metrics_on_testset(self, testset_name, config: Dict[str, Any]):
        intra_period = config.get("intra_period")

        use_i_frame_model = config.get("use_i_frame_model", None)
        if use_i_frame_model is None:
            use_i_frame_model = config.get("use_i_frame", True)

        params = EncoderTestParams(
            is_yuv420=config.get("yuv420", True),
            intra_period=intra_period,
            max_n_frames=config.get("max_n_frames"),
            calc_psnr=config.get("calc_psnr", True),
            calc_psnr_rgb=config.get("calc_psnr_rgb", False),
            calc_ssim=config.get("calc_ssim", False),
            calc_ssim_rgb=config.get("calc_ssim_rgb", False),
            calc_psnr_roi=config.get("calc_psnr_roi", False),
            calc_vif=config.get("calc_vif", False),
            calc_lpips=config.get("calc_lpips", False),
            calc_deqa_score_rgb=config.get("calc_deqa_score_rgb", False),
            calc_bits_estimates=False,
            use_i_frame_model=use_i_frame_model,
            decoder_folder_path=self.resolve_path(config.get("decoder_folder"), default_source="save_dir"),
            decoder_filename_format=config.get("decoder_filename_format"),
            decoder_format=config.get("decoder_format"),
            decoder_compression_options=config.get("decoder_compression_options"),
            verbose=config.get("verbose", 0),
            verbose_json=config.get("verbose_json", False),
            precomputed_masks_path=self.resolve_path(config.get("precomputed_masks_path"), default_source="data_mount"),
        )

        params.i_frame_q_index_list, params.p_frame_q_index_list = self._get_q_index_lists(config)

        clip_metrics = self.resolve_path(config.get("clip_metrics"), default_source="checkpoints_mount")
        if not clip_metrics:
            clip_metrics = None
        else:
            with open(clip_metrics, "rb") as f:
                clip_metrics = json.load(f)

        model = FrameReaderModelFactory(
            clip_folder=self.resolve_path(config["clip_folder"], default_source="checkpoints_mount"),
            clip_filename_format=config.get("clip_filename_format"),
            clip_format=config.get("clip_format") or "mp4",
            metrics=clip_metrics,
            seq_id_regexp=config.get("seq_id_regexp", ""),
            device=self.device,
        )

        metrics = self._run_encoder_test(testset_name, config=config, params=params, model=model)

        if params.decoder_folder_path:
            self._compute_decoder_metrics(config=config, encoder_params=params, encoder_metrics=metrics)

    @staticmethod
    def _get_q_index_lists(config):
        i_frame_q_index_list = config.get("i_frame_q_index_list")
        p_frame_q_index_list = config.get("p_frame_q_index_list")

        if i_frame_q_index_list is None and p_frame_q_index_list is None:
            raise ValueError("No q-points (i_frame_q_index_list and/or p_frame_q_index_list) are specified for testing")

        if i_frame_q_index_list is not None:
            i_frame_q_index_list = tuple(i_frame_q_index_list)
        if p_frame_q_index_list is not None:
            p_frame_q_index_list = tuple(p_frame_q_index_list)

        if i_frame_q_index_list is None:
            i_frame_q_index_list = p_frame_q_index_list
        elif p_frame_q_index_list is None:
            p_frame_q_index_list = i_frame_q_index_list

        return i_frame_q_index_list, p_frame_q_index_list

    def _compute_decoder_metrics(self, *, config: Dict[str, Any], encoder_params: EncoderTestParams, encoder_metrics):
        decoder_config = config.get("decoder_metrics")
        if decoder_config is None:
            return
        if not isinstance(decoder_config, dict):
            raise ValueError("decoder_metrics must be an YAML object")

        testset_name = decoder_config.get("name")
        if not testset_name:
            return

        params = EncoderTestParams(
            is_yuv420=encoder_params.is_yuv420,
            intra_period=encoder_params.intra_period,
            reset_period=encoder_params.reset_period,
            max_n_frames=encoder_params.max_n_frames,
            calc_psnr_rgb=decoder_config.get("calc_psnr_rgb", encoder_params.calc_psnr_rgb),
            calc_ssim=decoder_config.get("calc_ssim", encoder_params.calc_ssim),
            calc_ssim_rgb=decoder_config.get("calc_ssim_rgb", encoder_params.calc_ssim_rgb),
            calc_vif=decoder_config.get("calc_vif", encoder_params.calc_vif),
            frame_rate=decoder_config.get("frame_rate", encoder_params.frame_rate),
            use_i_frame_model=encoder_params.use_i_frame_model,
            i_frame_q_index_list=encoder_params.i_frame_q_index_list,
            p_frame_q_index_list=encoder_params.p_frame_q_index_list,
            verbose=decoder_config.get("verbose", encoder_params.verbose),
            verbose_json=decoder_config.get("verbose_json", encoder_params.verbose_json),
        )

        model = FrameReaderModelFactory(
            clip_folder=encoder_params.decoder_folder_path,
            clip_filename_format=encoder_params.decoder_filename_format,
            clip_format=encoder_params.decoder_format or "yuv",
            seq_id_regexp=decoder_config.get("seq_id_regexp"),
            metrics=encoder_metrics,
            device=self.device,
        )

        self._run_encoder_test(testset_name, config=decoder_config, params=params, model=model)

    def _run_encoder_test(self, testset_name: str, *, config: Dict[str, Any], params: EncoderTestParams, model):
        testset_desc_filename = self.resolve_path(config["config"], default_source=".")

        output_json_path = f"metrics_{testset_name}.json"
        output_json_path = os.path.join(self.save_dir, output_json_path)

        return run_encoder_test(
            params,
            rank=self.rank,
            testset_name=testset_name,
            testset_desc_filename=testset_desc_filename,
            testset_root=self.resolve_path(config.get("data_path")),
            i_frame_model=model,
            p_frame_model=model,
            auxiliary_models=self.auxiliary_models,
            output_path=output_json_path,
        )


if __name__ == "__main__":
    ComputeBenchmarkMetricApp().main()
