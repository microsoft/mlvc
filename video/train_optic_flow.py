# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from typing import Dict, Any, Optional
import os

import torch
from torch import nn

from src.datasets.video_dataset import FastVideoFolder
from src.utils.app import BaseTrainVideoApp
from src.models.legacy.video_net import ME_SpynetOptimised
from src.models.block_mc import block_mc_func
from src.utils.encoder_tester import EncoderTestParams
from src.utils.flow_tester import run_flow_test

from src.utils.stream_helper import get_state_dict


def reconstruction_loss(warped_frame, target_frame):
    return torch.sqrt(torch.nn.functional.mse_loss(warped_frame, target_frame))


class SpyNetWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.optic_flow = ME_SpynetOptimised(3, 3, 0, 0, out_2x=False)

    def forward(self, frame1, frame2):
        flow = self.optic_flow(frame1, frame2)
        return flow


class TrainFlowApp(BaseTrainVideoApp):
    def __init__(self):
        super().__init__()

    @property
    def model_config(self):
        return self.config["model"]["optic_flow"]

    def load_model(self, model_state, pretrained: Optional[bool] = None):
        model = self.create_model()

        if model_state is None and (pretrained is None or pretrained):
            checkpoint_path = self.pretrained_checkpoint_path
            if pretrained and checkpoint_path is None:
                raise ValueError("no pretrained checkpoint path is specified")
            if checkpoint_path is not None:
                model_state = get_state_dict(checkpoint_path)
                logging.info(f"pretrained weights loaded from {checkpoint_path}")

        if model_state is None:
            self.initialize_model(model)
        else:
            model.load_state_dict(model_state)

        model = model.to(self.device)
        return model

    def create_model(self):
        model = SpyNetWrapper()

        # print number of parameters
        if self.rank <= 0:
            print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

        return model

    def create_train_dataset(self):
        config = self.get_config_by_path("train.dataset", expected_type=dict)

        patch_w, patch_h = config["patch_size"]
        return FastVideoFolder(
            self.resolve_path(config["path"]),
            patch_w,
            patch_h,
            crop_method="random",
            frame_selection="random",
            max_frame_distance=config["max_frame_distance"],
            random_flip=config["random_flip"],
            frame_num=config["n_frames"],
        )

    def prepare_for_epoch(self, epoch):
        super().prepare_for_epoch(epoch)

        config = self.get_config_by_path("train", expected_type=dict)
        lr = config["learning_rate"]
        # loss_type = config["loss_type"]

        assert self.optimizer is not None
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        if self.rank <= 0:
            self.metrics_writer.add_scalar("lr", lr, epoch)

    def train_step(self, epoch, step, batch):
        assert self.optimizer is not None
        self.optimizer.zero_grad()

        frame1, frame2 = batch[:, :3, :, :], batch[:, 3:, :, :]

        # Note that order of flow arguments seems mixed up here, but it might be fine -
        # model just learn the opposite movement vectors
        assert self.model is not None
        flow = self.model(frame2, frame1)
        warped_frame1 = block_mc_func(frame1, flow)
        loss = reconstruction_loss(warped_frame1, frame2)
        loss.backward()

        metrics = {
            "loss": loss.detach().mean(),
        }

        metric_names = {n: i for i, n in enumerate(metrics.keys())}
        metric_values = torch.stack(tuple(metrics.values()))
        metric_values = self.all_reduce(metric_values)

        self.add_step_metric("loss", metric_values[metric_names["loss"]])
        self.add_step_info(metric_names, metric_values)

        self.optimizer_step()

    def test_model(self, epoch, model: torch.nn.Module):
        model.eval()
        self.run_benchmark_test(epoch, flow_model=model)

    def run_benchmark_test(self, epoch, *, flow_model: nn.Module):
        benchmark_test = self.get_config_by_path("benchmark_test")
        if benchmark_test is None:
            return False

        for name, config in benchmark_test.items():
            if epoch is not None and not self.is_testset_benchmark_epoch(config, epoch):
                continue

            self.run_benchmark_on_testset(epoch, name, config, flow_model=flow_model)

    def compare_test_with_anchor(self, anchor_results_path, test_results_path, testset_name, epoch, save_dir):
        with open(anchor_results_path, "r") as f:
            anchor_results = json.load(f)

        with open(test_results_path, "r") as f:
            test_results = json.load(f)

        sum_anchor_psnr = 0
        sum_test_psnr = 0
        count_test = 0

        for dataset in test_results:
            for video in test_results[dataset]:
                assert video in anchor_results[dataset], f"video {video} not found in anchor results"
                # Keep qp_num for compatibility
                for qp_num in test_results[dataset][video]:
                    clip_results = test_results[dataset][video][qp_num]
                    anchor_clip_results = anchor_results[dataset][video][qp_num]
                    for psnr in clip_results["psnr"]:
                        sum_test_psnr += psnr
                        count_test += 1

                    for psnr in anchor_clip_results["psnr"]:
                        sum_anchor_psnr += psnr

        mean_psnr = sum_test_psnr / count_test
        mean_anchor_psnr = sum_anchor_psnr / count_test
        psnr_ratio = mean_psnr / mean_anchor_psnr

        if (self.rank <= 0) and (self.metrics_writer is not None):
            self.metrics_writer.add_metrics(
                {
                    f"{testset_name}_psnr_ratio": psnr_ratio,
                    f"{testset_name}_mean_psnr": mean_psnr,
                },
                step=epoch,
            )

        out_file_path = os.path.join(save_dir, f"{testset_name}_psnr_ratio.json")
        with open(out_file_path, "a") as f:
            line = (
                f"epoch: {epoch}, "
                f"psnr_ratio: {psnr_ratio:.2f}, "
                f"mean_psnr: {mean_psnr:.2f}, "
                f"mean_anchor_psnr: {mean_anchor_psnr:.2f}\n"
            )
            f.write(line)

    def run_benchmark_on_testset(self, epoch, testset_name, config: Dict[str, Any], flow_model: nn.Module):
        i_frame_q_index_list = [21]
        p_frame_q_index_list = i_frame_q_index_list

        params = EncoderTestParams(
            is_yuv420=self.is_yuv420,
            i_frame_q_index_list=i_frame_q_index_list,
            p_frame_q_index_list=p_frame_q_index_list,
            intra_period=config["intra_period"],
            max_n_frames=config.get("max_n_frames"),
            decoder_folder_path=self.resolve_path(config.get("decoder_folder"), default_source="save_dir"),
            calc_ssim=config.get("calc_ssim", False),
            verbose=config.get("verbose", 0),
            verbose_json=config.get("verbose_json", False),
        )

        if epoch is not None:
            output_json_path = f"bmk_test_epo_{epoch}.json"
        else:
            output_json_path = "bmk_test.json"
        output_json_path = os.path.join(self.save_dir, output_json_path)

        testset_desc_filename = self.resolve_path(config["config"], default_source=".")
        results = run_flow_test(
            params,
            rank=self.rank,
            testset_name=testset_name,
            testset_desc_filename=testset_desc_filename,
            testset_root=self.resolve_path(config.get("data_path")),
            flow_model=flow_model,
            output_path=output_json_path,
        )

        if self.rank <= 0:
            self.log_benchmark_metrics(results, testset_name=testset_name, config=config, epoch=epoch)

            anchor_path = config.get("anchor")
            if anchor_path:
                # TODO loading test results sometimes fails?
                try:
                    self.compare_test_with_anchor(
                        epoch=epoch or 0,
                        anchor_results_path=self.resolve_path(anchor_path, default_source="checkpoints_mount"),
                        test_results_path=output_json_path,
                        testset_name=testset_name,
                        save_dir=self.save_dir,
                    )
                except json.decoder.JSONDecodeError:
                    logging.warning("Failed to compare test results with anchor - JSONDecodeError!")


if __name__ == "__main__":
    TrainFlowApp().main()
