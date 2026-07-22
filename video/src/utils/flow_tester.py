# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import time
from typing import Optional, Callable, List

import torch
import torch.nn as nn
import torch.utils.data

from .common import dump_json
from .stream_helper import get_padding_size
from .video_writer import PNGWriter, YUVWriter
from ..metrics.msssim import MS_SSIM
from ..transforms.functional import yuv_420_to_444
from ..models.block_mc import block_mc_func
from .encoder_tester import (
    EncoderTestParams,
    EncoderTestTask,
    EncoderTestDataset,
    EncoderTester,
    EncoderTestTaskProcessor,
    identity_collate_fn,
    create_frame_reader,
)


class FlowTestTaskProcessor(EncoderTestTaskProcessor):
    def __init__(
        self,
        task: EncoderTestTask,
        flow_model: nn.Module,
        read_one_frame: Optional[Callable] = None,
    ):
        self.task = task

        self.flow_model = flow_model

        self.device = next(iter(self.flow_model.parameters())).device

        if read_one_frame is None:
            read_one_frame = create_frame_reader(task)

        self.read_one_frame = read_one_frame

        rec_frame_writer = None
        if task.decoded_path:
            os.makedirs(task.decoded_path, exist_ok=True)

            if task.src_type == "png":
                rec_frame_writer = PNGWriter(task.decoded_path, task.src_width, task.src_height)
            elif task.src_type == "yuv420":
                rec_frame_writer = YUVWriter(
                    os.path.join(task.decoded_path, task.decoded_seq_name),
                    task.src_width,
                    task.src_height,
                )
            else:
                raise ValueError("Unknown source format")

        self.rec_frame_writer = rec_frame_writer

        if task.params.is_yuv420:
            self._process_rec = self._process_yuv420_rec
        else:
            self._process_rec = self._process_rgb_rec

        if task.params.calc_ssim:
            self._msssim = MS_SSIM(channels=1 if task.params.is_yuv420 else 3, data_range=1)
            self._msssim.to(self.device)
        else:
            self._msssim = None

        self.start_time = time.time()

        self.frame_pixel_num = 0
        self.frame_types = []
        self.frame_stats = {}

    def _process_frame(self, frame_idx, dpb):
        frame_start_time = time.time()

        task = self.task
        params = task.params
        if frame_idx > 0 and params.frame_step > 1:
            for _ in range(params.frame_step - 1):
                self.read_one_frame()

        image = self.read_one_frame()
        if params.is_yuv420:
            y, uv = image
            u, v = self._to_tensor(uv).chunk(2, dim=1)
            image = self._to_tensor(y), u, v
            x = yuv_420_to_444(image, mode="nearest")
        else:
            image = self._to_tensor(image).unsqueeze(0)
            x = image

        if params.float16:
            x = x.to(torch.float16)

        height = x.shape[2]
        width = x.shape[3]

        frame_pixel_num = width * height
        if self.frame_pixel_num == 0:
            self.frame_pixel_num = frame_pixel_num
        elif self.frame_pixel_num != frame_pixel_num:
            raise ValueError("Video frame size changed in the middle of clip")

        # pad if necessary
        padding = get_padding_size(height, width, 16)
        x = nn.functional.pad(x, padding, mode="replicate")

        if frame_idx % task.intra_period == 0:
            # Skip I-frame for flow

            dpb = dict(
                ref_frame=x,
                ref_feature=None,
                ref_mv_feature=None,
                ref_y=None,
                ref_mv_y=None,
            )
            self.frame_types.append(0)
        else:
            assert dpb is not None
            try:
                flow = self.flow_model(x, dpb["ref_frame"])
            except ValueError:
                logging.exception(f"flow inference failed for {self.task.src_path} - {task.rank_idx} - {frame_idx}")
                dpb = dict(
                    dpb=dict(
                        ref_frame=torch.zeros_like(x),
                        ref_feature=None,
                        ref_mv_feature=None,
                        ref_y=None,
                        ref_mv_y=None,
                    ),
                    bit=0,
                )

            self.frame_types.append(1)

            x_hat = block_mc_func(dpb["ref_frame"], flow)
            x_hat = nn.functional.pad(x_hat, tuple(-x for x in padding))

            psnr, msssim = self._process_rec(x_hat, image)
            frame_end_time = time.time()

            if params.verbose >= 2:
                extra_stat = ""
                if msssim is not None:
                    extra_stat = f", MS-SSIM: {msssim:.4f}"
                logging.info(
                    f"frame {frame_idx}, {frame_end_time - frame_start_time:.3f} seconds,PSNR: {psnr:.4f}{extra_stat}"
                )

            # Use actual image as reference for next frame
            dpb = dict(
                ref_frame=x,
                ref_feature=None,
                ref_mv_feature=None,
                ref_y=None,
                ref_mv_y=None,
            )

        return dpb

    def _generate_result(self):
        result = dict(
            ds_name=self.task.dataset_name,
            video_path=self.task.seq_name,
            frame_pixel_num=self.frame_pixel_num,
            n_frames=self.task.n_frames,
        )

        result["psnr"] = self.frame_stats["psnr"]
        result["msssim"] = self.frame_stats["msssim"]

        assert len(result["psnr"]) == result["n_frames"] - 1

        result["mse"] = [(255**2) / (10 ** (psnr / 10)) for psnr in result["psnr"]]

        result["ave_mse"] = sum(result["mse"]) / len(result["mse"])

        if self.task.params.verbose_json:
            result["frame_type"] = self.frame_types.copy()
            for name, stat_list in self.frame_stats.items():
                result[f"frame_{name}"] = stat_list.copy()

        result["i_frame_q_index"] = self.task.i_frame_q_index
        result["p_frame_q_index"] = self.task.p_frame_q_index
        result["decoded_seq_name"] = self.task.decoded_seq_name
        result["test_time"] = time.time() - self.start_time

        return result


class FlowTester(EncoderTester):
    def __init__(self, rank: int, params: EncoderTestParams, *, flow_model: nn.Module):
        super().__init__(rank, params, i_frame_model=nn.Module(), p_frame_model=nn.Module())

        self.flow_model = flow_model

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
            result = FlowTestTaskProcessor(
                task, flow_model=self.flow_model, read_one_frame=lambda: next(frame_iterator)
            ).run()

            yield task.dataset_name, task.seq_name, task.qp, result


def run_flow_test(
    params: EncoderTestParams,
    *,
    testset_name: str,
    testset_desc_filename: str,
    rank=-1,
    flow_model: nn.Module = nn.Module(),
    testset_root: Optional[str] = None,
    output_path: Optional[str] = None,
):
    tester = FlowTester(rank, params, flow_model=flow_model)
    result = tester.run(testset_name, testset_desc_filename, testset_root=testset_root)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wt", encoding="utf-8") as f:
            dump_json(result, f, indent=2, float_digits=6)

    return result
