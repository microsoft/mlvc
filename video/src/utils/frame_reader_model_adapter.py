# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import re
import subprocess
from typing import Optional, Dict, Any

import torch

from src.datasets.ffmpeg_dataset import FfmpegIterator
from src.transforms.functional import yuv_420_to_444
from .encoder_tester import EncoderTestTask
from .video_reader import YUVReader

__all__ = ["FrameReaderModelAdapter", "FrameReaderModelFactory"]


class FrameReaderModelAdapter(torch.nn.Module):
    def __init__(self, streamname: str, *, width: int, height: int, stream_format: str, bits: Optional[torch.Tensor]):
        super().__init__()

        if stream_format != "yuv":
            s_width, s_height, n_frames, s_bits = self._get_stream_info(streamname, bits is None)
            if width != s_width or height != s_height:
                raise ValueError(f"{streamname}: frame size mismatch: {s_width}x{s_height}, expected {width}x{height}")
            if bits is None:
                bits = s_bits
            assert bits is not None
            if n_frames > len(bits):
                raise ValueError(f"{streamname}: number of frames does not match: {n_frames} vs {len(bits)}")

        if bits is None:
            raise ValueError("No bits information is provided")

        if stream_format != "yuv":
            self.reader = FfmpegIterator(streamname, width=width, height=height, dst_format="420").__next__
        elif stream_format == "yuv":
            self.reader = YUVReader(streamname, width=width, height=height).read_one_frame
        else:
            raise ValueError(f"Unsupported stream format: {stream_format}")

        self.frame_index = 0
        self.bits = bits

        # dummy parameters to make it look like a model
        self.dummy = torch.nn.Parameter(torch.tensor(0.0))
        self.frame_index_map = (0,)

    @staticmethod
    def _get_stream_info(streamname, return_bits: bool):
        cmd = [
            "ffprobe",
            "-print_format",
            "json",
            "-strict",
            "-2",
            "-select_streams",
            "v",
            "-show_streams",
        ]
        if return_bits:
            cmd.append("-show_frames")
        cmd.append(streamname)

        rc = subprocess.run(cmd, capture_output=True)
        if rc.returncode != 0:
            if rc.stderr is not None and len(rc.stderr) > 0:
                error = ":\n" + rc.stderr.decode(encoding="utf-8", errors="ignore")
            else:
                error = ""

            raise ValueError(f"{streamname}: error running ffprobe, rc={rc.returncode}{error}")

        def to_int(v):
            if isinstance(v, str):
                v = int(v)
            return v

        try:
            info = json.loads(rc.stdout.decode(encoding="utf-8"))
            stream_info = info["streams"][0]
            width = to_int(stream_info["width"])
            height = to_int(stream_info["height"])
            n_frames = to_int(stream_info["nb_read_frames" if return_bits else "nb_frames"])

            bits = None
            if return_bits:
                stream_index = stream_info["index"]
                bits = list()
                for frame in info["frames"]:
                    if frame["stream_index"] != stream_index:
                        continue

                    bits.append(8 * float(frame["pkt_size"]))

                bits = torch.tensor(bits, dtype=torch.float64)

            return width, height, n_frames, bits
        except Exception as e:
            raise ValueError(f"{streamname}: invalid ffprobe output") from e

    def forward(self, x, **kwargs):
        _ = kwargs

        frame_index = self.frame_index
        result = self.reader()
        assert result is not None
        y, uv = result
        self.frame_index = frame_index + 1

        y = torch.from_numpy(y).to(x.device)
        uv = torch.from_numpy(uv).to(x.device)

        y = y[None, ...]
        u, v = uv[None, ...].chunk(2, 1)
        yuv = yuv_420_to_444((y, u, v), mode="nearest")
        yuv = torch.nn.functional.pad(yuv, (0, x.shape[-1] - yuv.shape[-1], 0, x.shape[-2] - yuv.shape[-2]))

        return dict(bits=self.bits[frame_index][None, ...], x_hat=yuv, dpb=dict())


class FrameReaderModelFactory:
    def __init__(
        self,
        *,
        clip_folder,
        clip_filename_format: Optional[str] = None,
        clip_format: str,
        metrics: Optional[Dict[str, Any]],
        seq_id_regexp: Optional[str] = None,
        device: torch.device,
    ):
        self.clip_folder = clip_folder
        self.clip_filename_format = clip_filename_format or "{sequence_id}_qp{qp}{ext}"
        self.clip_format = clip_format
        self.seq_id_regexp = re.compile(seq_id_regexp or "^(?:([^.]*)|(.*)[.][^.]*)$")
        self.device = device

        if metrics is not None:
            metrics = {n: self._process_dataset(v) for n, v in metrics.items()}
        self.metrics = metrics

    def _normalize_seq_id(self, seq_id):
        m = self.seq_id_regexp.match(seq_id)
        if not m:
            raise ValueError(f"Invalid sequence id: {seq_id}")

        return "".join(x for x in m.groups() if x) or m.group(0)

    def _process_dataset(self, metrics: Dict[str, Any]):
        return {self._normalize_seq_id(n): self._process_seq(n, v) for n, v in metrics.items()}

    def _process_seq(self, seq_id, metrics: Dict[str, Any]):
        return dict(seq_id=seq_id, bits=dict(self._process_qp(n, v) for n, v in metrics.items()))

    @staticmethod
    def _process_qp(qp, metrics: Dict[str, Any]):
        p_frame_q_index = metrics.get("p_frame_q_index")
        if isinstance(p_frame_q_index, int) and p_frame_q_index == metrics.get("i_frame_q_index"):
            # normalize qp description
            qp = str(p_frame_q_index)

        n_pixels = metrics["frame_pixel_num"]
        return qp, torch.tensor(metrics["frame_bpp"], dtype=torch.float64) * n_pixels

    def __call__(self, task: EncoderTestTask):
        assert len(task.q_points) == 1
        qp_desc = task.q_points[0].qp_desc

        seq_id = self._normalize_seq_id(task.seq_name)
        if self.metrics is not None:
            seq_info = self.metrics[task.dataset_name][seq_id]
            seq_id = os.path.splitext(seq_info["seq_id"])[0]
            bits = seq_info["bits"][qp_desc]
        else:
            bits = None

        target_id = os.path.splitext(task.seq_name)[0]
        streamname = self.clip_filename_format.format(
            sequence_id=seq_id, target_id=target_id, qp=qp_desc, ext="." + self.clip_format
        )
        streamname = os.path.join(self.clip_folder, task.dataset_name, streamname)
        return FrameReaderModelAdapter(
            streamname, width=task.src_width, height=task.src_height, stream_format=self.clip_format, bits=bits
        ).to(self.device)
