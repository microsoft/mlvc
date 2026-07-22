# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess
from typing import Optional

import numpy as np
import scipy.ndimage
import torch.utils.data

from src.utils.os import set_pipe_buffer_size


class FfmpegIterator:
    def __init__(self, video_path, *, width, height, dst_format: str = "444", ffmpeg_error_mode: Optional[str] = None):
        if dst_format not in ("444", "420"):
            raise ValueError(f"Unsupported decoded video format {dst_format}")

        # fmt: off
        args = [
            "ffmpeg",
            "-v", "quiet",
            "-threads", "1",
            "-i", video_path,
            "-vsync", "0",
            "-c:v", "rawvideo",
            "-f", "rawvideo",
            "-pix_fmt", "yuv420p",
            "-"
        ]
        # fmt: on
        self.video_path = video_path
        self.ffmpeg = subprocess.Popen(args, stdout=subprocess.PIPE)
        self.width = width
        self.height = height
        self.dst_format = dst_format
        self.ffmpeg_error_mode = ffmpeg_error_mode or "exception"
        self.n_frames = 0

        buffer_size = width * height * 3 // 2
        set_pipe_buffer_size(self.ffmpeg.stdout, buffer_size)

    def _read_frame(self, fd):
        y_size = self.width * self.height

        uv_width = (self.width + 1) // 2
        uv_height = (self.height + 1) // 2
        uv_size = 2 * uv_width * uv_height

        y = fd.read(y_size)
        if not len(y):
            return None

        if len(y) != y_size:
            raise EOFError(f"{self.video_path}: unexpected end of ffmpeg output")

        uv = fd.read(uv_size)
        if len(uv) != uv_size:
            raise EOFError(f"{self.video_path}: unexpected end of ffmpeg output")

        self.n_frames += 1

        y = np.frombuffer(y, dtype=np.uint8).reshape(1, self.height, self.width)
        uv = np.frombuffer(uv, dtype=np.uint8).reshape(2, uv_height, uv_width)

        y = y.astype(np.float32) / np.iinfo(y.dtype).max
        uv = uv.astype(np.float32) / np.iinfo(uv.dtype).max

        if self.dst_format == "420":
            return y, uv

        uv = np.asarray(scipy.ndimage.zoom(uv, (1, 2, 2), order=0))
        if uv.shape[-2] != y.shape[-2] or uv.shape[-1] != y.shape[-1]:
            uv = uv[..., : y.shape[-2], : y.shape[-1]]

        yuv = np.concatenate((y, uv), axis=0)
        return yuv

    def _process_eof(self, ffmpeg, check):
        try:
            ffmpeg.stdout.close()
            try:
                rc = ffmpeg.wait(60)
            except subprocess.TimeoutExpired:
                raise IOError(f"{self.video_path}: ffmpeg failed to terminate cleanly")

            if check and rc != 0:
                if self.ffmpeg_error_mode == "ignore":
                    pass
                else:
                    error = f"{self.video_path}: ffmpeg exit code {rc}, after {self.n_frames} frames"
                    if self.ffmpeg_error_mode == "warn" or (
                        self.ffmpeg_error_mode == "warn_if_nonempty" and self.n_frames > 0
                    ):
                        import logging

                        logging.warning(error)
                    else:
                        raise IOError(error)
        finally:
            ffmpeg.kill()
            pass

    def __iter__(self):
        return self

    def __next__(self):
        ffmpeg = self.ffmpeg
        if ffmpeg is None:
            raise StopIteration()

        yuv = self._read_frame(ffmpeg.stdout)
        if yuv is None:
            self.ffmpeg = None
            self._process_eof(ffmpeg, True)
            raise StopIteration()

        return yuv

    def close(self):
        ffmpeg = self.ffmpeg
        if ffmpeg is not None:
            self.ffmpeg = None
            self._process_eof(ffmpeg, False)


class FfmpegDataset(torch.utils.data.IterableDataset):
    def __init__(self, video_path, width, height, dst_format: str = "444", ffmpeg_error_mode: Optional[str] = None):
        super().__init__()
        self.video_path = video_path
        self.width = width
        self.height = height
        self.dst_format = dst_format
        self.ffmpeg_error_mode = ffmpeg_error_mode

    def __iter__(self) -> FfmpegIterator:
        return FfmpegIterator(
            self.video_path,
            width=self.width,
            height=self.height,
            dst_format=self.dst_format,
            ffmpeg_error_mode=self.ffmpeg_error_mode,
        )
