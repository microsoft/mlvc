# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import errno
import itertools
import logging
import os
import queue
import subprocess
import threading
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from .os import set_pipe_buffer_size
from ..transforms.functional import (
    ycbcr420_to_rgb,
    rgb_to_ycbcr420,
    ycbcr444_to_rgb,
    rgb_to_ycbcr444,
    ycbcr420_to_444,
    ycbcr444_to_420,
)


class VideoWriter:
    def __init__(self, dst_path, width, height):
        self.dst_path = dst_path
        self.width = width
        self.height = height

    def write_one_frame(self, rgb=None, y=None, uv=None, src_format="rgb"):
        """
        y is 1xhxw Y float numpy array, in the range of [0, 1]
        uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
        rgb is 3xhxw float numpy array, in the range of [0, 1]
        """
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()
        else:
            # noinspection PyBroadException
            try:
                self.close()
            except BaseException:
                logging.error(f"Exception while closing {self.dst_path}", exc_info=True)


class FrameFormatAdapter:
    def __init__(self, dst_format, bit_depth=8):
        if dst_format not in ("420", "444", "rgb"):
            raise ValueError(f"Unspported dst format: {dst_format}")
        self.dst_format = dst_format

        if 8 < bit_depth <= 16:
            self.dtype = np.uint16
            self.max_val = (1 << bit_depth) - 1
        elif bit_depth == 8:
            self.dtype = np.uint8
            self.max_val = 255
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")

    def __call__(self, *, src_format, rgb=None, y=None, uv=None):
        data = self._convert_layout(src_format=src_format, rgb=rgb, y=y, uv=uv)
        if isinstance(data, tuple):
            y, uv = data
            return self._quantize(y), self._quantize(uv)
        else:
            return self._quantize(data)

    def _convert_layout(self, *, src_format, rgb, y, uv):
        if src_format in ("420", "444"):
            if rgb is not None:
                raise ValueError("no rgb components are allowed for yuv format")

            if src_format == "420":
                if self.dst_format == "420":
                    return y, uv
                elif self.dst_format == "444":
                    return ycbcr420_to_444(y, uv, separate=True)
                else:
                    assert self.dst_format == "rgb"
                    return ycbcr420_to_rgb(y, uv)
            else:
                assert src_format == "444"
                if self.dst_format == "420":
                    return ycbcr444_to_420(y, uv)
                elif self.dst_format == "444":
                    return y, uv
                else:
                    assert self.dst_format == "rgb"
                    return ycbcr444_to_rgb(y, uv)
        elif src_format == "rgb":
            if y is not None or uv is not None:
                raise ValueError("no yuv components are allowed for rgb format")

            if self.dst_format == "420":
                return rgb_to_ycbcr420(rgb)
            elif self.dst_format == "444":
                return rgb_to_ycbcr444(rgb)
            else:
                assert self.dst_format == "rgb"
                return rgb
        else:
            raise ValueError(f"Unsupported src format: {src_format}")

    def _quantize(self, x: np.ndarray):
        return np.clip(np.rint(x * self.max_val), 0, self.max_val).astype(self.dtype)


class PNGWriter(VideoWriter):
    def __init__(self, dst_path, width, height):
        super().__init__(dst_path, width, height)
        self.format_adapter = FrameFormatAdapter("rgb")
        self.padding = 5
        self.current_frame_index = 1
        os.makedirs(dst_path, exist_ok=True)

    def write_one_frame(self, rgb=None, y=None, uv=None, src_format="rgb"):
        result = self.format_adapter(src_format=src_format, rgb=rgb, y=y, uv=uv)
        assert not isinstance(result, tuple)
        rgb = result
        rgb = rgb.transpose(1, 2, 0)

        png_path = os.path.join(self.dst_path, f"im{str(self.current_frame_index).zfill(self.padding)}.png")
        Image.fromarray(rgb).save(png_path)

        self.current_frame_index += 1

    def close(self):
        self.current_frame_index = 1


class RGBWriter(VideoWriter):
    def __init__(self, dst_path, width, height, dst_format="rgb", bit_depth=8):
        super().__init__(dst_path, width, height)

        if dst_format != "rgb":
            raise ValueError(f"Unsupported dst format: {dst_format}")

        self.format_adapter = FrameFormatAdapter(dst_format, bit_depth)
        self.file = open(dst_path, "wb")

    def write_one_frame(self, rgb=None, y=None, uv=None, src_format="rgb"):
        result = self.format_adapter(src_format=src_format, rgb=rgb, y=y, uv=uv)
        assert not isinstance(result, tuple)
        rgb = result
        self.file.write(rgb.tobytes())

    def close(self):
        self.file.close()


class YUVWriter(VideoWriter):
    def __init__(self, dst_path, width, height, dst_format="420", bit_depth=8):
        super().__init__(dst_path, width, height)

        if dst_format not in ("420", "444"):
            raise ValueError(f"Unsupported dst format: {dst_format}")

        self.format_adapter = FrameFormatAdapter(dst_format, bit_depth)
        self.file = open(dst_path, "wb")

    def write_one_frame(self, rgb=None, y=None, uv=None, src_format="420"):
        result = self.format_adapter(src_format=src_format, rgb=rgb, y=y, uv=uv)
        assert isinstance(result, tuple)
        y, uv = result
        self.file.write(y.tobytes())
        self.file.write(uv.tobytes())

    def close(self):
        self.file.close()


class _FFMpegWriterLoop:
    def __init__(self, filename, args, buffer_size):
        self.filename = filename
        self.args = args
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=5)
        self.closed = False
        self.exception = None
        self.rethrow_count = itertools.count()

    def write(self, *chunks):
        if self.closed:
            self.rethrow()
        else:
            self.queue.put(chunks)

    def close(self):
        if not self.closed:
            self.closed = True
            self.queue.put(None)

    def rethrow(self):
        exception = self.exception
        if exception is not None and next(self.rethrow_count) == 0:
            self.exception = None
            raise IOError(f"Error writing {self.filename} with ffmpeg") from exception

    def run(self):
        try:
            ffmpeg = subprocess.Popen(self.args, bufsize=0, stdin=subprocess.PIPE)
            done = False
            try:
                self._write_loop(ffmpeg.stdin)
                done = True
            except Exception as e:
                if isinstance(e, OSError) and e.errno in (errno.EPIPE, errno.EINVAL):
                    # UNIX: EPIPE means that process has died
                    # Windows: returns EINVAL instead
                    pass
                else:
                    # noinspection PyBroadException
                    try:
                        self._wait(ffmpeg, check=False)
                    except Exception:
                        logging.error(
                            f"{self.filename}: exception while waiting for ffmpeg to terminate", exc_info=True
                        )

                    raise

            self._wait(ffmpeg, check=True)
            if not done:
                raise IOError("ffmpeg process terminated before receiving all the data")
        except Exception as e:
            self.exception = e
        finally:
            self.closed = True

            while not (self.queue.empty()):
                try:
                    self.queue.get(block=False)
                except queue.Empty:
                    break

    def _write_loop(self, file):
        set_pipe_buffer_size(file, self.buffer_size)
        while True:
            msg = self.queue.get()
            if msg is None:
                break

            for data in msg:
                file.write(data)

        file.close()

    def _wait(self, ffmpeg: subprocess.Popen, check: bool):
        try:
            # make sure pipe is closed before waiting for process to terminate
            try:
                if ffmpeg.stdin is not None:
                    ffmpeg.stdin.close()
            except IOError:
                pass

            rc = ffmpeg.wait(60)
        except Exception as e:
            # noinspection PyBroadException
            try:
                ffmpeg.kill()
                ffmpeg.wait(60)
            except BaseException:
                if not check and isinstance(e, subprocess.TimeoutExpired):
                    # kill exception has preference over wait timeout
                    raise

                logging.error(f"{self.filename}: exception while sending ffmpeg death signal", exc_info=True)

            raise

        if check and rc != 0:
            raise subprocess.CalledProcessError(rc, self.args[0])


class FFMpegWriter(VideoWriter):
    def __init__(self, dst_path, *, width, height, fps, ffmpeg_options: Optional[Sequence[str]] = None):
        super().__init__(dst_path, width, height)

        self.format_adapter = FrameFormatAdapter("420")
        buffer_size = width * height * 3 * np.dtype(self.format_adapter.dtype).itemsize // 2

        # fmt: off
        args = [
            "ffmpeg",
            "-v", "quiet",
            "-f", "rawvideo",
            "-pix_fmt", "yuv420p",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-y",
        ]
        if ffmpeg_options is None:
            ffmpeg_options = [
                "-threads", "1",
                "-preset", "veryslow",
                "-keyint_min", "2",
                "-g", "30",
                "-sc_threshold", "0",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "17",
            ]
        # fmt: on
        args.extend(ffmpeg_options)
        args.append(dst_path)

        self._writer_loop = _FFMpegWriterLoop(dst_path, args, buffer_size)
        self._writer_thread = threading.Thread(
            name=f"ffmpeg to {os.path.basename(dst_path)}",
            target=self._writer_loop.run,
            daemon=True,
        )
        self._writer_thread.start()

    def __del__(self):
        self._writer_loop.close()

    def write_one_frame(self, rgb=None, y=None, uv=None, src_format="420"):
        y, uv = self.format_adapter(src_format=src_format, rgb=rgb, y=y, uv=uv)
        self._writer_loop.write(y.tobytes(), uv.tobytes())

    def close(self):
        self._writer_loop.close()
        self._writer_thread.join()
        self._writer_loop.rethrow()
