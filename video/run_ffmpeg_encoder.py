# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import concurrent.futures
import json
import logging
import numbers
import os
import subprocess
import time
from functools import cached_property

from src.utils.app import BaseApp


class FFMpegEncoderApp(BaseApp):
    def __init__(self):
        super().__init__()
        self._last_progress_time: float = 0.0

    @cached_property
    def yuv_path(self):
        return self.resolve_path(
            self.get_config_by_path("ffmpeg_encoder.yuv_path", expected_type=str), default_source="checkpoints_mount"
        )

    @cached_property
    def output_path(self):
        output_path = self.save_dir
        relative_path = self.get_config_by_path("ffmpeg_encoder.output_path", default=None)
        if relative_path:
            output_path = os.path.join(output_path, relative_path)

        return output_path

    def _load_clip_list(self):
        ds_config = self.resolve_path(
            self.get_config_by_path("ffmpeg_encoder.dataset_config", expected_type=str), default_source="data_mount"
        )

        with open(ds_config, "rt", encoding="utf-8") as f:
            testset_desc = json.load(f)

        fps = self.get_config_by_path("ffmpeg_encoder.fps", expected_type=numbers.Real)

        clip_path_cache = dict()
        clip_list = list()
        for dataset_name, dataset_desc in testset_desc["test_classes"].items():
            if not dataset_desc.get("test", True):
                continue
            for seq_name, seq_desc in dataset_desc["sequences"].items():
                clip_path = os.path.join(dataset_desc["base_path"], seq_name)
                clip_path = os.path.splitext(clip_path)[0]

                for clip_path in self._expand_clip_paths(clip_path_cache, clip_path):
                    clip_list.append(
                        dict(
                            filename=clip_path,
                            width=seq_desc["width"],
                            height=seq_desc["height"],
                            fps=fps,
                        )
                    )

        return clip_list

    def _expand_clip_paths(self, clip_path_cache, clip_path):
        clip_folder = os.path.dirname(clip_path)
        file_map = self._list_clip_files(clip_path_cache, clip_folder)

        file_list = list()

        prefix = os.path.basename(clip_path)
        for n, used in file_map.items():
            if not n.startswith(prefix):
                continue

            if used:
                raise ValueError(f"{os.path.join(clip_folder, n)} matches at least 2 sequences in dataset")

            file_list.append(n)

        if len(file_list) == 0:
            raise ValueError(f"{clip_path}*.yuv does not match any files")

        for n in file_list:
            file_map[n] = True

        return [os.path.join(clip_folder, n) for n in file_list]

    def _list_clip_files(self, clip_path_cache, folder: str):
        file_list = clip_path_cache.get(folder)
        if file_list is None:
            file_list = dict()
            for de in os.scandir(os.path.join(self.yuv_path, folder)):
                if de.name.endswith(".yuv") and de.is_file():
                    file_list[de.name] = False

            clip_path_cache[folder] = file_list

        return file_list

    def _run_ffmpeg(self, clip_meta):
        clip_path: str = clip_meta["filename"]
        output_path = os.path.splitext(clip_path)[0] + ".mp4"
        output_path = os.path.join(self.output_path, output_path)
        if not os.path.isabs(output_path):
            output_path = os.path.join(os.getcwd(), output_path)

        # fmt: off
        args = [
            "ffmpeg",
            "-v", "quiet",
            "-f", "rawvideo",
            "-pix_fmt", "yuv420p",
            "-s", f"{clip_meta['width']}x{clip_meta['height']}",
            "-r", str(clip_meta["fps"]),
            "-i", clip_path,
            "-threads", "1",
            "-preset", "veryslow",
            "-keyint_min", "2",
            "-g", "30",
            "-sc_threshold", "0",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "17",
            output_path
        ]
        # fmt: on

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            subprocess.run(args, cwd=self.yuv_path, stdin=None, stdout=None, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            if len(e.stderr) > 0:
                logging.error(
                    f"{clip_path}: ffmpeg error output:\n" + e.stderr.decode(encoding="utf-8", errors="replace")
                )
            else:
                logging.error(f"{clip_path}: ffmpeg failed with exit code {e.returncode}")

            raise
        except BaseException:
            logging.error(f"{clip_path}: processing failed")
            raise

    def _print_progress(self, idx, total):
        idx += 1
        t = time.time()
        if idx < total and t - self._last_progress_time < 10:
            return

        logging.info(f"processed {idx}/{total}")
        self._last_progress_time = t

    def run(self):
        if self.rank >= 0:
            raise ValueError("Distributed processing is not supported")

        clip_list = self._load_clip_list()

        process_count = self.get_config_by_path("ffmpeg_encoder.process_count", default=None)
        if process_count is None or process_count == 0:
            process_count = os.cpu_count()
        else:
            process_count = int(process_count)

        assert process_count is not None
        process_count = min(process_count, len(clip_list))

        logging.info(f"Running {process_count} ffmpeg processes in parallel")

        self._last_progress_time = time.time()
        if process_count <= 1:
            for idx, clip_meta in enumerate(clip_list):
                self._run_ffmpeg(clip_meta)
                self._print_progress(idx, len(clip_list))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=process_count) as executor:
                futures = [executor.submit(self._run_ffmpeg, clip_meta) for clip_meta in clip_list]
                try:
                    for idx, f in enumerate(concurrent.futures.as_completed(futures)):
                        f.result()
                        self._print_progress(idx, len(futures))
                except BaseException:
                    for f in futures:
                        f.cancel()
                    raise


if __name__ == "__main__":
    FFMpegEncoderApp().main()
