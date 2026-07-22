# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import concurrent.futures
import functools
import json
import logging
import os
import subprocess
import time

from src.utils.app import BaseApp


class FFProbeApp(BaseApp):
    def __init__(self):
        super().__init__()
        self._last_progress_time: float = 0.0

    def run(self):
        if self.rank >= 0:
            raise ValueError("Distributed processing is not supported")

        video_list_fn = self.resolve_path(self.get_config_by_path("ffprobe.video_list"), default_source=".")

        video_list = list()
        with open(video_list_fn, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                if len(line) == 0:
                    continue

                video_list.append(line)

        process_count = self.get_config_by_path("ffprobe.process_count", default=None)
        if process_count is None or process_count == 0:
            process_count = os.cpu_count()
        else:
            process_count = int(process_count)

        assert process_count is not None
        process_count = min(process_count, len(video_list))

        logging.info(f"Running {process_count} ffprobe processes in parallel")

        source_dir = self.resolve_path(self.get_config_by_path("ffprobe.source_dir", expected_type=str))
        output_frames = self.get_config_by_path("ffprobe.output_frames", expected_type=bool, default=False)
        output_dir = self.get_config_by_path("ffprobe.output_dir", expected_type=str, default="")
        output_dir = self.resolve_path(output_dir, default_source="save_dir")

        run_ffprobe = functools.partial(
            self._run_ffprobe, source_dir=source_dir, output_frames=output_frames, output_dir=output_dir
        )

        self._last_progress_time = time.time()
        if process_count <= 1:
            for idx, filename in enumerate(video_list):
                run_ffprobe(filename)
                self._print_progress(idx, len(video_list))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=process_count) as executor:
                futures = [executor.submit(run_ffprobe, filename) for filename in video_list]
                try:
                    for idx, f in enumerate(concurrent.futures.as_completed(futures)):
                        f.result()
                        self._print_progress(idx, len(futures))
                except BaseException:
                    for f in futures:
                        f.cancel()
                    raise

    def _print_progress(self, idx, total):
        idx += 1
        t = time.time()
        if idx < total and t - self._last_progress_time < 10:
            return

        logging.info(f"processed {idx}/{total}")
        self._last_progress_time = t

    @staticmethod
    def _run_ffprobe(filename, *, source_dir, output_frames, output_dir):
        if os.path.isabs(filename):
            raise ValueError(f"only relative paths are supported: {filename}")

        dst = os.path.splitext(filename)[0] + ".json"
        dst = os.path.join(output_dir, dst)

        args = ["ffprobe", "-print_format", "json", "-show_format", "-show_streams", "-select_streams", "v"]
        if output_frames:
            args.append("-show_frames")

        args.append(filename)
        try:
            rc = subprocess.run(args, cwd=source_dir, stdin=None, capture_output=True)
        except BaseException:
            logging.error(f"{filename}: processing failed")
            raise

        is_corrupted = False
        if rc.returncode:
            if len(rc.stderr) == 0:
                logging.error(f"{filename}: ffprobe failed with exit code {rc.returncode}")
                return

            error = rc.stderr.decode(encoding="utf-8", errors="replace")
            if rc.returncode != 1 or FFProbeApp._check_fatal_error(filename, error):
                logging.error(f"{filename}: ffprobe exit code {rc.returncode}, error output:\n" + error)
                return

            is_corrupted = True

        try:
            meta = rc.stdout.decode(encoding="utf-8")
        except UnicodeDecodeError:
            logging.warning(f"{filename}: invalid bytes in ffprobe output:\n" + str(rc.stdout))
            meta = rc.stdout.decode(encoding="utf-8", errors="replace")
            is_corrupted = True

        if is_corrupted:
            try:
                decoded = json.loads(meta)
            except ValueError as e:
                logging.error(f"{filename}: invalid ffprobe output: {e}")
                return

            if not isinstance(decoded, dict) or "streams" not in decoded:
                logging.warning(f"{filename}: skipping corrupted file, no streams are extracted")
                return

            if output_frames:
                frames = decoded.get("frames")
                if not isinstance(frames, list) or len(frames) == 0:
                    logging.warning(f"{filename}: skipping corrupted file, no frames are extracted")
                    return

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wt", encoding="utf-8") as f:
            f.write(meta)
            f.close()

    @staticmethod
    def _check_fatal_error(filename, error: str):
        # ffprobe error messages for corrupted file
        corrupted_file_errors = (
            "End of file",
            "Invalid data found when processing input",
        )

        prefix = filename + ": "
        for line in error.splitlines():
            if not line.startswith(prefix):
                continue

            line = line[len(prefix) :]
            if line in corrupted_file_errors:
                return False

        return True


if __name__ == "__main__":
    FFProbeApp().main()
