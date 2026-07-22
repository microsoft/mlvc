# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import concurrent.futures
import contextlib
import csv
import dataclasses
import logging
import os
import time
from functools import cached_property
from typing import Sequence, List

import PIL.Image
import numpy as np

from src.datasets.ffmpeg_dataset import FfmpegDataset
from src.transforms.functional import ycbcr444_to_rgb
from src.utils.app import BaseApp


@dataclasses.dataclass(frozen=True)
class ClipInfo:
    filename: str
    width: int
    height: int
    frame_rate: float
    n_frames: int


@dataclasses.dataclass(frozen=True)
class SequenceInfo:
    sequence_id: str
    start_frame: int
    n_frames: int
    scale_factor: int
    bbox_top: int
    bbox_bottom: int
    bbox_left: int
    bbox_right: int


@dataclasses.dataclass
class ClipRecord:
    clip_info: ClipInfo
    sequences: List[SequenceInfo] = dataclasses.field(default_factory=list)


class FrameSequenceExtractorApp(BaseApp):
    def __init__(self):
        super().__init__()
        self._processed_clip_file = None
        self._last_progress_time: float = 0.0

    def close(self):
        processed_clip_file = self._processed_clip_file
        if processed_clip_file is not None:
            self._processed_clip_file = None
            # noinspection PyBroadException
            try:
                processed_clip_file.close()
            except BaseException:
                logging.error("Exception closing processed clip list", exc_info=True)

        super().close()

    @cached_property
    def source_dir(self):
        source_dir = self.get_config_by_path("frame_sequence_extractor.source_dir", expected_type=str)
        return self.resolve_path(source_dir)

    @cached_property
    def output_dir(self):
        output_dir = self.get_config_by_path("frame_sequence_extractor.output_dir", expected_type=str)
        output_dir = self.resolve_path(output_dir, default_source="save_dir")
        return output_dir or self.save_dir

    @cached_property
    def compression_quality(self):
        return self.get_config_by_path("frame_sequence_extractor.compression.quality", expected_type=int, default=90)

    @cached_property
    def lossless_compression(self):
        return self.get_config_by_path(
            "frame_sequence_extractor.compression.lossless", expected_type=bool, default=False
        )

    @cached_property
    def processed_clip_list_filename(self):
        filename = self.get_config_by_path(
            "frame_sequence_extractor.processed_clip_list", expected_type=str, default=""
        )
        filename = self.resolve_path(filename, default_source="save_dir")
        return filename

    def run(self):
        if self.rank >= 0:
            raise ValueError("Distributed processing is not supported")

        clip_list = self._read_frame_sequences()

        clip_count = len(clip_list)
        sequence_count = sum(len(x.sequences) for x in clip_list)
        logging.info(f"Extracting {sequence_count} sequences from {clip_count} clips")

        clip_list = self._filter_processed_clips(clip_list)
        restart_from = clip_count - len(clip_list)
        if restart_from > 0:
            logging.info(f"Restarting from clip #{restart_from}")

        process_count = self.get_config_by_path("process_count", default=None)
        if process_count is None or process_count == 0:
            process_count = os.cpu_count()
        else:
            process_count = int(process_count)

        assert process_count is not None
        process_count = min(process_count, len(clip_list))
        logging.info(f"Running {process_count} ffmpeg processes in parallel")

        task_kwargs = dict(
            source_dir=self.source_dir,
            output_dir=self.output_dir,
            compression_quality=self.compression_quality,
            lossless_compression=self.lossless_compression,
        )

        self._last_progress_time = time.time()
        if process_count <= 1:
            for idx, clip_record in enumerate(clip_list):
                filename = self._extract_frame_sequences(clip_record, **task_kwargs)
                self._on_processed_clip(filename, restart_from + idx, clip_count)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=process_count) as executor:
                futures = [
                    executor.submit(self._extract_frame_sequences, clip_record, **task_kwargs)
                    for clip_record in clip_list
                ]
                try:
                    for idx, f in enumerate(concurrent.futures.as_completed(futures)):
                        filename = f.result()
                        self._on_processed_clip(filename, restart_from + idx, clip_count)
                except BaseException:
                    for f in futures:
                        f.cancel()
                    raise

    def _read_frame_sequences(self):
        filename = self.get_config_by_path("frame_sequence_extractor.frame_sequences_csv", expected_type=str)
        filename = self.resolve_path(filename, default_source=".")

        clip_record_map = dict()

        start, end = self._get_sequence_range()
        with open(filename, "rt", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, record in enumerate(reader):
                if idx < start:
                    continue
                if end is not None and idx >= end:
                    break

                clip_info = ClipInfo(
                    filename=record["filename"],
                    width=int(record["width"]),
                    height=int(record["height"]),
                    frame_rate=float(record["frame_rate"]),
                    n_frames=int(record["total_frames"]),
                )
                sequence_info = SequenceInfo(
                    sequence_id=record["sequence_id"],
                    start_frame=int(record["start_frame"]),
                    n_frames=int(record["n_frames"]),
                    scale_factor=int(record["scale_factor"]),
                    bbox_top=int(record["bbox_top"]),
                    bbox_bottom=int(record["bbox_bottom"]),
                    bbox_left=int(record["bbox_left"]),
                    bbox_right=int(record["bbox_right"]),
                )

                clip_record = clip_record_map.get(clip_info.filename)
                if clip_record is None:
                    clip_record_map[clip_info.filename] = clip_record = ClipRecord(clip_info)
                elif clip_info != clip_record.clip_info:
                    raise ValueError(f"Conflicting clip info for {clip_info.filename}")

                clip_record.sequences.append(sequence_info)

        clip_record_list = list(clip_record_map.values())
        for clip_record in clip_record_list:
            clip_record.sequences.sort(key=lambda x: x.start_frame)

        return clip_record_list

    def _get_sequence_range(self):
        sequence_range = self.get_config_by_path("frame_sequence_extractor.sequence_range", default=None)
        if sequence_range is None:
            return 0, None

        if isinstance(sequence_range, Sequence) and len(sequence_range) == 2:
            start, end = sequence_range
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end:
                return start, end

        raise ValueError(f"Invalid sequence range: {sequence_range}")

    def _filter_processed_clips(self, clip_list: List[ClipRecord]):
        processed_clip_list = self._load_processed_clip_list()
        if len(processed_clip_list) == 0:
            return clip_list

        processed_clip_set = set(processed_clip_list)
        new_clip_list = [x for x in clip_list if x.clip_info.filename not in processed_clip_set]
        assert len(new_clip_list) + len(processed_clip_list) == len(clip_list)
        return new_clip_list

    def _load_processed_clip_list(self):
        auto_restart = self.get_config_by_path(
            "frame_sequence_extractor.auto_restart", expected_type=bool, default=True
        )

        if auto_restart:
            if self.processed_clip_list_filename:
                try:
                    file = open(self.processed_clip_list_filename, "rt", encoding="utf-8")
                    with file:
                        return [n.rstrip("\n") for n in file]
                except FileNotFoundError:
                    pass

        processed_clip_list = self.get_config_by_path(
            "frame_sequence_extractor.initial_processed_clip_list", expected_type=str, default=""
        )
        processed_clip_list = self.resolve_path(processed_clip_list, default_source="checkpoints_mount")
        if not processed_clip_list:
            return list()

        with open(processed_clip_list, "rt", encoding="utf-8") as file:
            clip_list = [n.rstrip("\n") for n in file]

        if len(clip_list) > 0 and self.processed_clip_list_filename:
            assert self._processed_clip_file is None
            self._processed_clip_file = open(self.processed_clip_list_filename, "wt", encoding="utf-8")
            for filename in clip_list:
                self._processed_clip_file.write(f"{filename}\n")
            self._processed_clip_file.flush()

        return clip_list

    def _on_processed_clip(self, filename, idx, total):
        if self._processed_clip_file is None and self.processed_clip_list_filename:
            self._processed_clip_file = open(self.processed_clip_list_filename, "wt", encoding="utf-8")

        if self._processed_clip_file is not None:
            self._processed_clip_file.write(f"{filename}\n")
            self._processed_clip_file.flush()

        idx += 1
        t = time.time()
        if idx < total and t - self._last_progress_time < 10:
            return

        logging.info(f"processed {idx}/{total}")
        self._last_progress_time = t

    @staticmethod
    def _extract_frame_sequences(
        clip_record: ClipRecord, *, source_dir, output_dir, compression_quality, lossless_compression
    ):
        clip_info = clip_record.clip_info
        clip_filename = os.path.join(source_dir, clip_info.filename)
        with contextlib.closing(iter(FfmpegDataset(clip_filename, clip_info.width, clip_info.height))) as reader:
            clip_frame_idx = 0
            for sequence_info in clip_record.sequences:
                # skip to start frame
                while clip_frame_idx < sequence_info.start_frame:
                    next(reader)
                    clip_frame_idx += 1

                sequence_folder = os.path.join(output_dir, sequence_info.sequence_id)
                os.makedirs(sequence_folder, exist_ok=True)

                assert clip_frame_idx == sequence_info.start_frame
                for frame_idx in range(sequence_info.n_frames):
                    frame = next(reader)
                    clip_frame_idx += 1

                    # convert to RGB using ITU-R_BT.709 luma transform (Pillow uses ITU-R 601-2)
                    frame = ycbcr444_to_rgb(frame[:1], frame[1:])
                    frame = (frame * 255).round().clip(0, 255).astype(np.uint8)
                    frame = frame.transpose(1, 2, 0).flatten()

                    frame = PIL.Image.frombytes("RGB", (clip_info.width, clip_info.height), frame)

                    bbox_left = sequence_info.bbox_left
                    bbox_right = sequence_info.bbox_right
                    bbox_top = sequence_info.bbox_top
                    bbox_bottom = sequence_info.bbox_bottom
                    if sequence_info.scale_factor != 1:
                        bbox_right -= (bbox_right - bbox_left) % sequence_info.scale_factor
                        bbox_bottom -= (bbox_bottom - bbox_top) % sequence_info.scale_factor
                    if (
                        bbox_left != 0
                        or bbox_right != clip_info.width
                        or bbox_top != 0
                        or bbox_bottom != clip_info.height
                    ):
                        frame = frame.crop((bbox_left, bbox_top, bbox_right, bbox_bottom))

                    if sequence_info.scale_factor != 1:
                        width = (bbox_right - bbox_left) // sequence_info.scale_factor
                        height = (bbox_bottom - bbox_top) // sequence_info.scale_factor
                        frame = frame.resize((width, height), resample=PIL.Image.Resampling.BOX)

                    filename = os.path.join(sequence_folder, f"im{frame_idx:05d}.webp")
                    frame.save(filename, "webp", method=6, quality=compression_quality, lossless=lossless_compression)

        return clip_info.filename


if __name__ == "__main__":
    FrameSequenceExtractorApp().main()
