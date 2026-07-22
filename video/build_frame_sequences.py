# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import concurrent.futures
import json
import os
import sys
import traceback
from functools import cached_property
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


class SequenceBuilder:
    def __init__(self, config, process_count):
        self.config = config
        if process_count <= 0:
            process_count = min(8, (os.cpu_count() or 1) - 1)

        self.process_count = max(1, process_count)

    @cached_property
    def sequence_length(self):
        length = self.config.get("sequence_length", 64)
        return int(length) if length is not None else None

    @cached_property
    def patch_width(self):
        width = self.config.get("patch_width", 512)
        return int(width)

    @cached_property
    def patch_height(self):
        height = self.config.get("patch_height", 512)
        return int(height)

    @cached_property
    def weak_frame_limit(self):
        limit = self.config.get("weak_frame_limit", 5)
        return int(limit)

    @cached_property
    def scale_factors(self):
        scale_factors = dict()
        for f, w in self.config.get("scale_factors", {}).items():
            f = int(f)
            if f < 1:
                raise ValueError("Invalid scale factor")
            if f in scale_factors:
                raise ValueError("Duplicate scale factor")
            w = float(w)

            scale_factors[f] = w

        if len(scale_factors) == 0:
            scale_factors[1] = 1.0

        return tuple(sorted(scale_factors.items()))

    def _process_metrics(self, dirname, filename):
        filename = os.path.join(dirname, filename)
        try:
            with open(filename, "rb") as f:
                data = json.load(f)
        except Exception as e:
            tb = "".join(traceback.format_exception(e))
            print(f"{filename}: {tb}")
            return None

        data = list(self._flatten_dict(data, ""))

        clip_info = pd.DataFrame.from_records([{n: v for n, v in data if not isinstance(v, list)}])
        clip_filter = self.config.get("clip_filter")
        if clip_filter and clip_info.eval(clip_filter).sum() == 0:
            return None

        df = pd.DataFrame.from_dict({n: v for n, v in data if isinstance(v, list)})
        df = df.rename_axis(index="frame_index").reset_index()

        # hard scene breaks
        scene_break_condition = self.config.get("scene_break_condition", "`optic_flow.d_90` > 40")
        if scene_break_condition is not None:
            scene_breaks = df.eval(scene_break_condition)
        else:
            scene_breaks = pd.Series(False, index=df.index, dtype="bool")

        # weak frames (to be skipped at start and end of clip)
        weak_frame_condition = self.config.get(
            "weak_frame_condition", "(`optic_flow.d_90` < 2) or (`luma.rgb_std` < 0.02) or (`luma.y_max` < 0.1)"
        )
        if weak_frame_condition is not None:
            weak_frames = df.eval(weak_frame_condition)
        else:
            weak_frames = pd.Series(False, index=df.index, dtype="bool")

        # add frames around hard breaks
        weak_frames |= scene_breaks.rolling(self.weak_frame_limit, center=True).max().fillna(value=1) != 0

        # too many weak frames in a row break scene
        scene_breaks = scene_breaks | weak_frames.rolling(self.weak_frame_limit, center=True).min().fillna(value=0)
        # convert mask to scene indices
        df["scene"] = scene_breaks.cumsum()

        # clear frame index for weak frames
        df.loc[weak_frames, "frame_index"] = np.nan

        # build scenes
        df = df.groupby("scene").agg(
            start_frame=pd.NamedAgg("frame_index", "min"),
            last_frame=pd.NamedAgg("frame_index", "max"),
            bbox_top=pd.NamedAgg("bbox.top", "min"),
            bbox_bottom=pd.NamedAgg("bbox.bottom", "max"),
            bbox_left=pd.NamedAgg("bbox.left", "min"),
            bbox_right=pd.NamedAgg("bbox.right", "max"),
        )
        df = df[pd.notna(df["start_frame"])]
        if self.sequence_length is not None:
            df = df[df["last_frame"] - df["start_frame"] >= self.sequence_length - 1]
        df = df[df["bbox_bottom"] - df["bbox_top"] >= self.patch_height]
        df = df[df["bbox_right"] - df["bbox_left"] >= self.patch_width]
        df["start_frame"] = df["start_frame"].astype("int")
        df["last_frame"] = df["last_frame"].astype("int")
        df["n_frames"] = df["last_frame"] - df["start_frame"] + 1

        clip_info = pd.DataFrame(
            clip_info.values.repeat(len(df), axis=0),
            index=df.index,  # type: ignore[reportAttributeAccessIssue]
            columns=clip_info.columns,  # type: ignore[reportAttributeAccessIssue]
        )
        clip_info.rename(columns={"n_frames": "total_frames"}, inplace=True)
        df = pd.concat([clip_info, df], axis=1)  # type: ignore[reportCallIssue]
        return df

    def _flatten_dict(self, d, prefix):
        for n, v in d.items():
            if isinstance(v, dict):
                yield from self._flatten_dict(v, prefix + n + ".")
            else:
                yield prefix + n, v

    def build_scenes(self, metrics_folder, metrics_filelist):
        if not metrics_filelist:
            filenames = list(self._collect_filename(metrics_folder, ".json"))
        else:
            with open(metrics_filelist, "rt", encoding="utf-8") as f:
                filenames = (x.strip() for x in f)
                filenames = [(metrics_folder, x) for x in filenames if x]

        df_list = list()
        if self.process_count == 1:
            for d, fn in tqdm(filenames, file=sys.stdout):
                df = self._process_metrics(d, fn)
                if df is not None:
                    df_list.append(df)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.process_count) as pool:
                tasks = [pool.submit(self._process_metrics, d, fn) for d, fn in filenames]
                try:
                    for t in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks), file=sys.stdout):
                        df = t.result()
                        if df is not None:
                            df_list.append(df)
                except BaseException:
                    for t in tasks:
                        t.cancel()
                    raise

        return pd.concat(df_list, ignore_index=True).sort_values(["filename", "start_frame"], ignore_index=True)

    @staticmethod
    def _collect_filename(base_dir, ext):
        for dirname, _, filenames in os.walk(base_dir):
            for fn in filenames:
                if not fn.endswith(ext):
                    continue

                yield dirname, fn

    def generate_sequences(self, scenes):
        # calc how many sequences can fit into scene
        if self.sequence_length is None:
            n_candidates = pd.Series(1, index=scenes.index)
        else:
            n_candidates = scenes.n_frames // self.sequence_length
        # next after last candidate index for each scene
        scene_end = n_candidates.cumsum()

        # calculate scene index for each candidate
        scene_indices = np.zeros(scene_end.iloc[-1], dtype=np.int32)
        scene_indices[scene_end.iloc[:-1]] = 1
        scene_indices = scene_indices.cumsum()

        # calculate candidate weight based on number of frames in scene
        weights = (scenes.n_frames / n_candidates)[scene_indices]
        weights /= weights.sum()

        # randomly pick sequences to generate
        total_sequences = self.config["n_sequences"]
        if total_sequences is None:
            selected = scene_indices
        else:
            selected = scene_indices[np.random.choice(len(weights), size=total_sequences, replace=False, p=weights)]

        # number of sequences per scene
        n_sequences = np.bincount(selected, minlength=len(scenes))

        sequences = list()
        for t, count in zip(scenes.itertuples(), n_sequences):
            if count == 0:
                continue

            sequence_length = self.sequence_length
            if sequence_length is None:
                sequence_length = t.n_frames

            # max sequence shift from flushed-to-start layout
            max_shift = t.n_frames - count * sequence_length
            # pick random shifts
            if max_shift == 0:
                shifts = [0] * count
            else:
                shifts: Sequence[int] = sorted(np.random.choice(max_shift, size=count).tolist())
            for idx, start in enumerate(shifts):
                # sequence start positions
                start += t.start_frame + idx * sequence_length
                assert start + sequence_length <= t.last_frame + 1
                sequences.append(
                    dict(
                        filename=t.filename,
                        start_frame=start,
                        n_frames=sequence_length,
                        scale_factor=self._sample_scale_factor(t),
                        bbox_top=t.bbox_top,
                        bbox_bottom=t.bbox_bottom,
                        bbox_left=t.bbox_left,
                        bbox_right=t.bbox_right,
                        width=t.width,
                        height=t.height,
                        frame_rate=t.frame_rate,
                        total_frames=t.total_frames,
                    )
                )

        sequences = pd.DataFrame.from_records(sequences)
        # shuffle generated sequences
        sequences = sequences.iloc[np.random.permutation(len(sequences))].reset_index(drop=True)
        # and generate sequence folder paths
        sequences.index = sequences.index.to_series().apply(lambda x: f"{x // 1000:03d}/{x % 1000:03d}")
        sequences.rename_axis(index="sequence_id", inplace=True)
        return sequences

    def _sample_scale_factor(self, scene):
        width = scene.bbox_right - scene.bbox_left
        height = scene.bbox_bottom - scene.bbox_top

        scale_factors = self.scale_factors

        weights = list()
        for f, w in scale_factors:
            if self.patch_width * f > width or self.patch_height * f > height:
                break
            weights.append(w)

        if len(weights) == 0:
            return 1
        if len(weights) == 1:
            return scale_factors[0][0]

        weights = np.asarray(weights)
        weights /= weights.sum()
        return scale_factors[np.random.choice(len(weights), p=weights)][0]


def parse_args():
    parser = argparse.ArgumentParser(description="Build training dataset frame sequences")
    parser.add_argument("--config", help="YAML configuration file", required=True)
    parser.add_argument("--scenes_csv", help="CSV file to save extracted scenes")
    parser.add_argument("--process_count", type=int, default=0, help="level of concurrency")
    parser.add_argument("--metrics_filenames", help="file with metrics filenames")
    parser.add_argument("metrics_folder", help="Folder with clip metrics")
    parser.add_argument("sequences_csv", help="CSV file to save generated sequences")
    return parser.parse_args()


def main():
    args = parse_args()

    config_filename = os.path.abspath(args.config)
    with open(config_filename, "rt") as f:
        config = yaml.full_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration YAML in {args.config} must be a dictionary")

    builder = SequenceBuilder(config.get("sequence_builder", {}), args.process_count)
    scenes = builder.build_scenes(args.metrics_folder, args.metrics_filenames)
    if args.scenes_csv:
        scenes.to_csv(args.scenes_csv, index=False)

    sequences = builder.generate_sequences(scenes)
    sequences.to_csv(args.sequences_csv)


if __name__ == "__main__":
    exit(main())
