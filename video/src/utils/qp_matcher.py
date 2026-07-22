# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import itertools
import json
import re
from typing import Dict, Any, Iterable, List, Tuple

import numpy as np
import pandas as pd

__all__ = ["match_q_index_lists"]


def _create_bitrate_anchor(
    candidates: pd.DataFrame, anchor_kbits: Iterable[int], match_per_video: bool, fps: int = 30
) -> pd.DataFrame:
    if not match_per_video:
        records = [{"q_index": (i, i), "bits_per_frame": kbit * 1000 / fps} for i, kbit in enumerate(anchor_kbits)]
        anchor = pd.DataFrame.from_records(records)
    else:
        records = []
        for key, _ in candidates.groupby(["ds_name", "video_path"]):
            ds_name, video_path = key  # type: ignore[misc]
            vid_path_norm = re.sub(r"_(\d+)x(\d+)_", "_", str(video_path))
            for i, kbit in enumerate(anchor_kbits):
                records.append(
                    {
                        "ds_name": ds_name,
                        "video_path": vid_path_norm,
                        "q_index": (i, i),
                        "bits_per_frame": kbit * 1000 / fps,
                    }
                )
        anchor = pd.DataFrame.from_records(records)
    return anchor


def _read_metrics(
    metrics: str | Dict[str, Any], scenarios: None | str | Iterable[str] = None, group: bool = True
) -> pd.DataFrame:
    if isinstance(metrics, dict):
        metrics_ = metrics
    else:
        with open(metrics, "rb") as f:
            metrics_ = json.load(f)

        if not isinstance(metrics_, dict):
            raise ValueError(f"{metrics}: JSON must contain a dictionary (JSON object)")

    def list_all(m):
        for d in m.values():
            for s in d.values():
                for q in s.values():
                    yield q

    columns = ["ds_name", "video_path", "i_frame_q_index", "p_frame_q_index", "frame_pixel_num", "ave_all_frame_bpp"]
    df = pd.DataFrame.from_records(list_all(metrics_), columns=columns)
    if scenarios is not None:
        if isinstance(scenarios, str):
            scenarios_ = (scenarios,)
        else:
            scenarios_ = scenarios
        df = df.loc[df.ds_name.isin(scenarios_)].copy()

    if group:
        df = pd.DataFrame(
            {
                "q_index": np.fromiter(zip(df.i_frame_q_index, df.p_frame_q_index), dtype=object, count=len(df)),
                "bits_per_frame": df.frame_pixel_num * df.ave_all_frame_bpp,
            }
        )
        df = df.groupby("q_index").mean().reset_index()  # type: ignore[union-attr]
    else:
        # Remove resolution information so that matched QPs work for decoder_metrics as well
        df["video_path"] = [re.sub(r"_(\d+)x(\d+)_", "_", p) for p in df.video_path]
        df["q_index"] = list(zip(df.i_frame_q_index, df.p_frame_q_index))
        df["bits_per_frame"] = df.frame_pixel_num * df.ave_all_frame_bpp
    return df


def _apply_min_distance(q_index_list: List[Tuple[int, int]], min_distance: int) -> List[Tuple[int, int]]:
    if min_distance > 0 and len(q_index_list) > 1:

        def distance(x, y):
            return max(abs(a - b) for a, b in zip(x, y))

        q_index_list[1:-1] = [p[0] for p in itertools.pairwise(q_index_list[1:]) if distance(*p) >= min_distance]
        q_index_list[1:-1] = [p[1] for p in itertools.pairwise(q_index_list[:-1]) if distance(*p) >= min_distance]
    return q_index_list


def match_q_index_lists(
    *,
    candidate_metrics: str | Dict[str, Any],
    scenarios: None | str | Iterable[str] = None,
    anchor_kbits: None | Iterable[int] = None,
    anchor_metrics: None | str | Dict[str, Any] = None,
    anchor_q_index_list: Iterable[int] | None = None,
    min_distance: int = 0,
    match_per_video: bool = False,
    fps: int = 30,
):

    if anchor_metrics is None and anchor_kbits is None:
        raise ValueError("Either anchor_metrics or anchor_kbits must be provided")

    candidates = _read_metrics(candidate_metrics, scenarios=scenarios, group=not match_per_video)
    if anchor_metrics is None:
        assert anchor_kbits is not None
        anchor = _create_bitrate_anchor(candidates, anchor_kbits, match_per_video, fps=fps)
    else:
        anchor = _read_metrics(anchor_metrics, scenarios=scenarios, group=not match_per_video)

    if not match_per_video:
        candidates.columns = ["c_q_index", "c_value"]
        anchor.columns = ["a_q_index", "a_value"]
    else:
        anchor = anchor.rename(columns={"q_index": "a_q_index", "bits_per_frame": "a_value"})
        candidates = candidates.rename(columns={"q_index": "c_q_index", "bits_per_frame": "c_value"})

    if anchor_q_index_list is not None:
        anchor = anchor.loc[anchor.a_q_index.isin((q, q) for q in anchor_q_index_list)]

    if not match_per_video:
        df = candidates.merge(anchor, how="cross")
        df["distance"] = (df.c_value - df.a_value).abs()
        df = df.loc[df.groupby("a_q_index").distance.idxmin(), ["c_q_index"]]

        q_index_list = df.c_q_index.to_list()
        q_index_list.sort()
        q_index_list = _apply_min_distance(q_index_list, min_distance)
        i_frame_q_index_list = [x[0] for x in q_index_list]
        p_frame_q_index_list = [x[1] for x in q_index_list]
        return i_frame_q_index_list, p_frame_q_index_list
    else:
        df = candidates.merge(anchor, on="video_path")
        df["distance"] = (df.c_value - df.a_value).abs()

        # For each video and each anchor q_index choose minimal distance
        idx = df.groupby(["video_path", "a_q_index"]).distance.idxmin()
        df = df.loc[idx, ["video_path", "c_q_index"]]

        i_frame_q_index_list = {}
        p_frame_q_index_list = {}

        for v, sub in df.groupby("video_path"):
            q_pairs = sorted(sub.c_q_index.tolist())
            q_pairs = _apply_min_distance(q_pairs, min_distance)
            i_list = [p[0] for p in q_pairs]
            p_list = [p[1] for p in q_pairs]
            i_frame_q_index_list[v] = i_list
            p_frame_q_index_list[v] = p_list

    return i_frame_q_index_list, p_frame_q_index_list
