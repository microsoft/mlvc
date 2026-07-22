# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import re
import sys

from src.utils.qp_matcher import match_q_index_lists


def parse_list(value: str):
    value = value.strip()
    if not value:
        return None

    return [x for x in re.split(r"\s*,\s*|\s+", value) if x]


def parse_int_list(value: str):
    items = parse_list(value)
    if items is None:
        return None
    return [int(x) for x in items]


def parse_args():
    parser = argparse.ArgumentParser(description="Match model q-indices based on bitrate metrics")
    parser.add_argument("candidate_metrics", help="Benchmark metrics for the candidate model")
    parser.add_argument("anchor_metrics", help="Benchmark metrics for the anchor model")
    parser.add_argument(
        "--scenarios", type=parse_list, help="Comma or space separated list of scenarios to use in matching"
    )
    parser.add_argument(
        "--q-indices",
        "--q_indices",
        type=parse_int_list,
        help="Comma or space separated list of anchor q-indices to match",
    )
    parser.add_argument(
        "--min-qp-distance",
        "--min_qp_distance",
        type=int,
        default=3,
        help="Comma or space separated list of scenarios to use in matching",
    )
    parser.add_argument("--per_video", action="store_true", help="Match q-indices per video instead of globally")
    return parser.parse_args()


def main():
    args = parse_args()

    i_frame_q_index_list, p_frame_q_index_list = match_q_index_lists(
        candidate_metrics=args.candidate_metrics,
        anchor_metrics=args.anchor_metrics,
        scenarios=args.scenarios,
        anchor_q_index_list=args.q_indices,
        min_distance=args.min_qp_distance,
        match_per_video=args.per_video,
    )

    if not i_frame_q_index_list:
        print("No q-points found")
        return 1

    if i_frame_q_index_list == p_frame_q_index_list:
        print(f"Matched q-points: {i_frame_q_index_list}")
    else:
        print(f"Matched i-frame q-points: {i_frame_q_index_list}")
        print(f"Matched p-frame q-points: {p_frame_q_index_list}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
