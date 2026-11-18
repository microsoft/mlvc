# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Build dataset description from frame sequence CSV")
    parser.add_argument("frame_sequence_csv", help="source frame sequence CSV")
    parser.add_argument("dataset_description_json", help="output dataset description JSON")

    args = parser.parse_args()

    df = pd.read_csv(args.frame_sequence_csv)
    dataset_description = []
    for t in df.to_dict("records"):
        dataset_description.append(
            {
                "height": (t["bbox_bottom"] - t["bbox_top"]) // t["scale_factor"],
                "width": (t["bbox_right"] - t["bbox_left"]) // t["scale_factor"],
                "seq_length": t["n_frames"],
                "path": t["sequence_id"],
                "frames": [f"im{i:05d}.webp" for i in range(t["n_frames"])],
            }
        )

    with open(args.dataset_description_json, "wt", encoding="utf-8") as f:
        json.dump(dataset_description, f)
        f.close()


if __name__ == "__main__":
    sys.exit(main())
