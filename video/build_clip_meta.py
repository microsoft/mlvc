# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import concurrent.futures
import fractions
import json
import logging
import math
import os
from typing import Callable, List

import pandas as pd
from tqdm import tqdm


def build_meta_from_file(fn):
    with open(fn, "rb") as f:
        data = json.load(f)

    video_name = data["format"]["filename"]

    stream = None
    for s in data["streams"]:
        if s["codec_type"] == "video" and s.get("codec_name"):
            stream = s
            break

    if stream is None:
        print(f"{video_name} does not contain video stream")
        return None

    frames = data.get("frames")
    frame_rate = None
    duration = None
    payload_size = None
    if frames is not None:
        # select frames by stream index
        stream_index = stream["index"]
        frames = [x for x in frames if x["stream_index"] == stream_index]
        # convert time stamps to float
        missing_timestamp = False
        for f in frames:
            timestamp = f.get("best_effort_timestamp_time")
            if timestamp is None:
                missing_timestamp = True
            else:
                f["best_effort_timestamp_time"] = float(timestamp)

        n_frames = len(frames)

        if not missing_timestamp:
            # sort by timestamp
            frames.sort(key=lambda x: (x["best_effort_timestamp_time"], x.get("coded_picture_number", 0)))
            if n_frames >= 2:
                timespan = frames[-1]["best_effort_timestamp_time"] - frames[0]["best_effort_timestamp_time"]
                if timespan > 0:
                    frame_rate = (n_frames - 1) / timespan
                    duration = timespan + 1 / frame_rate

        payload_size = sum(int(x["pkt_size"]) for x in frames)
    else:
        n_frames = stream.get("nb_frames")
        if n_frames is not None:
            n_frames = int(n_frames)

    if n_frames is not None:
        if n_frames == 0:
            print(f"{video_name}: does not contain any frames")
            return None
        if n_frames < 2:
            print(f"{video_name}: contains only {n_frames} frames")
            return None

    if frame_rate is None:
        try:
            frame_rate = fractions.Fraction(stream["r_frame_rate"])
            frame_rate = float(frame_rate)
        except ValueError:
            print(f"{video_name}: can not parse frame rate")
            frame_rate = None

    if duration is None:
        duration = stream.get("duration")
        if duration is not None:
            try:
                duration = float(duration)
            except ValueError:
                print(f"{video_name}: can not parse clip duration")

    if duration is not None and frame_rate is not None:
        if n_frames is not None:
            if abs(duration - n_frames / frame_rate) >= 1:
                print(
                    f"{video_name}: duration mismatch: n_frames={n_frames}, frame_rate={frame_rate}"
                    f", duration={duration}, estimated={n_frames / frame_rate}"
                )
        elif frame_rate > 0:
            # estimate number of frames from duration
            n_frames = math.ceil(duration / frame_rate)

    if payload_size is not None and duration is not None:
        bit_rate = 8 * payload_size / duration
    else:
        bit_rate = stream.get("bit_rate")
        if bit_rate is not None:
            bit_rate = float(bit_rate)
        elif duration is not None:
            print(f"{video_name}: missing stream bit_rate, estimating from file size")
            bit_rate = int(data["format"]["size"]) * 8 / duration

    rotation = 0
    side_data_list = stream.get("side_data_list", [])
    for side_data in side_data_list:
        if side_data.get("side_data_type") == "Display Matrix":
            rotation = int(side_data.get("rotation", 0))
            break

    width = stream["width"]
    height = stream["height"]
    if rotation in (-90, 90, -270, 270):
        width, height = height, width

    return dict(
        filename=video_name,
        codec_name=stream["codec_name"],
        frame_rate=frame_rate,
        width=width,
        height=height,
        rotation=rotation,
        pix_fmt=stream["pix_fmt"],
        n_frames=n_frames,
        qp=int(stream.get("level", -1)),
        bit_rate=bit_rate,
    )


def build_meta_from_file_and_log_error(fn):
    try:
        return build_meta_from_file(fn)
    except BaseException:
        print(f"Exception while parsing {fn} meta")
        raise


def call_on_list(func: Callable, arglist: List, desc: str):
    result_list = list()

    max_workers = max(1, (os.cpu_count() or 1) - 1)
    max_workers = min(max_workers, 8, len(arglist))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(func, fn) for fn in arglist]
        try:
            for f in tqdm(concurrent.futures.as_completed(futures), desc=desc, total=len(futures)):
                rc = f.result()
                if rc is not None:
                    result_list.append(rc)
        except BaseException:
            for f in futures:
                f.cancel()
            raise

    return result_list


def build_meta(source_dir):
    filelist = list()
    for dirname, _, filenames in os.walk(source_dir):
        for fn in filenames:
            filelist.append(os.path.join(dirname, fn))

    if len(filelist) == 0:
        raise ValueError("No clip meta found")

    meta_list = call_on_list(build_meta_from_file_and_log_error, filelist, "building clip meta")

    df = pd.DataFrame.from_records(meta_list)
    df["bpp"] = df.bit_rate / (df.width * df.height * df.frame_rate)
    df.sort_values("filename", inplace=True)
    return df


def load_processed_json(fn):
    try:
        with open(fn, "rb") as f:
            data = json.load(f)
    except ValueError:
        logging.info(f"{fn}: correupted file, skipping...")
        return None

    return data["filename"]


def load_processed(source_dir):
    filelist = list()
    for dirname, _, filenames in os.walk(source_dir):
        for fn in filenames:
            filelist.append(os.path.join(dirname, fn))

    if len(filelist) == 0:
        return set()

    processed_list = call_on_list(load_processed_json, filelist, "loading processed clip meta")
    return set(processed_list)


def main():
    parser = argparse.ArgumentParser(description="Build clip meta from ffprobe output")
    parser.add_argument("source_dir", help="source folder with ffprobe output")
    parser.add_argument("meta", help="file to output meta")
    parser.add_argument("--split", type=int, default=1, help="split clip meta in chunks")
    parser.add_argument("--processed_dir", help="exclude processed clips")

    args = parser.parse_args()

    df = build_meta(args.source_dir)

    if args.processed_dir:
        processed = load_processed(args.processed_dir)
        if len(processed) > 0:
            df = df[~df.filename.isin(processed)]

    split = args.split
    if split <= 1:
        df.to_json(args.meta, orient="records")
    else:
        df["num_pixels"] = df["width"] * df["height"]
        df.sort_values(by=["num_pixels", "n_frames", "filename"], inplace=True)  # type: ignore[call-overload]
        df.drop(columns=["num_pixels"])

        for idx in range(split):
            meta_fn, meta_ext = os.path.splitext(args.meta)
            part = df.iloc[idx::split].sort_values("filename")
            part.to_json(f"{meta_fn}-{idx + 1}{meta_ext}", orient="records")


if __name__ == "__main__":
    exit(main())
