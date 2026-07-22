# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import json
import torch
import struct
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import NamedTuple, Any
from ._azure import download_blob
from ._env import get_env, get_required_env
from .types import ModelType, ValidationTestResults
from .const import DEFAULT_JOB_OUTPUTS_DIR, DEFAULT_TEST_DATA_DIR
from src.utils.video_reader import PNGReader, YUVReader


def get_namedtuple_fields(x: NamedTuple):
    # noinspection PyProtectedMember
    return x._fields


def iter_namedtuple(x):
    return zip(get_namedtuple_fields(x), x)


def to_torch_namedtuple(x: NamedTuple) -> NamedTuple:
    tensors = {k: torch.from_numpy(v) for k, v in iter_namedtuple(x)}
    return type(x)(**tensors)  # type: ignore


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def download_test_data(path: Path | str, base_path: Path | str = DEFAULT_TEST_DATA_DIR) -> Path:
    if Path(path).is_absolute():
        return Path(path)

    account = get_required_env("AZURE_TEST_DATA_STORAGE_ACCOUNT")

    return download_blob(
        account,
        get_env("AZURE_TEST_DATA_CONTAINER", "test-set"),
        path,
        Path(base_path).expanduser(),
        overwrite=False,
    )


def download_job_outputs(path: Path | str, base_path: Path | str = DEFAULT_JOB_OUTPUTS_DIR) -> Path:
    if Path(path).is_absolute():
        return Path(path)

    account = get_required_env("AZURE_JOB_OUTPUTS_STORAGE_ACCOUNT")

    return download_blob(
        account,
        get_env("AZURE_JOB_OUTPUTS_CONTAINER", "job-outputs"),
        path,
        Path(base_path).expanduser(),
        overwrite=False,
    )


def read_video_frames(
    video_path: str | Path, image_width: int, image_height: int, frame_count: int | None = 300
) -> list[tuple[np.ndarray, np.ndarray]]:
    video_path = Path(video_path)
    if video_path.suffix == ".yuv":
        reader = YUVReader(src_path=str(video_path), height=image_height, width=image_width)
    elif video_path.suffix == ".png":
        reader = PNGReader(src_path=str(video_path), height=image_height, width=image_width)
    else:
        raise NotImplementedError(f"Video format not supported: {video_path}")

    original_frames = []
    while True:
        y, uv = reader.read_one_frame(dst_format="420")  # type: ignore
        if y is None:
            break
        original_frames.append((y, uv))
    if frame_count is None:
        return original_frames

    # Repeat the frames forwards and backwards until we have enough frames
    res = []
    while len(res) < frame_count:
        res.extend(original_frames)
        res.extend(reversed(original_frames))
        res = res[:frame_count]
    return res


@dataclass(frozen=True)
class MlvcFrameHeader:
    q_index: int


def read_mlvc_bitstreams(file_path: Path | str) -> list[tuple[MlvcFrameHeader, bytes]]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError(f"MLVC data not found at {file_path}")

    res: list[tuple[MlvcFrameHeader, bytes]] = []
    with open(file_path, "rb") as f:
        while (frame_header_bytes := f.read(8)) != b"":
            if len(frame_header_bytes) != 8:
                raise ValueError("Failed to read frame header")
            q_index, payload_size = struct.unpack("<iI", frame_header_bytes)

            payload = f.read(payload_size)
            if len(payload) != payload_size:
                raise ValueError("Failed to read frame payload")
            res.append((MlvcFrameHeader(q_index=q_index), payload))

    return res


def save_mlvc_bitstreams(file_path: Path | str, frames: list[tuple[MlvcFrameHeader, bytes]]) -> None:
    with open(file_path, "wb") as f:
        for header, payload in frames:
            f.write(struct.pack("<iI", header.q_index, len(payload)))
            f.write(payload)


def calc_psnr(x1: np.ndarray, x2: np.ndarray) -> float:
    mse = (x1.astype(float) - x2.astype(float)) ** 2
    mse = mse.sum() / math.prod(mse.shape[2:])

    if not np.isfinite(mse):
        return -999.9
    if mse < 1e-10:
        return 999.9
    return -10 * np.log10(mse).item()


def parse_metrics_dataframe(metrics) -> pd.DataFrame:
    df_metrics = []
    for test_class, test_class_data in metrics.items():
        for sequence_name, sequence_data in test_class_data.items():
            for q_name, q_data in sequence_data.items():
                fps = q_data.get("fps", 30.0)
                frame_bpp = np.array(q_data.get("frame_bpp", []))
                frame_kbps = frame_bpp * q_data["frame_pixel_num"] * fps / 1000
                df_metrics.append(
                    {
                        "test_class": test_class,
                        "sequence_name": sequence_name,
                        "fps": fps,
                        "q_name": q_name,
                        "q_index": q_data.get("p_frame_q_index", -1),
                        "bpp": q_data["ave_all_frame_bpp"],
                        "kbps": q_data["frame_pixel_num"] * q_data["ave_all_frame_bpp"] * fps / 1000,
                        "psnr": q_data["ave_all_frame_psnr"],
                        "psnr_y": q_data["ave_all_frame_psnr_y"],
                        "psnr_u": q_data["ave_all_frame_psnr_u"],
                        "psnr_v": q_data["ave_all_frame_psnr_v"],
                        "frame_qp": q_data.get("frame_qp", []),
                        "frame_bpp": q_data.get("frame_bpp", []),
                        "frame_kbps": frame_kbps.tolist(),
                        "frame_psnr": q_data.get("frame_psnr", []),
                        "frame_psnr_y": q_data.get("frame_psnr_y", []),
                        "frame_psnr_u": q_data.get("frame_psnr_u", []),
                        "frame_psnr_v": q_data.get("frame_psnr_v", []),
                    }
                )
    df_metrics = pd.DataFrame(df_metrics)
    return df_metrics


def parse_metrics(
    data: dict[str, Any],
    only_common_clips: bool = True,
    filter_sequences: set[str] | list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    def _find_common_sequences(data) -> set[str]:
        res = None
        for df_metrics in data.values():
            sequences = df_metrics["sequence_name"].unique()
            res = set(sequences) if res is None else res.intersection(sequences)
        return res or set()

    res: dict[str, pd.DataFrame] = {}
    for name, value in data.items():
        if isinstance(value, (Path, str)):
            with open(value, "r") as f:
                metrics = json.load(f)
            if "params" in metrics and "results" in metrics:
                # This is a validation test result, convert it to the metrics format
                test_data = ValidationTestResults.from_dict(metrics)
                metrics = transform_validation_test_results(test_data)
        else:
            metrics = value
        res[name] = parse_metrics_dataframe(metrics)

    common_sequences = None
    if filter_sequences is not None:
        common_sequences = set(filter_sequences)
    elif only_common_clips:
        common_sequences = _find_common_sequences(res)

    if common_sequences is not None:
        print(f"Number of common sequences: {len(common_sequences)}")
        for name, df in res.items():
            res[name] = df.query("sequence_name in @common_sequences")
    return res


def transform_validation_test_results(test_data: ValidationTestResults) -> dict[str, Any]:
    output_data = {}
    for params, res in zip(test_data.params, test_data.results):
        video_name = Path(params.video_path).name
        test_class = Path(params.video_path).parent.name
        if test_class not in output_data:
            output_data[test_class] = {}
        if video_name not in output_data[test_class]:
            output_data[test_class][video_name] = {}

        if params.q_index is not None:
            q_name = f"q_index={params.q_index}"
        elif params.bitrate is not None:
            q_name = f"bitrate_kbps={params.bitrate / 1e3:.0f}"
        else:
            raise ValueError("Either q_index or bitrate must be provided in the test parameters")

        output_data[test_class][video_name][q_name] = {
            "ds_name": test_class,
            "video_path": video_name,
            "fps": params.fps,
            "frame_pixel_num": params.image_width * params.image_height,
            "i_frame_num": 1,
            "p_frame_num": res.psnr.count - 1,
            "ave_all_frame_bpp": res.bpp.mean,
            "ave_all_frame_psnr": res.psnr.mean,
            "ave_all_frame_psnr_y": res.psnr_y.mean,
            "ave_all_frame_psnr_u": res.psnr_u.mean,
            "ave_all_frame_psnr_v": res.psnr_v.mean,
            "frame_qp": res.q_index.values,
            "frame_bpp": res.bpp.values,
            "frame_psnr": res.psnr.values,
            "frame_psnr_y": res.psnr_y.values,
            "frame_psnr_u": res.psnr_u.values,
            "frame_psnr_v": res.psnr_v.values,
            "i_frame_q_index": params.q_index,
            "p_frame_q_index": params.q_index,
        }
    return output_data


def get_model_type_extension(model_type: ModelType) -> str:
    extensions = {
        ModelType.TORCH: "torch",
        ModelType.ONNX: "onnx",
        ModelType.COREML: "mlpackage",
        ModelType.OPENVINO: "xml",
    }
    if model_type not in extensions:
        raise ValueError(f"Unsupported model type: {model_type}")
    return extensions[model_type]


def _upsample_uv(img: np.ndarray, factor: int = 2) -> np.ndarray:
    res = np.repeat(np.repeat(img, factor, axis=-1), factor, axis=-2)
    return res


def _downsample_uv(img: np.ndarray, factor: int = 2) -> np.ndarray:
    height, width = img.shape[-2:]
    return (
        img.astype(np.float32, copy=False)
        .reshape(height // factor, factor, width // factor, factor)
        .mean(axis=(1, 3))
        .astype(img.dtype)
    )


def yuv_420_to_444(yuv420: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    y, uv = yuv420
    u = _upsample_uv(uv[0:1])
    v = _upsample_uv(uv[1:2])
    return np.stack([y, u, v], axis=1)


def yuv_444_to_420(yuv444: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert yuv444.shape[0] == 1
    y, u, v = yuv444[0]
    return (y, _downsample_uv(u), _downsample_uv(v))
