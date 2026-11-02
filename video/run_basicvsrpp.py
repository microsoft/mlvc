# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Apply BasicVSR++ restoration to image-sequence or raw-YUV datasets.

Image-sequence mode reads JPEG/PNG/WebP training manifests. YUV-test mode reads
benchmark test descriptions and writes aligned raw YUV420 targets. Both modes
use BasicVSR++ with is_low_res_input=False and preserve the input resolution.

Output is at the same resolution as input (no super-resolution).
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
import os
import time
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.models.restoration import basicvsrpp_arch
from src.models.restoration.basicvsrpp_inference import BasicVSRPPInference
from src.utils.app import BaseApp
from src.utils.distributed.operations import run_distributed
from src.utils.video_reader import YUVReader
from src.utils.video_writer import YUVWriter

logger = logging.getLogger(__name__)


# Supported image extensions and their PIL save formats.
_EXT_TO_FORMAT = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
}
# Canonical output extension per explicit format override.
_FORMAT_TO_EXT = {
    "jpeg": ".jpg",
    "png": ".png",
    "webp": ".webp",
}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_description_path(root: Path, *parts: str) -> Path:
    """Resolve description fields beneath an already-resolved dataset root."""
    for part in parts:
        # Recognize drive/UNC paths and both separator styles on either platform.
        description_path = PureWindowsPath(part)
        if description_path.anchor or ".." in description_path.parts:
            raise ValueError(f"Description paths must be relative and cannot contain '..': {part!r}")
    path = root.joinpath(*parts).resolve()
    if root not in path.parents:
        raise ValueError("Dataset paths must remain inside their respective roots")
    return path


def build_yuv_test_tasks(description: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(description.get("test_classes"), dict):
        raise ValueError("YUV test description is missing a 'test_classes' object")

    tasks = []
    for class_name, class_desc in description["test_classes"].items():
        if not class_desc.get("test", True):
            continue
        if class_desc.get("src_type") != "yuv420":
            raise ValueError(f"{class_name}: expected src_type yuv420, got {class_desc.get('src_type')}")
        base_path = class_desc["base_path"]
        for seq_name, seq_desc in class_desc["sequences"].items():
            tasks.append(
                {
                    "class_name": class_name,
                    "base_path": base_path,
                    "seq_name": seq_name,
                    "width": int(seq_desc["width"]),
                    "height": int(seq_desc["height"]),
                    "frames": int(seq_desc["frames"]),
                    "frame_rate": float(seq_desc.get("frame_rate", 30)),
                }
            )

    if not tasks:
        raise ValueError("No YUV test sequences selected")
    return tasks


def make_target_description(
    source_description: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    target_description = copy.deepcopy(source_description)
    target_description.pop("root_path", None)
    target_description["target_provenance"] = provenance
    return target_description


class RestoreVimeoBasicVSRPP(BaseApp):
    def __init__(self):
        super().__init__()
        self._device: torch.device | None = None
        self._processor: BasicVSRPPInference | None = None
        self._chunk_size = 32
        self._overlap = 0
        self._data_root: str | None = None
        self._output_root: str | None = None
        self._out_format = "auto"
        self._quality = 100
        self._webp_lossless = False
        self._source_config_path: str | None = None
        self._source_description: dict[str, Any] | None = None
        self._weights_path: str | None = None
        self._checkpoint_sha256: str | None = None

    def initialize(self, local_rank: int):
        super().initialize(local_rank)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        logger.info("Device: %s (cuda=%s)", device, torch.cuda.is_available())

        self._chunk_size = int(self.get_config_by_path("model.chunk_size", default=32))
        self._overlap = int(self.get_config_by_path("model.overlap", default=0))
        logger.info("chunk_size: %d, overlap: %d", self._chunk_size, self._overlap)

        weights_path = self.resolve_path(
            self.get_config_by_path("model.weights_path"),
            default_source="checkpoints_mount",
        )
        self._weights_path = weights_path
        mid_channels = int(self.get_config_by_path("model.mid_channels", default=64))
        num_blocks = int(self.get_config_by_path("model.num_blocks", default=15))

        logger.info("Loading BasicVSR++ weights from %s", weights_path)
        self._processor = BasicVSRPPInference.from_checkpoint(
            weights_path,
            device,
            mid_channels=mid_channels,
            num_blocks=num_blocks,
            chunk_size=self._chunk_size,
            overlap=self._overlap,
        )

        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated() / 1e6
            logger.info("VRAM after model load: %.1f MB", alloc)
        if self.rank <= 0:
            self._checkpoint_sha256 = sha256_file(weights_path)

    def _process_sequence(
        self,
        rgb_frames: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[float]]:
        assert self._processor is not None
        return self._processor.restore_sequence(rgb_frames)

    def _read_frames(self, src_dir: str, frame_names: list[str]) -> list[np.ndarray]:
        """Read image frames as list of 3×H×W float32 [0,1]."""
        frames = []
        for fname in frame_names:
            img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            frames.append(arr.transpose(2, 0, 1))  # CHW
        return frames

    def _resolve_output_name(self, src_fname: str) -> str:
        """Map an input frame filename to its output filename.

        auto: keep the original name (output format follows the input extension).
        explicit override: replace the extension with the chosen format's canonical
        extension so the encoded bytes and the filename always agree.
        """
        if self._out_format == "auto":
            return src_fname
        stem = os.path.splitext(src_fname)[0]
        return stem + _FORMAT_TO_EXT[self._out_format]

    def _write_frame(self, dst_path: str, frame_chw: np.ndarray) -> None:
        """Write a single CHW float32 [0,1] frame, encoding by output extension."""
        arr = (frame_chw.transpose(1, 2, 0) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        ext = os.path.splitext(dst_path)[1].lower()
        fmt = _EXT_TO_FORMAT.get(ext)
        if fmt is None:
            raise ValueError(f"Unsupported output extension {ext} for {dst_path}; supported: {sorted(_EXT_TO_FORMAT)}")

        if fmt == "JPEG":
            img.save(dst_path, "JPEG", quality=self._quality)
        elif fmt == "WEBP":
            if self._webp_lossless:
                img.save(dst_path, "WEBP", lossless=True)
            else:
                img.save(dst_path, "WEBP", quality=self._quality)
        else:  # PNG (lossless)
            img.save(dst_path, "PNG")

    def _process_image_task(self, task: dict) -> dict:
        """Process one image sequence: read → restore → write."""
        assert self._device is not None
        assert self._data_root is not None and self._output_root is not None
        seq_id = task["path"]
        frame_names = task["frames"]
        source_root = Path(self._data_root).resolve()
        output_root = Path(self._output_root).resolve()
        source_paths = [resolve_description_path(source_root, seq_id, name) for name in frame_names]
        output_paths = [
            resolve_description_path(output_root, seq_id, self._resolve_output_name(name)) for name in frame_names
        ]
        if set(source_paths) & set(output_paths):
            raise ValueError("Image source and output paths must differ")

        orig_frames = self._read_frames(str(source_root), [str(path.relative_to(source_root)) for path in source_paths])
        logger.info("  %s: read %d frames", seq_id, len(orig_frames))

        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        restored_frames, chunk_times = self._process_sequence(orig_frames)
        wall = time.perf_counter() - t0

        for output_path, rf in zip(output_paths, restored_frames):
            os.makedirs(output_path.parent, exist_ok=True)
            self._write_frame(str(output_path), rf)

        total_model_s = sum(chunk_times)
        result = {
            "seq": seq_id,
            "skipped": False,
            "frames": len(restored_frames),
            "chunks": len(chunk_times),
            "total_model_s": total_model_s,
            "wall_clock_s": wall,
            "fps_pure_model": len(restored_frames) / total_model_s if total_model_s > 0 else 0,
            "fps_with_io": len(restored_frames) / wall if wall > 0 else 0,
        }
        if self._device.type == "cuda":
            result["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1e6
        return result

    def _load_tasks(self) -> tuple[list[dict], str]:
        """Load image sequence task list from description_path.

        Returns (tasks, description_path).
        """
        desc_path_cfg = self.get_config_by_path("dataset.description_path")
        description_path = self.resolve_path(desc_path_cfg, default_source="data_mount")
        self._data_root = os.path.dirname(description_path)

        with open(description_path, "r") as f:
            manifest = json.load(f)

        if isinstance(manifest, list):
            tasks = manifest
        elif isinstance(manifest, dict):
            if "seqs" not in manifest or "frames" not in manifest:
                raise ValueError("Both 'seqs' and 'frames' keys required")
            frames = manifest["frames"]
            tasks = manifest["seqs"]
            for t in tasks:
                t["frames"] = frames
        else:
            raise ValueError("Invalid description.json format")

        return tasks, description_path

    def _run_image_sequences(self):
        tasks, description_path = self._load_tasks()

        self._out_format = str(self.get_config_by_path("output.format", default="auto")).lower()
        valid_formats = ("auto",) + tuple(_FORMAT_TO_EXT.keys())
        if self._out_format not in valid_formats:
            raise ValueError(f"output.format must be one of {valid_formats}, got {self._out_format!r}")
        self._quality = int(self.get_config_by_path("output.quality", default=100))
        self._webp_lossless = bool(self.get_config_by_path("output.webp_lossless", default=False))

        if self.rank <= 0:
            logger.info("Processing %d sequences", len(tasks))

        self._output_root = os.path.join(self.save_dir, "restored")
        os.makedirs(self._output_root, exist_ok=True)

        # Write resolved config for provenance
        if self.rank <= 0:
            with open(os.path.join(self.save_dir, "config_resolved.json"), "w") as f:
                json.dump(
                    {
                        "model": "BasicVSR++",
                        "chunk_size": self._chunk_size,
                        "overlap": self._overlap,
                        "output_format": self._out_format,
                        "quality": self._quality,
                        "webp_lossless": self._webp_lossless,
                        "total_sequences": len(tasks),
                        "device": str(self._device),
                        "data_root": self._data_root,
                    },
                    f,
                    indent=2,
                )

        start = time.perf_counter()
        results = run_distributed(
            self.rank,
            tasks,
            task_processor=self._process_image_task,
            progress_callback=lambda done, total, elapsed: logger.info(
                "rank %d: %d/%d sequences (%.1fs elapsed)",
                self.rank,
                done,
                total,
                elapsed,
            ),
            progress_interval=50,
        )
        elapsed = time.perf_counter() - start

        # Rank 0: write metadata and timing summary
        if self.rank <= 0:
            processed = [r for r in results if r and not r.get("skipped")]
            skipped = [r for r in results if r and r.get("skipped")]
            all_fps = [r["fps_pure_model"] for r in processed if r.get("fps_pure_model")]

            timing = {
                "model": "BasicVSR++",
                "device": str(self._device),
                "chunk_size": self._chunk_size,
                "overlap": self._overlap,
                "output_format": self._out_format,
                "quality": self._quality,
                "webp_lossless": self._webp_lossless,
                "total_sequences": len(tasks),
                "processed": len(processed),
                "skipped": len(skipped),
                "wall_clock_s": elapsed,
                "mean_fps_pure_model": float(np.mean(all_fps)) if all_fps else None,
                "sequences": {r["seq"]: r for r in results if r},
            }

            timing_path = os.path.join(self.save_dir, "timing.json")
            with open(timing_path, "w") as f:
                json.dump(timing, f, indent=2)

            logger.info(
                "Done. %d processed, %d skipped, %.1fs total, %.1f fps avg",
                len(processed),
                len(skipped),
                elapsed,
                float(np.mean(all_fps)) if all_fps else 0,
            )

        # Barrier to prevent NCCL teardown deadlocks
        if self.rank >= 0:
            torch.distributed.barrier()

    @staticmethod
    def _expected_yuv420_size(width: int, height: int, frames: int) -> int:
        return width * height * 3 // 2 * frames

    def _load_yuv_tasks(self) -> list[dict[str, Any]]:
        source_config = self.get_config_by_path("dataset.config")
        self._source_config_path = self.resolve_path(source_config, default_source="data_mount")
        self._data_root = self.resolve_path(self.get_config_by_path("dataset.data_path"))
        with open(self._source_config_path, "rt", encoding="utf-8") as file:
            description = json.load(file)
        self._source_description = description
        return build_yuv_test_tasks(description)

    def _read_yuv_frames(self, path: str, width: int, height: int, count: int) -> list[np.ndarray]:
        reader = YUVReader(path, width, height)
        frames = []
        try:
            for frame_index in range(count):
                frame = reader.read_one_frame(dst_format="rgb")
                if frame is None:
                    raise ValueError(f"{path}: ended before frame {frame_index} of {count}")
                frames.append(frame)
        finally:
            reader.close()
        return frames

    def _process_yuv_task(self, task: dict[str, Any]) -> dict[str, Any]:
        assert self._processor is not None
        assert self._data_root is not None and self._output_root is not None

        parts = (task["base_path"], task["seq_name"])
        source_root = Path(self._data_root).resolve()
        output_root = Path(self._output_root).resolve()
        source = resolve_description_path(source_root, *parts)
        output = resolve_description_path(output_root, *parts)
        if source == output:
            raise ValueError("YUV source and output paths must differ")
        source_path, output_path = str(source), str(output)
        expected_size = self._expected_yuv420_size(task["width"], task["height"], task["frames"])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        partial_path = output_path + ".partial"
        if os.path.exists(partial_path):
            os.remove(partial_path)

        frames = self._read_yuv_frames(source_path, task["width"], task["height"], task["frames"])
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        restored_frames, chunk_times = self._process_sequence(frames)
        elapsed = time.perf_counter() - start
        if len(restored_frames) != task["frames"]:
            raise RuntimeError(
                f"{task['class_name']}/{task['seq_name']}: restored {len(restored_frames)} "
                f"frames, expected {task['frames']}"
            )

        try:
            with YUVWriter(partial_path, task["width"], task["height"]) as writer:
                for frame in restored_frames:
                    writer.write_one_frame(rgb=frame, src_format="rgb")
            actual_size = os.path.getsize(partial_path)
            if actual_size != expected_size:
                raise RuntimeError(
                    f"{task['class_name']}/{task['seq_name']}: output size {actual_size}, expected {expected_size}"
                )
            os.replace(partial_path, output_path)
        finally:
            if os.path.exists(partial_path):
                os.remove(partial_path)

        model_seconds = sum(chunk_times)
        result = {
            **task,
            "skipped": False,
            "chunks": len(chunk_times),
            "model_seconds": model_seconds,
            "wall_seconds": elapsed,
            "model_fps": task["frames"] / model_seconds if model_seconds > 0 else None,
            "output_bytes": expected_size,
            "output_sha256": sha256_file(output_path),
        }
        if self._device is not None and self._device.type == "cuda":
            result["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1e6
        return result

    @staticmethod
    def _source_hashes() -> dict[str, str]:
        paths = [
            __file__,
            inspect.getsourcefile(BasicVSRPPInference),
            inspect.getsourcefile(basicvsrpp_arch.BasicVSRPlusPlusNet),
        ]
        return {os.path.basename(path): sha256_file(path) for path in paths if path is not None}

    def _run_yuv_testset(self):
        tasks = self._load_yuv_tasks()
        assert self._source_config_path is not None
        assert self._source_description is not None
        assert self._weights_path is not None

        self._output_root = os.path.join(self.save_dir, "restored")
        os.makedirs(self._output_root, exist_ok=True)
        logger.info("Processing %d YUV test sequences into %s", len(tasks), self._output_root)

        if self._checkpoint_sha256 is None:
            self._checkpoint_sha256 = sha256_file(self._weights_path)
        provenance = {
            "model": "BasicVSR++",
            "checkpoint": os.path.basename(self._weights_path),
            "checkpoint_sha256": self._checkpoint_sha256,
            "source_config": os.path.basename(self._source_config_path),
            "source_config_sha256": sha256_file(self._source_config_path),
            "chunk_size": self._chunk_size,
            "overlap": self._overlap,
            "source_hashes": self._source_hashes(),
        }
        manifest_path = os.path.join(self.save_dir, "restoration_manifest.json")

        start = time.perf_counter()
        results = run_distributed(
            self.rank,
            tasks,
            task_processor=self._process_yuv_task,
            progress_callback=lambda done, total, elapsed: logger.info(
                "rank %d: %d/%d sequences (%.1fs elapsed)",
                self.rank,
                done,
                total,
                elapsed,
            ),
            progress_interval=10,
            weight_func=lambda task: task["width"] * task["height"] * task["frames"],
        )
        elapsed = time.perf_counter() - start

        if self.rank <= 0:
            target_description = make_target_description(self._source_description, provenance)
            target_config_path = os.path.join(self._output_root, os.path.basename(self._source_config_path))
            with open(target_config_path, "wt", encoding="utf-8") as file:
                json.dump(target_description, file, indent=2)

            if any(result is None for result in results):
                raise RuntimeError("At least one YUV restoration task did not return a result")
            sequence_results = {
                f"{result['class_name']}/{result['seq_name']}": result for result in results if result is not None
            }
            summary = {
                "model": provenance,
                "source_config": self._source_config_path,
                "target_config": target_config_path,
                "total_sequences": len(tasks),
                "completed": len(sequence_results),
                "wall_seconds": elapsed,
                "sequences": sequence_results,
            }
            with open(manifest_path, "wt", encoding="utf-8") as file:
                json.dump(summary, file, indent=2)

        if self.rank >= 0:
            torch.distributed.barrier()

    def run(self):
        description_path = self.get_config_by_path("dataset.description_path", default=None)
        test_config = self.get_config_by_path("dataset.config", default=None)
        if bool(description_path) == bool(test_config):
            raise ValueError("Specify exactly one of dataset.description_path or dataset.config")
        if description_path:
            self._run_image_sequences()
        else:
            self._run_yuv_testset()


if __name__ == "__main__":
    RestoreVimeoBasicVSRPP().main()
