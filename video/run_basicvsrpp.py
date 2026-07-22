# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Apply BasicVSR++ denoising restoration to training data.

Reads image sequences (JPEG/PNG/WebP, description.json), restores through
BasicVSR++ (denoising mode, is_low_res_input=False), and writes restored
data to save_dir.

Output is at the same resolution as input (no super-resolution).
"""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import torch
from PIL import Image

from src.models.restoration.basicvsrpp_arch import load_basicvsrpp
from src.utils.app import BaseApp
from src.utils.distributed.operations import run_distributed

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
_IMAGE_EXTS = tuple(_EXT_TO_FORMAT.keys())


class RestoreVimeoBasicVSRPP(BaseApp):
    def __init__(self):
        super().__init__()
        self._device: torch.device | None = None
        self._model: torch.nn.Module | None = None
        self._chunk_size = 32
        self._overlap = 0
        self._data_root: str | None = None
        self._output_root: str | None = None
        self._out_format = "auto"
        self._quality = 100
        self._webp_lossless = False

    def initialize(self, local_rank: int):
        super().initialize(local_rank)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s (cuda=%s)", self._device, torch.cuda.is_available())

        self._chunk_size = int(self.get_config_by_path("model.chunk_size", default=32))
        self._overlap = int(self.get_config_by_path("model.overlap", default=0))
        logger.info("chunk_size: %d, overlap: %d", self._chunk_size, self._overlap)

        # Validate chunking config: the multi-chunk loop advances by
        # (chunk_size - 2 * overlap), which must be positive or it never terminates.
        if self._chunk_size <= 0:
            raise ValueError(f"model.chunk_size must be > 0, got {self._chunk_size}")
        if self._overlap < 0:
            raise ValueError(f"model.overlap must be >= 0, got {self._overlap}")
        if self._chunk_size - 2 * self._overlap <= 0:
            raise ValueError(
                "Invalid chunking configuration: model.chunk_size - 2 * model.overlap "
                f"must be > 0, got chunk_size={self._chunk_size}, overlap={self._overlap}"
            )

        weights_path = self.resolve_path(
            self.get_config_by_path("model.weights_path"),
            default_source="checkpoints_mount",
        )
        mid_channels = int(self.get_config_by_path("model.mid_channels", default=64))
        num_blocks = int(self.get_config_by_path("model.num_blocks", default=15))

        logger.info("Loading BasicVSR++ weights from %s", weights_path)
        self._model = load_basicvsrpp(
            weights_path,
            map_location="cpu",
            mid_channels=mid_channels,
            num_blocks=num_blocks,
            is_low_res_input=False,
        )
        self._model = self._model.to(self._device)
        for p in self._model.parameters():
            p.requires_grad_(False)

        if self._device.type == "cuda":
            alloc = torch.cuda.memory_allocated() / 1e6
            logger.info("VRAM after model load: %.1f MB", alloc)

    @torch.inference_mode()
    def _restore_chunk(
        self,
        rgb_frames: list[np.ndarray],
    ) -> tuple[list[np.ndarray], float]:
        """Restore a chunk of RGB frames.

        Args:
            rgb_frames: list of 3×H×W float32 arrays in [0,1].

        Returns:
            (restored_frames, elapsed_seconds)
        """
        assert self._device is not None and self._model is not None
        x = torch.stack([torch.from_numpy(f) for f in rgb_frames], dim=0)
        x = x.unsqueeze(0).to(self._device, non_blocking=True)

        if self._device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        y = self._model(x)

        if self._device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        y = y.clamp_(0.0, 1.0).squeeze(0).cpu().numpy().astype(np.float32)
        return [y[i] for i in range(y.shape[0])], elapsed

    def _process_sequence(
        self,
        rgb_frames: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[float]]:
        """Process a full sequence with overlapping chunks."""
        n_frames = len(rgb_frames)
        chunk_size = self._chunk_size
        overlap = self._overlap
        usable = chunk_size - 2 * overlap

        if n_frames <= chunk_size:
            restored, elapsed = self._restore_chunk(rgb_frames)
            return restored, [elapsed]

        restored_buf: list[np.ndarray | None] = [None] * n_frames
        chunk_times: list[float] = []

        i = 0
        while i < n_frames:
            chunk_start = max(0, i - overlap)
            chunk_end = min(n_frames, i + usable + overlap)
            chunk = rgb_frames[chunk_start:chunk_end]

            out_frames, elapsed = self._restore_chunk(chunk)
            chunk_times.append(elapsed)

            halo_left = i - chunk_start
            halo_right = chunk_end - min(i + usable, n_frames)
            keep_start = halo_left
            keep_end = len(out_frames) - halo_right if halo_right > 0 else len(out_frames)

            for j in range(keep_start, keep_end):
                frame_idx = i + (j - keep_start)
                if frame_idx < n_frames:
                    restored_buf[frame_idx] = out_frames[j]

            i += usable

        for idx in range(n_frames):
            if restored_buf[idx] is None:
                logger.warning("Frame %d was not restored; using original", idx)
                restored_buf[idx] = rgb_frames[idx]

        # Every slot is filled above, so the comprehension yields a list[np.ndarray].
        restored = [f for f in restored_buf if f is not None]
        return restored, chunk_times

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

    def _process_task(self, task: dict) -> dict:
        """Process one image sequence: read → restore → write."""
        assert self._device is not None
        assert self._data_root is not None and self._output_root is not None
        seq_id = task["path"]
        frame_names = task["frames"]
        expected_count = len(frame_names)
        src_dir = os.path.join(self._data_root, seq_id)
        dst_dir = os.path.join(self._output_root, seq_id)

        # Skip-existing
        if os.path.isdir(dst_dir):
            existing = [f for f in os.listdir(dst_dir) if f.lower().endswith(_IMAGE_EXTS)]
            if len(existing) >= expected_count:
                logger.info("Skipping %s — already has %d files", seq_id, len(existing))
                return {"seq": seq_id, "skipped": True, "frames": expected_count}

        os.makedirs(dst_dir, exist_ok=True)

        orig_frames = self._read_frames(src_dir, frame_names)
        logger.info("  %s: read %d frames", seq_id, len(orig_frames))

        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        restored_frames, chunk_times = self._process_sequence(orig_frames)
        wall = time.perf_counter() - t0

        for fname, rf in zip(frame_names, restored_frames):
            out_name = self._resolve_output_name(fname)
            self._write_frame(os.path.join(dst_dir, out_name), rf)

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

    def run(self):
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
            task_processor=self._process_task,
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


if __name__ == "__main__":
    RestoreVimeoBasicVSRPP().main()
