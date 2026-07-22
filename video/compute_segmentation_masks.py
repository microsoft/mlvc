# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import glob
import json
import os
from pathlib import Path, PurePosixPath
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
import logging
from time import time

from src.utils.app import BaseApp
from src.models.segment import SegmentationModel
from src.utils.distributed.operations import run_distributed
from src.utils.video_reader import YUVReader


def to_tensor(image):
    # Convert image to NumPy array
    np_image = np.array(image, dtype=np.float32) / 255.0
    # H x W x C to C x H x W
    np_image = np_image.transpose((2, 0, 1))
    # Convert to tensor
    return torch.from_numpy(np_image)


def resize_min_side(image, target=540):
    w, h = image.size
    m = min(w, h)
    if m == target:
        return image
    scale = target / m
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return image.resize((new_w, new_h), Image.Resampling.BILINEAR)


def pad_collate(batch):
    """
    Pads all image tensors in the batch to the same (Hmax,Wmax),
    returns (padded_images, paths, original_sizes).
    """
    imgs, paths, orig_sizes = zip(*batch)
    # compute max dims
    H = max(img.shape[1] for img in imgs)
    W = max(img.shape[2] for img in imgs)

    padded = []
    for img in imgs:
        _, h, w = img.shape
        # (left, right, top, bottom)
        pad = (0, W - w, 0, H - h)
        padded.append(torch.nn.functional.pad(img, pad))
    sizes = [(img.shape[1], img.shape[2], os[0], os[1]) for img, os in zip(imgs, orig_sizes)]
    return torch.stack(padded, 0), list(paths), sizes


class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        orig_w, orig_h = image.size
        image = resize_min_side(image, 540)
        image = to_tensor(image)
        return image, str(img_path), (orig_h, orig_w)


class CreateSegmentationMaskApp(BaseApp):
    def __init__(self):
        super().__init__()

        self.model = None
        self._device = None

    def initialize(self, local_rank: int):
        # First perform standard BaseApp initialization (distributed, flags, seeds)
        super().initialize(local_rank)

        # Then construct device and segmentation model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        has_dataset = self.get_config_by_path("dataset", default=None) is not None
        has_testset = self.get_config_by_path("testset", default=None) is not None
        if has_dataset and has_testset:
            raise ValueError(
                "Specify either 'dataset' (training PNG manifest) or 'testset' (YUV testset JSON), not both."
            )
        if not has_dataset and not has_testset:
            raise ValueError("Either 'dataset' or 'testset' configuration block is required.")

        self._mode = "dataset" if has_dataset else "testset"
        if self._mode == "dataset":
            self._dataset_batch_size = int(self.get_config_by_path("dataset.batch_size", expected_type=int))
            if self._dataset_batch_size != 1:
                raise ValueError("Only batch_size of 1 is supported.")
            dataset_rel_path = self.get_config_by_path("dataset.path", expected_type=str)
            self._dataset_path = self.resolve_path(dataset_rel_path)
        else:
            testset_config_rel = self.get_config_by_path("testset.config", expected_type=str)
            self._testset_config_path = self.resolve_path(testset_config_rel, default_source=".")
            data_path_rel = self.get_config_by_path("testset.data_path", default=None)
            self._testset_data_path = self.resolve_path(data_path_rel) if data_path_rel else None
            output_rel = self.get_config_by_path("testset.output_root", expected_type=str)
            self._testset_output_root = self.resolve_path(output_rel, default_source="save_dir")
            self._testset_resize_min_side = self.get_config_by_path("testset.resize_min_side", default=None)
            self._testset_max_n_frames = self.get_config_by_path("testset.max_n_frames", default=None)

        # Load segmentation model
        seg_path = self.resolve_path("pretrained/", default_source="checkpoints_mount")
        self.model = SegmentationModel(seg_path, config=self.get_config_by_path("segmentation_model"), device=device)
        self._device = device

    def run(self):
        if self._mode == "dataset":
            self._run_training_dataset()
        else:
            self._run_testset()

    def _run_training_dataset(self):
        data_manifest_path = self.resolve_path(self._dataset_path)
        data_root = Path(data_manifest_path).parent
        with open(data_manifest_path, "r") as f:
            data_manifest = json.load(f)
        logging.info(f"Found {len(data_manifest)} sequences in {data_manifest_path}")
        # Build per-sequence task list (each task is a list of image paths for that sequence)
        task_list = self._build_task_list(data_manifest, data_root)
        total_images = sum(len(t) for t in task_list)
        logging.info(f"Segmenting {total_images} images across {len(task_list)} tasks")

        # Run tasks (locally or distributed) with progress reporting
        start_time = time()
        results = run_distributed(
            self.rank,
            task_list,
            task_processor=lambda paths: self._process_image_paths(paths, data_root),
            progress_callback=lambda done, total, elapsed: self._log_progress(done, total, elapsed),
            progress_interval=100,
        )
        elapsed = time() - start_time
        processed = sum(r for r in results if r is not None)
        logging.info(f"Segmentation completed: {processed} images in {elapsed:.1f}s")

        # Synchronize all ranks
        if self.rank >= 0:
            torch.distributed.barrier()

        # Filter out sequences that had at least one failed frame
        if self.rank <= 0:
            self._filter_failed_sequences(data_manifest, data_manifest_path)

    def _run_testset(self):
        with open(self._testset_config_path, "r") as f:
            testset_desc = json.load(f)
        testset_root = (
            self._testset_data_path or testset_desc.get("root_path") or str(Path(self._testset_config_path).parent)
        )
        task_list = self._build_testset_task_list(testset_desc, testset_root)
        total_frames = sum(t["n_frames"] for t in task_list)
        logging.info(
            f"Segmenting {total_frames} frames across {len(task_list)} sequences "
            f"(output -> {self._testset_output_root})"
        )

        start_time = time()
        results = run_distributed(
            self.rank,
            task_list,
            task_processor=self._process_yuv_sequence,
            progress_callback=lambda done, total, elapsed: self._log_progress(done, total, elapsed),
            progress_interval=10,
        )
        elapsed = time() - start_time
        processed = sum(r for r in results if r is not None)
        logging.info(f"Testset mask generation completed: {processed} frames in {elapsed:.1f}s")

        if self.rank >= 0:
            torch.distributed.barrier()

    def _build_task_list(self, data_manifest, data_root: Path):
        """
        Build a list of tasks, where each task is a list of image paths for a single sequence.
        This keeps tasks reasonably sized and enables fair distribution.
        """
        task_list = []
        if isinstance(data_manifest, list):
            for seq in data_manifest:
                seq_path = seq["path"]
                img_indexes = seq["frames"]
                img_paths = [data_root / seq_path / img_index for img_index in img_indexes]
                task_list.append(img_paths)
        elif isinstance(data_manifest, dict):
            frames = data_manifest["frames"]
            for seq in data_manifest["seqs"]:
                seq_path = seq["path"]
                img_paths = [data_root / seq_path / img_index for img_index in frames]
                task_list.append(img_paths)
        else:
            raise ValueError("Unsupported data manifest format")
        return task_list

    def _build_testset_task_list(self, testset_desc, testset_root):
        """One task per YUV sequence in the testset JSON."""
        task_list = []
        max_n_frames = self._testset_max_n_frames
        for class_name, class_desc in testset_desc["test_classes"].items():
            if not class_desc.get("test", True):
                continue
            src_type = class_desc.get("src_type", "yuv420")
            if src_type != "yuv420":
                logging.warning(f"Skipping class {class_name}: src_type={src_type} (only yuv420 supported)")
                continue
            base_path = class_desc.get("base_path", "")
            for seq_name, seq_desc in class_desc["sequences"].items():
                n_frames = seq_desc["frames"]
                if max_n_frames is not None and max_n_frames > 0:
                    n_frames = min(n_frames, max_n_frames)
                task_list.append(
                    {
                        "class_name": class_name,
                        "base_path": base_path,
                        "seq_name": seq_name,
                        "seq_basename": os.path.splitext(seq_name)[0],
                        "yuv_path": str(Path(testset_root) / base_path / seq_name),
                        "width": seq_desc["width"],
                        "height": seq_desc["height"],
                        "n_frames": n_frames,
                    }
                )
        return task_list

    @staticmethod
    def _resize_min_side_tensor(x: torch.Tensor, target: int) -> torch.Tensor:
        """Resize a [B, C, H, W] tensor so min(H, W) == target."""
        _, _, h, w = x.shape
        m = min(h, w)
        if m == target:
            return x
        scale = target / m
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        return torch.nn.functional.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    def _process_yuv_sequence(self, task):
        """Generate per-frame masks for one YUV sequence. Returns number of frames processed."""
        yuv_path = task["yuv_path"]
        class_name = task["class_name"]
        base_path = task["base_path"]
        seq_basename = task["seq_basename"]
        width = task["width"]
        height = task["height"]
        n_frames = task["n_frames"]

        out_dir = Path(self._testset_output_root) / base_path / seq_basename
        out_dir.mkdir(parents=True, exist_ok=True)

        log_name = "failed_paths.txt" if self.rank < 0 else f"failed_paths.rank{self.rank}.txt"
        error_log_path = Path(self.save_dir, log_name)

        reader = YUVReader(yuv_path, width, height, src_format="420")
        processed = 0
        try:
            for frame_idx in range(n_frames):
                rgb_np = reader.read_one_frame(dst_format="rgb")
                if rgb_np is None:
                    logging.error(f"Premature EOF in {yuv_path} at frame {frame_idx}")
                    break
                try:
                    rgb = torch.from_numpy(rgb_np).unsqueeze(0).to(self._device)  # [1, 3, H, W]
                    assert self.model is not None
                    with torch.no_grad():
                        if self._testset_resize_min_side is not None:
                            inp = self._resize_min_side_tensor(rgb, int(self._testset_resize_min_side))
                            mask = self.model(inp, is_yuv420=False)
                            mask = torch.nn.functional.interpolate(mask, size=(height, width), mode="nearest")
                        else:
                            mask = self.model(rgb, is_yuv420=False)
                    arr = (mask[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255
                    Image.fromarray(arr).save(out_dir / f"{frame_idx:06d}_mask.png", format="PNG", optimize=True)
                    processed += 1
                except Exception as e:
                    logging.error(f"Error processing {yuv_path} frame {frame_idx}: {e}")
                    torch.cuda.empty_cache()
                    rel = f"{class_name}/{seq_basename}/{frame_idx:06d}"
                    with error_log_path.open("a") as f:
                        f.write(rel + "\n")
        finally:
            reader.close()

        return processed

    def _process_image_paths(self, image_paths, data_root: Path):
        """
        Process a single task: run segmentation on provided image paths.
        Returns the number of successfully processed images.
        """
        dataset = ImageDataset(image_paths)
        dataloader = DataLoader(
            dataset,
            batch_size=self._dataset_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=pad_collate,
        )
        # Per-rank error log to avoid concurrent writes to the same file in distributed runs
        log_name = "failed_paths.txt" if self.rank < 0 else f"failed_paths.rank{self.rank}.txt"
        error_log_path = Path(self.save_dir, log_name)
        # Pre-create masks root once to reduce per-image directory creation overhead
        masks_root = Path(self.save_dir, "masks")
        masks_root.mkdir(parents=True, exist_ok=True)
        processed = 0
        for batch, paths, sizes in dataloader:
            try:
                batch = batch.to(self._device)
                assert self.model is not None
                with torch.no_grad():
                    output = self.model(batch, is_yuv420=False).to("cpu")
            except Exception as e:
                logging.error(f"Error processing batch: {e}")
                torch.cuda.empty_cache()
                with error_log_path.open("a") as f:
                    for p in paths:
                        # log path relative to data_root using POSIX style for portability
                        rel = Path(p).relative_to(data_root).as_posix()
                        f.write(rel + "\n")
                continue
            finally:
                del batch

            if self._dataset_batch_size == 1:
                path = paths[0]
                resized_h, resized_w, orig_h, orig_w = sizes[0]
                cropped = output[:, :, :resized_h, :resized_w]
                up = torch.nn.functional.interpolate(cropped, size=(orig_h, orig_w), mode="nearest")[0]
                arr = up.numpy().astype(bool)[0].astype(np.uint8) * 255
                im = Image.fromarray(arr)
                # relative path for output naming consistency (as Path)
                rel = Path(path).relative_to(data_root)
                out_p = (masks_root / rel).with_name(rel.name + "_mask.png")
                out_p.parent.mkdir(parents=True, exist_ok=True)
                im.save(out_p, format="PNG", optimize=True)
                processed += 1
            else:
                # If ever enabling batch > 1, implement batched save here
                raise NotImplementedError("Batch size > 1 saving not implemented")

        return processed

    def _filter_failed_sequences(self, data_manifest, data_manifest_path):
        """
        Collect all failed-frame error logs from save_dir, derive the set of
        affected sequence paths, and write a filtered copy of the dataset
        description with those sequences removed.
        """
        error_logs = glob.glob(str(Path(self.save_dir, "failed_paths*.txt")))
        if not error_logs:
            logging.info("No failed-paths logs found; skipping description filtering.")
            return

        # Collect affected sequence paths from all rank logs
        failed_seqs: set[str] = set()
        for log_path in error_logs:
            with open(log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        failed_seqs.add(str(PurePosixPath(line).parent))

        if not failed_seqs:
            logging.info("Error logs are empty; skipping description filtering.")
            return

        logging.info(
            f"Found {len(failed_seqs)} sequence(s) with failed frames, removing from description: {sorted(failed_seqs)}"
        )

        # Filter depending on manifest format
        if isinstance(data_manifest, list):
            original_count = len(data_manifest)
            filtered = [seq for seq in data_manifest if seq["path"] not in failed_seqs]
        elif isinstance(data_manifest, dict):
            original_count = len(data_manifest["seqs"])
            filtered_seqs = [s for s in data_manifest["seqs"] if s["path"] not in failed_seqs]
            filtered = {**data_manifest, "seqs": filtered_seqs}
        else:
            logging.error("Unsupported manifest format; cannot filter.")
            return

        new_count = len(filtered) if isinstance(filtered, list) else len(filtered["seqs"])
        removed = original_count - new_count
        logging.info(f"Removed {removed} sequence(s) ({original_count} -> {new_count}).")

        # Write filtered description to save_dir
        manifest_name = Path(data_manifest_path).name
        stem = Path(manifest_name).stem
        suffix = Path(manifest_name).suffix
        out_path = Path(self.save_dir, stem + "_filtered" + suffix)
        with open(out_path, "w") as f:
            json.dump(filtered, f, indent=2)
        logging.info(f"Filtered description written to {out_path}")

    def _log_progress(self, done: int, total: int, elapsed: float):
        rank_prefix = ""
        if self.rank >= 0:
            rank_prefix = f"rank {self.rank}: "

        logging.info(f"{rank_prefix}sequences: {done}/{total} in {elapsed:.0f}s")


if __name__ == "__main__":
    CreateSegmentationMaskApp().main()
