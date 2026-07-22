# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
from pathlib import Path
import numpy as np
import cv2  # type: ignore[import-not-found]
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from PIL import Image
import logging
from time import time

from src.utils.app import BaseApp
from src.utils.common import AuxiliaryModels
from src.utils.distributed.operations import run_distributed
from src.utils.stream_helper import get_state_dict


def to_tensor(image):
    # Convert image to NumPy array
    np_image = np.array(image, dtype=np.float32) / 255.0
    # H x W x C to C x H x W
    np_image = np_image.transpose((2, 0, 1))
    # Convert to tensor
    return torch.from_numpy(np_image)


def compute_fft_hf_ratio(gray: np.ndarray, cutoff_ratio: float = 0.25) -> float:
    """Compute high-frequency energy ratio using FFT."""
    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(fft) ** 2
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    radius = cutoff_ratio * min(h, w) / 2
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    low_energy = magnitude[mask].sum()
    total_energy = magnitude.sum()
    return float((total_energy - low_energy) / (total_energy + 1e-8))


def compute_canny_edge_density(gray: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> float:
    """Compute edge pixel density using Canny."""
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return float(edges.sum() / 255 / edges.size)


def identity_collate(batch):
    # Keep items as-is to allow PIL Images without tensor stacking
    return batch


class ImageDataset(Dataset):
    def __init__(self, image_paths: list):
        # sort by filename to ensure stable order
        self.image_paths = sorted(image_paths, key=lambda p: Path(p).name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return image, str(img_path)


class FrameSequenceMetricApp(BaseApp):
    def __init__(self):
        super().__init__()
        self.deqa_model = None
        self._device = None
        self._auxiliary_models = None
        self._of_model = None
        self._of_quantiles = None
        # metric flags
        self._enable_deqa = True
        self._enable_optic_flow = True
        self._enable_texture = True
        self._fft_cutoff_ratio = 0.25
        self._texture_target_resolution = 256

    @property
    def auxiliary_models(self):
        if self._auxiliary_models is None:
            auxiliary = self.get_config_by_path("model.auxiliary", default=None)
            if auxiliary is not None:
                pretrained_path = self.resolve_path(auxiliary["path"], default_source="checkpoints_mount")
                auxiliary_model_config = auxiliary.get("config", {})
            else:
                pretrained_path = None
                auxiliary_model_config = {}
            self._auxiliary_models = AuxiliaryModels(pretrained_path, self._device, auxiliary_model_config)
        return self._auxiliary_models

    def initialize(self, local_rank: int):
        # First perform standard BaseApp initialization (distributed, flags, seeds)
        super().initialize(local_rank)
        # Use the auxiliary_models property defined on this class to obtain DeQA model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.rank <= 0:
            logging.info(f"Using device: {device}")
        self._device = device
        # Load dataset configuration
        self._dataset_batch_size = int(self.get_config_by_path("dataset.batch_size", expected_type=int))
        if self._dataset_batch_size != 1:
            raise ValueError("Only batch_size of 1 is supported.")
        dataset_rel_path = self.get_config_by_path("dataset.path", expected_type=str)
        self._dataset_path = self.resolve_path(dataset_rel_path)
        # Load metric flags from config
        self._enable_deqa = self.get_config_by_path("metrics.deqa", default=True)
        self._enable_optic_flow = self.get_config_by_path("metrics.optic_flow", default=True)
        self._enable_texture = self.get_config_by_path("metrics.texture_complexity", default=True)
        self._fft_cutoff_ratio = self.get_config_by_path("metrics.fft_cutoff_ratio", default=0.25)
        self._texture_target_resolution = self.get_config_by_path("metrics.texture_target_resolution", default=256)
        # Load models conditionally
        if self._enable_deqa:
            self.deqa_model = self.auxiliary_models.deqa_score_model
        # Attempt to load optic flow model (optional)
        if self._enable_optic_flow:
            try:
                self._load_optic_flow_model()
            except Exception as e:
                if self.rank <= 0:
                    logging.warning(f"Optic flow disabled: {e}")

    # --- Optic flow helpers ---
    def _load_optic_flow_model(self):
        type_full_name = self.get_config_by_path("model.optic_flow.type", expected_type=str, default=None)
        if type_full_name is None:
            raise ValueError("model.optic_flow.type missing")
        params = self.get_config_by_path("model.optic_flow.params", expected_type=dict, default=dict())
        ckpt_path = self.get_config_by_path("model.optic_flow.ckpt_path", expected_type=str, default=None)
        if ckpt_path is None:
            raise ValueError("model.optic_flow.ckpt_path missing")
        ckpt_path = self.resolve_path(ckpt_path, default_source="checkpoints_mount")
        module_name, sep, type_name = type_full_name.rpartition(".")
        if not sep or not module_name:
            raise ValueError(f"Invalid optic flow type: {type_full_name}")
        from importlib import import_module

        module = import_module(module_name)
        model_type = getattr(module, type_name, None)
        if not isinstance(model_type, type) or not issubclass(model_type, nn.Module):
            raise ValueError(f"Optic flow model class invalid: {type_full_name}")
        model = model_type(**params)
        state = get_state_dict(ckpt_path)
        prefix = self.get_config_by_path("model.optic_flow.ckpt_module_prefix", expected_type=str, default="")
        if prefix:
            if not prefix.endswith("."):
                prefix += "."

            def remove_prefix(n):
                if len(n) <= len(prefix) or not n.startswith(prefix):
                    raise ValueError(f"Unexpected key in state dict: {n}")
                return n[len(prefix) :]

            state = {remove_prefix(n): v for n, v in state.items()}
        model.load_state_dict(state)
        model.to(self._device)
        model.eval()
        self._of_model = model
        quantiles = self.get_config_by_path("model.optic_flow.quantiles", expected_type=list, default=[])
        if len(quantiles) == 0:
            quantiles = list(range(10, 100, 10))
        self._of_quantiles = quantiles
        if self.rank <= 0:
            logging.info(f"Optic flow model loaded from {ckpt_path}; quantiles={quantiles}")

    def _process_optic_flow(self, frame1: torch.Tensor, frame2: torch.Tensor):
        # Pad to multiples of 8 like reference implementation
        def calc_pad(x, g):
            r = x % g
            return g - r if r > 0 else 0

        pad_r = calc_pad(frame1.size(-1), 8)
        pad_b = calc_pad(frame1.size(-2), 8)
        if pad_r or pad_b:
            frame1 = nn.functional.pad(frame1, (0, pad_r, 0, pad_b), "replicate")
            frame2 = nn.functional.pad(frame2, (0, pad_r, 0, pad_b), "replicate")
        assert self._of_model is not None
        with torch.inference_mode():
            d: torch.Tensor = self._of_model(frame1, frame2)
        if pad_r or pad_b:
            d = d[..., : d.size(-2) - pad_b, : d.size(-1) - pad_r]
        d = (d**2).sum(dim=1).sqrt().flatten(1)
        var, mean = torch.var_mean(d, dim=-1)
        std = torch.sqrt(var)
        metrics = {"d_mean": mean.squeeze(0), "d_std": std.squeeze(0)}
        # quantiles
        assert self._of_quantiles is not None
        q_tensor = torch.as_tensor(self._of_quantiles, dtype=torch.float, device=d.device).mul_(0.01)
        q_values = d.quantile(q_tensor, dim=-1).squeeze(0).cpu()
        for q, qv in zip(self._of_quantiles, q_values):
            name = "d_" + ("{:.4f}".format(q).rstrip("0").rstrip(".").replace(".", ""))
            metrics[name] = qv
        return metrics

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

    def _process_sequence(self, image_paths, data_root: Path):
        """
        Process a single sequence: compute DeQA score for each frame.
        Returns an object {sequence: rel_seq_path_posix, scores: [float,...]}.
        """
        dataset = ImageDataset(image_paths)
        dataloader = DataLoader(
            dataset,
            batch_size=self._dataset_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=identity_collate,
        )
        # Per-rank error log to avoid concurrent writes to the same file in distributed runs
        log_name = "failed_paths.txt" if self.rank < 0 else f"failed_paths.rank{self.rank}.txt"
        error_log_path = Path(self.save_dir, log_name)

        scores = []
        seq_rel_path = None
        # optic flow accumulators (N-1 pairs)
        of_d_mean = []
        of_d_std = []
        of_quantiles = dict()
        prev_tensor = None
        # texture complexity accumulators
        fft_hf_ratios = []
        canny_densities = []
        for batch in dataloader:
            # batch is a list of single item: (PIL.Image, path)
            assert len(batch) == 1, "Only batch_size==1 supported"
            pil_image, img_path = batch[0]
            if self._enable_deqa:
                assert self.deqa_model is not None
                try:
                    # DeQA model expects a list of PIL images and returns per-frame scores (tensors)
                    with torch.inference_mode():
                        score_list = self.deqa_model([pil_image])
                    # Convert first score to float
                    score = float(score_list[0].cpu())
                except Exception as e:
                    logging.info(f"Error scoring frame {img_path}: {e}")
                    torch.cuda.empty_cache()
                    with error_log_path.open("a") as f:
                        rel = Path(img_path).relative_to(data_root).as_posix()
                        f.write(rel + "\n")
                    score = None

                if score is not None:
                    scores.append(score)

            if seq_rel_path is None:
                # parent folder of the image, relative to data_root
                seq_rel_path = Path(img_path).parent.relative_to(data_root).as_posix()

            # Texture complexity metrics (FFT + Canny)
            if self._enable_texture:
                try:
                    gray = np.array(pil_image.convert("L"), dtype=np.uint8)
                    # Resize to target resolution for fair comparison across datasets
                    if self._texture_target_resolution is not None:
                        target = self._texture_target_resolution
                        h, w = gray.shape
                        scale = target / min(h, w)
                        new_h, new_w = int(h * scale), int(w * scale)
                        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    fft_hf_ratios.append(compute_fft_hf_ratio(gray.astype(np.float32), self._fft_cutoff_ratio))
                    canny_densities.append(compute_canny_edge_density(gray))
                except Exception as e:
                    logging.error(f"Texture metrics failed for {img_path}: {e}")

            # Optic flow (requires previous frame tensor and loaded model)
            if self._of_model is not None:
                try:
                    cur_tensor = to_tensor(pil_image).unsqueeze(0).to(self._device, non_blocking=True)
                    if prev_tensor is not None:
                        metrics = self._process_optic_flow(prev_tensor, cur_tensor)
                        of_d_mean.append(float(metrics["d_mean"].cpu().numpy()))
                        of_d_std.append(float(metrics["d_std"].cpu().numpy()))
                        for k, v in metrics.items():
                            if k.startswith("d_") and k not in ("d_mean", "d_std"):
                                of_quantiles.setdefault(k, []).append(float(v.cpu().numpy()))
                    prev_tensor = cur_tensor
                except Exception as e:
                    logging.error(f"Optic flow failed for frame {img_path}: {e}")
                    torch.cuda.empty_cache()
        result = {"sequence": seq_rel_path or "", "scores": scores}
        # Save per-sequence results immediately
        base_dir = Path(self.save_dir)
        # Save DeQA scores
        if self._enable_deqa and scores:
            deqa_dir = base_dir / "deqa_scores"
            deqa_path = deqa_dir / f"{(seq_rel_path or 'sequence').replace('/', os.sep)}.json"
            deqa_path.parent.mkdir(parents=True, exist_ok=True)
            with open(deqa_path, "w", encoding="utf-8") as f:
                json.dump({"sequence": result["sequence"], "scores": result["scores"]}, f, ensure_ascii=False)

        # Save optic flow metrics if available
        if self._of_model is not None:
            of_dir = base_dir / "optical_flow"
            of_path = of_dir / f"{(seq_rel_path or 'sequence').replace('/', os.sep)}.json"
            of_path.parent.mkdir(parents=True, exist_ok=True)
            of_payload = {
                "sequence": result["sequence"],
                "metrics": {"d_mean": of_d_mean, "d_std": of_d_std, **of_quantiles},
            }
            with open(of_path, "w", encoding="utf-8") as f:
                json.dump(of_payload, f, ensure_ascii=False)

        # Save texture complexity metrics
        if self._enable_texture and (fft_hf_ratios or canny_densities):
            tex_dir = base_dir / "texture_complexity"
            tex_path = tex_dir / f"{(seq_rel_path or 'sequence').replace('/', os.sep)}.json"
            tex_path.parent.mkdir(parents=True, exist_ok=True)
            tex_payload = {
                "sequence": result["sequence"],
                "fft_hf_ratio": fft_hf_ratios,
                "canny_density": canny_densities,
            }
            with open(tex_path, "w", encoding="utf-8") as f:
                json.dump(tex_payload, f, ensure_ascii=False)

        # Return only compact summary to avoid high memory usage in distributed aggregator
        return {
            "sequence": result["sequence"],
            "n_scores": len(scores),
            "n_of_pairs": len(of_d_mean) if self._of_model is not None else 0,
            "n_texture": len(fft_hf_ratios),
        }

    def run(self):
        data_manifest_path = self.resolve_path(self._dataset_path)
        data_root = Path(data_manifest_path).parent
        with open(data_manifest_path, "r") as f:
            data_manifest = json.load(f)
        if self.rank <= 0:
            logging.info(f"Found {len(data_manifest)} sequences in {data_manifest_path}")
        # Build per-sequence task list (each task is a list of image paths for that sequence)
        task_list = self._build_task_list(data_manifest, data_root)
        total_images = sum(len(t) for t in task_list)
        if self.rank <= 0:
            logging.info(f"Scoring {total_images} images across {len(task_list)} sequences")

        # Run tasks (locally or distributed) with progress reporting
        start_time = time()
        results = run_distributed(
            self.rank,
            task_list,
            task_processor=lambda paths: self._process_sequence(paths, data_root),
            progress_callback=lambda done, total, elapsed: self._log_progress(done, total, elapsed),
            progress_interval=100,
        )
        elapsed = time() - start_time
        if self.rank <= 0:
            processed_deqa = sum(r.get("n_scores", 0) for r in results if isinstance(r, dict))
            processed_of = sum(r.get("n_of_pairs", 0) for r in results if isinstance(r, dict))
            logging.info(f"Number of sequences processed: {len(results)}")
            logging.info(f"Per-sequence saving completed in {elapsed:.1f}s")
            logging.info(f"Total frames scored: {processed_deqa}")
            logging.info(f"Total optic flow pairs processed: {processed_of}")

        # Synchronize all ranks before exiting to prevent NCCL teardown deadlocks
        if self.rank >= 0:
            torch.distributed.barrier()

    def _log_progress(self, done: int, total: int, elapsed: float):
        rank_prefix = ""
        if self.rank >= 0:
            rank_prefix = f"rank {self.rank}: "

        logging.info(f"{rank_prefix}sequences: {done}/{total} in {elapsed:.0f}s")


if __name__ == "__main__":
    FrameSequenceMetricApp().main()
