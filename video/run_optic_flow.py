# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import itertools
import json
import logging
import numbers
import os
import sys
import tempfile
import time
from datetime import timedelta
from functools import cached_property

import torch
import torch.distributed
import torch.nn as nn
import torch.utils.data

from src.datasets.ffmpeg_dataset import FfmpegDataset
from src.datasets.iterable_datasets import BatchBuilderDataset, FramePairDataset, NamedChunkDataset
from src.transforms.functional import ycbcr2rgb
from src.utils.app import BaseApp
from src.utils.stream_helper import get_state_dict


class OpticFlowApp(BaseApp):
    def __init__(self):
        super().__init__()
        self._wait_store = None
        self._last_progress_time: float = 0.0
        self._device = None
        self._model = None
        self._clip_metrics = dict()
        self._clip_meta = dict()

    @property
    def device(self):
        device = self._device
        if device is None:
            is_cuda_available = torch.cuda.is_available()
            logging.info(f"cuda: is_available() = {is_cuda_available}, device_count = {torch.cuda.device_count()}")
            self._device = device = torch.device("cuda" if is_cuda_available else "cpu")
        return device

    @property
    def model(self):
        model = self._model
        if model is None:
            self._model = model = self._load_model()

        return model

    @cached_property
    def clip_folder(self):
        clip_folder = self.get_config_by_path("optic_flow.dataset.clip_folder", expected_type=str)
        clip_folder = self.resolve_path(clip_folder, default_source="data_mount")
        return clip_folder

    @cached_property
    def skipped_metrics(self):
        skipped_metrics = self.get_config_by_path("optic_flow.skipped_metrics", default=None)
        if skipped_metrics is None:
            return set()

        if not isinstance(skipped_metrics, list):
            raise ValueError("skipped metrics must be a list")

        return set(skipped_metrics)

    @cached_property
    def merge_metrics_from(self):
        merge_metrics_from = self.get_config_by_path("optic_flow.merge_metrics_from", expected_type=str, default="")
        merge_metrics_from = self.resolve_path(merge_metrics_from, default_source="checkpoints_mount")
        return merge_metrics_from or None

    @cached_property
    def output_path(self):
        output_path = self.save_dir
        relative_path = self.get_config_by_path("optic_flow.output_path", default=None)
        if relative_path:
            output_path = os.path.join(output_path, relative_path)

        return output_path

    @cached_property
    def black_threshold(self):
        return self.get_config_by_path("optic_flow.black_threshold", expected_type=numbers.Real, default=0.01)

    @cached_property
    def quantiles(self):
        quantiles = self.get_config_by_path("optic_flow.quantiles", expected_type=list, default=[])
        if len(quantiles) == 0:
            quantiles = list(range(10, 100, 10))

        return quantiles

    def _init_wait_store(self):
        """
        Initialize separate distributed store to wait for all GPUs
        """
        if self.rank < 0:
            return

        # we create a separate store as:
        # - NCCL default timeout is too short
        # - there is no public API to get torch.distributed store
        if self.rank == 0:
            folder = "/dev/shm" if sys.platform.startswith("linux") else None
            fd, filename = tempfile.mkstemp(prefix="wait_store", dir=folder)
            os.close(fd)
            filename_list = [filename]
        else:
            filename_list = [""]

        torch.distributed.broadcast_object_list(filename_list)

        self._wait_store = torch.distributed.FileStore(filename_list[0], self.world_size)  # pyright: ignore[reportPrivateImportUsage]

    def _wait_all(self):
        if self.rank < 0:
            return

        assert self._wait_store is not None

        self._wait_store.set(f"ready{self.rank}", "1")
        self._wait_store.wait([f"ready{idx}" for idx in range(self.world_size)], timedelta(days=1))

    def _load_model(self):
        type_full_name: str = self.get_config_by_path("model.optic_flow.type", expected_type=str)
        params = self.get_config_by_path("model.optic_flow.params", expected_type=dict, default=dict())

        checkpoint_path = self.get_config_by_path("model.optic_flow.ckpt_path", expected_type=str)
        checkpoint_path = self.resolve_path(checkpoint_path, default_source="checkpoints_mount")

        module_name, sep, type_name = type_full_name.rpartition(".")
        if sep is None or not module_name:
            raise ValueError(f"Invalid optic_flow model type name: {type_full_name}")

        from importlib import import_module

        module = import_module(module_name)

        model_type = getattr(module, type_name, None)
        if not isinstance(model_type, type):
            raise ValueError(f"Model type {type_name} is not found in module {module_name}")
        if not issubclass(model_type, nn.Module):
            raise ValueError(f"Model type {type_full_name} is not nn.Module")

        model = model_type(**params)

        model_state = get_state_dict(checkpoint_path)
        module_prefix = self.get_config_by_path("model.optic_flow.ckpt_module_prefix", expected_type=str, default="")
        if module_prefix:
            if not module_prefix.endswith("."):
                module_prefix += "."

            def remove_prefix(n):
                if len(n) <= len(module_prefix) or not n.startswith(module_prefix):
                    raise ValueError(f"Unexpected key in model state: {n}")
                return n[len(module_prefix) :]

            model_state = {remove_prefix(n): v for n, v in model_state.items()}

        logging.info(f"pretrained weights loaded from {checkpoint_path}")
        model.load_state_dict(model_state)
        model.to(self.device)
        model.eval()
        return model

    def _load_clip_list(self):
        clip_list_fn = self.resolve_path(
            self.get_config_by_path("optic_flow.dataset.clip_meta"), default_source=None, expand_cwd=True
        )
        if not os.path.isabs(clip_list_fn):
            clip_list_fn = os.path.join(self.clip_folder, clip_list_fn)

        with open(clip_list_fn, "rt", encoding="utf-8") as f:
            clip_list = json.load(f)

        if not isinstance(clip_list, list):
            raise ValueError(f"Invalid clip list: {clip_list_fn}")

        def sortkey(meta):
            return -meta["width"], -meta["height"], -(meta.get("n_frames") or 0), meta["filename"]

        clip_list.sort(key=sortkey)
        return clip_list

    def _create_dataset(self, clip_list):
        batch_tensor_size = self.get_config_by_path(
            "optic_flow.dataset.batch_tensor_size", expected_type=int, default=1
        )
        clip_list = (
            (clip_meta["filename"], self._make_clip_dataset(clip_meta, batch_tensor_size)) for clip_meta in clip_list
        )
        return NamedChunkDataset(clip_list)

    def _make_clip_dataset(self, clip_meta, batch_tensor_size):
        filename = os.path.join(self.clip_folder, clip_meta["filename"])
        ds = FfmpegDataset(filename, clip_meta["width"], clip_meta["height"], ffmpeg_error_mode="warn_if_nonempty")
        ds = FramePairDataset(ds)
        ds = BatchBuilderDataset(ds, batch_tensor_size=batch_tensor_size, batch_size_step=8)
        return ds

    def _process_batch(self, clip_name: str, batch: torch.Tensor):
        frame1, frame2 = batch
        frame1 = frame1.to(self.device, non_blocking=True)
        frame2 = frame2.to(self.device, non_blocking=True)

        metrics = dict()
        if "optic_flow" not in self.skipped_metrics:
            metrics["optic_flow"] = self._process_optic_flow(frame1, frame2)

        if "bbox" not in self.skipped_metrics:
            metrics["bbox"] = self._find_bbox(frame1[:, 0])

        if "luma" not in self.skipped_metrics:
            metrics["luma"] = self._process_luma(frame1)

        self._add_metrics(clip_name, frame1.size(0), metrics)

    def _process_optic_flow(self, frame1, frame2):
        metrics = dict()

        def calc_pad(x, g):
            r = x % g
            return g - r if r > 0 else 0

        pad_r = calc_pad(frame1.size(-1), 8)
        pad_b = calc_pad(frame1.size(-2), 8)

        if pad_r or pad_b:
            frame1 = nn.functional.pad(frame1, (0, pad_r, 0, pad_b), "replicate")
            frame2 = nn.functional.pad(frame2, (0, pad_r, 0, pad_b), "replicate")

        d: torch.Tensor = self.model(frame1, frame2)
        if pad_r or pad_b:
            d = d[..., : d.size(-2) - pad_b, : d.size(-1) - pad_r]

        d = (d**2).sum(dim=1).sqrt().flatten(1)

        std, mean = torch.var_mean(d, dim=-1)
        std = torch.sqrt(std)

        metrics["d_mean"] = mean
        metrics["d_std"] = std
        self._calc_quantiles(metrics, "d_", d, self.quantiles)

        return metrics

    @staticmethod
    def _calc_quantiles(metrics, prefix, x, q):
        qv = torch.as_tensor(q).to(dtype=torch.float, device=x.device, non_blocking=True).mul_(0.01)
        qv = x.quantile(qv, dim=-1)
        qv = qv.cpu()

        for q, qv in zip(q, qv):
            n = prefix + "{:.4f}".format(q).rstrip("0").rstrip(".").replace(".", "")
            metrics[n] = qv

    def _find_bbox(self, frame: torch.Tensor):
        rows = torch.greater_equal(frame.max(dim=-1)[0], self.black_threshold)
        cols = torch.greater_equal(frame.max(dim=-2)[0], self.black_threshold)

        is_not_black, top = rows.float().max(dim=-1)
        bottom = rows.size(-1) - torch.flip(rows, dims=(-1,)).float().argmax(-1)
        bottom = torch.where(torch.not_equal(is_not_black, 0), bottom, top)

        is_not_black, left = cols.float().max(dim=-1)
        right = cols.size(-1) - torch.flip(cols, dims=(-1,)).float().argmax(-1)
        right = torch.where(torch.not_equal(is_not_black, 0), right, left)

        return dict(
            top=top,
            bottom=bottom,
            left=left,
            right=right,
        )

    def _process_luma(self, frame):
        y = frame[:, 0]
        rgb = ycbcr2rgb(frame)
        b = (rgb**2).sum(dim=1).sqrt()

        metrics = dict()
        self._calc_luma_metrics(metrics, "y_", y.flatten(1), self.quantiles)
        self._calc_luma_metrics(metrics, "b_", b.flatten(1), self.quantiles)

        # RGB dispersion
        rgb = rgb.flatten(2)
        d = rgb - rgb.mean(dim=-1, keepdim=True)
        d = (d * d).sum(dim=1).mean(dim=-1).sqrt()
        metrics["rgb_std"] = d

        return metrics

    def _calc_luma_metrics(self, metrics, prefix, x, q):
        metrics[f"{prefix}max"] = x.max(dim=-1)[0]

        std, mean = torch.var_mean(x, dim=-1)
        std = torch.sqrt(std)
        metrics[f"{prefix}mean"] = mean
        metrics[f"{prefix}std"] = std

        self._calc_quantiles(metrics, prefix, x, q)

    def _add_metrics(self, clip_name, n_frames, metrics):
        clip_metrics = self._clip_metrics.get(clip_name)
        if clip_metrics is None:
            self._clip_metrics[clip_name] = clip_metrics = self._init_clip_metrics(clip_name)
            clip_metrics["n_frames"] = 1

            def make_skeleton(d, m):
                for n, v in m.items():
                    if isinstance(v, dict):
                        d[n] = s = dict()
                        make_skeleton(s, v)
                    else:
                        d[n] = list()

            make_skeleton(clip_metrics, metrics)

        clip_metrics["n_frames"] += n_frames

        def add(d, m):
            for n, v in m.items():
                s = d[n]
                if isinstance(v, dict):
                    add(s, v)
                else:
                    s.extend(v.detach().cpu().numpy().tolist())

        add(clip_metrics, metrics)

    def _init_clip_metrics(self, clip_name):
        merge_metrics_from = self.merge_metrics_from
        if not merge_metrics_from:
            return self._clip_meta[clip_name].copy()

        metrics_fn = self._make_metrics_filename(merge_metrics_from, clip_name)
        with open(metrics_fn, "rb") as f:
            return json.load(f)

    def _write_metrics(self, clip_name):
        metrics = self._clip_metrics.pop(clip_name)

        metrics_fn = self._make_metrics_filename(self.output_path, clip_name)

        os.makedirs(os.path.dirname(metrics_fn), exist_ok=True)
        with open(metrics_fn, "wt", encoding="utf-8") as f:
            # noinspection PyTypeChecker
            json.dump(metrics, f)
            f.close()

    @staticmethod
    def _make_metrics_filename(folder, clip_name):
        metrics_fn = os.path.splitext(clip_name)[0] + ".json"
        metrics_fn = os.path.join(folder, metrics_fn)
        return metrics_fn

    def _print_progress(self, idx, total):
        t = time.time()
        if idx < total and t - self._last_progress_time < 10:
            return

        logging.info(f"processed {idx}/{total}")
        self._last_progress_time = t

    @torch.inference_mode()
    def run(self):
        self._init_wait_store()

        clip_list = self._load_clip_list()

        total_clips = len(clip_list)
        if self.rank < 0:
            logging.info(f"processing {total_clips} clips")
            n_clips = total_clips
        else:
            clip_list = tuple(itertools.islice(clip_list, self.rank, None, self.world_size))
            n_clips = len(clip_list)
            logging.info(f"processing {n_clips} of {total_clips} clips")

        # remember process clip metadata
        self._clip_meta.update((clip_meta["filename"], clip_meta) for clip_meta in clip_list)

        self._last_progress_time = start_time = time.time()

        n_workers = self.get_config_by_path("optic_flow.dataset.n_workers", expected_type=int, default=1)
        loader = torch.utils.data.DataLoader(
            self._create_dataset(clip_list), batch_size=None, shuffle=False, num_workers=n_workers
        )

        n_processed = 0

        for idx, (clip_name, is_last, batch) in enumerate(loader):
            self._process_batch(clip_name, batch)
            if is_last:
                self._write_metrics(clip_name)
                n_processed += 1
                self._print_progress(n_processed, n_clips)

                # clear torch memory cache to avoid fragmentation
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        assert n_processed == n_clips
        assert len(self._clip_metrics) == 0

        if self.rank >= 0:
            self._wait_all()

            all_n_clips = torch.as_tensor(n_clips, dtype=torch.long, device=self.device)
            torch.distributed.all_reduce(all_n_clips)
            assert all_n_clips.item() == total_clips

        logging.info(f"{total_clips} processed in {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    OpticFlowApp().main()
