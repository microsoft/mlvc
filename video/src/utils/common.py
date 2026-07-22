# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import logging
import math
import numbers
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Mapping, Sequence, Dict, Any, BinaryIO, Tuple, Iterable, List
from unittest.mock import patch
from functools import cached_property

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from .retry import retry

__all__ = [
    "get_training_lambdas",
    "ddp_sync_state_dict",
    "ddp_sync_model",
    "ddp_sync_optimizer",
    "avg_per_rate",
    "create_folder",
    "remove_nan_grad",
    "remove_nan_grad_and_clamp",
    "dump_json",
    "configure_logging",
    "multiprocessing_init",
    "apply_type_annotations_to_config_value",
    "apply_type_annotations_to_config",
    "set_config_value_by_path",
    "sync_random_seed",
    "create_metrics_writer",
    "write_epoch_marker",
    "read_epoch_marker",
    "compare_test_with_anchor",
    "get_average_bd_rates",
    "write_bd_rate_to_file",
    "get_optimizer",
    "GradNormTracker",
    "AuxiliaryModels",
]


class AuxiliaryModels:
    def __init__(self, pretrained_path, device, config: dict):
        self.pretrained_path = pretrained_path
        self.device = device
        self.config = config

        if self.pretrained_path:
            torch.hub.set_dir(self.pretrained_path)

    def _load_lpips_model(self):
        from src.losses.perceptual import PerceptualModel

        lpips_config = {
            "base_model": "vgg16",
            "normalize_input": True,
            "use_lpips_aggregation": True,
            "use_lpips_weights": True,
        }
        model = PerceptualModel(
            pretrained_models_path=self.pretrained_path,
            config=lpips_config,
        )
        model.to(self.device)
        return model

    def _create_perceptual_model(self):
        from src.losses.perceptual import PerceptualModel

        perceptual_config = self.config["perceptual"]
        model = PerceptualModel(pretrained_models_path=self.pretrained_path, config=perceptual_config)
        model.to(self.device)
        return model

    def _load_segmentation_model(self):
        from src.models.segment import SegmentationModel

        segmentation_config = self.config.get("segmentation", {})
        model = SegmentationModel(self.pretrained_path, config=segmentation_config, device=self.device)
        model.to(self.device)
        return model

    def _load_lip_reading_model(self):
        from src.vsr.auto_avsr.api import AutoAVSRAPI

        lr_config = self.config["lip_reading"]
        ckpt_path = os.path.join(self.pretrained_path, lr_config["ckpt_name"])
        logging.info(f"Loaded lip reading model from {ckpt_path}")
        model = AutoAVSRAPI(ckpt_path, self.device)
        return model

    def _load_deqa_score_model(self):
        from src.metrics.deqa.scorer import Scorer

        deqa_config = self.config["deqa_score"]
        ckpt_path = os.path.join(self.pretrained_path, deqa_config["ckpt_name"])
        model = Scorer(pretrained=ckpt_path, device=self.device)
        logging.info(f"Loaded DeQA-Score model from {ckpt_path}")
        self._deqa_max_num_frames = deqa_config.get("max_num_frames", 30)
        logging.info(f"DeQA-Score max num frames: {self._deqa_max_num_frames}")
        return model

    @cached_property
    def perceptual_model(self):
        if self.pretrained_path is None:
            raise ValueError("Pretrained path is required for perceptual model")
        return self._create_perceptual_model()

    @cached_property
    def perceptual_model_lpips_reference(self):
        """For eval metric, use reference LPIPS"""
        if self.pretrained_path is None:
            raise ValueError("Pretrained path is required for LPIPS model")
        return self._load_lpips_model()

    @cached_property
    def segmentation_model(self):
        if self.pretrained_path is None:
            raise ValueError("Pretrained path is required for segmentation model")
        return self._load_segmentation_model()

    @cached_property
    def lip_reading_model(self):
        if self.pretrained_path is None:
            raise ValueError("Pretrained path is required for lip reading model")
        return self._load_lip_reading_model()

    @cached_property
    def deqa_score_model(self):
        if self.pretrained_path is None:
            raise ValueError("Pretrained path is required for DeQA-Score model")
        return self._load_deqa_score_model()

    def offload_deqa_score_model(self):
        if "deqa_score_model" not in self.__dict__:
            return
        else:
            # Delete cached instance to free memory
            del self.__dict__["deqa_score_model"]
            logging.info("Offloaded DeQA-Score model (deleted cached instance).")

    def offload_heavy_models(self):
        self.offload_deqa_score_model()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_training_lambdas(lmbdas, qp_num):
    all_lmbdas = np.linspace(np.log(lmbdas[0]), np.log(lmbdas[1]), qp_num)
    all_lmbdas = np.exp(all_lmbdas)
    return all_lmbdas


def downsample_mask(mask, factor: int):
    assert factor in [2, 4, 8, 16]
    return torch.nn.functional.avg_pool2d(mask, kernel_size=factor, stride=factor)


# New method version to replace obsolete, when code conversion completes
def ddp_sync_state_dict(state_dict):
    wait_list = list()

    bcast_device = None
    if torch.distributed.get_backend() == torch.distributed.Backend.NCCL:
        bcast_device = torch.device("cuda", torch.cuda.current_device())
    sync_list = list()

    def process_dict(obj: dict):
        for _, value in sorted(obj.items()):
            if isinstance(value, dict):
                process_dict(value)
            elif isinstance(value, torch.Tensor):
                if bcast_device is not None and value.device.type != bcast_device.type:
                    # move CPU tensors to broadcast device
                    bcast_value = value.to(bcast_device)
                    sync_list.append((value, bcast_value))
                    value = bcast_value

                wait_list.append(torch.distributed.broadcast(value, src=0, async_op=True))

    process_dict(state_dict)
    for op in wait_list:
        op.wait()

    for tensor, bcast_tensor in sync_list:
        tensor[...] = bcast_tensor[...]


def ddp_sync_model(model):
    ddp_sync_state_dict(model.state_dict())


def ddp_sync_optimizer(optimizer):
    ddp_sync_state_dict(optimizer.state_dict()["state"])


def avg_per_rate(x, B, anchor_num):
    y = x.reshape((anchor_num, B))
    return torch.sum(y, dim=1) / B


def create_folder(path, print_if_create=False):
    if not os.path.exists(path):
        os.makedirs(path)
        if print_if_create:
            print(f"created folder: {path}")


def remove_nan_grad(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is not None:
            p.grad.data.nan_to_num_(0.0, 0.0, 0.0)


def remove_nan_grad_and_clamp(parameters, max_value):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        g = p.grad
        if g is None:
            continue
        g.data.nan_to_num_(0.0, 0.0, 0.0).clamp_(-max_value, max_value)


@patch("json.encoder.c_make_encoder", None)
def dump_json(obj, fid, float_digits=-1, **kwargs):
    # noinspection PyProtectedMember
    of = json.encoder._make_iterencode  # type: ignore[attr-defined]  # pylint: disable=W0212

    # noinspection PyShadowingNames
    def inner(*args, **kwargs):
        args = list(args)
        # fifth argument is float formater which we will replace
        args[4] = lambda o: format(o, ".%df" % float_digits)
        return of(*args, **kwargs)

    with patch("json.encoder._make_iterencode", wraps=inner):
        json.dump(obj, fid, **kwargs)


def configure_logging(
    *, level: Optional[int] = None, log_format: Optional[str] = None, file_name: Optional[str] = None
):
    if log_format is None:
        log_format = "%(asctime)s P%(process)05d %(levelname).1s: %(message)s"

    if level is None:
        # Logging level DEBUG should be set only selectively.
        # Otherwise there is too much output from azureml packages
        level = logging.INFO

    warn_handler = logging.StreamHandler(stream=sys.stderr)
    warn_handler.setLevel(logging.WARNING)

    info_handler = logging.StreamHandler(stream=sys.stdout)
    info_handler.addFilter(lambda record: record.levelno < logging.WARNING)

    handlers = [info_handler, warn_handler]

    if file_name is not None:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        handlers.append(logging.FileHandler(file_name, encoding="utf-8"))

    # noinspection PyArgumentList
    logging.basicConfig(format=log_format, level=level, handlers=handlers, force=True)


def multiprocessing_init():
    """
    Multiprocessing intitialization
    """
    import torch.multiprocessing as mp

    mp_start_method = "forkserver"
    if mp_start_method in mp.get_all_start_methods():
        logging.info(f"Switching to multiprocessing strategy {mp_start_method}")
        mp.set_start_method(mp_start_method)
        mp.set_forkserver_preload(["torch"])

    mp.freeze_support()


config_type_annotation_regex = re.compile("@([sbfin]):(.*)")


def apply_type_annotations_to_config_value(value):
    """
    Apply type annotation to JSON/YAML config value

    Format: @[sbfin]:...
        s - string, b - bool, f - float, i - integer, n - null

    Required for compatibility with deficient type support in some YAML parsers (e.g. AzureDevOps)
    """
    m = config_type_annotation_regex.match(value)
    if m is None:
        return value

    t = m.group(1)
    value = m.group(2)

    if t == "b":
        value = bool(value)
    elif t == "f":
        value = float(value)
    elif t == "i":
        value = int(value)
    elif t == "n":
        if value not in ("null", "None", "nil"):
            raise ValueError(f"Can not convert {value} to None")
        value = None

    return value


def apply_type_annotations_to_config(value):
    """
    Apply type annotation to JSON/YAML config (single value, dictionary or list)

    Format: @[sbfin]:...
        s - string, b - bool, f - float, i - integer, n - null

    Required for compatibility with deficient type support in some YAML parsers (e.g. AzureDevOps)
    """
    if isinstance(value, Mapping):
        return {n: apply_type_annotations_to_config(v) for n, v in value.items()}
    elif isinstance(value, str):
        return apply_type_annotations_to_config_value(value)
    elif isinstance(value, Sequence):
        return [apply_type_annotations_to_config(v) for v in value]
    else:
        return value


def set_config_value_by_path(config: Dict[str, Any], name, value, *, overwrite: bool = True):
    """
    Set value in structured config by path
    """
    name_list = name.split(".")
    for idx, n in enumerate(name_list[:-1]):
        subconfig = config.get(n, None)
        if subconfig is None:
            config[n] = subconfig = dict()
        elif not isinstance(subconfig, dict):
            raise ValueError(f"Config does not have an object with path {'.'.join(name_list[: idx + 1])}")

        config = subconfig

    if overwrite or name_list[-1] not in config:
        config[name_list[-1]] = value


def sync_random_seed(seed):
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)


def create_metrics_writer(rank: int = -1):
    from .mlflow_metrics_writer import DebugMetricsWriter, MlflowMetricsWriter, get_mlflow_run_id

    if rank > 0:
        return DebugMetricsWriter()

    run_id = get_mlflow_run_id()
    if run_id is None:
        logging.warning("Unable to get AzureML/MLflow run context, not reporting metrics to AzureML")
        return DebugMetricsWriter()

    return MlflowMetricsWriter(run_id)


@retry
def write_epoch_marker(filename: str, epoch: int, *, is_first_checkpoint: Optional[bool] = None):
    if is_first_checkpoint is None:
        is_first_checkpoint = epoch == 0
    mode = "wb" if is_first_checkpoint else "rb+"
    with open(filename, mode) as f:
        f: BinaryIO
        f.write(np.asarray(epoch, dtype=np.int64).tobytes())
        f.close()


@retry
def read_epoch_marker(filename: str):
    try:
        f = open(filename, "rb")
    except FileNotFoundError:
        return None

    with f:
        data = np.frombuffer(f.read(), np.int64)
        assert data.shape == (1,)
        return int(data[0])


def compare_test_with_anchor(
    *,
    epoch: int,
    anchor_results_path: str,
    test_results_path: str,
    testset_name: str,
    save_dir: str,
    metrics_writer=None,
    frame_type: str = "default",
    distortion_metrics: Sequence[str] = ("psnr",),
):
    output_path = os.path.join(save_dir, f"{testset_name}_bd_rate.jsonl")
    distortion_metrics = " ".join(distortion_metrics)

    command_line = (
        f"{sys.executable} ./compare_rd_video.py --compare_between class"
        f" --compare_frame_type {frame_type}"
        " --base_method anchor"
        f" --log_paths anchor {anchor_results_path}"
        f" test_epo{epoch} {test_results_path}"
        f" --output_path {output_path}"
        " --auto_test 1"
        " --plot_rd_curve 0 --plot_path ./test_room/figs/"
        f" --distortion_metrics {distortion_metrics}"
    )
    print(command_line)
    os.system(command_line)

    all_results = get_average_bd_rates(output_path)
    for metric_name, results in all_results.items():
        last_result = results[-1]
        epoch = last_result.epoch
        curr_bd_rate = last_result.bd_rate
        best_bd_rate = min(result.bd_rate for result in results)

        if metrics_writer is not None:
            metric_log_name = f"{testset_name}_bd_rate"
            if metric_name != "psnr":
                metric_log_name += f"_{metric_name}"
            metrics_writer.add_scalar(metric_log_name, curr_bd_rate, epoch)

        bd_rate_file_name = f"{testset_name}_best_bd_rate"
        if metric_name != "psnr":
            bd_rate_file_name += f"_{metric_name}"
        bd_rate_file_name += ".txt"
        bd_rate_file_path = os.path.join(save_dir, bd_rate_file_name)
        write_bd_rate_to_file(epoch, curr_bd_rate, best_bd_rate, bd_rate_file_path)


def get_epoch_number(s):
    match = re.search(r"\d+$", s)
    assert match is not None
    return int(s[match.start() : match.end()])


@dataclass(frozen=True)
class BDRateResult:
    metric_name: str
    epoch: int
    bd_rate: float


def get_average_bd_rates(log_path):
    # List of lines with metric -> frame_type -> method -> dataset -> average bd rate
    data: List[Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = []
    with open(log_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    results = defaultdict(list)
    for result in data:
        for metric_name, metric_results in result.items():
            all_results = metric_results["all"]
            for method, method_results in all_results.items():
                epoch = get_epoch_number(method)
                bd_rate_sum = 0
                bd_rate_count = 0
                for _, bd_rate in method_results.items():
                    bd_rate_sum += bd_rate
                    bd_rate_count += 1
                avg_bd_rate = bd_rate_sum / bd_rate_count
                results[metric_name].append(BDRateResult(metric_name, epoch, avg_bd_rate))
    return results


def write_bd_rate_to_file(epoch, curr_bd_rate, best_bd_rate, output_file_path):
    with open(output_file_path, "a") as f:
        line = f"epoch: {epoch}, curr_bd_rate: {curr_bd_rate:.2f}, best_bd_rate: {best_bd_rate:.2f}\n"
        f.write(line)


def get_optimizer(parameters, lr=1e-4, **kwargs):
    return optim.AdamW(parameters, lr=lr, **kwargs)


class GradNormTracker:
    def __init__(
        self,
        *,
        total_l2_norm_limit: Optional[float] = None,
        parameter_initial_norm,
        parameter_norm_limit_factor: float = 5.0,
        parameter_norm_alpha: float = 0.05,
        parameter_norm_min_ratio: float = 0.0,
        parameter_exclude_list: Optional[Sequence[str]] = None,
    ):
        if total_l2_norm_limit is not None:
            if not isinstance(total_l2_norm_limit, numbers.Real) or total_l2_norm_limit <= 0:
                raise ValueError(f"Invalid total_l2_norm_limit value: {total_l2_norm_limit}")

            self.total_l2_norm_limit: Optional[torch.Tensor] = torch.tensor(total_l2_norm_limit, dtype=torch.float64)
        else:
            self.total_l2_norm_limit: Optional[torch.Tensor] = None

        if not isinstance(parameter_initial_norm, numbers.Real) or parameter_initial_norm <= 0:
            raise ValueError(f"Invalid parameter_initial_norm value: {parameter_initial_norm}")
        self.parameter_initial_norm = torch.tensor(parameter_initial_norm, dtype=torch.float64)

        if not isinstance(parameter_norm_limit_factor, numbers.Real) or parameter_norm_limit_factor <= 1:
            raise ValueError(f"Invalid parameter_norm_limit_factor value: {parameter_norm_limit_factor}")
        self.parameter_norm_limit_factor = torch.tensor(parameter_norm_limit_factor, dtype=torch.float64)

        if not isinstance(parameter_norm_alpha, numbers.Real) or not (0 < float(parameter_norm_alpha) < 1):
            raise ValueError(f"Invalid parameter_norm_alpha value: {parameter_norm_alpha}")
        self.parameter_norm_alpha = float(parameter_norm_alpha)

        if not isinstance(parameter_norm_min_ratio, numbers.Real) or not (0 <= float(parameter_norm_min_ratio)):
            raise ValueError(f"Invalid parameter_norm_min_ratio value: {parameter_norm_min_ratio}")
        self.parameter_norm_min_ratio = float(parameter_norm_min_ratio)

        self.parameter_exclude_list = tuple(parameter_exclude_list) if parameter_exclude_list is not None else tuple()

        self.parameter_map = dict()
        self.parameter_running_norm = None
        self.parameter_mask: Optional[torch.Tensor] = None
        self.history = list()

    @torch.no_grad()
    def track_and_clip_(self, parameters: Iterable[Tuple[str, nn.Parameter]]):
        gradients, running_norm, grad_norm, weight_norm = self._collect_grad_norms(parameters)
        if len(gradients) == 0:
            return 0.0, 1.0

        assert running_norm is not None

        self.history.append(grad_norm)

        if not torch.all(torch.isfinite(grad_norm)):
            for g in gradients:
                torch.zero_(g)

            return math.nan if torch.any(torch.isnan(grad_norm)) else math.inf, 0.0

        # compute total l2 norm
        l2_norm = torch.linalg.norm(grad_norm[:, 1], 2.0)
        l2_norm_value = l2_norm

        # replace zero gradients with running_norm to avoid changing it
        grad_norm = torch.where(torch.eq(grad_norm, 0), running_norm, grad_norm)

        norm_limit = self.parameter_norm_limit_factor * running_norm
        if self.parameter_mask is None:
            masked_grad_norm = grad_norm
        else:
            assert self.parameter_mask is not None
            masked_grad_norm = grad_norm * self.parameter_mask
        scale = torch.min(norm_limit / torch.maximum(masked_grad_norm, norm_limit))

        # limit impact of huge gradients on running average
        norm_limit *= self.parameter_norm_limit_factor
        torch.minimum(grad_norm, norm_limit, out=grad_norm)

        grad_norm -= running_norm
        torch.add(running_norm, grad_norm, alpha=self.parameter_norm_alpha, out=running_norm)
        if weight_norm is not None and self.parameter_norm_min_ratio > 0:
            weight_norm *= self.parameter_norm_min_ratio
            torch.maximum(running_norm, weight_norm, out=running_norm)

        total_l2_norm_limit = self.total_l2_norm_limit
        if total_l2_norm_limit is not None and l2_norm > total_l2_norm_limit:
            torch.divide(total_l2_norm_limit, l2_norm, out=l2_norm)
            torch.minimum(scale, l2_norm)

        scale_value = scale.item()
        if scale < 1:
            for g in gradients:
                scale = scale.to(g.device)
                g.mul_(scale)

        return l2_norm_value, scale_value

    def _collect_grad_norms(
        self, parameters: Iterable[Tuple[str, nn.Parameter]]
    ) -> Tuple[list, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        parameter_map = self.parameter_map
        gradients = list()
        norms = list()
        weight_norms = list() if self.parameter_norm_min_ratio > 0 else None
        indices = list()

        new_params = None
        running_norm = self.parameter_running_norm
        parameter_count = len(running_norm) if running_norm is not None else 0

        for n, p in parameters:
            g = p.grad
            if g is None:
                continue

            gradients.append(g)
            g = g.flatten()

            norms.extend(self._calc_norm(g))
            if weight_norms is not None:
                weight_norms.extend(self._calc_norm(p.flatten()))

            idx = parameter_map.get(n)
            if idx is None:
                idx = len(parameter_map)
                parameter_map[n] = idx
            indices.append(idx)

            if idx >= parameter_count:
                if new_params is None:
                    new_params = list()

                new_params.append((idx, p))

        indices = torch.tensor(indices, dtype=torch.int64)

        if new_params is not None:
            new_running_norm = torch.empty((len(parameter_map), 2), dtype=torch.float64)
            if parameter_count > 0:
                assert running_norm is not None
                new_running_norm[:parameter_count] = running_norm
            torch.fill_(new_running_norm[parameter_count:], self.parameter_initial_norm)

            for idx, p in new_params:
                # adjust initial l2 norm to number of parameters
                new_running_norm[idx, 1] *= math.sqrt(p.numel())

            self.parameter_running_norm = running_norm = new_running_norm
            self.parameter_mask = self._rebuild_parameter_mask()

        if weight_norms is not None:
            norms.extend(weight_norms)

        norms = torch.stack(norms).cpu()
        if weight_norms is not None:
            norms, weight_norms = torch.chunk(norms, 2)
            weight_norms = self._reorder_norms_tensor(running_norm, weight_norms, indices)

        norms = self._reorder_norms_tensor(running_norm, norms, indices)
        return gradients, running_norm, norms, weight_norms

    @staticmethod
    def _reorder_norms_tensor(running_norm, src, indices):
        dst = torch.zeros_like(running_norm)
        dst[indices] = src.reshape(-1, 2)
        return dst

    @staticmethod
    def _calc_norm(g):
        return (
            torch.linalg.norm(g, np.inf, dtype=torch.float64),
            torch.linalg.norm(g, 2, dtype=torch.float64),
        )

    def _rebuild_parameter_mask(self):
        mask = None
        for idx, n in enumerate(self.parameter_map.keys()):
            excluded = False
            for prefix in self.parameter_exclude_list:
                if n.startswith(prefix) and (len(n) == len(prefix) or n[len(prefix)] == "."):
                    excluded = True
                    break

            if not excluded:
                continue

            if mask is None:
                mask = torch.ones((len(self.parameter_map), 1), dtype=torch.float64)

            mask[idx] = 0.0

        return mask

    def truncate_history(self):
        self.history.clear()

    def state_dict(self):
        return dict(
            parameter_map=self.parameter_map,
            parameter_running_norm=self.parameter_running_norm,
        )

    def load_state_dict(self, state_dict):
        self.parameter_map = dict(state_dict["parameter_map"])
        self.parameter_mask = self._rebuild_parameter_mask()
        self.parameter_running_norm = state_dict["parameter_running_norm"].cpu().clone()
