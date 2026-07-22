# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import abc
import argparse
import copy
import datetime
import io
import itertools
import json
import logging
import math
import numbers
import os.path
import re
import sys
import time
from functools import cached_property
from numbers import Number, Real
from typing import Dict, Any, Set, Optional, Sequence, List, Union, Callable, Mapping

import numpy as np
import torch
import torch.distributed
import torch.utils.data
import yaml
from torch.nn.parallel import DistributedDataParallel

from .common import (
    apply_type_annotations_to_config,
    configure_logging,
    multiprocessing_init,
    set_config_value_by_path,
    sync_random_seed,
    create_metrics_writer,
    get_optimizer,
    ddp_sync_optimizer,
    ddp_sync_model,
    write_epoch_marker,
    read_epoch_marker,
    compare_test_with_anchor,
    remove_nan_grad,
    remove_nan_grad_and_clamp,
    get_training_lambdas,
    GradNormTracker,
    AuxiliaryModels,
)
from .encoder_tester import EncoderTestParams, run_encoder_test
from .mount_point import MountPointResolver
from .profiling import summarize_model_layers
from .qp_matcher import match_q_index_lists
from .stream_helper import get_state_dict, PaddingMode, PaddingAlignment

__all__ = ["BaseApp", "BaseTrainApp", "BaseTrainVideoApp"]

# undefined value
UNDEFINED = object()


class BaseApp(abc.ABC):
    def __init__(self):
        self._config = None
        self._rank = -1
        self._local_rank = -1
        self._world_size = 1
        self._cmdline_only_arguments = None
        self._metrics_writer = None

    @property
    def config(self):
        config = self._config
        if config is None:
            raise RuntimeError("config is not initialized")
        return config

    def get_config_by_path(self, key_path, *, default=UNDEFINED, expected_type=None) -> Any:
        config = self.config

        components = key_path.split(".")
        for idx, n in enumerate(components):
            if isinstance(config, dict):
                config = config.get(n, UNDEFINED)
            elif config is None:
                # empty YAML object
                config = UNDEFINED
            else:
                raise KeyError(f"Config value {'.'.join(components[:idx])} is not an object")

            if config is UNDEFINED:
                if default is UNDEFINED:
                    raise KeyError(f"Config key {'.'.join(components[: idx + 1])} does not exist")
                return default

        if expected_type is not None and not isinstance(config, expected_type):
            if expected_type is dict:
                raise ValueError(f"Config value {key_path} must be an object")
            elif expected_type is list:
                raise ValueError(f"Config value {key_path} must be a list")
            else:
                raise ValueError(f"Config value {key_path} is {type(config)}, while {expected_type} is expected")

        return config

    @property
    def rank(self):
        return self._rank

    @property
    def local_rank(self):
        return self._local_rank

    @property
    def world_size(self):
        return self._world_size

    @cached_property
    def metrics_writer(self):
        metrics_writer = self._metrics_writer
        if metrics_writer is None:
            self._metrics_writer = metrics_writer = create_metrics_writer(self.rank)
        return metrics_writer

    def main(self):
        self._config = self.parse_cmdline()
        self.configure_init_logging(level=logging.getLevelName(self.config.get("log_level", "INFO")))

        rc = -1
        # noinspection PyBroadException
        try:
            local_rank = self.determine_local_rank()
            self.configure_logging(local_rank)

            logging.info(f"Running with config: {self.config}")

            self.initialize(local_rank)
            rc = self.run()
        except BaseException:
            logging.exception("Exception in main:")
            if self.config["reraise_exceptions"]:
                raise
        finally:
            # noinspection PyBroadException
            try:
                self.close()
            except Exception:
                logging.exception(f"Unexpected exception in {type(self)}::close()")
                rc = -1

        quit(rc)

    # noinspection PyMethodMayBeStatic
    def configure_init_logging(self, **kwargs):
        configure_logging(**kwargs)

    def configure_logging(self, local_rank: int):
        if local_rank > 0:
            for handler in logging.getLogger().handlers:
                if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                    handler.setLevel(logging.WARNING)

        log_dir = self.config.get("log_dir")
        if log_dir:
            log_dir = self.resolve_path(log_dir, default_source=None)
            os.makedirs(log_dir, exist_ok=True)

            timestamp = datetime.datetime.now(datetime.timezone.utc)
            log_file_name = os.path.join(log_dir, f"l-{max(0, local_rank)}-{timestamp:%Y_%m_%d-%H_%M_%S}.log")

            root_logger = logging.getLogger()
            handler = logging.FileHandler(log_file_name)
            handler.setFormatter(root_logger.handlers[0].formatter)
            root_logger.addHandler(handler)

    def parse_cmdline(self):
        parser = argparse.ArgumentParser()
        self.configure_cmdline_only_arguments(parser)
        # noinspection PyProtectedMember
        self._cmdline_only_arguments = set(x.dest for x in parser._option_string_actions.values())
        self.configure_cmdline_arguments(parser)

        cmd_args, _ = parser.parse_known_args()
        initial_args = cmd_args

        config = self.load_config(cmd_args)
        self.apply_config_overrides(config, cmd_args)

        self._add_args_from_config(parser, "--", config)
        cmd_args = parser.parse_args()
        self._check_cmdline_consistent(initial_args, cmd_args)

        self.apply_cmdline_arguments(config, cmd_args)
        config = self.post_process_config(config)
        return config

    # noinspection PyMethodMayBeStatic
    def configure_cmdline_only_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--config", required=True, help="YAML configuration file")
        parser.add_argument("--overrides_file", required=False, help="YAML or JSON file with config overrides")
        parser.add_argument("--overrides", required=False, help="YAML string with config overrides")
        parser.add_argument("--store_regions", help="list of mount regions", required=False)
        parser.add_argument("--data_mount", action="append", help="Data blob storage mountpoint")
        parser.add_argument("--checkpoints_mount", action="append", help="Checkpoint blob storage mountpoint")
        parser.add_argument("--aux_mount_names", required=False, help="Auxiliary blob storage mountpoint names")
        parser.add_argument("--aux_mount", action="append", help="Auxiliary blob storage mountpoint")
        parser.add_argument("--add-to-system-path", action="append", help="Folder to add to system path")

    # noinspection PyMethodMayBeStatic
    def configure_cmdline_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--distributed_backend", help="distributed backend to use for multi-processing run")
        parser.add_argument(
            "--reraise-exceptions",
            action="store_true",
            help="re-raise exceptions to let debugger stop at raise statement",
        )
        parser.add_argument("--save_dir", type=str, default="test_room", help="Path to save models")

    # noinspection PyMethodMayBeStatic
    def load_config(self, cmd_args: argparse.Namespace):
        config_filename = os.path.abspath(cmd_args.config)
        with open(config_filename, "rt") as f:
            config = yaml.full_load(f)

        if not isinstance(config, dict):
            raise ValueError(f"Configuration YAML in {cmd_args.config} must be a dictionary")

        config["config"] = config_filename
        return config

    def apply_config_overrides(self, config, cmd_args: argparse.Namespace):
        mount_resolver = MountPointResolver(cmd_args)
        for mount_name in "data_mount", "checkpoints_mount":
            config[mount_name] = mount_resolver.resolve_mount(mount_name)

        aux_mount_names = cmd_args.aux_mount_names.split(",") if cmd_args.aux_mount_names else []
        aux_mount_list = cmd_args.aux_mount if cmd_args.aux_mount is not None else []
        if len(aux_mount_names) != len(aux_mount_list):
            raise ValueError(
                f"Number of auxiliary mount point names ({len(aux_mount_names)})"
                f" does not match number of auxiliary mounts {len(cmd_args.aux_mount)}"
            )

        config["aux_mounts"] = dict(zip(aux_mount_names, aux_mount_list))

        overrides_file = cmd_args.overrides_file
        if overrides_file:
            with open(overrides_file, "rt", encoding="utf-8") as f:
                if overrides_file.lower().endswith(".json"):
                    # YAML is supposed to be superset of JSON, but PyYAML implementation
                    # does not parse floating point number in scientific notation without dot (i.e. 1e-6),
                    # which is perfectly valid floating point number in JSON.
                    # To avoid misinterpretation of JSON produced by scripts (python or pwsh)
                    # parse JSON files using JSON parser
                    overrides_config = json.load(f)
                else:
                    overrides_config = yaml.load(f, Loader=yaml.FullLoader)

            self._apply_overrides_config(config, overrides_config)
        del cmd_args.overrides_file  # type: ignore[attr-defined]

        overrides = cmd_args.overrides
        if overrides:
            with io.StringIO(overrides) as f:
                overrides_config = yaml.load(f, Loader=yaml.FullLoader)

            self._apply_overrides_config(config, overrides_config)
        del cmd_args.overrides  # type: ignore[attr-defined]

        add_to_system_path = cmd_args.add_to_system_path
        if add_to_system_path is not None and len(add_to_system_path) > 0:
            system_path = os.environ.get("PATH", "")
            if system_path and not system_path.endswith(os.pathsep):
                system_path += os.pathsep
            system_path += os.pathsep.join(add_to_system_path)
            os.environ["PATH"] = system_path

    @staticmethod
    def _apply_overrides_config(config, overrides_config):
        if not isinstance(overrides_config, dict):
            raise ValueError("Overrides configuration must be a dictionary")

        overrides_config = apply_type_annotations_to_config(overrides_config)

        for name, value in overrides_config.items():
            set_config_value_by_path(config, name, value)

    def _add_args_from_config(
        self,
        parser: argparse.ArgumentParser,
        prefix: str,
        config: Dict[str, Any],
        explicit_args: Optional[Set[str]] = None,
    ):
        """
        Add arguments for configuration settings found in config
        """
        if explicit_args is None:
            explicit_args = set()
            # noinspection PyProtectedMember
            for n, a in parser._option_string_actions.items():
                explicit_args.add(n)
                explicit_args.add("--" + a.dest)

        for name, option in config.items():
            if isinstance(option, dict):
                self._add_args_from_config(parser, prefix + name + ".", option, explicit_args)
                continue

            if not isinstance(option, (Number, str, bool)):
                # skip all unsupported types for now
                continue

            option_name = prefix + str(name)
            if option_name in explicit_args:
                continue

            option_type = type(option)
            if option_type is bool:

                def parse_bool(s):
                    v = s.lower()
                    if v in ("true", "on", "yes", "1"):
                        return True
                    if v in ("false", "off", "no", "0"):
                        return False
                    raise ValueError(f"Can not convert value to bool: '{s}'")

                option_type = parse_bool

            parser.add_argument(option_name, type=option_type, default=argparse.SUPPRESS)

    @staticmethod
    def _check_cmdline_consistent(initial_args, cmd_args):
        initial_args = vars(initial_args)
        cmd_args = vars(cmd_args)
        for n, v in initial_args.items():
            if n not in cmd_args or cmd_args[n] != v:
                raise ValueError(f"Inconsistent command line: {n} changed when command line is fully parsed")

    def apply_cmdline_arguments(self, config, cmd_args: argparse.Namespace):
        assert self._cmdline_only_arguments is not None
        for name, value in vars(cmd_args).items():
            if name not in self._cmdline_only_arguments:
                set_config_value_by_path(config, name, value, overwrite=value is not None)

    @staticmethod
    def post_process_config(config):
        return config

    def determine_local_rank(self):
        if self.config.get("distributed_backend") is None:
            return -1
        else:
            try:
                return int(os.environ["LOCAL_RANK"])
            except Exception as e:
                raise ValueError("Unable to determine local rank") from e

    def initialize(self, local_rank: int):
        multiprocessing_init()
        if local_rank >= 0:
            self.distributed_init(local_rank)
        self.set_cuda_flags()
        self.set_torch_flags()
        self.initialize_seed()

    def set_cuda_flags(self):
        import torch.cuda

        if not torch.cuda.is_available():
            return

        matmul_allow_tf32 = self.get_config_by_path("cuda.matmul_allow_tf32", default=None)
        if matmul_allow_tf32 is not None:
            import torch.backends.cuda

            logging.info(f"Setting torch.backends.cuda.matmul.allow_tf32 = {matmul_allow_tf32}")
            torch.backends.cuda.matmul.allow_tf32 = matmul_allow_tf32

        cudnn_allow_tf32 = self.get_config_by_path("cuda.cudnn_allow_tf32", default=None)
        if cudnn_allow_tf32 is not None:
            import torch.backends.cudnn

            logging.info(f"Setting torch.backends.cudnn.allow_tf32 = {cudnn_allow_tf32}")
            torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32

        cudnn_benchmark = self.get_config_by_path("cuda.cudnn_benchmark", default=None)
        if cudnn_benchmark is not None:
            import torch.backends.cudnn

            logging.info(f"Setting torch.backends.cudnn.benchmark = {cudnn_benchmark}")
            torch.backends.cudnn.benchmark = cudnn_benchmark

    def set_torch_flags(self):
        use_deterministic_algorithms = self.get_config_by_path("torch.use_deterministic_algorithms", default=None)
        if use_deterministic_algorithms is not None:
            torch.use_deterministic_algorithms(use_deterministic_algorithms)

    def distributed_init(self, local_rank):
        distributed_backend = self.config["distributed_backend"]

        logging.info(f"Initializing distributed process group with {distributed_backend} backend...")

        if distributed_backend == "nccl":
            cuda_visible_devices = str(local_rank)
            # we use the only visible CUDA device
            device_id = 0
        else:
            cuda_visible_devices = "-1"
            device_id = -1

        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        torch.distributed.init_process_group(distributed_backend, device_id=device_id)

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        self._rank = rank
        self._local_rank = local_rank
        self._world_size = world_size

        logging.info(
            "Initialized distributed process group: "
            f"world_size = {world_size}, local_rank = {local_rank}, rank = {rank}"
        )

        if sys.platform.startswith("linux"):
            # initialize CUDA kernel cache to avoid sporadic warnings from CUDA
            kernel_cache_path = "/dev/shm/kernel_cache"
            os.environ["PYTORCH_KERNEL_CACHE_PATH"] = kernel_cache_path
            os.makedirs(kernel_cache_path, exist_ok=True)

    def initialize_seed(self):
        seed = self.config.get("seed")
        if seed is None:
            seed = torch.seed()
            logging.info(f"Random seed is {seed}")
        else:
            if isinstance(seed, Sequence):
                seed = seed[max(0, self.rank) % len(seed)]
            elif self.rank > 0:
                seed += self.rank
            logging.info(f"Setting random seed to {seed}")
            torch.manual_seed(seed)

        sync_random_seed(seed)

    def close(self):
        metrics_writer = self._metrics_writer
        if metrics_writer is not None:
            self._metrics_writer = None
            if hasattr(metrics_writer, "flush"):
                metrics_writer.flush()

        if self.rank >= 0:
            torch.distributed.destroy_process_group()

    def resolve_path(self, path, *, default_source: Optional[str] = "data_mount", expand_cwd=False) -> str:
        if not path:
            # empty strings and None are unchanged
            return path

        # using ~ instead of $ as expansion marker to avoid accidental expansion by shell
        m = re.match(r"^~\{([\w.-]+)}[/\\]*(.*)$", path)
        if m:
            source = m.group(1)
            if source not in (".", "data_mount", "checkpoints_mount", "save_dir") and source not in self.aux_mounts:
                raise ValueError(f"Unknown data source type in '{path}'")

            path = m.group(2)
        elif os.path.isabs(path):
            source = None
        else:
            source = default_source

        if source == ".":
            if expand_cwd:
                path = os.path.join(os.getcwd(), path)
            source = None

        if source is not None:
            source_dir = self.aux_mounts.get(source)
            if source_dir is None:
                source_dir = getattr(self, source)
            if not source_dir:
                raise ValueError(f"Undefined data source {source}")

            path = os.path.join(source_dir, path)

        if os.path.sep != "/":
            path = path.replace(os.path.sep, "/")

        return path

    @property
    def data_mount(self):
        return self.config["data_mount"]

    @property
    def checkpoints_mount(self):
        return self.config["checkpoints_mount"]

    @cached_property
    def save_dir(self):
        return self.resolve_path(self.config["save_dir"], default_source="checkpoints_mount")

    @property
    def aux_mounts(self):
        return self.config["aux_mounts"]

    def all_reduce(self, x: torch.Tensor):
        if self.rank >= 0:
            torch.distributed.all_reduce(x)
            x /= self.world_size

        return x

    def all_gather_tensors(self, *x: torch.Tensor):
        world_size = self.world_size
        if world_size <= 1:
            return x

        ops = list()
        result = list()
        for t in x:
            s = t.shape[0] * world_size, *t.shape[1:]
            r = torch.empty(s, dtype=t.dtype, device=t.device)
            result.append(r)
            r_list = list(r.chunk(world_size, dim=0))
            ops.append(torch.distributed.all_gather(r_list, t, async_op=True))

        for op in reversed(ops):
            op.wait()

        return tuple(result)

    @abc.abstractmethod
    def run(self):
        pass


class SchedulePlayer:
    def __init__(
        self,
        *,
        config: Dict[str, Any],
        config_path: str,
        schedule: List[Dict[str, Any]],
        schedule_path: str,
        immutable_path_set: Set[str],
    ):
        self.config = config
        self.config_path = config_path
        self.schedule = schedule
        self.schedule_path = schedule_path
        self.immutable_path_set = immutable_path_set
        self.frozen_path_set = set()
        self.epoch = 0
        self.cursor = 0
        self.stash = dict()

    def freeze(self, frozen_config: Dict[str, Any], frozen_config_path: str):
        change_set = dict()
        self._apply_object(
            self.config,
            frozen_config,
            config_path=self.config_path,
            step_path=frozen_config_path,
            change_set=change_set,
            dry_run=None,
            stash=None,
        )

        self.frozen_path_set |= set(n for n, is_leaf in change_set.items() if is_leaf)

    def apply(self, *, epoch: Optional[int], dry_run: bool = True) -> Dict[str, bool]:
        last_epoch = self.epoch
        change_set = dict()

        if epoch is not None and epoch < last_epoch:
            return change_set

        for idx in range(self.cursor, len(self.schedule)):
            step = self.schedule[idx]
            if not isinstance(step, dict):
                raise ValueError(f"step #{idx} in {self.schedule_path} is not an object")

            step_epoch = step.get("epoch")
            if step_epoch is None:
                raise ValueError(f"step #{idx} in {self.schedule_path} does not have an epoch")
            if not isinstance(step_epoch, int):
                raise ValueError(f"epoch value in step #{idx} in {self.schedule_path} is not an integer")
            if step_epoch < last_epoch:
                raise ValueError(f"epoch {step_epoch} in step #{idx} in {self.schedule_path} is out of order")

            if epoch is not None and step_epoch > epoch:
                break

            if not dry_run:
                self._apply_stash(step_epoch, change_set=change_set, dry_run=dry_run)

            step = copy.copy(step)
            del step["epoch"]

            step_path = f"{self.schedule_path}[epoch={step_epoch}]"
            self._apply_object(
                self.config,
                step,
                config_path=self.config_path,
                step_path=step_path,
                change_set=change_set,
                dry_run=dry_run,
                stash=False,
            )

            last_epoch = step_epoch
            if not dry_run:
                self.cursor = idx + 1
                self.epoch = last_epoch

        if dry_run or epoch is not None:
            # apply stash once for dry run
            self._apply_stash(epoch or last_epoch, change_set=change_set, dry_run=dry_run)

        if epoch is not None:
            self.epoch = epoch

        return change_set

    def _apply_object(
        self,
        config: Dict[str, Any],
        step: Dict[str, Any],
        *,
        config_path: str,
        step_path: str,
        change_set: Dict[str, bool],
        dry_run: Optional[bool],
        stash: Optional[bool],
    ):
        for n, v in step.items():
            self._apply_item(
                config,
                n,
                v,
                config_path=config_path,
                step_path=step_path,
                change_set=change_set,
                dry_run=dry_run,
                stash=stash,
            )

    def _apply_item(
        self,
        config,
        key_path: str,
        value,
        *,
        config_path: str,
        step_path: str,
        change_set: Dict[str, bool],
        dry_run: Optional[bool],
        stash: Optional[bool],
    ):

        if stash is not None and key_path.startswith("^"):
            stash = True
            key_path = key_path[1:]

        parent = None
        n = None
        for n in key_path.split("."):
            if not n:
                raise ValueError(f"empty key path component in schedule {step_path}")
            if not isinstance(config, dict) or n not in config:
                raise ValueError(
                    f"schedule {step_path} requests an update to {n}, which is not defined in {config_path}"
                )

            parent = config
            config = config[n]
            if config_path:
                config_path += "."
            config_path += n
            step_path += "." + n

            if config_path in self.immutable_path_set:
                raise ValueError(f"can not apply {step_path}: {config_path} is immutable")

        if parent is None:
            raise ValueError(f"empty key path in schedule {step_path}")

        if isinstance(config, dict):
            if not isinstance(value, dict):
                raise ValueError(f"schedule {step_path} requests to replace object {config_path} with a value")

            self._apply_object(
                config,
                value,
                config_path=config_path,
                step_path=step_path,
                change_set=change_set,
                dry_run=dry_run,
                stash=stash,
            )
        else:
            if isinstance(value, dict):
                raise ValueError(f"schedule {step_path} requests to replace {config_path} with an object")

            if dry_run is not None and config_path in self.frozen_path_set:
                # skip frozen
                return

            if not dry_run and stash:
                self._add_to_stash(config_path, config)

            if dry_run is not None and value == config:
                # optimize changes
                return

            self._add_to_change_set(change_set, config_path)
            if not dry_run:
                parent[n] = value

    def _add_to_stash(self, key_path, value):
        config_path = self.config_path
        if config_path:
            assert key_path.startswith(config_path) and key_path[len(config_path)] == "."
            key_path = key_path[len(config_path) + 1 :]

        self.stash[key_path] = value

    def _add_to_change_set(self, change_set: Dict[str, bool], key_path: str, is_leaf=True):
        value = change_set.get(key_path)
        if value is not None:
            assert value == is_leaf
            return

        change_set[key_path] = is_leaf

        parent, sep, name = key_path.rpartition(".")
        if not sep:
            return

        self._add_to_change_set(change_set, parent, False)

    def _apply_stash(self, epoch, *, change_set: Dict[str, bool], dry_run: bool):
        assert self.epoch <= epoch
        if epoch == self.epoch:
            return

        stash = self.stash
        if len(stash) == 0:
            return

        self._apply_object(
            self.config,
            stash,
            config_path=self.config_path,
            step_path="stash",
            change_set=change_set,
            dry_run=dry_run,
            stash=False,
        )
        if not dry_run:
            stash.clear()


class BatchSkipException(ValueError):
    """
    Common ancestor for batch skipping exception
    """

    pass


class NonFiniteGradError(BatchSkipException):
    """
    Exception for non-finite gradients
    """

    pass


class TooLargeGradError(BatchSkipException):
    """
    Exception for too large gradients
    """

    pass


class BaseTrainApp(BaseApp):
    model: Optional[torch.nn.Module]
    optimizer: Optional[torch.optim.Optimizer]
    grad_norm_tracker: Optional[GradNormTracker]

    def __init__(self):
        super().__init__()

        self.model = None
        self.optimizer = None
        self.grad_norm_tracker = None

        self._train_dataloader = None
        self._device = None
        self._epoch_steps = 0
        self._training_schedule = list()
        self._benchmarked_epoch = set()

        # metrics to log to AzureML
        self._step_metrics = dict()

        # info to print to log
        self._step_info_timestamp = 0.0

        self._auxiliary_models = None

    def configure_cmdline_arguments(self, parser: argparse.ArgumentParser):
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument("--mode", default="train", choices=self.get_supported_modes(), help="script run mode")
        self.configure_mode_argument_group(mode_group)
        super().configure_cmdline_arguments(parser)
        parser.add_argument("--model.ckpt_path", help="checkpoint to start form")

    @staticmethod
    def get_supported_modes():
        return "train", "validate", "revalidate"

    @staticmethod
    def configure_mode_argument_group(mode_group):
        mode_group.add_argument(
            "--validate", dest="mode", action="store_const", const="validate", help="run model validation"
        )

    @property
    def device(self):
        device = self._device
        if device is None:
            is_cuda_available = torch.cuda.is_available()
            logging.info(f"cuda: is_available() = {is_cuda_available}, device_count = {torch.cuda.device_count()}")
            self._device = device = torch.device("cuda" if is_cuda_available else "cpu")
        return device

    @property
    def model_config(self):
        return self.config["model"]

    @abc.abstractmethod
    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    def initialize_model(self, model):
        _ = self, model

    @property
    def pretrained_checkpoint_path(self):
        checkpoint_path = self.model_config.get("ckpt_path")
        if checkpoint_path is not None:
            checkpoint_path = self.resolve_path(checkpoint_path, default_source="checkpoints_mount")

        return checkpoint_path

    def load_model(self, model_state, pretrained: Optional[bool] = None):
        model = self.create_model()
        summarize_model_layers(model)

        if model_state is None and (pretrained is None or pretrained):
            checkpoint_path = self.pretrained_checkpoint_path
            if pretrained and checkpoint_path is None:
                raise ValueError("no pretrained checkpoint path is specified")
            if checkpoint_path is not None:
                model_state = get_state_dict(checkpoint_path)
                logging.info(f"pretrained weights loaded from {checkpoint_path}")

        if model_state is None:
            self.initialize_model(model)
        else:
            model.load_state_dict(model_state)

        model = model.to(self.device)
        return model

    def make_train_model(self, model):
        if self.world_size > 1:
            model = self.make_distributed_model(model)

        return model

    @staticmethod
    def make_distributed_model(model):
        return DistributedDataParallel(model, find_unused_parameters=False)

    @staticmethod
    def get_unwrapped_model(model: Optional[torch.nn.Module]):
        if isinstance(model, DistributedDataParallel):
            model = model.module
        return model

    def create_optimizer(self):
        assert self.model is not None
        return get_optimizer(self.model.parameters())

    def load_optimizer(self, train_state):
        optimizer = self.create_optimizer()
        if train_state is not None:
            optimizer.load_state_dict(train_state["optimizer"])

        return optimizer

    def get_epoch_marker_name(self):
        return self.config["train"].get("epoch_marker") or "epoch_marker"

    def get_ckpt_name_template(self):
        return self.config["train"].get("ckpt_name") or "model_epo{}.ckpt"

    def get_state_name_template(self):
        return self.config["train"].get("state_name") or "optimizer_epo{}.ckpt"

    def restore_train_state(self):
        epoch, model_state, train_state = None, None, None
        if self.config["train"]["restart"]:
            save_dir = self.save_dir

            epoch_marker_name = os.path.join(save_dir, self.get_epoch_marker_name())
            epoch = read_epoch_marker(epoch_marker_name)
            if epoch is not None:
                ckpt_name = os.path.join(save_dir, self.get_ckpt_name_template().format(epoch))
                model_state = torch.load(ckpt_name, map_location="cpu", weights_only=True)

                state_name = os.path.join(save_dir, self.get_state_name_template().format(epoch))
                train_state = torch.load(state_name, map_location="cpu", weights_only=True)

                logging.info(f"train state loaded at epoch {epoch} from {ckpt_name}, {state_name}")

        return epoch, model_state, train_state

    def load_initial_train_state(self):
        state_path = self.config["train"].get("initial_state_path")
        if state_path is None:
            return None

        state_path = self.resolve_path(state_path, default_source="checkpoints_mount")
        train_state = torch.load(state_path, map_location="cpu", weights_only=True)

        logging.info(f"initialized train state from {state_path}")
        return train_state

    def create_grad_norm_tracker(self, train_state):
        config = self.get_config_by_path("train.grad_norm_tracker", default=None)
        if config is None:
            return None

        config = copy.copy(config)
        if not config.pop("enabled", True):
            return None

        if "total_l2_norm_limit" not in config:
            grad_max_norm = self.get_config_by_path("train.grad_max_norm", default=0, expected_type=numbers.Number)
            if grad_max_norm > 0:
                config["total_l2_norm_limit"] = grad_max_norm

        grad_norm_tracker = GradNormTracker(**config)
        if train_state is not None:
            tracker_state = train_state.get("grad_norm_tracker")
            if tracker_state is not None:
                grad_norm_tracker.load_state_dict(tracker_state)

        return grad_norm_tracker

    def initialize_train(self, model_state, train_state):
        model = self.load_model(model_state)
        train_config = self.config["train"]
        model = self.disable_grad_for_modules(model, modules_to_train=train_config.get("modules_to_train"))
        self.model = self.make_train_model(model)

        self.optimizer = self.load_optimizer(train_state)
        self.grad_norm_tracker = self.create_grad_norm_tracker(train_state)

        self.prepare_for_training()

    def initialize_training_schedule(self):
        for config_path in self.get_config_keys_with_training_schedule():
            config = self.get_config_by_path(config_path)
            if not isinstance(config, dict):
                continue

            schedule = config.get("schedule")
            schedule_path = f"{config_path}.schedule"
            if schedule is not None and not isinstance(schedule, list):
                raise ValueError(f"{schedule_path} must be a list")

            frozen = config.get("frozen")
            frozen_path = f"{config_path}.frozen"
            if frozen is not None and not isinstance(frozen, dict):
                raise ValueError(f"{frozen_path} must be an object")

            if schedule is None and frozen is None:
                continue

            player = SchedulePlayer(
                config=config,
                config_path=config_path,
                schedule=schedule,  # type: ignore[arg-type]
                schedule_path=schedule_path,
                immutable_path_set={schedule_path, frozen_path},
            )

            if frozen is not None:
                player.freeze(frozen, frozen_path)

            player.apply(epoch=None, dry_run=True)
            self._training_schedule.append(player)

    @staticmethod
    def get_config_keys_with_training_schedule():
        return ("train",)

    @property
    def train_dataloader(self):
        train_dataloader = self._train_dataloader
        if train_dataloader is None:
            self._train_dataloader = train_dataloader = self.create_train_dataloader()

        return train_dataloader

    @abc.abstractmethod
    def create_train_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError()

    def create_train_dataloader(self):
        dataset = self.create_train_dataset()

        config = self.get_config_by_path("train.dataset", expected_type=dict)

        batch_size = config["batch_size"]
        if batch_size % self.world_size != 0:
            raise ValueError(f"Invalid batch size: {batch_size}")
        batch_size = batch_size // self.world_size

        sampler = None
        if self.world_size > 1:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, drop_last=True)

        extra_args = dict()

        n_workers = config["n_workers"]
        if n_workers > 0:
            extra_args["num_workers"] = n_workers
            extra_args["prefetch_factor"] = 10
            extra_args["persistent_workers"] = self.can_train_dataloader_use_persistent_workers()
            extra_args["pin_memory"] = True
            extra_args["worker_init_fn"] = self._data_worker_init_fn

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, drop_last=True, **extra_args
        )

        logging.info(f"created train dataloader with batch_size = {batch_size}, n_workers = {n_workers}")
        return dataloader

    @staticmethod
    def _data_worker_init_fn(worker_id):
        _ = worker_id

        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None
        seed = worker_info.seed
        sync_random_seed(seed)

    # noinspection PyMethodMayBeStatic
    def can_train_dataloader_use_persistent_workers(self):
        return True

    @staticmethod
    def is_train_dataloader_changed(change_set: Dict[str, bool]):
        return "train.dataset" in change_set

    def discard_train_dataloader(self):
        self._train_dataloader = None

    def get_train_state(self, epoch):
        assert self.optimizer is not None
        train_state = dict(
            epoch=epoch,
            optimizer=self.optimizer.state_dict(),
        )
        if self.grad_norm_tracker is not None:
            train_state["grad_norm_tracker"] = self.grad_norm_tracker.state_dict()

        return train_state

    def is_checkpoint_retained(self, epoch):
        if epoch in self._benchmarked_epoch:
            return True

        ckpt_per_epoch = self.get_config_by_path("train.ckpt_per_epoch", default=1, expected_type=int)
        if ckpt_per_epoch > 0 and (epoch + 1) % ckpt_per_epoch == 0:
            return True

        return False

    def checkpoint(self, epoch, *, is_first_checkpoint: bool):
        assert self.model is not None
        model = self.get_unwrapped_model(self.model)
        assert model is not None

        save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)

        ckpt_name = os.path.join(save_dir, self.get_ckpt_name_template().format(epoch))
        torch.save(model.state_dict(), ckpt_name)

        state_name = os.path.join(save_dir, self.get_state_name_template().format(epoch))
        torch.save(self.get_train_state(epoch), state_name)

        self.on_checkpoint(epoch, is_first_checkpoint=is_first_checkpoint)

        epoch_marker_name = os.path.join(save_dir, self.get_epoch_marker_name())
        write_epoch_marker(epoch_marker_name, epoch, is_first_checkpoint=is_first_checkpoint)

        return ckpt_name

    def on_checkpoint(self, epoch, *, is_first_checkpoint: bool):
        _ = is_first_checkpoint
        self._save_gradient_history(epoch)

    def prune_checkpoint(self, epoch):
        if self.is_checkpoint_retained(epoch):
            return

        save_dir = self.save_dir
        state_name = os.path.join(save_dir, self.get_state_name_template().format(epoch))
        try:
            os.unlink(state_name)
        except FileNotFoundError:
            pass
        except NotADirectoryError:
            pass

        ckpt_name = os.path.join(save_dir, self.get_ckpt_name_template().format(epoch))
        try:
            os.unlink(ckpt_name)
        except FileNotFoundError:
            pass
        except NotADirectoryError:
            pass

    def _save_gradient_history(self, epoch):
        grad_norm_tracker = self.grad_norm_tracker
        if grad_norm_tracker is None:
            return

        if self.get_config_by_path("train.save_gradient_history", default=False, expected_type=bool):
            torch.save(
                dict(
                    parameter_map=grad_norm_tracker.parameter_map,
                    history=grad_norm_tracker.history,
                ),
                os.path.join(self.save_dir, f"gradient_history-{epoch}.ckpt"),
            )

        grad_norm_tracker.truncate_history()

    @abc.abstractmethod
    def test_model(self, epoch, model: torch.nn.Module):
        raise NotImplementedError()

    @staticmethod
    def disable_grad_for_modules(model: torch.nn.Module, modules_to_train: Optional[Set[str]] = None):
        assert not isinstance(model, DistributedDataParallel), "model must be unwrapped before disabling grad"

        if modules_to_train is None:
            return model

        i = 0
        i_g = 0
        for n, p in model.named_parameters():
            module_name = n.split(".")[0]
            if module_name in modules_to_train:
                i_g += 1
            else:
                p.requires_grad = False
            i += 1

        if len(modules_to_train) > 0:
            assert i_g > 0, "No parameters to finetune"

        logging.info(f"Requiring grad for {i_g} parameters out of {i}")
        return model

    def prepare_for_training(self):
        pass

    @property
    def epoch_steps(self):
        return self._epoch_steps

    def prepare_for_epoch(self, epoch):
        self._step_info_timestamp = time.time()
        self.reset_step_info()

    @staticmethod
    def get_ddp_sync_steps(epoch_steps):
        ddp_sync_period = 2000

        # round to the nearest
        n_ddp_syncs = (epoch_steps + ddp_sync_period // 2) // ddp_sync_period
        if n_ddp_syncs <= 1:
            return epoch_steps

        # round up
        return (epoch_steps + n_ddp_syncs - 1) // n_ddp_syncs

    def train_one_epoch(self, epoch):
        assert self.model is not None
        assert self.optimizer is not None
        train_dataloader = self.train_dataloader
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)  # type: ignore[union-attr]

        epoch_steps = len(train_dataloader)
        limit = self.get_epoch_steps(epoch)
        if limit and limit < epoch_steps:
            train_dataloader = itertools.islice(train_dataloader, limit)
            epoch_steps = limit

        self.model.train()

        ddp_sync_steps = 0
        if self.rank >= 0:
            ddp_sync_steps = self.get_ddp_sync_steps(epoch_steps)

        self._epoch_steps = epoch_steps
        try:
            self.prepare_for_epoch(epoch)
            device = self.device

            processed = 0
            for step, d in enumerate(train_dataloader):
                if isinstance(d, dict):
                    for k in d:
                        if torch.is_tensor(d[k]):
                            d[k] = d[k].to(device, non_blocking=True)
                elif torch.is_tensor(d):
                    d = d.to(device, non_blocking=True)
                else:
                    raise ValueError(f"Unsupported batch type: {type(d)}")
                try:
                    self.train_step(epoch, step, d)
                    processed += 1
                except NonFiniteGradError:
                    self.on_nonfinite_grad(epoch, step, d)
                except BatchSkipException as e:
                    logging.info(f"epoch {epoch}, step {step}: {e}")

                self.post_step_metrics(step)
                self.log_step_info(epoch, step, len(d))

                if ddp_sync_steps > 0 and (step + 1 == self.epoch_steps or (step + 1) % ddp_sync_steps == 0):
                    logging.info("running DDP model and optimizer sync")
                    ddp_sync_model(self.model)
                    ddp_sync_optimizer(self.optimizer)

        finally:
            self._epoch_steps = 0

        if processed == 0:
            raise ValueError(f"All batches in {epoch} epoch were skipped.")

    def get_epoch_steps(self, epoch):
        _ = epoch
        return self.config["train"].get("epoch_steps")

    @abc.abstractmethod
    def train_step(self, epoch, step, batch):
        raise NotImplementedError()

    def optimizer_step(self):
        assert self.model is not None
        assert self.optimizer is not None
        model = self.get_unwrapped_model(self.model)
        assert model is not None

        grad_max_value = self.get_config_by_path("train.grad_max_value", default=0, expected_type=numbers.Number)
        if grad_max_value > 0:
            remove_nan_grad_and_clamp(model.parameters(), grad_max_value)

        if self.grad_norm_tracker is not None:
            total_norm, grad_scale = self.grad_norm_tracker.track_and_clip_(model.named_parameters())
            if not math.isfinite(total_norm):
                raise NonFiniteGradError("non-finite gradients")

            self.add_step_metric("grad_norm", total_norm)
            self.add_step_metric("grad_scale", grad_scale)

            min_grad_scale = self.get_config_by_path("train.min_grad_scale", default=0, expected_type=numbers.Number)
            if min_grad_scale > 0:
                # sync scale value to garantee all ranks skip or process batch
                if self.world_size > 1:
                    grad_scale_tensor = torch.tensor(grad_scale, dtype=torch.float64, device=self.device)
                    torch.distributed.broadcast(grad_scale_tensor, src=0)
                    grad_scale = grad_scale_tensor.item()

                if grad_scale < min_grad_scale:
                    raise TooLargeGradError(f"grad scale is too small ({grad_scale}), skipping batch...")
        else:
            grad_max_norm = self.get_config_by_path("train.grad_max_norm", default=0, expected_type=numbers.Number)
            if grad_max_norm > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)
                if not total_norm.isfinite():
                    raise NonFiniteGradError("non-finite gradients")

                self.add_step_metric("grad_norm", total_norm)
            else:
                remove_nan_grad(model.parameters())

        self.optimizer.step()

    def on_nonfinite_grad(self, epoch, step, batch):
        assert self.model is not None
        if any(not torch.all(torch.isfinite(p)) for p in self.model.parameters()):
            self.on_nonrecoverable_error(epoch, step, batch)
            raise ValueError(f"Epoch {epoch}, step {step}: non-finite model parameters are detected, stopping training")

        if not self.get_config_by_path("train.skip_nonfinite_grad", default=False, expected_type=bool):
            self.on_nonrecoverable_error(epoch, step, batch)
            raise NonFiniteGradError(
                f"Epoch {epoch}, step {step}: non-finite gradient norm is detected, stopping training"
            )

        if self.rank <= 0:
            logging.warning(f"Epoch {epoch}: skipping batch {step} with non-finite gradients")

    def on_nonrecoverable_error(self, epoch, step, batch):
        _ = step
        if self.rank <= 0:
            os.makedirs(self.save_dir, exist_ok=True)

            assert self.model is not None
            model = self.get_unwrapped_model(self.model)
            assert model is not None
            ckpt_name = os.path.join(self.save_dir, "nonrecoverable_model.ckpt")
            torch.save(model.state_dict(), ckpt_name)

            state_name = os.path.join(self.save_dir, "nonrecoverable_optimizer.ckpt")
            torch.save(self.get_train_state(epoch), state_name)

        if batch is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            if self.rank >= 0:
                batch_name = f"nonrecoverable_batch_{self.rank}.ckpt"
            else:
                batch_name = "nonrecoverable_batch.ckpt"

            batch_name = os.path.join(self.save_dir, batch_name)
            torch.save(batch, batch_name)

    def add_step_metric(self, name: str, value):
        if self.rank > 0:
            return
        if isinstance(value, torch.Tensor):
            value = value.item()
        else:
            assert isinstance(value, Real)

        value_list = self._step_metrics.get(name)
        if value_list is None:
            self._step_metrics[name] = value_list = list()

        value_list.append(value)

    def post_step_metrics(self, step):
        if len(self._step_metrics) == 0:
            return

        if step is not None:
            frequency = self.get_config_by_path("train.step_metric_frequency", default=1, expected_type=int)
            if frequency > 1 and step + 1 != self.epoch_steps and (step + 1) % frequency != 0:
                return

        metrics = {}
        for name, value_list in self._step_metrics.items():
            if len(value_list) == 0:
                continue

            if len(value_list) == 1:
                metrics[name] = value_list[0]
            else:
                metrics[name] = np.asarray(value_list).mean()

        self.metrics_writer.add_metrics(metrics)
        for value_list in self._step_metrics.values():
            value_list.clear()

    def log_step_info(self, epoch, step, batch_size):
        if self.rank > 0:
            return

        if (step + 1) % 100 != 0 and step + 1 != self.epoch_steps:
            return

        t0 = self._step_info_timestamp
        t1 = time.time()

        info = self.get_step_info()
        info = " |\t".join(f"{n}: {v}" for n, v in info.items())

        self.reset_step_info()
        self._step_info_timestamp = t1

        n_samples = self.epoch_steps * batch_size * self.world_size
        processed = (step + 1) * batch_size * self.world_size
        percent = 100 * processed / n_samples
        logging.info(f"Epoch {epoch}: [{processed}/{n_samples} ({percent:.0f}%)] time: {t1 - t0:.1f}s,\t{info}")

    def get_step_info(self):
        lr = self.get_config_by_path("train.learning_rate")
        return dict(lr=f"{lr:.6f}")

    def reset_step_info(self):
        pass

    @staticmethod
    def is_benchmark_epoch(epoch):
        return True

    def train(self):
        os.makedirs(self.save_dir, exist_ok=True)

        self.initialize_training_schedule()

        train_config = self.config["train"]
        start_epoch = train_config.get("start_epoch", 0)
        is_first_checkpoint = True

        state_epoch, model_state, train_state = self.restore_train_state()
        if state_epoch is not None and start_epoch <= state_epoch:
            is_first_checkpoint = False
            self._apply_training_schedule(state_epoch)
            start_epoch = state_epoch + 1
        else:
            self._apply_training_schedule(start_epoch)
            model_state = None
            train_state = self.load_initial_train_state()

        self.initialize_train(model_state, train_state=train_state)
        assert self.model is not None

        n_epoch = train_config["n_epoch"]
        for epoch in range(start_epoch, n_epoch):
            self._apply_training_schedule(epoch)
            self.train_one_epoch(epoch)

            if self.rank <= 0:
                self.checkpoint(epoch, is_first_checkpoint=is_first_checkpoint)
                if is_first_checkpoint:
                    is_first_checkpoint = False
                else:
                    self.prune_checkpoint(epoch - 1)

            if self.is_benchmark_epoch(epoch):
                eval_model = self.get_unwrapped_model(self.model)
                assert eval_model is not None
                self.test_model(epoch, eval_model)
                self._benchmarked_epoch.add(epoch)

    def _apply_training_schedule(self, epoch):
        change_set = dict()
        for schedule in self._training_schedule:
            change_set.update(schedule.apply(epoch=epoch, dry_run=False))

        if len(change_set) == 0:
            return

        changed_values = {n: self.get_config_by_path(n) for n, is_leaf in change_set.items() if is_leaf}
        logging.info(f"Epoch {epoch}: enacted scheduled changes: {changed_values}")

        self.on_config_changed(change_set)

    def on_config_changed(self, change_set: Dict[str, bool]):
        if self.is_train_dataloader_changed(change_set):
            self.discard_train_dataloader()

    def validate(self):
        os.makedirs(self.save_dir, exist_ok=True)
        model = self.load_model(None, pretrained=True)
        model.eval()
        self.test_model(None, model)

    def revalidate(self):
        os.makedirs(self.save_dir, exist_ok=True)
        checkpoints_folder = self.get_config_by_path("checkpoints_folder", expected_type=str)
        checkpoints_folder = self.resolve_path(checkpoints_folder, default_source="checkpoints_mount")

        start_epoch = self.get_config_by_path("train.start_epoch", default=0, expected_type=int)
        n_epoch = self.get_config_by_path("train.n_epoch", expected_type=int)

        self.initialize_training_schedule()

        for epoch in range(start_epoch, n_epoch):
            self._apply_training_schedule(epoch)

            if not self.is_benchmark_epoch(epoch):
                continue

            model_path = self.get_ckpt_name_template().format(epoch)
            model_path = os.path.join(checkpoints_folder, model_path)

            if not os.path.exists(model_path):
                logging.info(f"Skipping epoch {epoch}, model checkpoint is not found")
                continue

            logging.info(f"Testing epoch {epoch} checkpoint {model_path}")

            model = self.create_model()
            model.load_state_dict(get_state_dict(model_path))
            model.eval()
            model.to(self.device)

            self.test_model(epoch, model)

    def run(self):
        mode = self.config["mode"]
        if mode == "train":
            self.train()
        elif mode == "validate":
            self.validate()
        elif mode == "revalidate":
            self.revalidate()
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")

    def close(self):
        self._train_dataloader = None
        self.optimizer = None
        self.model = None
        super().close()


class BaseTrainVideoApp(BaseTrainApp, abc.ABC):
    def __init__(self):
        super().__init__()

        self._step_info_names = None
        self._step_info_values = None
        self._step_info_steps = None
        self._step_info_dict = None

        self.lambda_min = None
        self.lambda_max = None
        self.qp_num = None
        self.training_lambdas = None

    @property
    def is_yuv420(self):
        return bool(self.model_config["yuv420"])

    def get_config_keys_with_training_schedule(self):
        key_list: list[str] = list(super().get_config_keys_with_training_schedule())

        benchmark_test = self.get_config_by_path("benchmark_test")
        if isinstance(benchmark_test, dict):
            for testset_name in benchmark_test.keys():
                key_list.append(f"benchmark_test.{testset_name}")

        return key_list

    def is_benchmark_epoch(self, epoch):
        benchmark_test = self.get_config_by_path("benchmark_test")
        if benchmark_test is None:
            return False

        for config in benchmark_test.values():
            if self.is_testset_benchmark_epoch(config, epoch):
                return True

        return False

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
            self._auxiliary_models = AuxiliaryModels(pretrained_path, self.device, auxiliary_model_config)
        return self._auxiliary_models

    @staticmethod
    def is_testset_benchmark_epoch(config, epoch):
        test_interval = config.get("test_interval")
        if test_interval is None or test_interval == 1:
            return True
        if test_interval == 0:
            return False
        return (epoch + 1) % test_interval == 0

    def add_step_info(
        self, keys: Dict[str, int], values: torch.Tensor, *, qp_num: int = 1, q_index: Optional[torch.Tensor] = None
    ):
        if self.rank > 0:
            return

        n_bins = max(min(qp_num, 4), 1)
        if self._step_info_names is None:
            self._step_info_names = keys
            self._step_info_steps = torch.zeros(n_bins, dtype=torch.int)
            self._step_info_values = torch.zeros((n_bins, len(keys)), dtype=torch.float64)
        else:
            assert self._step_info_names == keys
            assert self._step_info_steps is not None
            assert self._step_info_values is not None
            assert n_bins == self._step_info_steps.shape[0]

        if q_index is not None:
            q_index = torch.div(q_index.cpu().long() * n_bins, qp_num, rounding_mode="trunc")
            values = values.cpu().to(torch.float64)

            self._step_info_steps.scatter_add_(0, q_index, torch.ones(q_index.shape, dtype=torch.int))
            self._step_info_values.scatter_add_(0, q_index[..., None].expand(values.shape), values)
        else:
            assert n_bins == 1
            self._step_info_steps += values.shape[0]
            self._step_info_values += values.sum(dim=0, keepdim=True)

    def get_step_info(self):
        info_names = self._step_info_names
        if info_names is None:
            return super().get_step_info()

        assert self._step_info_steps is not None
        assert self._step_info_values is not None
        bin_values = self._step_info_values / torch.clamp_min(self._step_info_steps, 1)[..., None]
        total_values = self._step_info_values.sum(0) / torch.clamp_min(self._step_info_steps.sum(0), 1)

        def format_value(idx):
            total = total_values[idx]
            precision = 2 - math.floor(math.log10(max(abs(total), 1e-6)))
            precision = min(max(precision, 0), 5)
            if bin_values.shape[1] <= 1:
                return f"{total:.{precision}f}"
            else:
                min_v = bin_values[0, idx]
                max_v = bin_values[-1, idx]
                return f"{total:.{precision}f} ({min_v:.{precision}f}/{max_v:.{precision}f})"

        info = {n: format_value(i) for n, i in info_names.items()}
        info.update(super().get_step_info())
        return info

    def reset_step_info(self):
        self._step_info_steps = None
        self._step_info_names = None
        self._step_info_values = None

    def run_benchmark_test(
        self,
        epoch,
        *,
        i_frame_model: Union[None, torch.nn.Module, Callable[[], torch.nn.Module]],
        p_frame_model: Union[None, torch.nn.Module, Callable[[], torch.nn.Module]],
        i_frame_qp_map=None,
    ):
        benchmark_test = self.get_config_by_path("benchmark_test")
        if benchmark_test is None:
            return False

        for name, config in benchmark_test.items():
            if not config:
                continue

            if epoch is not None and not self.is_testset_benchmark_epoch(config, epoch):
                continue

            self.run_benchmark_on_testset(
                epoch,
                name,
                config,
                i_frame_model=i_frame_model,
                p_frame_model=p_frame_model,
                i_frame_qp_map=i_frame_qp_map,
            )

    def run_benchmark_on_testset(
        self,
        epoch,
        testset_name,
        config: Dict[str, Any],
        *,
        i_frame_model: Union[None, torch.nn.Module, Callable[[], torch.nn.Module]],
        p_frame_model: Union[None, torch.nn.Module, Callable[[], torch.nn.Module]],
        i_frame_qp_map=None,
    ):

        if p_frame_model is not None:
            intra_period = config.get("intra_period")
            reset_period = config.get("reset_period")
        else:
            intra_period = 1
            reset_period = None

        use_i_frame_model = config.get("use_i_frame_model", None)
        if use_i_frame_model is None:
            use_i_frame_model = config.get("use_i_frame", True)

        if not use_i_frame_model:
            i_frame_pass_count = config.get("i_frame_pass_count", 1)
            i_frame_qp_shift = config.get("i_frame_qp_shift")
        else:
            i_frame_pass_count = 1
            i_frame_qp_shift = None

        params = EncoderTestParams(
            is_yuv420=self.is_yuv420,
            bitrate_list=[float(v) for v in config.get("bitrate_list", [])] if "bitrate_list" in config else None,
            intra_period=intra_period,
            reset_period=reset_period,
            max_n_frames=config.get("max_n_frames"),
            decoder_folder_path=self.resolve_path(config.get("decoder_folder"), default_source="save_dir"),
            stream_folder_path=self.resolve_path(config.get("stream_folder"), default_source="save_dir"),
            calc_psnr_rgb=config.get("calc_psnr_rgb", False),
            calc_ssim=config.get("calc_ssim", False),
            calc_ssim_rgb=config.get("calc_ssim_rgb", False),
            calc_psnr_roi=config.get("calc_psnr_roi", False),
            calc_vif=config.get("calc_vif", False),
            calc_lpips=config.get("calc_lpips", False),
            calc_lrwer_rgb=config.get("calc_lrwer_rgb", False),
            calc_deqa_score_rgb=config.get("calc_deqa_score_rgb", False),
            calc_hrd_metrics=config.get("calc_hrd_metrics", False),
            encode_bit_stream=config.get("encode_bit_stream", False),
            calc_bits_estimates=config.get("calc_bits_estimates", False),
            frame_rate=config.get("frame_rate"),
            use_decoder=config.get("use_decoder", False),
            decoder_format=config.get("decoder_format"),
            decoder_compression_options=config.get("decoder_compression_options"),
            lr_model_data_path=self.resolve_path(config.get("lr_model_data_path"), default_source="data_mount"),
            precomputed_masks_path=self.resolve_path(config.get("precomputed_masks_path"), default_source="data_mount"),
            use_i_frame_model=use_i_frame_model,
            i_frame_pass_count=i_frame_pass_count,
            i_frame_qp_shift=i_frame_qp_shift,
            ltr_period=config.get("ltr_period"),
            ltr_start_idx=config.get("ltr_start_idx", 0),
            ltr_qp_shift=config.get("ltr_qp_shift", 0),
            max_batch_size=config.get("max_batch_size", 1),
            verbose=config.get("verbose", 0),
            verbose_json=config.get("verbose_json", False),
            padding_mode=PaddingMode(config.get("padding_mode", "replicate")),
            padding_alignment=PaddingAlignment(config.get("padding_alignment", "bottom_right")),
            padding_resolution=config.get("padding_resolution"),
            calc_ane_divergence=config.get("calc_ane_divergence", False),
            ane_exact_conv=config.get("ane_exact_conv", False),
        )

        if params.use_i_frame_model:
            # instantiate i_frame_model
            if i_frame_model is not None and not isinstance(i_frame_model, torch.nn.Module):
                i_frame_model = i_frame_model()
        else:
            i_frame_model = None

        if not params.use_i_frame_model or params.intra_period != 1:
            # instantiate p_frame_model
            if p_frame_model is not None and not isinstance(p_frame_model, torch.nn.Module):
                p_frame_model = p_frame_model()
        else:
            p_frame_model = None

        if params.bitrate_list is None:
            params.i_frame_q_index_list, params.p_frame_q_index_list = self._get_q_index_lists(
                testset_name, config, i_frame_model, p_frame_model, i_frame_qp_map=i_frame_qp_map
            )
        elif (
            config.get("i_frame_q_index_list") is not None
            or config.get("p_frame_q_index_list") is not None
            or config.get("q_index_matching") is not None
        ):
            raise ValueError("bitrate_list and q-index configuration items are mutually exclusive")

        metrics = self._run_encoder_test(
            testset_name,
            config=config,
            params=params,
            epoch=epoch,
            i_frame_model=i_frame_model,
            p_frame_model=p_frame_model,
        )

        if params.decoder_folder_path:
            self._compute_decoder_metrics(config=config, encoder_params=params, encoder_metrics=metrics, epoch=epoch)

        # make sure that metrics written by rank #0 are visible to all ranks
        if self.rank >= 0:
            torch.distributed.barrier()

    def _get_q_index_lists(self, testset_name, config, i_frame_model, p_frame_model, i_frame_qp_map):
        i_frame_q_index_list = config.get("i_frame_q_index_list")
        p_frame_q_index_list = config.get("p_frame_q_index_list") if p_frame_model is not None else None

        q_index_matching = config.get("q_index_matching")
        if q_index_matching is not None:
            match = self._match_q_index_lists(q_index_matching)
            if match is not None:
                if i_frame_q_index_list is not None or p_frame_q_index_list is not None:
                    raise ValueError(
                        f"{testset_name}: q_index_matching and i/p_frame_qindex_list"
                        " configuration items are mutually exclusive"
                    )

            assert match is not None
            i_frame_q_index_list, p_frame_q_index_list = match
            if len(i_frame_q_index_list) == 0:
                raise ValueError(f"{testset_name}: q_index_matching failed, no q-points found")

            if not (isinstance(i_frame_q_index_list, dict) and isinstance(p_frame_q_index_list, dict)):
                if i_frame_q_index_list == p_frame_q_index_list:
                    logging.info(f"{testset_name}: matched q-indices: {i_frame_q_index_list}")
                else:
                    logging.info(f"{testset_name}: matched i-frame q-indices: {i_frame_q_index_list}")
                    logging.info(f"{testset_name}: matched p-frame q-indices: {p_frame_q_index_list}")
            else:
                logging.info(f"{testset_name}: matched q-indices per video")

            return i_frame_q_index_list, p_frame_q_index_list

        if i_frame_q_index_list is None and p_frame_q_index_list is None:
            raise ValueError("No q-points (i_frame_q_index_list and/or p_frame_q_index_list) are specified for testing")

        if i_frame_q_index_list is not None:
            i_frame_q_index_list = tuple(i_frame_q_index_list)
        if p_frame_q_index_list is not None:
            p_frame_q_index_list = tuple(p_frame_q_index_list)

        if i_frame_qp_map is not None:
            if p_frame_q_index_list is None:
                raise ValueError("q-points for p-frame model are not specified for testing")

            if i_frame_q_index_list is None:
                i_frame_q_index_list = tuple(int(i_frame_qp_map[x]) for x in p_frame_q_index_list)
        elif p_frame_model is not None and i_frame_model is not None:
            if p_frame_q_index_list is None:
                p_frame_q_index_list = self._rescale_q_index_list(
                    i_frame_q_index_list, i_frame_model.get_qp_num(), p_frame_model.get_qp_num()
                )
            if i_frame_q_index_list is None:
                i_frame_q_index_list = self._rescale_q_index_list(
                    p_frame_q_index_list, p_frame_model.get_qp_num(), i_frame_model.get_qp_num()
                )

        return i_frame_q_index_list, p_frame_q_index_list

    @staticmethod
    def _rescale_q_index_list(q_index_list, qp_num, new_num):
        if qp_num == new_num:
            return q_index_list

        return tuple((x * (new_num - 1) + qp_num // 2) // (qp_num - 1) for x in q_index_list)

    def _match_q_index_lists(self, config):
        candidates_filename = self.resolve_path(config.get("candidates"), default_source="checkpoints_mount")
        if not candidates_filename:
            return None

        anchor_filename = self.resolve_path(config["anchor"], default_source="checkpoints_mount")
        if not anchor_filename:
            raise ValueError("No anchor metrics file is specified")
        anchor_q_index_list = config.get("anchor_q_index_list")
        scenarios = config.get("scenarios")
        min_qp_distance = config.get("min_qp_distance", 3)
        match_per_video = config.get("match_per_video", False)
        return match_q_index_lists(
            candidate_metrics=candidates_filename,
            anchor_metrics=anchor_filename,
            anchor_q_index_list=anchor_q_index_list,
            scenarios=scenarios,
            min_distance=min_qp_distance,
            match_per_video=match_per_video,
        )

    def _run_encoder_test(
        self,
        testset_name: str,
        *,
        config: Mapping[str, Any],
        params: EncoderTestParams,
        epoch: Optional[int],
        i_frame_model,
        p_frame_model,
    ):
        if epoch is not None:
            output_json_path = f"metrics_{testset_name}_epo_{epoch}.json"
        else:
            output_json_path = f"metrics_{testset_name}.json"
        output_json_path = os.path.join(self.save_dir, output_json_path)

        testset_desc_filename = self.resolve_path(config["config"], default_source=".")
        results = run_encoder_test(
            params,
            rank=self.rank,
            testset_name=testset_name,
            testset_desc_filename=testset_desc_filename,
            testset_root=self.resolve_path(config.get("data_path")),
            i_frame_model=i_frame_model,
            p_frame_model=p_frame_model,
            auxiliary_models=self.auxiliary_models,
            output_path=output_json_path,
        )

        if self.rank <= 0:
            self.log_benchmark_metrics(results, testset_name=testset_name, config=config, epoch=epoch)

            anchor_path = config.get("anchor")
            frame_type = "all" if not params.use_i_frame_model else config.get("frame_type", "default")
            if anchor_path:
                compare_test_with_anchor(
                    epoch=epoch or 0,
                    anchor_results_path=self.resolve_path(anchor_path, default_source="checkpoints_mount"),
                    test_results_path=output_json_path,
                    testset_name=testset_name,
                    save_dir=self.save_dir,
                    metrics_writer=self.metrics_writer,
                    frame_type=frame_type,
                    distortion_metrics=config.get("distortion_metrics", ("psnr",)),
                )

            if config.get("calc_ane_divergence", False):
                self._log_ane_divergence_metrics(results, testset_name=testset_name, epoch=epoch)

        return results

    def _compute_decoder_metrics(
        self, *, config: Dict[str, Any], encoder_params: EncoderTestParams, encoder_metrics, epoch: Optional[int]
    ):
        decoder_config = config.get("decoder_metrics")
        if decoder_config is None:
            return
        if not isinstance(decoder_config, Mapping):
            raise ValueError("decoder_metrics must be an YAML object")

        testset_name = decoder_config.get("name")
        if not testset_name:
            return

        bitrate_list = encoder_params.bitrate_list
        if bitrate_list is not None:
            i_frame_q_index_list = None
            p_frame_q_index_list = None
        else:
            i_frame_q_index_list = encoder_params.i_frame_q_index_list
            p_frame_q_index_list = encoder_params.p_frame_q_index_list

        params = EncoderTestParams(
            is_yuv420=encoder_params.is_yuv420,
            bitrate_list=bitrate_list,
            intra_period=encoder_params.intra_period,
            reset_period=encoder_params.reset_period,
            max_n_frames=encoder_params.max_n_frames,
            calc_psnr_rgb=decoder_config.get("calc_psnr_rgb", encoder_params.calc_psnr_rgb),
            calc_ssim=decoder_config.get("calc_ssim", encoder_params.calc_ssim),
            calc_ssim_rgb=decoder_config.get("calc_ssim_rgb", encoder_params.calc_ssim_rgb),
            calc_vif=decoder_config.get("calc_vif", encoder_params.calc_vif),
            calc_hrd_metrics=decoder_config.get("calc_hrd_metrics", encoder_params.calc_hrd_metrics),
            frame_rate=decoder_config.get("frame_rate", encoder_params.frame_rate),
            use_i_frame_model=encoder_params.use_i_frame_model,
            i_frame_q_index_list=i_frame_q_index_list,
            p_frame_q_index_list=p_frame_q_index_list,
            verbose=decoder_config.get("verbose", encoder_params.verbose),
            verbose_json=decoder_config.get("verbose_json", encoder_params.verbose_json),
        )

        from .frame_reader_model_adapter import FrameReaderModelFactory

        model = FrameReaderModelFactory(
            clip_folder=encoder_params.decoder_folder_path,
            clip_filename_format=encoder_params.decoder_filename_format,
            clip_format=encoder_params.decoder_format or "yuv",
            seq_id_regexp=decoder_config.get("seq_id_regexp"),
            metrics=encoder_metrics,
            device=self.device,
        )

        self._run_encoder_test(
            testset_name, config=decoder_config, params=params, epoch=epoch, i_frame_model=model, p_frame_model=model
        )

    def log_benchmark_metrics(self, results, *, testset_name: str, config: Mapping[str, Any], epoch: Optional[int]):
        log_metrics = config.get("log_metrics")
        if log_metrics is None or len(log_metrics) == 0:
            return

        metrics = dict()
        for ds in results.values():
            for s in ds.values():
                for rate, r in enumerate(s.values()):
                    for metric_name in log_metrics:
                        name = f"{testset_name}_{metric_name}{rate:03d}"
                        value_list = metrics.get(name)
                        if value_list is None:
                            metrics[name] = value_list = list()

                        if metric_name in r:
                            value_list.append(r[metric_name])
                        else:
                            value_list.append(r[f"ave_{metric_name}"])

        self.metrics_writer.add_metrics(
            {name: np.asarray(value_list).mean() for name, value_list in metrics.items()},
            step=epoch,
        )

    def _log_ane_divergence_metrics(self, results, *, testset_name: str, epoch: Optional[int]):
        """Log the ANE divergence metrics that need non-mean aggregation, per rate."""
        from collections import defaultdict

        gpu_by_rate = defaultdict(list)
        ane_by_rate = defaultdict(list)
        feat_max_by_rate = defaultdict(list)
        for ds in results.values():
            for s in ds.values():
                for rate, r in enumerate(s.values()):
                    g = r.get("ave_p_frame_ane_psnr_gpu")
                    a = r.get("ave_p_frame_ane_psnr_ane")
                    fm = r.get("ave_p_frame_ane_feature_max_abs_diff")
                    if g is not None:
                        gpu_by_rate[rate].append(g)
                    if a is not None:
                        ane_by_rate[rate].append(a)
                    if fm is not None:
                        feat_max_by_rate[rate].append(fm)

        metrics = {}
        for rate in sorted(gpu_by_rate):
            gpu, ane = gpu_by_rate[rate], ane_by_rate.get(rate, [])
            if not (gpu and ane):
                continue
            avg_drop = float(np.mean(gpu)) - float(np.mean(ane))
            max_drop = float(np.max([g - a for g, a in zip(gpu, ane)]))
            metrics[f"{testset_name}_ane_psnr_drop{rate:03d}"] = avg_drop
            metrics[f"{testset_name}_ane_psnr_max_drop{rate:03d}"] = max_drop

        for rate in sorted(feat_max_by_rate):
            metrics[f"{testset_name}_ane_feature_max_diff{rate:03d}"] = float(np.max(feat_max_by_rate[rate]))

        self.metrics_writer.add_metrics(metrics, step=epoch)

    def _unpack_bpp(self, bin_index):
        assert self._step_info_dict is not None
        if bin_index is None:
            bin_index = ...

        names = self._step_info_dict["names"]
        bpp_y_ind = names["bpp_y"]
        bpp_z_ind = names["bpp_z"]

        values = self._step_info_dict["values"]
        bpp_y = values[bin_index, bpp_y_ind]
        bpp_z = values[bin_index, bpp_z_ind]
        steps = self._step_info_dict["steps"]
        if not torch.all(steps > 0):
            return None
        steps = steps[bin_index]

        if bin_index is Ellipsis:
            bpp_y = bpp_y.sum()
            bpp_z = bpp_z.sum()
            steps = steps.sum()

        bpp = (bpp_y + bpp_z) / steps
        return bpp

    def control_lambdas(self):
        controller_active = self.get_config_by_path("train.controller.active", default=False)
        freeze = self.get_config_by_path("train.controller.freeze_lambdas", default=False)
        if (not controller_active) or freeze:
            return

        self._step_info_dict = {
            "steps": self._step_info_steps,
            "names": self._step_info_names,
            "values": self._step_info_values,
        }
        # Broadcast info information to all ranks
        if torch.distributed.is_initialized():
            data_list = [self._step_info_dict if self.rank == 0 else None]
            torch.distributed.broadcast_object_list(data_list, src=0)
            self._step_info_dict = data_list[0]

        assert self._step_info_dict is not None
        if self._step_info_dict["steps"] is None:
            return

        def get_factor(bpp_target, bpp, k_p):
            # log(lmd) = log(lmd) + k_p * log(bpp_target / bpp)
            # or lmd = lmd * (bpp_target / bpp) ^ k_p
            factor = torch.pow(bpp_target / bpp, k_p).item()

            min_threshold = self.get_config_by_path("train.controller.min_threshold", default=None)
            if min_threshold is not None:
                if abs(factor - 1) < min_threshold:
                    factor = 1
            return factor

        k_p = self.get_config_by_path("train.controller.gain", default=1.0e-2)
        bpp_target = self.get_config_by_path("train.controller.bpp_target")
        assert self.lambda_min is not None
        assert self.lambda_max is not None
        if isinstance(bpp_target, list):
            bpp_target_low = bpp_target[0]
            bpp_target_high = bpp_target[1]

            bpp_low = self._unpack_bpp(bin_index=0)
            bpp_high = self._unpack_bpp(bin_index=-1)

            # Skip update if there is little data
            if (bpp_low is None) or (bpp_high is None):
                return

            factor_low = get_factor(bpp_target_low, bpp_low, k_p)
            factor_high = get_factor(bpp_target_high, bpp_high, k_p)

            self.lambda_min *= factor_low
            self.lambda_max *= factor_high

        else:
            bpp = self._unpack_bpp(bin_index=None)
            if bpp is None:
                return
            factor = get_factor(bpp_target, bpp, k_p)

            self.lambda_min *= factor
            self.lambda_max *= factor

        assert self.qp_num is not None
        lambdas = get_training_lambdas((self.lambda_min, self.lambda_max), self.qp_num)
        self.training_lambdas = torch.from_numpy(lambdas).float()

    def initialize_lambdas(self, model):
        self.qp_num = model.get_qp_num()
        controller_active = self.get_config_by_path("train.controller.active", default=False)
        if not controller_active or self.training_lambdas is None:
            self.lambda_min, self.lambda_max = self.get_config_by_path("train.loss_lambdas")
            lambdas = get_training_lambdas((self.lambda_min, self.lambda_max), self.qp_num)
            self.training_lambdas = torch.from_numpy(lambdas).float()

        control_at_step = self.get_config_by_path("train.controller.at_step", default=False)
        if controller_active:
            if not control_at_step:
                self.control_lambdas()

        assert self.lambda_min is not None and self.lambda_max is not None
        min_lmd = round(self.lambda_min, 2)
        max_lmd = round(self.lambda_max, 2)
        logging.info(f"Starting epoch with lambdas: {min_lmd}-{max_lmd}. {controller_active=}, {control_at_step=} ")
