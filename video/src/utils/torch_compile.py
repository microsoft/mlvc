# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""torch.compile settings, compilation, cache resets and GPU memory diagnostics for training apps.

``CompileSettings`` is a pure view of the ``train.compile_*`` config keys. ``TorchCompileManager``
applies them to modules and owns the process-wide compiler cache lifecycle; it reads the config on
every call so training-schedule updates take effect without rebuilding the manager.
"""

import contextlib
import dataclasses
import gc
import logging
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, overload

import torch
import torch.nn as nn

_UNSET = object()


def _lookup(config: Optional[Mapping], key_path: str, default: Any = None) -> Any:
    value: Any = config
    for component in key_path.split("."):
        if not isinstance(value, Mapping):
            return default
        value = value.get(component, _UNSET)
        if value is _UNSET:
            return default
    return value


def _typed(config: Optional[Mapping], key_path: str, default: Any, expected_type) -> Any:
    value = _lookup(config, key_path, default)
    if not isinstance(value, expected_type):
        raise ValueError(f"Config value {key_path} is {type(value)}, while {expected_type} is expected")
    return value


@dataclasses.dataclass(frozen=True)
class CompileSettings:
    """All torch.compile parameters used for training, parsed from the ``train.compile_*`` config keys."""

    ENABLED_KEY = "train.compile_model"
    BACKEND_KEY = "train.compile_backend"
    MODE_KEY = "train.compile_mode"
    OPTIONS_KEY = "train.compile_options"
    CONFIG_KEYS = (BACKEND_KEY, MODE_KEY, OPTIONS_KEY, ENABLED_KEY)

    enabled: bool = True
    backend: str = "inductor"
    # from_config defaults this to "reduce-overhead" for inductor without options.
    mode: Optional[str] = None
    options: Optional[dict] = None
    fullgraph: bool = True
    # Static shapes; callers normalize input strides (see with_standard_strides) to avoid recompiles.
    dynamic: bool = False

    def __post_init__(self):
        if self.mode is not None and self.options is not None:
            raise ValueError(f"{self.MODE_KEY} and {self.OPTIONS_KEY} cannot both be set (torch.compile limitation)")

    @classmethod
    def from_config(cls, config: Optional[Mapping]) -> "CompileSettings":
        """Parses settings from the root app config (the mapping that contains ``train``)."""
        enabled = _typed(config, cls.ENABLED_KEY, True, bool)
        backend = _typed(config, cls.BACKEND_KEY, "inductor", str)
        options = _typed(config, cls.OPTIONS_KEY, None, (dict, type(None)))
        # torch.compile accepts either mode or options; only default the mode for inductor without options.
        default_mode = "reduce-overhead" if options is None and backend == "inductor" else None
        mode = _typed(config, cls.MODE_KEY, default_mode, (str, type(None)))
        return cls(enabled=enabled, backend=backend, mode=mode, options=options)

    @property
    def uses_cudagraphs(self) -> bool:
        if not self.enabled:
            return False
        if self.backend == "cudagraphs":
            return True
        if self.backend != "inductor":
            return False
        if self.options is not None:
            return bool(self.options.get("triton.cudagraphs", False))
        return self.mode in ("reduce-overhead", "max-autotune")

    def compile_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``nn.Module.compile`` / ``torch.compile``."""
        return {
            "backend": self.backend,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "options": self.options,
            "mode": self.mode,
        }


@overload
def with_standard_strides(tensor: torch.Tensor) -> torch.Tensor: ...


@overload
def with_standard_strides(tensor: None) -> None: ...


def with_standard_strides(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Returns tensor with row-major strides, copying only when its strides differ.

    Frame views such as ``videos[:, i]`` keep ``stride(0) == n_frames * C * H * W``. torch.compile guards
    on exact strides, so every new clip length would otherwise trigger a recompile. ``contiguous()`` is not
    enough: with batch size 1 the view already reports ``is_contiguous()`` and is returned unchanged.
    """
    if tensor is None:
        return None
    expected_stride = []
    step = 1
    for size in reversed(tensor.shape):
        expected_stride.append(step)
        step *= size
    if tensor.stride() == tuple(reversed(expected_stride)):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def log_gpu_memory(device: torch.device, stage: str, rank: int = 0):
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    gib = 2**30
    allocated = torch.cuda.memory_allocated(device) / gib
    reserved = torch.cuda.memory_reserved(device) / gib
    free, total = (value / gib for value in torch.cuda.mem_get_info(device))
    logging.info(
        f"GPU memory {stage} (rank={rank}, device={device}): "
        f"allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB, free={free:.2f} GiB, total={total:.2f} GiB"
    )


def log_gpu_memory_pools(device: torch.device, rank: int = 0):
    """Break reserved memory down by allocator pool; pool (0, 0) is the default pool, others are CUDA graph pools."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    gib = 2**30
    pools: dict[tuple, list[int]] = {}
    for segment in torch.cuda.memory_snapshot():
        if segment["device"] != device.index and device.index is not None:
            continue
        stats = pools.setdefault(tuple(segment["segment_pool_id"]), [0, 0, 0, 0])
        stats[0] += segment["total_size"]
        stats[1] += segment["active_size"]
        stats[2] += 1
        stats[3] = max(stats[3], segment["total_size"] - segment["active_size"])
    for pool_id, (reserved, active, count, largest_idle) in sorted(pools.items(), key=lambda item: -item[1][0]):
        logging.info(
            f"GPU memory pool {pool_id} (rank={rank}): reserved={reserved / gib:.2f} GiB, "
            f"active={active / gib:.2f} GiB, segments={count}, max idle in one segment={largest_idle / gib:.2f} GiB"
        )


def reset_compiler(device: torch.device, rank: int = 0):
    """Reset torch.compile caches and return the freed CUDA graph pool memory to the driver."""
    log_gpu_memory(device, "before torch.compile reset", rank)
    torch.compiler.reset()
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    log_gpu_memory(device, "after torch.compile reset", rank)
    log_gpu_memory_pools(device, rank)


class TorchCompileManager:
    """Compiles training modules and resets process-wide compiler caches when compiled graphs become stale."""

    def __init__(
        self,
        config: Callable[[], Optional[Mapping]],
        device: Callable[[], torch.device],
        rank: Callable[[], int] = lambda: 0,
        extra_reset_keys: Iterable[str] = (),
    ):
        """
        Args:
            config: returns the current root app config; read on every call.
            device: returns the training device.
            rank: returns the distributed rank, used for logging only.
            extra_reset_keys: non-compile config keys whose change invalidates compiled graphs
                (e.g. keys read by Python-level guards inside the compiled forward).
        """
        self._config = config
        self._device = device
        self._rank = rank
        self.reset_keys = tuple(extra_reset_keys) + CompileSettings.CONFIG_KEYS

    @property
    def settings(self) -> CompileSettings:
        return CompileSettings.from_config(self._config())

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def compile(self, module: nn.Module, name: str) -> nn.Module:
        """Compiles ``module.forward`` in place, preserving module identity and parameters."""
        settings = self.settings
        if not settings.enabled:
            return module
        if self._device().type != "cuda":
            raise RuntimeError(f"{CompileSettings.ENABLED_KEY} requires a CUDA device")
        kwargs = settings.compile_kwargs()
        logging.info(f"Compiling {name} forward pass with torch.compile {kwargs}")
        module.compile(**kwargs)
        return module

    def reset(self):
        reset_compiler(self._device(), self._rank())

    def on_config_changed(self, change_set: Mapping[str, bool], model: Optional[nn.Module]) -> bool:
        """Resets caches and recompiles (or disables compilation of) ``model`` when a reset key changed.

        Returns True when a reset happened.
        """
        triggers = [key for key in self.reset_keys if key in change_set]
        if not triggers or model is None:
            return False
        enabled = self.enabled
        if not enabled and CompileSettings.ENABLED_KEY not in change_set:
            return False

        config = self._config()
        changed = {key: _lookup(config, key) for key in triggers}
        logging.info(f"Resetting process-wide torch.compile caches after config change: {changed}")
        self.reset()
        if enabled:
            self.compile(model, "training model")
        else:
            logging.info("Disabling torch.compile for training model")
            model.compile(disable=True)
        return True

    @contextlib.contextmanager
    def eval_scope(self, free_cudagraph_pools: bool = False) -> Iterator[None]:
        """Runs compiled modules eagerly; optionally frees CUDA graph pools first to make room for large eval models."""
        if free_cudagraph_pools and self.settings.uses_cudagraphs:
            logging.info("Resetting process-wide torch.compile caches before evaluation with CUDA graphs")
            self.reset()
        # Force eager mode to avoid recompiling the module; this has no noticeable performance penalty.
        with torch.compiler.set_stance("force_eager"):
            yield

    def detach_carried(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Detaches tensors carried into the next compiled call.

        Compiled outputs may live in buffers that the next call overwrites (CUDA graph replay), so they are
        cloned when compilation is enabled.
        """
        clone = self.enabled
        return {n: v.detach().clone() if clone else v.detach() for n, v in tensors.items()}

    def own_accumulated_grads(self, parameters: Iterable[torch.Tensor]):
        """Moves gradients that will be accumulated over further backward passes out of CUDA graph buffers.

        The first backward stores the compiled backward output as ``param.grad``; with CUDA graphs that buffer is
        overwritten by the next replay, so it is cloned before running more compiled forward/backward passes.
        """
        if not self.settings.uses_cudagraphs:
            return
        for p in parameters:
            if p.grad is not None:
                p.grad = p.grad.clone()
