# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Learning rate policies configured by ``train.lr_policy``.

A policy is a list of segments, each starting at its ``epoch`` and lasting until the next one starts. A linear or
cosine segment goes from ``start_lr`` to ``end_lr``. The last segment is constant, so the policy does not depend on
``train.n_epoch``. E.g. a warmup and a cosine decay, restarted at epoch 384::

    train:
      start_epoch: 344
      lr_policy:
        - { epoch: 344,   type: linear,   start_lr: 0,     end_lr: 5.e-5 }
        - { epoch: 345,   type: cosine,   start_lr: 5.e-5, end_lr: 1.e-6 }
        - { epoch: 384,   type: linear,   start_lr: 0,     end_lr: 5.e-5 }
        - { epoch: 384.5, type: cosine,   start_lr: 5.e-5, end_lr: 1.e-6 }
        - { epoch: 398,   type: constant, lr: 5.e-6 }

Epochs are absolute and may be fractional, and the first segment must start no later than ``train.start_epoch``. The
rate is a function of the fractional epoch ``epoch + (step + 1) / epoch_steps``, so it changes every step and a
restarted run continues on the same curve.

The policy is built once, when training starts, so ``train.schedule`` cannot change ``train.lr_policy``, nor
``train.learning_rate`` when a policy is configured.
"""

import dataclasses
import itertools
import math
from typing import Any

_SEGMENT_KEYS = {
    "linear": {"epoch", "type", "start_lr", "end_lr"},
    "cosine": {"epoch", "type", "start_lr", "end_lr"},
    "constant": {"epoch", "type", "lr"},
}


@dataclasses.dataclass(frozen=True)
class LRPolicySegment:
    """From ``start_lr`` at ``start_epoch`` to ``end_lr`` at ``end_epoch``: linear, cosine or constant."""

    type: str
    start_epoch: float
    end_epoch: float
    start_lr: float
    end_lr: float

    def lr_at(self, epoch: float) -> float:
        if self.type == "constant":
            return self.start_lr
        progress = min(max((epoch - self.start_epoch) / (self.end_epoch - self.start_epoch), 0.0), 1.0)
        if self.type == "cosine":
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.start_lr + (self.end_lr - self.start_lr) * progress

    def __str__(self) -> str:
        end = f"{self.end_epoch:g}" if math.isfinite(self.end_epoch) else ""
        lrs = f"{self.start_lr:g}" if self.type == "constant" else f"{self.start_lr:g} -> {self.end_lr:g}"
        return f"{self.start_epoch:g}-{end}: {self.type} {lrs}"


@dataclasses.dataclass(frozen=True)
class LRPolicy:
    """Consecutive segments; each one lasts until the next one starts, and the last one is constant."""

    segments: tuple[LRPolicySegment, ...]

    def lr_at(self, epoch: float) -> float:
        """Learning rate at a fractional epoch.

        A step's rate is taken where the step ends, so a step ending exactly where a segment ends is the last step of
        that segment.
        """
        return next(segment for segment in self.segments if epoch <= segment.end_epoch).lr_at(epoch)

    def __str__(self) -> str:
        return "; ".join(str(segment) for segment in self.segments)


def create_lr_policy(config: Any, *, start_epoch: float = 0) -> LRPolicy:
    """Policy for a ``train.lr_policy`` config, a list of segments."""
    if not (isinstance(config, list) and config and all(isinstance(segment, dict) for segment in config)):
        raise ValueError(f"train.lr_policy must be a non-empty list of segments, got {config!r}")
    for segment in config:
        keys = _SEGMENT_KEYS.get(segment.get("type"))
        if keys is None:
            raise ValueError(f"train.lr_policy segment type must be one of {tuple(_SEGMENT_KEYS)}: {dict(segment)}")
        if set(segment) != keys:
            raise ValueError(f"A {segment['type']} train.lr_policy segment takes {sorted(keys)}: {dict(segment)}")
    if config[-1]["type"] != "constant":
        raise ValueError(f"The last train.lr_policy segment must be constant, got {dict(config[-1])}")

    epochs = [float(segment["epoch"]) for segment in config]
    if not (
        all(math.isfinite(epoch) for epoch in epochs)
        and epochs[0] >= 0
        and all(end > start for start, end in itertools.pairwise(epochs))
    ):
        raise ValueError(f"train.lr_policy epochs must be finite, >= 0 and increasing, got {epochs}")
    if epochs[0] > start_epoch:
        raise ValueError(
            f"train.lr_policy starts at epoch {epochs[0]:g}, after train.start_epoch {start_epoch}: "
            "it defines no learning rate for the epochs in between"
        )

    segments = []
    for segment, start, end in zip(config, epochs, epochs[1:] + [math.inf]):
        if segment["type"] == "constant":
            start_lr = end_lr = float(segment["lr"])
        else:
            start_lr, end_lr = float(segment["start_lr"]), float(segment["end_lr"])
        if not all(math.isfinite(lr) and lr >= 0 for lr in (start_lr, end_lr)):
            raise ValueError(f"train.lr_policy learning rates must be finite and >= 0: {dict(segment)}")
        segments.append(LRPolicySegment(segment["type"], start, end, start_lr, end_lr))
    return LRPolicy(tuple(segments))
