# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Stores the parameters of each top-level submodule in one flat tensor to cut per-parameter CPU overhead in training.

A submodule that a step does not use gets no gradient, so the optimizer skips it as it skips its parameters. Each
element of a top-level ModuleList or ModuleDict gets its own flat tensor, since models often select one per call.
"""

from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn

# Parameter offsets are aligned so that the views keep the alignment that vectorized CUDA kernels require.
_ALIGNMENT = 32


class _Layout:
    """Places a group of parameters in one flat tensor."""

    def __init__(self, params: list[nn.Parameter]):
        self.shapes = [p.shape for p in params]
        # Chunk sizes for flat.split(): one chunk per parameter plus padding chunks between them. split() keeps
        # the backward a single concatenation; slicing would materialize a full-size gradient per parameter.
        self.chunks: list[int] = []
        self.chunk_index: list[int] = []
        self.numel = 0
        for p in params:
            numel = p.numel()
            padding = -numel % _ALIGNMENT
            self.chunk_index.append(len(self.chunks))
            self.chunks.append(numel)
            if padding:
                self.chunks.append(padding)
            self.numel += numel + padding

    def views(self, flat: torch.Tensor) -> list[torch.Tensor]:
        chunks = flat.split(self.chunks)
        return [chunks[i].view(shape) for i, shape in zip(self.chunk_index, self.shapes)]


class FlatParameters(nn.Module):
    """Runs ``module`` with its parameters taken from one flat trainable tensor per top-level submodule.

    The original parameters stay registered on ``module`` (so ``state_dict`` and eager evaluation are unchanged),
    but become frozen views of the flat storage. Optimizer updates of ``flats`` are therefore visible through them.
    Only calls through this wrapper produce gradients; ``flats`` are the tensors to optimize.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        named = list(module.named_parameters())
        if not named:
            raise ValueError("module has no parameters to flatten")
        dtype, device = named[0][1].dtype, named[0][1].device
        containers = {n for n, m in module.named_children() if isinstance(m, (nn.ModuleList, nn.ModuleDict))}
        groups: dict[str, list[int]] = {}
        for index, (name, p) in enumerate(named):
            if p.dtype != dtype or p.device != device:
                raise ValueError(f"all parameters must share dtype and device, {name} is {p.dtype} on {p.device}")
            if not p.requires_grad:
                raise ValueError(f"parameter {name} is frozen; flat parameters support training all parameters only")
            parts = name.split(".", 2)
            groups.setdefault(".".join(parts[:2]) if parts[0] in containers else parts[0], []).append(index)

        self._names = [name for name, _ in named]
        # indices into named_parameters() order of the parameters in each flat tensor
        self._groups = list(groups.values())
        self._layouts = [_Layout([named[i][1] for i in group]) for group in self._groups]

        flats = []
        for group, layout in zip(self._groups, self._layouts):
            flat = torch.zeros(layout.numel, dtype=dtype, device=device)
            with torch.no_grad():
                for i, view in zip(group, layout.views(flat)):
                    view.copy_(named[i][1])
            flats.append(nn.Parameter(flat))
        self.flats = nn.ParameterList(flats)

        for group, layout, flat in zip(self._groups, self._layouts, self.flats):
            for i, view in zip(group, layout.views(flat.data)):
                p = named[i][1]
                p.data = view
                p.requires_grad_(False)

    def _parameter_views(self) -> dict[str, torch.Tensor]:
        views = {}
        # indexing, since dynamo cannot trace iteration over a ParameterList
        for k, (group, layout) in enumerate(zip(self._groups, self._layouts)):
            views.update((self._names[i], view) for i, view in zip(group, layout.views(self.flats[k])))
        return views

    def forward(self, *args, **kwargs):
        return torch.func.functional_call(self.module, self._parameter_views(), args, kwargs, strict=False)

    def clear_grads(self, names: Iterable[str]):
        """Sets the gradients of the flat tensors that hold the named parameters to None.

        The optimizer then skips them, as it skips parameters without gradients. Each flat tensor must hold only
        named parameters.
        """
        names = set(names)
        for k, group in enumerate(self._groups):
            group_names = {self._names[i] for i in group}
            if group_names.isdisjoint(names):
                continue
            if not group_names <= names:
                raise ValueError(
                    f"cannot clear gradients of {sorted(group_names & names)} without the other parameters of their "
                    f"flat tensor: {sorted(group_names - names)}"
                )
            self.flats[k].grad = None

    def flatten_optimizer_state(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        """Converts the state dict of an optimizer over ``module.parameters()`` (one group) to one over ``flats``."""
        groups = state_dict["param_groups"]
        if len(groups) != 1 or len(groups[0]["params"]) != len(self._names):
            raise ValueError(
                f"expected one optimizer parameter group with {len(self._names)} parameters, "
                f"got {[len(g['params']) for g in groups]}"
            )
        param_ids = groups[0]["params"]
        per_param = [state_dict["state"].get(param_id) for param_id in param_ids]

        state = {}
        for flat_id, (group, layout) in enumerate(zip(self._groups, self._layouts)):
            present = [per_param[i] for i in group if per_param[i]]
            if not present:
                continue
            flat_state: dict[str, Any] = {}
            for key, value in present[0].items():
                if not torch.is_tensor(value):
                    flat_state[key] = value
                elif key == "step":
                    # Parameters can only have different step counts if some were skipped.
                    flat_state[key] = torch.stack([s[key] for s in present]).max()
                else:
                    flat_value = value.new_zeros(layout.numel)
                    for i, view in zip(group, layout.views(flat_value)):
                        if per_param[i]:
                            if per_param[i][key].shape != view.shape:
                                raise ValueError(
                                    f"optimizer state {key!r} of {self._names[i]} has shape "
                                    f"{tuple(per_param[i][key].shape)}, expected {tuple(view.shape)}"
                                )
                            view.copy_(per_param[i][key])
                    flat_state[key] = flat_value
            state[flat_id] = flat_state

        return {"state": state, "param_groups": [dict(groups[0], params=list(range(len(self._groups))))]}

    def unflatten_optimizer_state(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        """Converts the state dict of an optimizer over ``flats`` (one group) to one over ``module.parameters()``."""
        groups = state_dict["param_groups"]
        if len(groups) != 1 or len(groups[0]["params"]) != len(self._groups):
            raise ValueError(f"expected one optimizer parameter group with {len(self._groups)} flat parameters")

        state = {}
        for flat_id, group, layout in zip(groups[0]["params"], self._groups, self._layouts):
            flat_state = state_dict["state"].get(flat_id)
            if not flat_state:
                continue
            per_param: list[dict[str, Any]] = [{} for _ in group]
            for key, value in flat_state.items():
                if torch.is_tensor(value) and value.dim() > 0:
                    for s, view in zip(per_param, layout.views(value)):
                        s[key] = view.clone()
                else:
                    for s in per_param:
                        s[key] = value.clone() if torch.is_tensor(value) else value
            state.update(zip(group, per_param))

        return {
            "state": dict(sorted(state.items())),
            "param_groups": [dict(groups[0], params=list(range(len(self._names))))],
        }
