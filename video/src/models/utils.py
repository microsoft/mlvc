# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import contextlib
from typing import Optional

import torch
from torch import nn
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint as _torch_checkpoint


class LowerBound(Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx, x, bound):
        bound = torch.tensor(bound, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x, bound)
        return torch.clamp_min(x, bound)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        mask = torch.logical_or(torch.greater(x, bound), torch.less(grad_output, 0))
        return mask.to(grad_output.dtype) * grad_output, None


class UpperBound(Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx, x, bound):
        bound = torch.tensor(bound, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x, bound)
        return torch.clamp_max(x, bound)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        mask = torch.logical_or(torch.less(x, bound), torch.greater(grad_output, 0))
        return mask.to(grad_output.dtype) * grad_output, None


def apply_upper_lower_bound(x: torch.Tensor, *, lower: Optional[float] = None, upper: Optional[float] = None):
    if lower is None and upper is None:
        return x

    if x.requires_grad:
        if lower is not None:
            x_out: torch.Tensor = LowerBound.apply(x, lower)  # type: ignore[reportAssignmentType]
            x = x_out
        if upper is not None:
            x_out = UpperBound.apply(x, upper)  # type: ignore[reportAssignmentType]
            x = x_out
        return x
    else:
        if upper is None:
            assert lower is not None
            return torch.clamp_min(x, lower)
        elif lower is None:
            return torch.clamp_max(x, upper)
        else:
            return torch.clamp(x, lower, upper)


def _preserve_function_modes_context_fn():
    """Re-apply the active TorchFunctionMode(s) during checkpoint recompute.
    Currently needed for ANE simulation. No-op when no mode is active.
    """
    modes = [torch._C._get_function_stack_at(i) for i in range(torch._C._len_torch_function_stack())]

    def context_fn():
        @contextlib.contextmanager
        def recompute():
            with contextlib.ExitStack() as stack:
                for mode in modes:
                    stack.enter_context(mode)
                yield

        return contextlib.nullcontext(), recompute()

    return context_fn


class CkptModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_ckpt = False

    def set_use_ckpt(self, use_ckpt=True):
        self.use_ckpt = use_ckpt

    def internal_forward(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        if self.use_ckpt:
            return _torch_checkpoint(
                self.custom_forward(self.internal_forward),
                *args,
                **kwargs,
                preserve_rng_state=False,
                use_reentrant=False,
                context_fn=_preserve_function_modes_context_fn(),
            )
        return self.internal_forward(*args, **kwargs)

    @staticmethod
    def custom_forward(f):
        def custom_func(*args, **kwargs):
            return f(*args, **kwargs)

        return custom_func
