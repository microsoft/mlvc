# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ANE arithmetic simulation as a TorchFunctionMode."""

import contextlib

import torch
import torch.nn.functional as F

from . import ane_simulator as A

__all__ = ["ANESimMode", "ane_simulation", "drift_simulation"]

_MAX_FP16 = 65504.0

# Op categories that can be selectively simulated via ane_simulation(simu=...).
_SIMU_OPS = frozenset({"conv", "add", "mul", "sigmoid", "pixel_shuffle", "relu", "clamp", "leaky_relu", "silu"})


def _as_tensor(v, ref):
    return v if torch.is_tensor(v) else torch.as_tensor(v, dtype=ref.dtype, device=ref.device)


class ANESimMode(torch.overrides.TorchFunctionMode):
    """Intercept torch ops and replace them with ANE-simulated equivalents.

    Args:
        differentiable: use STE variants (exact ANE forward value, GPU-op
            gradient) so the enclosed forward is trainable.
        exact_conv: simulate conv with bit-exact P=16 fixed-point accumulation
            (slow) instead of the fast fp16 ``F.conv2d`` round-trip.
        simu: iterable of op categories to simulate (default: all of
            ``_SIMU_OPS``). Ops not selected pass through to plain torch - e.g.
            ``simu=("conv",)`` simulates only convolution.
    """

    def __init__(self, differentiable: bool = False, exact_conv: bool = False, simu=None):
        super().__init__()
        self.exact_conv = exact_conv
        if simu is None:
            self.simu = _SIMU_OPS
        else:
            if isinstance(simu, str):
                simu = (simu,)
            self.simu = frozenset(simu)
            unknown = self.simu - _SIMU_OPS
            if unknown:
                raise ValueError(f"ane_simulation: unknown ops {sorted(unknown)}; valid: {sorted(_SIMU_OPS)}")
        if differentiable:
            self._mul, self._add = A.ane_mul_ste, A.ane_add_ste
            self._sigmoid, self._pixel_shuffle = A.ane_sigmoid_ste, A.ane_pixel_shuffle_ste
            self._leaky_relu = A.ane_leaky_relu_ste
            self._conv_exact = A.ane_conv2d_ste
        else:
            self._mul = lambda a, b: A.ane_mul(a, b).float()
            self._add = lambda a, b: A.ane_add(a, b).float()
            self._sigmoid = lambda x: A.ane_sigmoid(x).float()
            self._pixel_shuffle = lambda x, f: A.ane_pixel_shuffle(x, f).float()
            self._leaky_relu = lambda x, s: A.ane_leaky_relu(x, s).float()
            self._conv_exact = lambda i, w, b, **k: A.ane_conv2d(i, w, b, **k).float()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        sim = self.simu

        # ANE fp16 simulation applies only to floating-point math.
        # Integer/bool tensors are index/shape bookkeeping (e.g. q_index through shift_qp);
        # routing them through ane_* would cast to float and break tensor indexing.
        if any(torch.is_tensor(a) and not a.is_floating_point() for a in args):
            return func(*args, **kwargs)

        if "mul" in sim and name in ("mul", "__mul__", "__rmul__", "multiply"):
            a, b = args[0], args[1]
            r = a if torch.is_tensor(a) else b
            return self._mul(_as_tensor(a, r), _as_tensor(b, r))
        if "add" in sim and name in ("add", "__add__", "__radd__"):
            a, b = args[0], args[1]
            r = a if torch.is_tensor(a) else b
            return self._add(_as_tensor(a, r), _as_tensor(b, r))
        if "add" in sim and name in ("sub", "__sub__", "subtract"):  # a - b
            a, b = args[0], args[1]
            r = a if torch.is_tensor(a) else b
            return self._add(_as_tensor(a, r), -_as_tensor(b, r))
        if "add" in sim and name in ("rsub", "__rsub__"):  # a.__rsub__(b) == b - a
            a, b = args[0], args[1]
            return self._add(_as_tensor(b, a), -_as_tensor(a, a))
        if "sigmoid" in sim and name == "sigmoid":
            return self._sigmoid(args[0])
        if "pixel_shuffle" in sim and name == "pixel_shuffle":
            return self._pixel_shuffle(args[0], args[1] if len(args) > 1 else kwargs["upscale_factor"])
        # In-place variants (relu_/clamp_/leaky_relu_) return a fresh tensor, not mutate args[0].
        if "relu" in sim and name in ("relu", "relu_"):
            return A.ane_clamp(args[0].half(), 0.0, _MAX_FP16).float()
        if "leaky_relu" in sim and name in ("leaky_relu", "leaky_relu_"):
            slope = args[1] if len(args) > 1 else kwargs.get("negative_slope", 0.01)
            return self._leaky_relu(args[0], slope)
        if "silu" in sim and name == "silu":
            x = args[0]
            return self._mul(self._sigmoid(x), x)
        if "clamp" in sim and name in ("clamp", "clamp_"):
            x = args[0]
            mn = kwargs.get("min", args[1] if len(args) > 1 else None)
            mx = kwargs.get("max", args[2] if len(args) > 2 else None)
            return A.ane_clamp(x.half(), -_MAX_FP16 if mn is None else mn, _MAX_FP16 if mx is None else mx).float()
        if "conv" in sim and name in ("conv2d", "convolution"):
            # conv2d and convolution have different positional layouts past `padding`:
            #   conv2d:      (..., dilation, groups)
            #   convolution: (..., dilation, transposed, output_padding, groups)
            inp, w = args[0], args[1]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            stride = args[3] if len(args) > 3 else kwargs.get("stride", 1)
            padding = args[4] if len(args) > 4 else kwargs.get("padding", 0)
            dilation = args[5] if len(args) > 5 else kwargs.get("dilation", 1)
            if name == "convolution":
                transposed = args[6] if len(args) > 6 else kwargs.get("transposed", False)
                output_padding = args[7] if len(args) > 7 else kwargs.get("output_padding", 0)
                groups = args[8] if len(args) > 8 else kwargs.get("groups", 1)
            else:  # conv2d
                transposed, output_padding = False, 0
                groups = args[6] if len(args) > 6 else kwargs.get("groups", 1)

            if transposed or output_padding not in (0, (0, 0)):
                raise NotImplementedError(
                    f"ANE conv simulation does not support transposed convolution "
                    f"(transposed={transposed}, output_padding={output_padding})"
                )
            if dilation not in (1, (1, 1)):
                raise NotImplementedError(f"ANE conv simulation does not support dilation={dilation}")

            if self.exact_conv:
                return self._conv_exact(inp, w, bias, stride=stride, padding=padding, groups=groups)
            # Fast path: fp16 F.conv2d round-trip.
            return (
                F.conv2d(
                    inp.half().float(),
                    w.half().float(),
                    bias.half().float() if bias is not None else None,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                )
                .half()
                .float()
            )

        return func(*args, **kwargs)


def ane_simulation(differentiable: bool = False, exact_conv: bool = False, simu=None) -> ANESimMode:
    return ANESimMode(differentiable=differentiable, exact_conv=exact_conv, simu=simu)


def drift_simulation(
    mode: str = "ane",
    *,
    device_type: str = "cuda",
    differentiable: bool = True,
    simu=None,
    exact_conv: bool = False,
):
    if mode == "ane":
        return ane_simulation(differentiable=differentiable, simu=simu, exact_conv=exact_conv)
    if mode == "fp16":
        return torch.autocast(device_type=device_type, dtype=torch.float16)
    if mode == "bf16":
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    if mode == "fp32":
        return contextlib.nullcontext()
    raise ValueError(f"unknown drift_mode {mode!r}; expected 'ane', 'fp16', 'bf16', or 'fp32'")
