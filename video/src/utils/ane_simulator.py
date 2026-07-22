# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""PyTorch simulation of ANE FP16 arithmetic."""

import torch
import torch.nn.functional as F

from typing import Union

__all__ = [
    "ane_conv2d",
    "ane_add",
    "ane_mul",
    "ane_leaky_relu",
    "ane_clamp",
    "ane_sigmoid",
    "ane_approx_sigmoid",
    "ane_pixel_shuffle",
    "ane_pixel_unshuffle",
    # STE (straight-through estimator) variants for differentiable training.
    "ane_conv2d_ste",
    "ane_add_ste",
    "ane_mul_ste",
    "ane_leaky_relu_ste",
    "ane_sigmoid_ste",
    "ane_approx_sigmoid_ste",
    "ane_pixel_shuffle_ste",
]

P = 16  # Binary point for fixed-point accumulator
_COMPUTE_DTYPE = torch.float32


def _round_fp16_ties_away(values):
    """Round to FP16 using TiesToAway"""
    dt = _COMPUTE_DTYPE
    values = values.to(dt)
    signs = values.sign()
    abs_val = values.abs()
    is_zero = abs_val == 0

    # FP16 biased exponent via frexp: frexp(x) = (m, e) with 0.5 <= |m| < 1.
    _, raw_exp = torch.frexp(abs_val.clamp(min=2.0 ** (-24)))
    biased_exp = (raw_exp + 14).clamp(0, 30)

    # Scale maps the FP16 unit grid to integers; max(1, ·) unifies subnormals.
    effective_exp = biased_exp.clamp(min=1)
    scale = torch.pow(torch.tensor(2.0, dtype=dt, device=values.device), (25 - effective_exp).to(dt))

    q = torch.floor(abs_val * scale + 0.5)  # TiesToAway for positive magnitudes
    abs_result = q / scale

    inf = torch.tensor(float("inf"), dtype=dt, device=values.device)
    abs_result = torch.where(abs_result > 65504, inf, abs_result)
    abs_result = torch.where(is_zero, torch.zeros_like(abs_result), abs_result)

    result = (signs * abs_result).to(torch.float16)
    zero = torch.tensor(0.0, dtype=torch.float16, device=values.device)
    return torch.where(result == zero, zero, result)  # canonicalize -0 -> +0


def _round_to_11sig_ties_away(values):
    """Round float64 values to 11 significant bits (10 frac + implicit 1), TiesToAway."""
    v = values.to(torch.float64).clone()
    nonzero = v != 0
    if not bool(nonzero.any()):
        return v
    sign = v[nonzero].sign()
    va = v[nonzero].abs()
    e = va.log2().floor()
    ulp = torch.exp2((e - 10).to(torch.float64))
    m = va / ulp  # in [1024, 2048)
    mr = (m + 0.5).floor()  # TiesToAway (all positive)
    carry = mr >= 2048
    mr = torch.where(carry, mr / 2, mr)
    ulp = torch.where(carry, ulp * 2, ulp)
    v[nonzero] = sign * mr * ulp
    return v


def _get_biased_exp(x_fp16):
    """Extract biased FP16 exponent (0-31), clamped to min 1 for subnormals."""
    raw = (x_fp16.contiguous().view(torch.int16).to(torch.int32) >> 10) & 0x1F
    return torch.clamp(raw, min=1)


def ane_conv2d(
    input_fp16,
    weight_fp16,
    bias_fp16=None,
    stride: Union[int, tuple] = 1,
    padding: Union[int, tuple] = 0,
    groups: int = 1,
):
    """Simulate ANE Conv2d with P=16 fixed-point accumulation.

    Args:
        input_fp16:  (N, C_in, H, W) float16 tensor
        weight_fp16: (C_out, C_in/groups, kH, kW) float16 tensor
        bias_fp16:   (C_out,) float16 tensor or None
        stride:      int or tuple of int
        padding:     int or tuple of int
        groups:      int

    Returns:
        (N, C_out, H_out, W_out) float16 tensor
    """
    x = input_fp16.to(torch.float16)
    w = weight_fp16.to(torch.float16)

    N, _, H_in, W_in = x.shape
    C_out, C_per_group, kH, kW = w.shape
    K = C_per_group * kH * kW

    stride_t = (stride, stride) if isinstance(stride, int) else tuple(stride)
    padding_t = (padding, padding) if isinstance(padding, int) else tuple(padding)

    H_out = (H_in + 2 * padding_t[0] - kH) // stride_t[0] + 1
    W_out = (W_in + 2 * padding_t[1] - kW) // stride_t[1] + 1
    L = H_out * W_out
    C_out_per_group = C_out // groups

    # im2col: skip unfold for 1x1 stride-1 no-padding (most common case)
    if kH == 1 and kW == 1 and stride_t == (1, 1) and padding_t == (0, 0):
        cols = x.reshape(N, groups, K, L)
    else:
        cols = F.unfold(x.float(), (kH, kW), padding=padding_t, stride=stride_t)
        cols = cols.to(torch.float16).reshape(N, groups, K, L)

    w_grouped = w.reshape(groups, C_out_per_group, K)

    # Biased exponents for product flush - computed once for all chunks.
    e_x = _get_biased_exp(cols)  # (N, groups, K, L)
    e_w = _get_biased_exp(w_grouped)  # (groups, C_out_per_group, K)
    need_flush = (e_x.min() + e_w.min()).item() < 14

    # Chunk over output channels to bound the 5D products tensor (~16M elems).
    max_elements = 16_000_000
    cout_chunk = max(1, max_elements // max(N * K * L, 1))
    cout_chunk = min(cout_chunk, C_out_per_group)

    cols_compute = cols.to(_COMPUTE_DTYPE)  # (N, groups, K, L)

    out = torch.empty(N, groups, C_out_per_group, L, dtype=torch.float16, device=x.device)

    for c0 in range(0, C_out_per_group, cout_chunk):
        c1 = min(c0 + cout_chunk, C_out_per_group)
        wc = w_grouped[:, c0:c1].to(_COMPUTE_DTYPE)  # (groups, chunk, K)

        # Products: (N, groups, chunk, K, L)
        prods = cols_compute[:, :, None, :, :] * wc[None, :, :, :, None]

        scaled = prods * (2**P)
        q = torch.where(scaled >= 0, torch.floor(scaled + 0.5), torch.ceil(scaled - 0.5))

        if need_flush:
            e_wc = e_w[:, c0:c1]
            combined = e_x[:, :, None, :, :] + e_wc[None, :, :, :, None]
            q = torch.where(combined < 14, torch.zeros_like(q), q)

        s = q.to(torch.int32).sum(dim=3)  # (N, groups, chunk, L)

        if bias_fp16 is not None:
            bc = bias_fp16.to(torch.float16).reshape(groups, C_out_per_group)[:, c0:c1]
            bs = bc.to(_COMPUTE_DTYPE) * (2**P)
            bq = torch.where(bs >= 0, torch.floor(bs + 0.5), torch.ceil(bs - 0.5)).to(torch.int32)
            s = s + bq[None, :, :, None]

        acc = s.to(_COMPUTE_DTYPE) / (2**P)
        out[:, :, c0:c1] = _round_fp16_ties_away(acc)

    return out.reshape(N, C_out, H_out, W_out)


def ane_add(a, b):
    """ANE elementwise add: FP16 add with TiesToAway rounding."""
    a = a.to(torch.float16).to(_COMPUTE_DTYPE)
    b = b.to(torch.float16).to(_COMPUTE_DTYPE)
    return _round_fp16_ties_away(a + b)


def ane_mul(a, b):
    """ANE elementwise multiply: TiesToAway with double-rounding for subnormals."""
    a = a.to(torch.float16)
    b = b.to(torch.float16)
    product = a.to(_COMPUTE_DTYPE) * b.to(_COMPUTE_DTYPE)

    # Subnormal products also go through the ANE multiplier's 11-bit pre-rounding.
    is_sub = product.abs() < (2.0**-14)  # FP16 smallest normal = 2^-14
    if bool(is_sub.any()):
        product = product.clone()
        product[is_sub] = _round_to_11sig_ties_away(product[is_sub]).to(_COMPUTE_DTYPE)

    return _round_fp16_ties_away(product)


def ane_leaky_relu(x, negative_slope=0.01):
    """Passthrough for >=0, FP16 multiply by slope for <0 with TiesToAway."""
    x = x.to(torch.float16)
    slope = torch.tensor(negative_slope, dtype=torch.float16, device=x.device)
    neg = ane_mul(x, slope.expand_as(x))
    zero = torch.tensor(0.0, dtype=torch.float16, device=x.device)
    return torch.where(x >= zero, x, neg)


def ane_clamp(x, min_val, max_val):
    """Clamp: comparison + select, no arithmetic. Exact."""
    x = x.to(torch.float16)
    return torch.clamp(
        x, min=torch.tensor(min_val, dtype=torch.float16).item(), max=torch.tensor(max_val, dtype=torch.float16).item()
    )


def ane_sigmoid(x):
    """Simulate ANE sigmoid: piecewise-linear LUT at 0.5 spacing, FP16 TiesToAway.

    35-entry LUT (knots at -8.5, -8, ..., 0, ..., 8, 8.5).
    Positive x: y = y_lo + round(round(2*dx) * dy)
    Negative x: y = y_hi - round(round(2*dx_hi) * dy)
    Outside [-8.5, 8.5]: saturates to 0/1.
    """
    x = x.to(torch.float16)
    device = x.device

    knots = torch.arange(-8.5, 9.0, 0.5, dtype=torch.float64, device=device)
    lut = (1.0 / (1.0 + torch.exp(-knots))).to(torch.float16)

    dt = _COMPUTE_DTYPE
    x_c = x.to(dt)
    x_clamped = torch.clamp(x_c, -8.499, 8.499)

    seg = torch.floor((x_clamped + 8.5) * 2.0).to(torch.long)
    seg = torch.clamp(seg, 0, len(knots) - 2)

    y_lo = lut[seg]
    y_hi = lut[seg + 1]
    knot_lo_f16 = (seg.to(dt) * 0.5 - 8.5).to(torch.float16)
    knot_hi_f16 = ((seg + 1).to(dt) * 0.5 - 8.5).to(torch.float16)

    dy = _round_fp16_ties_away(y_hi.to(dt) - y_lo.to(dt))
    is_neg = x < torch.tensor(0.0, dtype=torch.float16, device=device)

    # Positive path: y_lo + round(t * dy)
    dx = _round_fp16_ties_away(x.to(dt) - knot_lo_f16.to(dt))
    t = _round_fp16_ties_away(dx.to(dt) * 2.0)
    t_dy = _round_fp16_ties_away(t.to(dt) * dy.to(dt))
    result_pos = _round_fp16_ties_away(y_lo.to(dt) + t_dy.to(dt))

    # Negative path: y_hi - round(omt * dy)
    dx_hi = _round_fp16_ties_away(knot_hi_f16.to(dt) - x.to(dt))
    omt = _round_fp16_ties_away(dx_hi.to(dt) * 2.0)
    omt_dy = _round_fp16_ties_away(omt.to(dt) * dy.to(dt))
    result_neg = _round_fp16_ties_away(y_hi.to(dt) - omt_dy.to(dt))

    return torch.where(is_neg, result_neg, result_pos)


def ane_approx_sigmoid(x):
    """Simulate ANE approx_sigmoid: 0.2 * (relu(x + 2.5) - relu(x - 2.5))."""
    x = x.to(torch.float16)
    device = x.device

    def f16(v):
        return torch.tensor(v, dtype=torch.float16, device=device)

    xp = ane_clamp(ane_add(x, f16(2.5)), 0.0, 65504.0)  # relu(x + 2.5)
    xm = ane_clamp(ane_add(x, f16(-2.5)), 0.0, 65504.0)  # relu(x - 2.5)
    diff = ane_add(xp, ane_mul(f16(-1.0), xm))  # relu(x+2.5) - relu(x-2.5)
    return ane_mul(f16(0.2), diff)


def ane_pixel_shuffle(x, factor):
    """ANE pixel_shuffle via conv engine - values go through P=16 round-trip."""
    x = x.to(torch.float16)
    xf = x.to(_COMPUTE_DTYPE)
    scaled = xf * (2**P)
    q = torch.where(scaled >= 0, torch.floor(scaled + 0.5), torch.ceil(scaled - 0.5)).to(torch.int32)
    rounded = _round_fp16_ties_away(q.to(_COMPUTE_DTYPE) / (2**P))
    return F.pixel_shuffle(rounded.float(), factor).to(torch.float16)


def ane_pixel_unshuffle(x, factor):
    """Pure data reordering - exact (no ANE arithmetic)."""
    x = x.to(torch.float16).float()
    return F.pixel_unshuffle(x, factor).to(torch.float16)


# ---------------------------------------------------------------------------
# STE (Straight-Through Estimator) variants for differentiable training.
# ---------------------------------------------------------------------------


def ane_conv2d_ste(
    input_fp16,
    weight_fp16,
    bias_fp16=None,
    stride: Union[int, tuple] = 1,
    padding: Union[int, tuple] = 0,
    groups: int = 1,
):
    gpu = (
        F.conv2d(
            input_fp16.to(torch.float16).float(),
            weight_fp16.to(torch.float16).float(),
            bias_fp16.to(torch.float16).float() if bias_fp16 is not None else None,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        .half()
        .float()
    )
    with torch.no_grad():
        ane = ane_conv2d(input_fp16, weight_fp16, bias_fp16, stride=stride, padding=padding, groups=groups).float()
    return gpu + (ane - gpu).detach()


def ane_add_ste(a, b):
    gpu = a.float() + b.float()
    with torch.no_grad():
        ane = ane_add(a, b).float()
    return gpu + (ane - gpu).detach()


def ane_mul_ste(a, b):
    gpu = a.float() * b.float()
    with torch.no_grad():
        ane = ane_mul(a, b).float()
    return gpu + (ane - gpu).detach()


def ane_leaky_relu_ste(x, negative_slope=0.01):
    gpu = F.leaky_relu(x.float(), negative_slope)
    with torch.no_grad():
        ane = ane_leaky_relu(x, negative_slope).float()
    return gpu + (ane - gpu).detach()


def ane_sigmoid_ste(x):
    gpu = torch.sigmoid(x.float())
    with torch.no_grad():
        ane = ane_sigmoid(x).float()
    return gpu + (ane - gpu).detach()


def ane_approx_sigmoid_ste(x):
    gpu = 0.2 * (F.relu(x.float() + 2.5) - F.relu(x.float() - 2.5))
    with torch.no_grad():
        ane = ane_approx_sigmoid(x).float()
    return gpu + (ane - gpu).detach()


def ane_pixel_shuffle_ste(x, factor):
    gpu = F.pixel_shuffle(x.float(), factor)
    with torch.no_grad():
        ane = ane_pixel_shuffle(x, factor).float()
    return gpu + (ane - gpu).detach()
