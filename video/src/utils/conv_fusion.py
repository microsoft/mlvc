# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn


def _subpel_params(subpel):
    """Extract (weight, bias, factor) from a SubpelConv2x-style block (1x1 conv + PixelShuffle)."""
    conv, ps = subpel.conv[0], subpel.conv[1]
    assert isinstance(ps, nn.PixelShuffle), "expected PixelShuffle as the second sub-layer"
    assert conv.kernel_size == (1, 1) and conv.padding == (0, 0)
    assert conv.stride == (1, 1) and conv.dilation == (1, 1) and conv.groups == 1
    return conv.weight, conv.bias, ps.upscale_factor


def _compose_subpel(wA, bA, rA, wB, bB, rB):
    """Compose (convA + PixelShuffle(rA)) then (convB + PixelShuffle(rB)) into one 1x1 conv + PixelShuffle(rA*rB)."""
    in_ch = wA.shape[1]
    mid_ch = wB.shape[1]
    out_ch = wB.shape[0] // (rB * rB)
    assert wA.shape[0] == mid_ch * rA * rA, "channel mismatch between composed blocks"
    device, dtype = wA.device, wA.dtype

    WA = wA.view(mid_ch, rA, rA, in_ch)  # [mid, ia, ja, in]
    WB = wB.view(out_ch, rB, rB, mid_ch)  # [out, ib, jb, mid]
    bA_ = torch.zeros(mid_ch, rA, rA, dtype=dtype, device=device) if bA is None else bA.view(mid_ch, rA, rA)
    bB_ = torch.zeros(out_ch, rB, rB, dtype=dtype, device=device) if bB is None else bB.view(out_ch, rB, rB)

    R = rA * rB
    wF = torch.zeros(out_ch, R, R, in_ch, dtype=dtype, device=device)
    bF = torch.zeros(out_ch, R, R, dtype=dtype, device=device)
    for ia in range(rA):
        for ja in range(rA):
            wA_sub, bA_sub = WA[:, ia, ja, :], bA_[:, ia, ja]  # [mid,in], [mid]
            for ib in range(rB):
                for jb in range(rB):
                    i, j = rB * ia + ib, rB * ja + jb  # subpixel offset in the PS(R) grid
                    wB_sub = WB[:, ib, jb, :]  # [out, mid]
                    wF[:, i, j, :] = wB_sub @ wA_sub
                    bF[:, i, j] = wB_sub @ bA_sub + bB_[:, ib, jb]
    return wF.reshape(out_ch * R * R, in_ch, 1, 1), bF.reshape(out_ch * R * R), R


def fuse_subpel_chain(subpels) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Fuse a chain of N SubpelConv2x (1x1 conv + PixelShuffle(2)) into one conv + PixelShuffle(2**N)."""
    assert len(subpels) >= 1
    w, b, r = _subpel_params(subpels[0])
    for sp in subpels[1:]:
        w2, b2, r2 = _subpel_params(sp)
        w, b, r = _compose_subpel(w, b, r, w2, b2, r2)
    if b is None:
        b = torch.zeros(w.shape[0], dtype=w.dtype, device=w.device)
    return w, b, r


def _compose_stride_convs(w1, b1, s1, w2, b2, s2):
    """Compose convA (s1xs1 stride s1) then convB (s2xs2 stride s2) into one SxS stride S conv (S = s1*s2)."""
    out_ch, mid_ch = w2.shape[0], w2.shape[1]
    in_ch = w1.shape[1]
    assert w1.shape[0] == mid_ch, "channel mismatch between composed convs"
    device, dtype = w1.device, w1.dtype

    S = s1 * s2
    wF = torch.zeros(out_ch, in_ch, S, S, dtype=dtype, device=device)
    for c in range(s2):
        for d in range(s2):
            w2_sub = w2[:, :, c, d]  # [out, mid]
            for a in range(s1):
                for b in range(s1):
                    i, j = s1 * c + a, s1 * d + b  # input-kernel position in the SxS grid
                    wF[:, :, i, j] = w2_sub @ w1[:, :, a, b]

    bF = b2.clone() if b2 is not None else torch.zeros(out_ch, dtype=dtype, device=device)
    if b1 is not None:
        bF = bF + w2.sum(dim=(2, 3)) @ b1
    return wF, bF, S


def fuse_conv_chain(convs) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Fuse a chain of N non-overlapping stride-s convs (kernel s x s) into one stride-s^N conv."""
    assert len(convs) >= 1
    for conv in convs:
        s = conv.stride[0]
        assert conv.kernel_size == (s, s) and conv.stride == (s, s), "expected non-overlapping square kernel==stride"
        assert conv.padding == (0, 0) and conv.dilation == (1, 1) and conv.groups == 1

    w, b, s = convs[0].weight, convs[0].bias, convs[0].stride[0]
    for conv in convs[1:]:
        w, b, s = _compose_stride_convs(w, b, s, conv.weight, conv.bias, conv.stride[0])
    if b is None:
        b = torch.zeros(w.shape[0], dtype=w.dtype, device=w.device)
    return w, b, s


if __name__ == "__main__":
    torch.manual_seed(0)

    print("Testing fuse_conv_chain (N stride-2 convs -> one stride-2^N conv)...")
    for n in (2, 3):
        convs = [nn.Conv2d(16, 16, kernel_size=2, stride=2) for _ in range(n)]
        seq = nn.Sequential(*convs)
        w, b, stride = fuse_conv_chain(convs)
        fused = nn.Conv2d(16, 16, kernel_size=stride, stride=stride)
        assert fused.bias is not None
        fused.weight.data, fused.bias.data = w, b
        x = torch.randn(1, 16, 2**n, 2**n)
        err = (seq(x) - fused(x)).abs().max().item()
        print(f"  N={n}: max error {err:.2e}")
        assert err < 1e-5, "fuse_conv_chain failed!"

    print("Testing fuse_subpel_chain (N SubpelConv2x -> one conv + PixelShuffle(2^N))...")

    class SubpelConv2x(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch * 4, 1), nn.PixelShuffle(2))

        def forward(self, x):
            return self.conv(x)

    for n in (2, 3):
        subpels = [SubpelConv2x(16, 16) for _ in range(n)]
        seq = nn.Sequential(*subpels)
        w, b, factor = fuse_subpel_chain(subpels)
        fused = nn.Sequential(nn.Conv2d(16, 16 * factor * factor, 1), nn.PixelShuffle(factor))
        fused[0].weight.data, fused[0].bias.data = w, b
        x = torch.randn(1, 16, 4, 4)
        err = (seq(x) - fused(x)).abs().max().item()
        print(f"  N={n}: max error {err:.2e}")
        assert err < 1e-5, "fuse_subpel_chain failed!"

    print("Tests passed!")
