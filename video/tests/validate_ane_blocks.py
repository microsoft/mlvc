# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Validate ANE simulation."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
import warnings
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.utils.ane_simulator as ane

warnings.filterwarnings("ignore")


def _randn_fp16(*shape):
    return torch.from_numpy(np.random.randn(*shape).astype(np.float16))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
H, W = 128, 128
NUM_SEEDS = 10
PAD_CH = 64  # ballast channel width (forces ANE scheduling)

# Model-realistic channel sizes from dmc61sbr_mini_reglu config
FEATURE_CH = 48
HIDDEN_CH = 192
RECON_CH = 192
Z_CH = 48


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    name: str
    config: str
    mismatch: int
    total: int
    max_abs_err: float
    mean_abs_err: float
    ref_type: str = "SIM"

    @property
    def rate(self):
        return 100 * self.mismatch / self.total if self.total > 0 else 0.0

    @property
    def status(self):
        return "PASS" if self.mismatch == 0 else "FAIL"


def build_and_run_ane(target_module, input_dict, output_names):
    """Build a CoreML model wrapping target_module + ballast, run on ANE.

    Args:
        target_module: nn.Module to test (already .eval(), weights set).
        input_dict: dict of {name: torch.float16 tensor} for the target module.
        output_names: list of output names from the target module.

    Returns:
        dict of {name: torch.float16 tensor} with ANE outputs.
    """
    n_target_inputs = len(input_dict)

    class _Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.target = target_module
            self.p1 = nn.Conv2d(PAD_CH, PAD_CH, 1, bias=False)
            self.p2 = nn.Conv2d(PAD_CH, PAD_CH, 1, bias=False)
            self.p3 = nn.Conv2d(PAD_CH, PAD_CH, 1, bias=False)

        def _ballast(self, pad):
            return self.p3(self.p2(self.p1(pad)))

    class _Wrap1(_Wrap):
        def forward(self, x, pad):
            return self.target(x), self._ballast(pad)

    class _Wrap2(_Wrap):
        def forward(self, x, y, pad):
            return self.target(x, y), self._ballast(pad)

    m = _Wrap1() if n_target_inputs == 1 else _Wrap2()
    m.eval()

    # Pre-quantize ALL weights to FP16
    with torch.no_grad():
        for p in m.parameters():
            p.data = p.data.half().float()

    pad_t = torch.zeros(1, PAD_CH, H, W, dtype=torch.float16)

    # Build full input dict
    target_keys = list(input_dict.keys())
    full_input = {}
    full_input["x"] = input_dict[target_keys[0]]
    if n_target_inputs == 2:
        full_input["y"] = input_dict[target_keys[1]]
    full_input["pad"] = pad_t

    # Trace (needs float32 tensors)
    trace_inputs = tuple(v.float() for v in full_input.values())
    with torch.no_grad():
        tr = torch.jit.trace(m, trace_inputs)

    # Convert to numpy for CoreML
    np_input = {k: v.numpy() for k, v in full_input.items()}

    ct_inputs = []
    for name, arr in np_input.items():
        ct_inputs.append(ct.TensorType(shape=arr.shape, dtype=np.float16, name=name))

    ct_outputs = [
        ct.TensorType(dtype=np.float16, name="out"),
        ct.TensorType(dtype=np.float16, name="pout"),
    ]

    cm = ct.convert(
        tr,
        inputs=ct_inputs,
        outputs=ct_outputs,
        minimum_deployment_target=ct.target.macOS13,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )
    assert isinstance(cm, ct.models.MLModel)

    ml = ct.models.MLModel(
        cm.get_spec(),
        weights_dir=cm.weights_dir,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    preds = ml.predict(np_input)
    out_np = np.array(preds["out"]).astype(np.float16)
    return {"out": torch.from_numpy(out_np)}


def compare(ane_out, ref_out):
    """Compare ANE and reference outputs (both torch float16 tensors)."""
    ane = ane_out.to(torch.float32)
    ref = ref_out.to(torch.float32)
    diff = (ane - ref).abs()
    mismatch = int((ane_out != ref_out).sum().item())
    return mismatch, ane_out.numel(), float(diff.max().item()), float(diff.mean().item())


def run_test(name, config, make_module_and_inputs, sim_fn, num_seeds=NUM_SEEDS):
    """Run a single test across multiple seeds and return TestResult."""
    total_mm, total_n = 0, 0
    max_err = 0.0
    sum_mean_err = 0.0

    for seed in range(num_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        module, input_dict, output_names = make_module_and_inputs()
        module.eval()

        ane_out = build_and_run_ane(module, input_dict, output_names)
        ref = sim_fn(module, input_dict)

        mm, n, mx, mn = compare(ane_out["out"], ref["out"])
        total_mm += mm
        total_n += n
        max_err = max(max_err, mx)
        sum_mean_err += mn

    mean_err = sum_mean_err / num_seeds if num_seeds > 0 else 0.0
    return TestResult(name, config, total_mm, total_n, max_err, mean_err)


# ---------------------------------------------------------------------------
# Atomic op tests
# ---------------------------------------------------------------------------
def test_conv1x1(c_in, c_out, bias):
    def make():
        m = nn.Conv2d(c_in, c_out, 1, bias=bias)
        x = _randn_fp16(1, c_in, H, W)
        return m, {"x": x}, ["out"]

    return make


def test_depthwise_conv3x3(channels, bias):
    def make():
        m = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias)
        x = _randn_fp16(1, channels, H, W)
        return m, {"x": x}, ["out"]

    return make


def test_strided_conv2x2(c_in, c_out, bias):
    def make():
        m = nn.Conv2d(c_in, c_out, 2, stride=2, bias=bias)
        x = _randn_fp16(1, c_in, H, W)
        return m, {"x": x}, ["out"]

    return make


def test_strided_conv3x3(c_in, c_out, bias):
    def make():
        m = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=bias)
        x = _randn_fp16(1, c_in, H, W)
        return m, {"x": x}, ["out"]

    return make


def test_leaky_relu(channels, negative_slope=0.01):
    def make():
        m = nn.LeakyReLU(negative_slope=negative_slope)
        x = _randn_fp16(1, channels, H, W)
        return m, {"x": x}, ["out"]

    return make


def test_relu1(channels):
    class ReLU1(nn.Module):
        def forward(self, x):
            return torch.clamp(x, min=0.0, max=1.0)

    def make():
        m = ReLU1()
        x = _randn_fp16(1, channels, H, W)
        return m, {"x": x}, ["out"]

    return make


def test_sigmoid(channels):
    def make():
        m = nn.Sigmoid()
        x = _randn_fp16(1, channels, H, W)
        return m, {"x": x}, ["out"]

    return make


def test_pixel_shuffle(channels, factor):
    def make():
        m = nn.PixelShuffle(factor)
        c_in = channels * factor * factor
        x = _randn_fp16(1, c_in, H, W)
        return m, {"x": x}, ["out"]

    return make


def test_pixel_unshuffle(channels, factor):
    class PixelUnshuffle(nn.Module):
        def __init__(self, f):
            super().__init__()
            self.f = f

        def forward(self, x):
            return F.pixel_unshuffle(x, self.f)

    def make():
        m = PixelUnshuffle(factor)
        x = _randn_fp16(1, channels, H * factor, W * factor)
        return m, {"x": x}, ["out"]

    return make


def test_elementwise_mul(channels):
    class ScalarMul(nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha = nn.Parameter(torch.randn(1))

        def forward(self, x):
            return x * self.alpha

    def make():
        m = ScalarMul()
        x = _randn_fp16(1, channels, H, W)
        return m, {"x": x}, ["out"]

    return make


def test_elementwise_add(channels):
    class TensorAdd(nn.Module):
        def forward(self, x, y):
            return x + y

    def make():
        m = TensorAdd()
        x = _randn_fp16(1, channels, H, W)
        y = _randn_fp16(1, channels, H, W)
        return m, {"x": x, "y": y}, ["out"]

    return make


# ---------------------------------------------------------------------------
# Composite block tests
# ---------------------------------------------------------------------------
def test_gated_ffn(channels):
    class GatedFFN(nn.Module):
        def __init__(self):
            super().__init__()
            mid = channels * 4
            self.expand = nn.Conv2d(channels, mid, 1)
            self.project = nn.Conv2d(mid // 2, channels, 1)

        def forward(self, x):
            h = self.expand(x)
            gates, values = h.chunk(2, dim=1)
            gates = torch.clamp(gates, min=0.0, max=1.0)
            return self.project(gates * values)

    def make():
        m = GatedFFN()
        x = _randn_fp16(1, channels, H, W)
        return m, {"x": x}, ["out"]

    return make


def test_depth_conv_block(in_ch, out_ch):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = _make_depth_conv_block(in_ch, out_ch)

        def forward(self, x):
            return self.block(x)

    def make():
        m = Block()
        x = _randn_fp16(1, in_ch, H, W)
        return m, {"x": x}, ["out"]

    return make


def test_subpel_conv2x(in_ch, out_ch):
    class SubpelConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch * 4, 1)
            self.ps = nn.PixelShuffle(2)

        def forward(self, x):
            return self.ps(self.conv(x))

    def make():
        m = SubpelConv()
        x = _randn_fp16(1, in_ch, H // 2, W // 2)
        return m, {"x": x}, ["out"]

    return make


def test_residual_block_stride2(in_ch, out_ch):
    class ResBlockDown(nn.Module):
        def __init__(self):
            super().__init__()
            self.down = nn.Conv2d(in_ch, out_ch, 2, stride=2)
            self.block = _make_depth_conv_block(out_ch, out_ch)

        def forward(self, x):
            return self.block(self.down(x))

    def make():
        m = ResBlockDown()
        x = _randn_fp16(1, in_ch, H, W)
        return m, {"x": x}, ["out"]

    return make


def test_residual_block_upsample(in_ch, out_ch):
    class ResBlockUp(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch * 4, 1)
            self.ps = nn.PixelShuffle(2)
            self.block = _make_depth_conv_block(out_ch, out_ch)

        def forward(self, x):
            return self.block(self.ps(self.conv(x)))

    def make():
        m = ResBlockUp()
        x = _randn_fp16(1, in_ch, H // 2, W // 2)
        return m, {"x": x}, ["out"]

    return make


def test_memory_cell(channels):
    class MemoryCell(nn.Module):
        def forward(self, feature, memory):
            state, forget_gate, output_gate = torch.chunk(feature, 3, dim=1)
            forget_gate = torch.sigmoid(forget_gate)
            output_gate = torch.sigmoid(output_gate)
            memory = forget_gate * memory + (1 - forget_gate) * state
            feature = output_gate * memory
            return feature

    def make():
        m = MemoryCell()
        x = _randn_fp16(1, channels * 3, H, W)
        mem = _randn_fp16(1, channels, H, W)
        return m, {"feature": x, "memory": mem}, ["out"]

    return make


def test_memory_cell_approx_sigmoid(channels):
    class MemoryCellApproxSigmoid(nn.Module):
        @staticmethod
        def approx_sigmoid(x):
            return 0.2 * (F.relu(x + 2.5) - F.relu(x - 2.5))

        def forward(self, feature, memory):
            state, forget_gate, output_gate = torch.chunk(feature, 3, dim=1)
            forget_gate = self.approx_sigmoid(forget_gate)
            output_gate = self.approx_sigmoid(output_gate)
            memory = forget_gate * memory + (1 - forget_gate) * state
            feature = output_gate * memory
            return feature

    def make():
        m = MemoryCellApproxSigmoid()
        x = _randn_fp16(1, channels * 3, H, W)
        mem = _randn_fp16(1, channels, H, W)
        return m, {"feature": x, "memory": mem}, ["out"]

    return make


def test_hyper_encoder_mini(in_ch, mid_ch, out_ch, num_layers):
    class HyperEncMini(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            for i in range(num_layers):
                ic = in_ch if i == 0 else mid_ch
                oc = out_ch if i == num_layers - 1 else mid_ch
                layers.append(nn.Conv2d(ic, oc, 2, stride=2))
            self.conv = nn.Sequential(*layers)

        def forward(self, x):
            return self.conv(x)

    def make():
        m = HyperEncMini()
        x = _randn_fp16(1, in_ch, H, W)
        return m, {"x": x}, ["out"]

    return make


def test_hyper_decoder_mini(in_ch, mid_ch, out_ch, num_layers):
    class HyperDecMini(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            for i in range(num_layers):
                ic = in_ch if i == 0 else mid_ch
                oc = out_ch if i == num_layers - 1 else mid_ch
                layers.append(nn.Conv2d(ic, oc * 4, 1))
                layers.append(nn.PixelShuffle(2))
            self.conv = nn.Sequential(*layers)

        def forward(self, x):
            return self.conv(x)

    def make():
        m = HyperDecMini()
        x = _randn_fp16(1, in_ch, H // (2**num_layers), W // (2**num_layers))
        return m, {"x": x}, ["out"]

    return make


# ---------------------------------------------------------------------------
# DepthConvBlock helper
# ---------------------------------------------------------------------------
def _make_depth_conv_block(in_ch, out_ch):
    class DepthConvBlockSimple(nn.Module):
        def __init__(self):
            super().__init__()
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None
            mid_ch = out_ch
            self.dc = nn.Sequential(
                nn.Conv2d(out_ch, mid_ch, 1),
                nn.LeakyReLU(),
                nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch),
                nn.Conv2d(mid_ch, out_ch, 1),
            )
            ffn_mid = out_ch * 4
            self.ffn_expand = nn.Conv2d(out_ch, ffn_mid, 1)
            self.ffn_project = nn.Conv2d(ffn_mid // 2, out_ch, 1)

        def forward(self, x):
            if self.adaptor is not None:
                x = self.adaptor(x)
            dc_out = self.dc(x)
            dc_out = dc_out + x
            h = self.ffn_expand(dc_out)
            gates, values = h.chunk(2, dim=1)
            gates = torch.clamp(gates, min=0.0, max=1.0)
            ffn_out = self.ffn_project(gates * values)
            out = ffn_out + dc_out
            return out

    return DepthConvBlockSimple()


# ---------------------------------------------------------------------------
# ANE simulation reference functions for atomic ops
# ---------------------------------------------------------------------------
def _sim_conv(module, input_dict):
    w = module.weight.data.to(torch.float16)
    b = module.bias.data.to(torch.float16) if module.bias is not None else None
    x = next(iter(input_dict.values()))
    return {"out": ane.ane_conv2d(x, w, b, stride=module.stride, padding=module.padding, groups=module.groups)}


def _sim_leaky_relu(module, input_dict):
    x = next(iter(input_dict.values()))
    slope = module.negative_slope
    return {"out": ane.ane_leaky_relu(x, negative_slope=slope)}


def _sim_relu1(module, input_dict):
    x = next(iter(input_dict.values()))
    return {"out": ane.ane_clamp(x, 0.0, 1.0)}


def _sim_pixel_shuffle(factor):
    def sim(module, input_dict):
        x = next(iter(input_dict.values()))
        return {"out": ane.ane_pixel_shuffle(x, factor)}

    return sim


def _sim_pixel_unshuffle(factor):
    def sim(module, input_dict):
        x = next(iter(input_dict.values()))
        return {"out": ane.ane_pixel_unshuffle(x, factor)}

    return sim


def _sim_scalar_mul(module, input_dict):
    x = next(iter(input_dict.values()))
    alpha = module.alpha.data.to(torch.float16)
    return {"out": ane.ane_mul(x, alpha)}


def _sim_tensor_add(module, input_dict):
    keys = list(input_dict.keys())
    return {"out": ane.ane_add(input_dict[keys[0]], input_dict[keys[1]])}


def _sim_sigmoid(module, input_dict):
    x = next(iter(input_dict.values()))
    return {"out": ane.ane_sigmoid(x)}


def _sim_memory_cell(module, input_dict):
    keys = list(input_dict.keys())
    feature = input_dict[keys[0]]
    memory = input_dict[keys[1]]
    c = memory.shape[1]
    state = feature[:, :c]
    fg_in = feature[:, c : 2 * c]
    og_in = feature[:, 2 * c : 3 * c]
    forget_gate = ane.ane_sigmoid(fg_in)
    output_gate = ane.ane_sigmoid(og_in)
    one = torch.ones_like(forget_gate, dtype=torch.float16)
    neg_one = torch.tensor(-1.0, dtype=torch.float16)
    one_minus_fg = ane.ane_add(one, ane.ane_mul(neg_one, forget_gate))
    mem_new = ane.ane_add(
        ane.ane_mul(forget_gate, memory),
        ane.ane_mul(one_minus_fg, state),
    )
    return {"out": ane.ane_mul(output_gate, mem_new)}


def _sim_memory_cell_approx_sigmoid(module, input_dict):
    keys = list(input_dict.keys())
    feature = input_dict[keys[0]]
    memory = input_dict[keys[1]]
    c = memory.shape[1]
    state = feature[:, :c]
    fg_in = feature[:, c : 2 * c]
    og_in = feature[:, 2 * c : 3 * c]
    forget_gate = ane.ane_approx_sigmoid(fg_in)
    output_gate = ane.ane_approx_sigmoid(og_in)
    one = torch.ones_like(forget_gate, dtype=torch.float16)
    neg_one = torch.tensor(-1.0, dtype=torch.float16)
    one_minus_fg = ane.ane_add(one, ane.ane_mul(neg_one, forget_gate))
    mem_new = ane.ane_add(
        ane.ane_mul(forget_gate, memory),
        ane.ane_mul(one_minus_fg, state),
    )
    return {"out": ane.ane_mul(output_gate, mem_new)}


# ---------------------------------------------------------------------------
# ANE simulation reference functions for composite blocks
# ---------------------------------------------------------------------------
def _sim_conv_layer(conv, x):
    w = conv.weight.data.to(torch.float16)
    b = conv.bias.data.to(torch.float16) if conv.bias is not None else None
    s = conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride
    p = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
    g = conv.groups
    return ane.ane_conv2d(x, w, b, stride=s, padding=p, groups=g)


def _sim_gated_ffn(module, input_dict):
    x = next(iter(input_dict.values()))
    h = _sim_conv_layer(module.expand, x)
    c = h.shape[1]
    gates, values = h[:, : c // 2], h[:, c // 2 :]
    gates = ane.ane_clamp(gates, 0.0, 1.0)
    gated = ane.ane_mul(gates, values)
    return {"out": _sim_conv_layer(module.project, gated)}


def _sim_depth_conv_block_inner(block, x):
    if block.adaptor is not None:
        x = _sim_conv_layer(block.adaptor, x)

    dc = _sim_conv_layer(block.dc[0], x)
    dc = ane.ane_leaky_relu(dc)
    dc = _sim_conv_layer(block.dc[2], dc)
    dc = _sim_conv_layer(block.dc[3], dc)
    dc = ane.ane_add(dc, x)

    h = _sim_conv_layer(block.ffn_expand, dc)
    c = h.shape[1]
    gates, values = h[:, : c // 2], h[:, c // 2 :]
    gates = ane.ane_clamp(gates, 0.0, 1.0)
    gated = ane.ane_mul(gates, values)
    ffn = _sim_conv_layer(block.ffn_project, gated)
    return ane.ane_add(ffn, dc)


def _sim_depth_conv_block(module, input_dict):
    x = next(iter(input_dict.values()))
    return {"out": _sim_depth_conv_block_inner(module.block, x)}


def _sim_subpel_conv2x(module, input_dict):
    x = next(iter(input_dict.values()))
    h = _sim_conv_layer(module.conv, x)
    return {"out": ane.ane_pixel_shuffle(h, 2)}


def _sim_res_block_stride2(module, input_dict):
    x = next(iter(input_dict.values()))
    h = _sim_conv_layer(module.down, x)
    return {"out": _sim_depth_conv_block_inner(module.block, h)}


def _sim_res_block_upsample(module, input_dict):
    x = next(iter(input_dict.values()))
    h = _sim_conv_layer(module.conv, x)
    h = ane.ane_pixel_shuffle(h, 2)
    return {"out": _sim_depth_conv_block_inner(module.block, h)}


def _sim_hyper_enc_mini(module, input_dict):
    x = next(iter(input_dict.values()))
    for layer in module.conv:
        x = _sim_conv_layer(layer, x)
    return {"out": x}


def _sim_hyper_dec_mini(module, input_dict):
    x = next(iter(input_dict.values()))
    for layer in module.conv:
        if isinstance(layer, nn.Conv2d):
            x = _sim_conv_layer(layer, x)
        elif isinstance(layer, nn.PixelShuffle):
            x = ane.ane_pixel_shuffle(x, layer.upscale_factor)
    return {"out": x}


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------
def build_test_suite():
    tests = []

    for c_in in [48, 96, 192]:
        tests.append(("conv1x1_nobias", f"{c_in}→{c_in}", test_conv1x1(c_in, c_in, bias=False), _sim_conv))
        tests.append(("conv1x1_bias", f"{c_in}→{c_in}", test_conv1x1(c_in, c_in, bias=True), _sim_conv))

    for ch in [FEATURE_CH, FEATURE_CH * 2, HIDDEN_CH]:
        tests.append(("dw_conv3x3_nobias", f"ch={ch}", test_depthwise_conv3x3(ch, bias=False), _sim_conv))
        tests.append(("dw_conv3x3_bias", f"ch={ch}", test_depthwise_conv3x3(ch, bias=True), _sim_conv))

    for c_in, c_out in [(Z_CH, Z_CH), (Z_CH, HIDDEN_CH)]:
        tests.append(("strided_conv2x2", f"{c_in}→{c_out}", test_strided_conv2x2(c_in, c_out, bias=True), _sim_conv))

    tests.append(
        ("strided_conv3x3", f"{HIDDEN_CH}→{Z_CH}", test_strided_conv3x3(HIDDEN_CH, Z_CH, bias=True), _sim_conv)
    )

    tests.append(("leaky_relu", f"ch={HIDDEN_CH} s=0.01", test_leaky_relu(HIDDEN_CH, 0.01), _sim_leaky_relu))
    tests.append(("leaky_relu", f"ch={HIDDEN_CH} s=0.125", test_leaky_relu(HIDDEN_CH, 0.125), _sim_leaky_relu))
    tests.append(("leaky_relu", f"ch={HIDDEN_CH} s=0.3", test_leaky_relu(HIDDEN_CH, 0.3), _sim_leaky_relu))
    tests.append(("relu1", f"ch={HIDDEN_CH}", test_relu1(HIDDEN_CH), _sim_relu1))
    tests.append(("sigmoid", f"ch={FEATURE_CH}", test_sigmoid(FEATURE_CH), _sim_sigmoid))

    tests.append(("pixel_shuffle_2", f"ch={FEATURE_CH}", test_pixel_shuffle(FEATURE_CH, 2), _sim_pixel_shuffle(2)))
    tests.append(
        ("pixel_unshuffle_2", f"ch={FEATURE_CH}", test_pixel_unshuffle(FEATURE_CH, 2), _sim_pixel_unshuffle(2))
    )
    tests.append(("pixel_unshuffle_8", "ch=3", test_pixel_unshuffle(3, 8), _sim_pixel_unshuffle(8)))

    tests.append(("elementwise_mul", f"ch={HIDDEN_CH}", test_elementwise_mul(HIDDEN_CH), _sim_scalar_mul))
    tests.append(("elementwise_add", f"ch={HIDDEN_CH}", test_elementwise_add(HIDDEN_CH), _sim_tensor_add))

    for ch in [FEATURE_CH, HIDDEN_CH]:
        tests.append(("gated_ffn", f"ch={ch}", test_gated_ffn(ch), _sim_gated_ffn))

    tests.append(
        (
            "depth_conv_block",
            f"{FEATURE_CH}→{FEATURE_CH}",
            test_depth_conv_block(FEATURE_CH, FEATURE_CH),
            _sim_depth_conv_block,
        )
    )
    tests.append(
        (
            "depth_conv_block",
            f"{HIDDEN_CH}→{HIDDEN_CH}",
            test_depth_conv_block(HIDDEN_CH, HIDDEN_CH),
            _sim_depth_conv_block,
        )
    )
    tests.append(
        (
            "depth_conv_block",
            f"{HIDDEN_CH}→{FEATURE_CH}",
            test_depth_conv_block(HIDDEN_CH, FEATURE_CH),
            _sim_depth_conv_block,
        )
    )

    tests.append(("subpel_conv2x", f"{Z_CH}→{HIDDEN_CH}", test_subpel_conv2x(Z_CH, HIDDEN_CH), _sim_subpel_conv2x))
    tests.append(
        ("subpel_conv2x", f"{FEATURE_CH}→{FEATURE_CH}", test_subpel_conv2x(FEATURE_CH, FEATURE_CH), _sim_subpel_conv2x)
    )

    tests.append(
        (
            "res_block_stride2",
            f"{HIDDEN_CH}→{HIDDEN_CH}",
            test_residual_block_stride2(HIDDEN_CH, HIDDEN_CH),
            _sim_res_block_stride2,
        )
    )

    tests.append(
        (
            "res_block_upsample",
            f"{Z_CH}→{HIDDEN_CH}",
            test_residual_block_upsample(Z_CH, HIDDEN_CH),
            _sim_res_block_upsample,
        )
    )

    tests.append(("memory_cell", f"ch={FEATURE_CH}", test_memory_cell(FEATURE_CH), _sim_memory_cell))
    tests.append(
        (
            "memory_cell_asig",
            f"ch={FEATURE_CH}",
            test_memory_cell_approx_sigmoid(FEATURE_CH),
            _sim_memory_cell_approx_sigmoid,
        )
    )

    tests.append(
        (
            "hyper_enc_mini",
            f"{Z_CH}→{Z_CH}→{Z_CH} (2L)",
            test_hyper_encoder_mini(Z_CH, Z_CH, Z_CH, 2),
            _sim_hyper_enc_mini,
        )
    )

    tests.append(
        (
            "hyper_dec_mini",
            f"{Z_CH}→{Z_CH}→{Z_CH} (2L)",
            test_hyper_decoder_mini(Z_CH, Z_CH, Z_CH, 2),
            _sim_hyper_dec_mini,
        )
    )

    return tests


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def print_results(results):
    print()
    print(
        f"{'Test':<24s} {'Config':<22s} {'Ref':>4s} {'Mismatch':>12s} {'Rate':>10s} {'MaxErr':>12s} {'MeanErr':>12s} {'Status':>8s}"
    )
    print("-" * 110)

    for r in results:
        print(
            f"{r.name:<24s} {r.config:<22s} "
            f"{r.ref_type:>4s} "
            f"{r.mismatch:>6d}/{r.total:<6d} "
            f"{r.rate:>9.4f}% "
            f"{r.max_abs_err:>12.6f} "
            f"{r.mean_abs_err:>12.8f} "
            f"{r.status:>8s}"
        )

    total_mm = sum(r.mismatch for r in results)
    total_n = sum(r.total for r in results)
    total_pct = 100 * total_mm / total_n if total_n > 0 else 0
    n_fail = sum(1 for r in results if r.mismatch > 0)
    n_pass = len(results) - n_fail

    print("-" * 110)
    print(f"{'TOTAL':<24s} {'':22s} {'':>4s} {total_mm:>6d}/{total_n:<6d} {total_pct:>9.4f}%")
    print(f"\n{len(results)} tests: {n_pass} bitexact, {n_fail} with mismatches")


if __name__ == "__main__":
    print("=" * 110)
    print("Validating dmc61sbr_mini_reglu building blocks on ANE (PyTorch simulation)")
    print(f"Spatial size: {H}x{W}, Seeds per test: {NUM_SEEDS}")
    print("=" * 110)

    tests = build_test_suite()
    results = []

    for i, (name, config, make_fn, sim_fn) in enumerate(tests):
        label = f"[{i + 1}/{len(tests)}] {name} ({config}) [SIM]"
        print(f"  {label}...", end="", flush=True)
        r = run_test(name, config, make_fn, sim_fn)
        results.append(r)
        print(f" {r.rate:.4f}% ({r.mismatch}/{r.total}) {r.status}")

    print_results(results)

    n_fail = sum(1 for r in results if r.mismatch > 0)
    sys.exit(1 if n_fail > 0 else 0)
