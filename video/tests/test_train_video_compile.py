# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""TrainVideoApp integration with torch.compile; compile logic itself is covered by test_torch_compile_manager.py."""

import copy

import pytest
import torch

from src.models.common_model import CompressionModel
from src.models.utils import CkptModule
from src.utils.ane_mode import ane_simulation
from train_video import FrameLossStep, TrainVideoApp


class TrainableModule(CkptModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.25))

    def internal_forward(self, value):
        return torch.sin(value * self.weight)


class MaskedModule(CompressionModel):
    def __init__(self, layout):
        super().__init__(y_distribution="gaussian", z_channel=0)
        self.layout = layout
        self.weight = torch.nn.Parameter(torch.tensor(1.25))

    def masks_for(self, value) -> list[torch.Tensor]:
        batch, channels, height, width = value.shape
        if self.layout == "spatial":
            return self.get_mask(height, width, value.dtype, value.device)
        if self.layout == "dual":
            return self.get_mask_dual(batch, channels, height, width, value.dtype, value.device)
        return self.get_mask_four_parts(batch, channels, height, width, value.dtype, value.device)

    def forward(self, value):
        return torch.sin(value * self.masks_for(value)[0] * self.weight)


N_FRAMES = 7
# None trains frame by frame; (N_FRAMES,) is plain cascade; (3, 5, N_FRAMES) is cascade_partition_frames: [3, 5].
CASCADE_FRAME_RANGES = [
    pytest.param(None, id="by_frame"),
    pytest.param((N_FRAMES,), id="cascade"),
    pytest.param((3, 5, N_FRAMES), id="cascade_partition_3_5"),
]


class RecurrentFrameModule(torch.nn.Module):
    """Minimal P-frame model carrying ref_frame/ref_feature in the dpb like DMC.

    ``adaptor_i`` runs only on the first P-frame (``ref_feature is None``), like DMC's ``feature_adaptor_i``. Its
    parameters get gradients from a single compiled call, so autograd stores the compiled backward output as
    ``param.grad`` without copying it; with CUDA graphs that buffer must be cloned before the next cascade partition.
    """

    def __init__(self, frame_block, channels=4):
        super().__init__()
        self.adaptor_i = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.feature_conv = torch.nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.frame_block = frame_block

    def forward(self, x, dpb, *, q_index, fa_idx):
        ref_feature = dpb["ref_feature"]
        if ref_feature is None:
            ref_feature = self.adaptor_i(dpb["ref_frame"])
        feature = torch.tanh(self.feature_conv(torch.cat((ref_feature, x), dim=1)))
        x_hat = self.frame_block(feature) + dpb["ref_frame"] * 0.5
        return {"x_hat": x_hat, "dpb": {"ref_frame": x_hat, "ref_feature": feature}}


class FrameLoss:
    bpp_weight = 1.0

    def __call__(self, rd, target, lambdas, mask):
        mse = (rd["x_hat"] - target).pow(2).flatten(1).mean(dim=1)
        return {"loss": lambdas * mse, "mse": mse}


def _make_train_app(model, train_config, *, compile_model, cascade_frame_ranges):
    app = TrainVideoApp()
    app._config = {
        "train": {
            **train_config,
            "compile_model": compile_model,
            "cascade": cascade_frame_ranges is not None,
            "grad_max_norm": 1.0,
        }
    }
    app._device = torch.device("cuda")
    app.use_drift_loss = False
    app.frame_index_map = (0,)
    app.cascade_frame_ranges = cascade_frame_ranges
    app.loss_func = FrameLoss()  # pyright: ignore[reportAttributeAccessIssue]
    app.torch_compile.compile(model, "training model")
    app.model = app.make_train_model(model)
    app.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return app


def assert_compiled_training_matches_eager(frame_block, train_config, cascade_frame_ranges, sizes):
    """Trains eager and compiled copies through the real per-frame / per-clip steps and compares every step."""
    torch.manual_seed(0)
    model = RecurrentFrameModule(frame_block).cuda()
    reference = copy.deepcopy(model)
    apps = [
        _make_train_app(m, train_config, compile_model=compiled, cascade_frame_ranges=cascade_frame_ranges)
        for m, compiled in ((reference, False), (model, True))
    ]

    for height, width in sizes:
        videos = torch.rand(2, N_FRAMES, 4, height, width, device="cuda")
        q_index = torch.zeros(2, N_FRAMES, dtype=torch.long, device="cuda")
        lambdas = torch.ones(2, N_FRAMES, device="cuda")
        results = []
        for app in apps:
            step = app._step_by_frame if cascade_frame_ranges is None else app._step_by_clip
            dpb = {"ref_frame": videos[:, 0], "ref_feature": None}
            results.append(step(videos, dpb=dpb, q_index=q_index, lambdas=lambdas))

        (expected_names, expected_values), (names, values) = results
        assert names == expected_names
        torch.testing.assert_close(values, expected_values)
        for (name, param), expected_param in zip(model.named_parameters(), reference.parameters()):
            torch.testing.assert_close(param.grad, expected_param.grad, msg=lambda m: f"{name} grad: {m}")
            torch.testing.assert_close(param, expected_param, msg=lambda m: f"{name}: {m}")


@pytest.fixture(autouse=True)
def reset_compiler():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.fixture
def app():
    app = TrainVideoApp()
    app._config = {"train": {"cascade": False, "compile_loss": False}}
    app._device = torch.device("cuda")
    return app


@pytest.fixture
def compile_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.nn.Module, "compile", lambda module, **kwargs: calls.append((module, kwargs)))
    return calls


@pytest.mark.parametrize("change", [{"cascade": True}, {"compile_backend": "inductor"}])
def test_scheduled_compile_policy_preserves_module_identity(app, compile_calls, monkeypatch, change):
    if "compile_backend" in change:
        app.config["train"]["compile_backend"] = "cudagraphs"
    app.config["train"]["schedule"] = [{"epoch": 1, **change}]
    app.config["benchmark_test"] = {}
    app.initialize_training_schedule()
    model = TrainableModule()
    forward, weight, keys = model.forward, model.weight, tuple(model.state_dict())
    resets = []
    monkeypatch.setattr(torch.compiler, "reset", lambda: resets.append(True))

    assert app.torch_compile.compile(model, "training") is model
    app.model = app.make_train_model(model)

    if "cascade" in change:
        # cascade changes still reset and recompile, but no longer change the compile policy.
        policies = ((0, "inductor", "reduce-overhead", None), (1, "inductor", "reduce-overhead", None))
    else:
        policies = ((0, "cudagraphs", None, None), (1, "inductor", "reduce-overhead", None))
    for epoch, backend, mode, options in policies:
        app._apply_training_schedule(epoch)

        assert len(resets) == epoch
        assert len(compile_calls) == epoch + 1
        assert compile_calls[-1] == (
            model,
            {
                "backend": backend,
                "fullgraph": True,
                "dynamic": False,
                "options": options,
                "mode": mode,
            },
        )
        assert app.get_unwrapped_model(app.model) is model
        assert model.forward == forward
        assert model.weight is weight
        assert tuple(model.state_dict()) == keys


@pytest.mark.parametrize(
    "key", ["compile_fast_cudagraph_recording", "flat_parameters", "compile_loss", "ddp_gradient_as_bucket_view"]
)
def test_scheduled_startup_only_options_fail_before_training(app, tmp_path, monkeypatch, key):
    app.config["save_dir"] = str(tmp_path)
    app.config["benchmark_test"] = {}
    app.config["train"][key] = False
    app.config["train"]["schedule"] = [{"epoch": 5, key: True}]
    monkeypatch.setattr(app, "restore_train_state", lambda: pytest.fail("training started"))

    with pytest.raises(ValueError, match=f"{key} is immutable"):
        app.train()


@pytest.mark.parametrize("has_model", [True, False])
def test_schedule_changes_reset_once_without_replacing_parameters(app, compile_calls, monkeypatch, has_model):
    model = TrainableModule()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    app.model = app.make_train_model(model) if has_model else None
    resets = []
    monkeypatch.setattr(torch.compiler, "reset", lambda: resets.append(True))

    app.on_config_changed({"train.learning_rate": True})
    assert resets == compile_calls == []
    for change in (
        {"train.dataset.patch_size": True},
        {"train.dataset.batch_size": True},
        {"train.cascade": True},
        {"train.use_backprop_checkpoints": True},
        {"train.compile_backend": True},
        {"train.dataset.patch_size": True, "train.cascade": True, "train.compile_backend": True},
    ):
        app.on_config_changed(change)
    assert len(resets) == len(compile_calls) == (6 if has_model else 0)
    assert optimizer.param_groups[0]["params"][0] is model.weight
    if has_model:
        assert app.get_unwrapped_model(app.model) is model


@pytest.mark.parametrize("enabled", [None, True, False])
def test_frame_inputs_get_standard_strides_only_when_compiling(app, monkeypatch, enabled):
    if enabled is not None:
        app.config["train"]["compile_model"] = enabled
    monkeypatch.setattr(app, "use_drift_loss", False, raising=False)
    app.frame_index_map = (0,)
    app.loss_func = lambda rd, target, lambdas, mask: {"loss": rd["out"]}
    videos = torch.rand(1, 3, 3, 8, 8)
    q_index = torch.zeros(1, 3, dtype=torch.long)
    seen = {}

    def model(x, dpb, *, q_index, fa_idx):
        seen.update(x=x, ref_frame=dpb["ref_frame"], q_index=q_index)
        return {"out": x.mean().reshape(1), "dpb": dpb}

    app._base_frame_forward(
        model,
        videos[:, 1],
        dpb={"ref_frame": videos[:, 0]},
        q_index=q_index,
        frame_idx=1,
        lambdas=torch.ones(1, 3),
    )

    normalized = enabled is not False
    assert (seen["x"].stride(0) == 3 * 8 * 8) == normalized
    assert (seen["ref_frame"].stride(0) == 3 * 8 * 8) == normalized
    assert (seen["q_index"].stride(0) == 1) == normalized


@pytest.mark.parametrize(
    "benchmark_test,expected_resets",
    [
        (None, 0),
        ({"regular": {"test_interval": 1}}, 0),
        ({"deqa": {"calc_deqa_score_rgb": True, "test_interval": 1, "deqa_test_interval": 2}}, 0),
        ({"deqa": {"calc_deqa_score_rgb": True, "test_interval": 0, "deqa_test_interval": 1}}, 1),
        ({"off": None, "deqa": {"calc_deqa_score_rgb": True}}, 1),
    ],
)
def test_testing_resets_compiler_only_before_deqa(app, monkeypatch, benchmark_test, expected_resets):
    app.config["train"]["compile_backend"] = "cudagraphs"
    app.config["benchmark_test"] = benchmark_test
    resets = []
    monkeypatch.setattr(torch.compiler, "reset", lambda: resets.append(True))
    monkeypatch.setattr(app, "_test_model", lambda epoch, model: None)

    app.test_model(0, TrainableModule())

    assert len(resets) == expected_resets


@pytest.mark.parametrize(
    "config,benchmark_epochs,deqa_epochs",
    [
        ({"test_interval": 2}, {1, 3, 5}, set()),
        ({"test_interval": 2, "calc_deqa_score_rgb": True}, {1, 3, 5}, {1, 3, 5}),
        ({"test_interval": 1, "calc_deqa_score_rgb": True, "deqa_test_interval": 3}, set(range(6)), {2, 5}),
        ({"test_interval": 0, "calc_deqa_score_rgb": True, "deqa_test_interval": 3}, {2, 5}, {2, 5}),
        ({"test_interval": 2, "calc_deqa_score_rgb": True, "deqa_test_interval": 0}, {1, 3, 5}, set()),
        ({"test_interval": 0, "deqa_test_interval": 1}, set(), set()),
    ],
)
def test_deqa_test_interval_is_independent_of_test_interval(config, benchmark_epochs, deqa_epochs):
    assert {e for e in range(6) if TrainVideoApp.is_testset_benchmark_epoch(config, e)} == benchmark_epochs
    assert {e for e in range(6) if TrainVideoApp.is_testset_deqa_epoch(config, e)} == deqa_epochs
    # validation / revalidation run without an epoch and compute every enabled metric
    assert TrainVideoApp.is_testset_deqa_epoch(config, None) == config.get("calc_deqa_score_rgb", False)


def test_deqa_metrics_are_not_logged_on_regular_test_epochs(app):
    logged = []
    app._metrics_writer = type("Writer", (), {"add_metrics": lambda self, metrics, step: logged.append(metrics)})()
    results = {"class": {"video": {21: {"ave_p_frame_psnr": 30.0}}}}
    config = {"log_metrics": ["p_frame_psnr", "all_frame_deqa_score_rgb"]}

    app.log_benchmark_metrics(results, testset_name="set", config=config, epoch=0, skip_deqa=True)
    assert logged == [{"set_p_frame_psnr000": 30.0}]
    with pytest.raises(KeyError):
        app.log_benchmark_metrics(results, testset_name="set", config=config, epoch=0)


def test_ane_simulation_with_compiled_checkpoint_raises():
    model = TrainableModule()
    model.set_use_ckpt(True)
    value = torch.full((2,), 0.5, requires_grad=True)
    with ane_simulation(differentiable=True):
        model(value).sum().backward()
    model.compile(backend="aot_eager", fullgraph=True, dynamic=False)

    with ane_simulation(differentiable=True), pytest.raises(RuntimeError, match="ANE simulation is not supported"):
        model(value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled backends require CUDA")
@pytest.mark.parametrize("backend", ["cudagraphs", "inductor"])
@pytest.mark.parametrize("cascade_frame_ranges", CASCADE_FRAME_RANGES)
def test_compiled_checkpoint_shapes_values_and_gradients(backend, cascade_frame_ranges):
    """Compiled training with a checkpointed block matches eager across shape changes and cascade partitions.

    Partitioned cascade runs a backward per partition before one optimizer step, so the dpb carried into the next
    partition and the accumulated gradients must not alias CUDA graph buffers that the next replay overwrites.
    """
    frame_block = TrainableModule()
    frame_block.set_use_ckpt(True)
    assert_compiled_training_matches_eager(
        frame_block, {"compile_backend": backend}, cascade_frame_ranges, sizes=((8, 8), (12, 8), (8, 8))
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled backends require CUDA")
@pytest.mark.parametrize("backend", ["cudagraphs", "inductor"])
@pytest.mark.parametrize("cascade_frame_ranges", CASCADE_FRAME_RANGES)
@pytest.mark.parametrize("layout", ["spatial", "dual", "four_parts"])
def test_cached_masks_survive_compiled_replay_with_assume_constant_result(backend, cascade_frame_ranges, layout):
    """Regress the cached-mask lifetime bug fixed by assume_constant_result.

    The decorator keeps mask creation outside CUDA graph capture, so cached masks
    remain valid across repeated compiled calls and input-shape changes.
    """
    # Exercise capture/replay and shape changes; the original bug failed on the third call.
    sizes = [(height, 7) for height in (5,) * 5 + (7,) * 5 + (5,) * 5]
    assert_compiled_training_matches_eager(
        MaskedModule(layout), {"compile_backend": backend}, cascade_frame_ranges, sizes=sizes
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled backends require CUDA")
@pytest.mark.parametrize("backend", [None, "cudagraphs", "inductor"])
@pytest.mark.parametrize("cascade_frame_ranges", CASCADE_FRAME_RANGES)
@pytest.mark.parametrize(
    "options",
    [
        {"flat_parameters": True, "compile_loss": True},
        {"flat_parameters": True, "compile_loss": True, "compile_skip_guard_eval_after_steps": 1},
    ],
    ids=["flat_parameters_compile_loss", "skip_guard_eval"],
)
def test_training_options_train_like_per_parameter_eager(backend, cascade_frame_ranges, options):
    """Flat parameters, a compiled loss and skipped guards give the same steps and checkpoints as eager training."""
    torch.manual_seed(0)
    model = RecurrentFrameModule(TrainableModule()).cuda()
    reference = copy.deepcopy(model)
    apps = []
    for m, train_config in (
        (reference, {"compile_model": False}),
        (model, {"compile_model": backend is not None, "compile_backend": backend or "inductor", **options}),
    ):
        app = TrainVideoApp()
        app._config = {
            "train": {
                **train_config,
                "cascade": cascade_frame_ranges is not None,
                "grad_max_norm": 1.0,
                "learning_rate": 0.01,
            }
        }
        app._device = torch.device("cuda")
        app.use_drift_loss = False
        app.frame_index_map = (0,)
        app.cascade_frame_ranges = cascade_frame_ranges
        app.loss_func = FrameLoss()  # pyright: ignore[reportAttributeAccessIssue]
        app.model = app.make_train_model(m)
        app.optimizer = app.create_optimizer()
        apps.append(app)
    assert apps[1].flat_parameters is not None
    assert isinstance(apps[1]._compiled_model, FrameLossStep)

    # shapes change only during the guard evaluation warmup
    sizes = ((8, 8), (8, 8), (8, 8)) if "compile_skip_guard_eval_after_steps" in options else ((8, 8), (12, 8), (8, 8))
    with apps[1].torch_compile.epoch_scope() as on_step:
        for i, (height, width) in enumerate(sizes):
            on_step(i)
            videos = torch.rand(2, N_FRAMES, 4, height, width, device="cuda")
            q_index = torch.zeros(2, N_FRAMES, dtype=torch.long, device="cuda")
            lambdas = torch.ones(2, N_FRAMES, device="cuda")
            results = []
            for app in apps:
                step = app._step_by_frame if cascade_frame_ranges is None else app._step_by_clip
                dpb = {"ref_frame": videos[:, 0], "ref_feature": None}
                results.append(step(videos, dpb=dpb, q_index=q_index, lambdas=lambdas))

            (expected_names, expected_values), (names, values) = results
            assert names == expected_names
            torch.testing.assert_close(values, expected_values)
            for (name, param), expected_param in zip(model.named_parameters(), reference.parameters()):
                torch.testing.assert_close(param, expected_param, msg=lambda m: f"{name}: {m}")

    expected_state, state = (app.get_train_state(0)["optimizer"] for app in apps)
    assert state["param_groups"] == expected_state["param_groups"]
    torch.testing.assert_close(state["state"], expected_state["state"])


class QScaledFrameModule(RecurrentFrameModule):
    """Adds a parameter whose gradient the model clears without a rate loss, like DMC's q_encoder etc."""

    def __init__(self, frame_block):
        super().__init__(frame_block)
        self.q_scale = torch.nn.Parameter(torch.tensor(1.5))

    def forward(self, x, dpb, *, q_index, fa_idx):
        rd = super().forward(x, dpb, q_index=q_index, fa_idx=fa_idx)
        rd["x_hat"] = rd["x_hat"] * self.q_scale
        return rd

    def clear_grad_for_y_q(self):
        self.q_scale.grad = None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled backends require CUDA")
@pytest.mark.parametrize("backend", [None, "cudagraphs", "inductor"])
def test_flat_parameters_clear_grad_for_y_q_like_per_parameter_eager(backend):
    torch.manual_seed(0)
    model = QScaledFrameModule(TrainableModule()).cuda()
    reference = copy.deepcopy(model)
    apps = []
    for m, train_config in (
        (reference, {"compile_model": False}),
        (
            model,
            {"compile_model": backend is not None, "compile_backend": backend or "inductor", "flat_parameters": True},
        ),
    ):
        app = TrainVideoApp()
        app._config = {"train": {**train_config, "cascade": False, "grad_max_norm": 1.0, "learning_rate": 0.01}}
        app._device = torch.device("cuda")
        app.use_drift_loss = False
        app.frame_index_map = (0,)
        loss_func = FrameLoss()
        loss_func.bpp_weight = 0.0
        app.loss_func = loss_func  # pyright: ignore[reportAttributeAccessIssue]
        app.clear_grad_for_y_q = True
        app.model = app.make_train_model(m)
        app.optimizer = app.create_optimizer()
        apps.append(app)

    for _ in range(2):
        videos = torch.rand(2, N_FRAMES, 4, 8, 8, device="cuda")
        q_index = torch.zeros(2, N_FRAMES, dtype=torch.long, device="cuda")
        lambdas = torch.ones(2, N_FRAMES, device="cuda")
        for app in apps:
            dpb = {"ref_frame": videos[:, 0], "ref_feature": None}
            app._step_by_frame(videos, dpb=dpb, q_index=q_index, lambdas=lambdas)

    assert model.q_scale.item() == 1.5
    for (name, param), expected_param in zip(model.named_parameters(), reference.parameters()):
        torch.testing.assert_close(param, expected_param, msg=lambda m: f"{name}: {m}")


def test_flat_parameters_reject_eager_training():
    app = TrainVideoApp()
    app._config = {"train": {"flat_parameters": True, "compile_model": False}}
    app.model = torch.nn.Linear(2, 2)
    app.grad_norm_tracker = None
    with pytest.raises(ValueError, match="does not support train.compile_model: false$"):
        app._check_flat_parameters_support({"transfer_weights": "none"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused optimizers require CUDA")
@pytest.mark.parametrize("flat_parameters", [False, True])
def test_fused_optimizer_resumes_from_non_fused_state(flat_parameters):
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2).cuda()
    saved = torch.optim.AdamW(model.parameters(), lr=0.1)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    saved.step()

    app = TrainVideoApp()
    app._config = {
        "train": {
            "flat_parameters": flat_parameters,
            "compile_model": False,
            "grad_max_norm": 1.0,
            "learning_rate": 0.1,
            "optimizer": {"params": {"fused": True}},
        }
    }
    app._device = torch.device("cuda")
    app.model = app.make_train_model(model)
    optimizer = app.optimizer = app.load_optimizer({"optimizer": saved.state_dict()})
    assert all(group["fused"] for group in optimizer.param_groups)

    for p in app.get_trainable_parameters():
        p.grad = torch.ones_like(p)
    app.optimizer_step()
    assert all(s["step"].device.type == "cuda" and s["step"].item() == 2 for s in optimizer.state.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled backends require CUDA")
@pytest.mark.parametrize("backend", [None, "cudagraphs", "inductor"])
def test_drift_loss_continues_without_flat_parameters_and_compiled_loss(backend):
    """When a schedule enables a drift loss, training continues on the plain model with the same optimizer state."""
    optimize_ddp = torch._dynamo.config.optimize_ddp
    torch.manual_seed(0)
    model = RecurrentFrameModule(TrainableModule()).cuda()
    reference = copy.deepcopy(model)
    apps = []
    for m, train_config in (
        (reference, {"compile_model": False}),
        (
            model,
            {
                "compile_model": backend is not None,
                "compile_backend": backend or "inductor",
                "flat_parameters": True,
                "compile_loss": True,
            },
        ),
    ):
        app = TrainVideoApp()
        app._config = {"train": {**train_config, "cascade": False, "grad_max_norm": 1.0, "learning_rate": 0.01}}
        app._device = torch.device("cuda")
        app.use_drift_loss = False
        app.frame_index_map = (0,)
        app.loss_func = FrameLoss()  # pyright: ignore[reportAttributeAccessIssue]
        app.model = app.make_train_model(m)
        app.optimizer = app.create_optimizer()
        apps.append(app)

    def train_step():
        videos = torch.rand(2, N_FRAMES, 4, 8, 8, device="cuda")
        q_index = torch.zeros(2, N_FRAMES, dtype=torch.long, device="cuda")
        lambdas = torch.ones(2, N_FRAMES, device="cuda")
        for app in apps:
            dpb = {"ref_frame": videos[:, 0], "ref_feature": None}
            app._step_by_frame(videos, dpb=dpb, q_index=q_index, lambdas=lambdas)

    train_step()
    apps[1]._remove_model_wrappers("drift losses")
    assert apps[1].flat_parameters is None and apps[1]._compiled_model is model
    assert all(p.requires_grad for p in model.parameters())
    assert torch._dynamo.config.optimize_ddp == optimize_ddp
    train_step()

    for (name, param), expected_param in zip(model.named_parameters(), reference.parameters()):
        torch.testing.assert_close(param, expected_param, msg=lambda m: f"{name}: {m}")
    expected_state, state = (app.get_train_state(0)["optimizer"] for app in apps)
    torch.testing.assert_close(state["state"], expected_state["state"])
