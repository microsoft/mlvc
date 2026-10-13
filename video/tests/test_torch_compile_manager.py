# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for src.utils.torch_compile; independent of TrainVideoApp and runnable without a GPU."""

import pytest
import torch

from src.utils.torch_compile import CompileSettings, TorchCompileManager, reset_compiler, with_standard_strides

CUDA = torch.device("cuda")
CPU = torch.device("cpu")
GUARD_KEYS = ("train.cascade",)


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.25))

    def forward(self, value):
        return torch.sin(value * self.weight)


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.fixture
def config():
    return {"train": {}}


@pytest.fixture
def manager(config):
    return TorchCompileManager(config=lambda: config, device=lambda: CUDA, extra_reset_keys=GUARD_KEYS)


@pytest.fixture
def compile_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.nn.Module, "compile", lambda module, **kwargs: calls.append((module, kwargs)))
    return calls


@pytest.fixture
def resets(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.compiler, "reset", lambda: calls.append(True))
    return calls


# ---- CompileSettings -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("root", [None, {}, {"train": None}, {"train": {}}])
def test_settings_defaults(root):
    settings = CompileSettings.from_config(root)
    assert settings == CompileSettings(enabled=True, backend="inductor", mode="reduce-overhead", options=None)
    assert settings.compile_kwargs() == {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "options": None,
        "mode": "reduce-overhead",
    }


@pytest.mark.parametrize(
    "train,expected",
    [
        ({"compile_backend": "cudagraphs"}, CompileSettings(backend="cudagraphs", mode=None)),
        ({"compile_backend": "inductor", "compile_mode": "default"}, CompileSettings(mode="default")),
        ({"compile_options": {"triton.cudagraphs": False}}, CompileSettings(options={"triton.cudagraphs": False})),
        ({"compile_mode": None}, CompileSettings(mode=None)),
        ({"compile_model": False}, CompileSettings(enabled=False, mode="reduce-overhead")),
    ],
)
def test_settings_from_config(train, expected):
    assert CompileSettings.from_config({"train": train}) == expected


def test_settings_reject_mode_with_options():
    with pytest.raises(ValueError, match="cannot both be set"):
        CompileSettings.from_config({"train": {"compile_mode": "default", "compile_options": {}}})
    with pytest.raises(ValueError, match="cannot both be set"):
        CompileSettings(mode="default", options={})


@pytest.mark.parametrize(
    "train",
    [{"compile_model": "yes"}, {"compile_backend": 1}, {"compile_mode": 1}, {"compile_options": ["x"]}],
)
def test_settings_validate_types(train):
    with pytest.raises(ValueError, match="Config value train.compile_"):
        CompileSettings.from_config({"train": train})


@pytest.mark.parametrize(
    "train,expected",
    [
        ({"compile_backend": "cudagraphs"}, True),
        ({}, True),  # default: inductor + reduce-overhead
        ({"compile_mode": "max-autotune"}, True),
        ({"compile_mode": "default"}, False),
        ({"compile_options": {"triton.cudagraphs": False}}, False),
        ({"compile_options": {"triton.cudagraphs": True}}, True),
        ({"compile_options": {}}, False),
        ({"compile_backend": "eager"}, False),
        ({"compile_model": False, "compile_backend": "cudagraphs"}, False),
    ],
)
def test_settings_uses_cudagraphs(train, expected):
    assert CompileSettings.from_config({"train": train}).uses_cudagraphs is expected


def test_settings_forward_fullgraph_and_dynamic():
    kwargs = CompileSettings(fullgraph=False, dynamic=True).compile_kwargs()
    assert (kwargs["fullgraph"], kwargs["dynamic"]) == (False, True)


# ---- TorchCompileManager.compile ------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["cudagraphs", "inductor"])
def test_compile_uses_current_settings_and_preserves_identity(manager, config, compile_calls, backend):
    model = Linear()
    config["train"]["compile_backend"] = backend

    assert manager.compile(model, "model") is model
    assert compile_calls == [(model, CompileSettings.from_config(config).compile_kwargs())]

    # The manager does not cache settings; config updates (e.g. training schedule) apply on the next call.
    config["train"]["compile_backend"] = "eager"
    manager.compile(model, "model")
    assert compile_calls[-1][1]["backend"] == "eager"


def test_compile_disabled_and_cpu_validation(config, compile_calls):
    model = Linear()
    manager = TorchCompileManager(config=lambda: config, device=lambda: CPU)
    config["train"]["compile_model"] = False
    assert manager.compile(model, "disabled") is model
    assert compile_calls == []

    config["train"]["compile_model"] = True
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        manager.compile(model, "invalid")
    assert compile_calls == []


def test_compile_real_module_matches_eager(config):
    config["train"].update(compile_backend="eager", compile_mode=None)
    reference, model = Linear(), Linear()
    manager = TorchCompileManager(config=lambda: config, device=lambda: CUDA)
    manager.compile(model, "model")

    value = torch.linspace(-1, 1, 5)
    torch.testing.assert_close(model(value), reference(value))


# ---- TorchCompileManager.on_config_changed / reset ------------------------------------------------------------


def test_reset_keys_combine_guard_and_compile_keys(manager):
    assert manager.reset_keys == GUARD_KEYS + CompileSettings.CONFIG_KEYS


@pytest.mark.parametrize("change", [{"train.cascade": True}, {"train.compile_backend": True}])
def test_config_change_resets_and_recompiles(manager, compile_calls, resets, change):
    model = Linear()
    assert manager.on_config_changed(change, model) is True
    assert len(resets) == 1
    assert compile_calls == [(model, CompileSettings.from_config({}).compile_kwargs())]


def test_unrelated_config_change_or_missing_model_is_ignored(manager, compile_calls, resets):
    assert manager.on_config_changed({"train.learning_rate": True}, Linear()) is False
    assert manager.on_config_changed({"train.cascade": True}, None) is False
    assert resets == compile_calls == []


def test_guard_change_while_disabled_is_ignored(manager, config, compile_calls, resets):
    config["train"]["compile_model"] = False
    assert manager.on_config_changed({"train.cascade": True, "train.compile_backend": True}, Linear()) is False
    assert resets == compile_calls == []


def test_disabling_compilation_resets_and_disables(manager, config, compile_calls, resets):
    model = Linear()
    config["train"]["compile_model"] = False
    assert manager.on_config_changed({"train.compile_model": True}, model) is True
    assert len(resets) == 1
    assert compile_calls == [(model, {"disable": True})]


@pytest.mark.parametrize("device,empties", [(CPU, 0), (CUDA, 1)])
def test_reset_compiler_frees_cuda_cache_only_on_cuda(monkeypatch, resets, device, empties):
    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # skip memory logging
    reset_compiler(device)
    assert len(resets) == 1
    assert len(emptied) == empties


# ---- TorchCompileManager.eval_scope / detach_carried ----------------------------------------------------------


@pytest.mark.parametrize(
    "free,train,expected_resets", [(True, {}, 1), (False, {}, 0), (True, {"compile_mode": "default"}, 0)]
)
def test_eval_scope_resets_only_when_freeing_cudagraph_pools(manager, config, resets, free, train, expected_resets):
    config["train"].update(train)
    with manager.eval_scope(free_cudagraph_pools=free):
        pass
    assert len(resets) == expected_resets


def test_eval_scope_runs_compiled_modules_eagerly(manager):
    traces = []

    def counting_backend(graph_module, example_inputs):
        traces.append(graph_module)
        return graph_module.forward

    model = Linear()
    model.compile(backend=counting_backend, fullgraph=True, dynamic=False)
    model(torch.ones(2))
    assert len(traces) == 1

    with torch._dynamo.config.patch(error_on_recompile=True), manager.eval_scope():
        model(torch.ones(3))  # new shape would recompile outside the eager stance
    assert len(traces) == 1


@pytest.mark.parametrize("enabled", [True, False])
def test_detach_carried_clones_only_when_compiling(manager, config, enabled):
    config["train"]["compile_model"] = enabled
    value = torch.ones(2, requires_grad=True) * 2

    carried = manager.detach_carried({"ref": value})

    assert not carried["ref"].requires_grad
    assert (carried["ref"].data_ptr() != value.data_ptr()) == enabled
    torch.testing.assert_close(carried["ref"], value)


@pytest.mark.parametrize("train,cloned", [({}, True), ({"compile_mode": "default"}, False)])
def test_own_accumulated_grads_clones_only_with_cudagraphs(manager, config, train, cloned):
    config["train"].update(train)
    weight = torch.nn.Parameter(torch.ones(2))
    weight.grad = torch.full((2,), 3.0)
    no_grad = torch.nn.Parameter(torch.ones(2))
    grad_ptr = weight.grad.data_ptr()

    manager.own_accumulated_grads([weight, no_grad])

    assert (weight.grad.data_ptr() != grad_ptr) == cloned
    torch.testing.assert_close(weight.grad, torch.full((2,), 3.0))
    assert no_grad.grad is None


# ---- with_standard_strides ------------------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 4])
def test_with_standard_strides_normalizes_frame_views(batch_size):
    frame_stride = 3 * 8 * 8
    for n_frames in (2, 3, 5):
        videos = torch.rand(batch_size, n_frames, 3, 8, 8, requires_grad=True)
        frame = videos[:, 1]
        assert frame.stride(0) == n_frames * frame_stride

        normalized = with_standard_strides(frame)

        assert normalized.stride() == (frame_stride, 64, 8, 1)
        torch.testing.assert_close(normalized, frame)
        normalized.sum().backward()
        grad = videos.grad
        assert grad is not None
        assert grad[:, 1].eq(1).all() and grad[:, 0].eq(0).all()

    standard = torch.rand(batch_size, 3, 8, 8)
    assert with_standard_strides(standard) is standard
    assert with_standard_strides(None) is None


def test_clip_length_change_does_not_recompile_with_standard_strides():
    compiled = torch.compile(lambda value: value * 2, backend="eager", fullgraph=True, dynamic=False)
    with torch._dynamo.config.patch(error_on_recompile=True):
        for n_frames in (2, 3, 4, 5):
            videos = torch.rand(1, n_frames, 3, 8, 8)
            for frame_idx in range(n_frames):
                compiled(with_standard_strides(videos[:, frame_idx]))
