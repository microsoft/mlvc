# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy

import pytest
import torch
import torch.nn as nn

from src.utils.flat_parameters import FlatParameters


class FirstFrameModel(nn.Module):
    """``adaptor[0]`` is used only on the first call and ``adaptor[1]`` on the others, like per-frame-type adaptors."""

    def __init__(self):
        super().__init__()
        # odd parameter sizes so that views need padding to stay aligned
        self.adaptor = nn.ModuleList(nn.Conv2d(3, 3, 3, padding=1) for _ in range(2))
        self.body = nn.Sequential(nn.Conv2d(3, 5, 3), nn.ReLU(), nn.Conv2d(5, 7, 1))
        self.scale = nn.Parameter(torch.tensor(1.5))

    def forward(self, x, first: bool):
        x = self.adaptor[0 if first else 1](x)
        return self.body(x) * self.scale


def test_training_through_flat_parameters_matches_per_parameter_training():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 6, 6)
    reference = FirstFrameModel()
    model = copy.deepcopy(reference)
    state_dict_keys = list(model.state_dict())

    flat = FlatParameters(model)
    assert len(flat.flats) == 4
    assert all(not p.requires_grad for p in model.parameters())
    flat_ranges = [(f.data_ptr(), f.data_ptr() + f.numel() * f.element_size()) for f in flat.flats]
    for p in model.parameters():
        start = next(start for start, end in flat_ranges if start <= p.data_ptr() < end)
        assert (p.data_ptr() - start) % 128 == 0

    optimizers = [
        torch.optim.AdamW(reference.parameters(), lr=0.1),
        torch.optim.AdamW(flat.flats.parameters(), lr=0.1),
    ]
    for step in range(4):
        for optimizer in optimizers:
            optimizer.zero_grad()
        reference(x, step == 0).square().mean().backward()
        flat(x, step == 0).square().mean().backward()
        # an adaptor gets no gradient when unused, so AdamW skips it exactly as it skips its parameters;
        # flat tensors follow named_parameters() order: scale, adaptor.0, adaptor.1, body
        assert (flat.flats[1].grad is None) == (step > 0)
        assert (flat.flats[2].grad is None) == (step == 0)
        for optimizer in optimizers:
            optimizer.step()

    # optimizer updates of the flat parameters are visible through the original parameters
    assert list(model.state_dict()) == state_dict_keys
    for (name, p), expected in zip(model.state_dict().items(), reference.state_dict().values()):
        torch.testing.assert_close(p, expected, msg=lambda m: f"{name}: {m}")

    # the flat optimizer state converts to the per-parameter layout and back
    per_param_state = flat.unflatten_optimizer_state(optimizers[1].state_dict())
    expected_state = optimizers[0].state_dict()
    assert per_param_state["param_groups"] == expected_state["param_groups"]
    torch.testing.assert_close(per_param_state["state"], expected_state["state"])

    restored = torch.optim.AdamW(flat.flats.parameters(), lr=0.1)
    restored.load_state_dict(flat.flatten_optimizer_state(expected_state))
    torch.testing.assert_close(restored.state_dict(), optimizers[1].state_dict())


def test_clear_grads_skips_whole_flat_tensors_only():
    model = FirstFrameModel()
    flat = FlatParameters(model)
    flat(torch.rand(1, 3, 6, 6), True).sum().backward()

    # flat tensors follow named_parameters() order: scale, adaptor.0, adaptor.1, body
    flat.clear_grads(["scale"])
    assert flat.flats[0].grad is None and flat.flats[1].grad is not None
    with pytest.raises(ValueError, match="adaptor.0.bias"):
        flat.clear_grads(["adaptor.0.weight"])
