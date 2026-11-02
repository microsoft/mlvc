# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the indexed CUDA graph tree alias lookup in src.utils.cudagraph_patches."""

import copy

import pytest
import torch

from torch._inductor import cudagraph_trees

from src.utils import cudagraph_patches

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require a GPU")


def test_indexed_lookup_matches_original_scan_for_chained_calls(monkeypatch):
    assert cudagraph_patches.patch_cudagraph_tree_alias_lookup()
    # cross-check every lookup against torch's original linear scan
    node_cls = cudagraph_trees.CUDAGraphNode
    for name, original in (
        ("_is_alias_of_live_recorded_tensor", cudagraph_patches._original_is_alias_of_live_recorded_tensor),
        ("_is_cuda_graph_recorded_tensor", cudagraph_patches._original_is_cuda_graph_recorded_tensor),
    ):

        def checked(self, t, indexed=getattr(node_cls, name), original=original):
            result = indexed(self, t)
            assert result == original(self, t)
            return result

        monkeypatch.setattr(node_cls, name, checked)
    torch.compiler.reset()

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(4, 4, 3, padding=1), torch.nn.LeakyReLU(), torch.nn.Conv2d(4, 4, 3, padding=1)
    ).cuda()
    reference = copy.deepcopy(model)
    compiled = torch.compile(model, backend="cudagraphs", fullgraph=True)
    x = torch.randn(2, 4, 16, 16, device="cuda")

    def clip_loss(module, n_frames):
        # chained calls keep earlier outputs alive for backward, like cascaded frames
        h, loss = x, x.new_zeros(())
        for _ in range(n_frames):
            h = module(h)
            loss = loss + h.square().mean()
        return loss

    # warm-up, recording and replay of one clip length, then a longer clip that extends the tree path
    for n_frames in (3, 3, 3, 5, 5):
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        clip_loss(compiled, n_frames).backward()
        clip_loss(reference, n_frames).backward()
        for p, q in zip(model.parameters(), reference.parameters()):
            assert p.grad is not None and q.grad is not None
            torch.testing.assert_close(p.grad, q.grad)
    torch.compiler.reset()
