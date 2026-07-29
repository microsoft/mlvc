# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from pathlib import Path

import pytest
import torch
from PIL import Image


def test_deqa_inference_matches_expected_score():
    checkpoint_value = os.getenv("DEQA_CHECKPOINT_PATH")
    if checkpoint_value is None:
        pytest.skip("Set DEQA_CHECKPOINT_PATH to run the real-weight DeQA test")

    checkpoint_path = Path(checkpoint_value)
    assert checkpoint_path.is_dir(), f"DeQA checkpoint not found: {checkpoint_path}"

    from src.metrics.deqa.scorer import Scorer

    repository_root = Path(__file__).resolve().parents[2]
    image = Image.open(repository_root / "assets" / "frame_comparison_full-hevc_qsv.png").convert("RGB")
    h265_crop = image.crop((3, 37, 722, 441))
    mlvc_crop = image.crop((743, 37, 1462, 441))
    scorer = Scorer(str(checkpoint_path), device="cpu").eval()
    assert scorer.model.config._attn_implementation == "sdpa"

    scores = scorer([h265_crop, mlvc_crop])

    torch.testing.assert_close(
        scores,
        torch.tensor([1.9541015625, 3.787109375], dtype=scores.dtype),
        rtol=0,
        atol=1e-3,
    )
