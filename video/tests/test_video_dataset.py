# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.datasets.video_dataset import FastVideoFolder


def _write_sequence(root, size, value):
    root.mkdir(parents=True)
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = value
    image[:, width // 2 :, 1] = 255
    image[height // 2 :, :, 2] = 255
    for name in ("0.png", "1.png"):
        Image.fromarray(image).save(root / name)
        Image.fromarray(image[:, :, 1]).save(root / f"{name}_mask.png")
    path = root / "description.json"
    path.write_text(
        json.dumps([dict(path=".", seq_length=2, width=width, height=height, frames=["0.png", "1.png"])]),
        encoding="utf-8",
    )
    return path, Image.fromarray(image)


@pytest.fixture
def make_dataset(tmp_path):
    datasets = []

    def make(source_size, target_size=None, **options):
        source, image = _write_sequence(tmp_path / "input", source_size, 50)
        target, target_image = None, None
        if target_size is not None:
            target, target_image = _write_sequence(tmp_path / "target", target_size, 170)
        dataset = FastVideoFolder(
            str(source),
            32,
            32,
            frame_num=2,
            frame_selection="fix",
            precomputed_masks_path=".",
            target_description_path=str(target) if target else None,
            **options,
        )
        datasets.append(dataset)
        return dataset, image, target_image

    yield make
    for dataset in datasets:
        if dataset._thread_pool is not None:
            dataset._thread_pool.shutdown(wait=True)


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize(
    "source_size,target_size,resolution,expected_size",
    [
        ((240, 144), (160, 96), 72, (120, 72)),
        ((240, 144), (160, 96), 120, (200, 120)),
        ((160, 96), (240, 144), 120, (200, 120)),
        ((160, 96), (240, 144), 180, (300, 180)),
        ((160, 96), (240, 144), 96, (160, 96)),
        ((160, 96), (240, 144), None, (160, 96)),
        ((144, 240), (96, 160), 120, (120, 200)),
        ((161, 120), (80, 60), 90, (121, 90)),
    ],
)
def test_exact_resize_pairs_and_masks(make_dataset, threads, source_size, target_size, resolution, expected_size):
    dataset, image, target = make_dataset(
        source_size,
        target_size,
        exact_resizing=True,
        target_resolution=resolution,
        thread_count=threads,
    )
    sample = dataset[0]
    width, height = expected_size
    x, y = (width - 32) // 2, (height - 32) // 2
    for key, original in (("video", image), ("target_video", target)):
        expected = np.asarray(original.resize(expected_size, Image.Resampling.BOX).crop((x, y, x + 32, y + 32)))
        expected = torch.from_numpy(expected.copy().transpose(2, 0, 1)).float() / 255
        torch.testing.assert_close(sample[key], expected.repeat(2, 1, 1))
    torch.testing.assert_close(sample["mask"], sample["video"][1::3])


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("resolution,default_size,exact_size", [(72, (160, 96), (120, 72)), (40, (80, 48), (67, 40))])
def test_resize_preserves_unpaired_and_default_behavior(make_dataset, exact, resolution, default_size, exact_size):
    dataset, image, _ = make_dataset((160, 96), target_resolution=resolution, exact_resizing=exact)
    sample = dataset[0]
    size = exact_size if exact else default_size
    x, y = (size[0] - 32) // 2, (size[1] - 32) // 2
    expected = np.asarray(image.resize(size, Image.Resampling.BOX).crop((x, y, x + 32, y + 32)))
    torch.testing.assert_close(sample["video"][:3], torch.from_numpy(expected.copy().transpose(2, 0, 1)).float() / 255)
    assert "target_video" not in sample


@pytest.mark.parametrize("exact", [False, True])
def test_original_target_fallback(make_dataset, exact):
    target_size = (240, 144) if exact else (160, 96)
    dataset, _, _ = make_dataset(
        (160, 96), target_size, exact_resizing=exact, target_resolution=72, target_prob=0, padding_simulation_prob=1
    )
    sample = dataset[0]
    assert sample["target_video"] is sample["video"]
    padding = sample["video"][0] == 0.5
    assert (sample["mask"][:, padding] == 0).all()


@pytest.mark.parametrize(
    "exact,target_size,error", [(False, (240, 144), "key 'width'"), (True, (96, 160), "aspect ratio")]
)
def test_rejects_incompatible_pair_geometry(make_dataset, exact, target_size, error):
    with pytest.raises(ValueError, match=error):
        make_dataset((160, 96), target_size, exact_resizing=exact)


def test_exact_resize_keeps_frame_alignment(tmp_path):
    source, _ = _write_sequence(tmp_path / "input", (160, 96), 50)
    target, _ = _write_sequence(tmp_path / "target", (240, 144), 170)
    description = json.loads(target.read_text(encoding="utf-8"))
    description[0]["frames"].reverse()
    target.write_text(json.dumps(description), encoding="utf-8")
    with pytest.raises(ValueError, match="'frames' list differs"):
        FastVideoFolder(str(source), 32, 32, target_description_path=str(target), exact_resizing=True)


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize(
    "color,alignment", [("gray", "bottom_right"), ("black", "top_left"), ("gray", "both"), ("black", "random")]
)
def test_exact_resize_shares_zoom_flip_crop_and_padding(make_dataset, monkeypatch, threads, color, alignment):
    dataset, _, _ = make_dataset(
        (160, 96),
        (240, 144),
        exact_resizing=True,
        target_resolution=144,
        thread_count=threads,
        min_zoom_factor=0.75,
        max_zoom_factor=0.75,
        random_flip=True,
        crop_method="random",
        resampling_method="NEAREST",
        padding_simulation_prob=1,
        padding_simulation_min=0.25,
        padding_simulation_max=0.25,
        padding_simulation_color=color,
        padding_simulation_alignment=alignment,
    )
    monkeypatch.setattr("src.datasets.video_dataset.random.choice", lambda values: values[0])
    monkeypatch.setattr("src.datasets.video_dataset.random.randint", lambda low, high: (low + high) // 2)
    sample = dataset[0]
    value = 0.0 if color == "black" else 0.5
    padding = sample["video"][0] == value
    expected_padding = torch.zeros((32, 32), dtype=torch.bool)
    if alignment == "top_left":
        expected_padding[:8, :] = True
        expected_padding[:, :8] = True
    elif alignment == "both":
        expected_padding[:4, :] = expected_padding[-4:, :] = True
        expected_padding[:, :4] = expected_padding[:, -4:] = True
    else:
        expected_padding[-8:, :] = True
        expected_padding[:, -8:] = True
    torch.testing.assert_close(padding, expected_padding)
    assert (sample["target_video"][:, padding] == value).all()
    assert (sample["mask"][:, padding] == 0).all()
    torch.testing.assert_close(sample["video"][1::3, ~padding], sample["target_video"][1::3, ~padding])
    torch.testing.assert_close(sample["video"][1::3, ~padding], sample["mask"][:, ~padding])


@pytest.mark.parametrize("allow_tiling", [False, True])
def test_exact_resize_below_patch_size(make_dataset, allow_tiling):
    dataset, _, _ = make_dataset(
        (160, 96),
        (240, 144),
        exact_resizing=True,
        target_resolution=16,
        allow_tiling=allow_tiling,
    )
    sample = dataset[0]
    assert sample["video"].shape == sample["target_video"].shape == (6, 32, 32)
    assert sample["mask"].shape == (2, 32, 32)
    torch.testing.assert_close(sample["video"][1::3], sample["mask"])
    torch.testing.assert_close(sample["video"][1::3], sample["target_video"][1::3])


def test_target_shape_error_precedes_padding(make_dataset, monkeypatch):
    dataset, _, _ = make_dataset((160, 96), (240, 144), exact_resizing=True, padding_simulation_prob=1)
    load = dataset._load_seq_frames_sync

    def load_short_target(**kwargs):
        frames = load(**kwargs)
        return frames[:, :-1] if kwargs["seq_root_folder_path"] == dataset.target_root_folder_path else frames

    monkeypatch.setattr(dataset, "_load_seq_frames_sync", load_short_target)
    with pytest.raises(ValueError, match="Target shape .* differs from input shape"):
        dataset[0]
