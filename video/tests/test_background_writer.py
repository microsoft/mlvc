# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for asynchronous save_dir writes in BaseTrainApp (train.async_save_dir_writes)."""

import os
import threading

import pytest
import torch

from src.utils.app import BaseTrainApp
from src.utils.common import read_epoch_marker


class _App(BaseTrainApp):
    def run(self):
        pass

    def create_model(self):
        raise NotImplementedError()

    def create_train_dataset(self):
        raise NotImplementedError()

    def test_model(self, epoch, model):
        pass

    def train_step(self, epoch, step, batch):
        pass


@pytest.fixture
def app(tmp_path):
    app = _App()
    app._config = {"save_dir": str(tmp_path / "save_dir"), "train": {"async_save_dir_writes": True}}
    app.model = torch.nn.Linear(2, 2)
    app.optimizer = torch.optim.SGD(app.model.parameters(), lr=0.1)
    yield app
    app.close_background_writer()


def test_async_checkpoint_is_published_in_the_background(app, tmp_path):
    save_dir = tmp_path / "save_dir"
    gate = threading.Event()
    # hold the writer so the checkpoint stays queued
    app.background_writer.submit("blocker", lambda: gate.wait(5))

    app.checkpoint(0, is_first_checkpoint=True)
    assert read_epoch_marker(str(save_dir / "epoch_marker")) is None
    assert not (save_dir / "model_epo0.ckpt").exists()

    gate.set()
    local_dir = app.local_output_dir
    app.close_background_writer()
    assert read_epoch_marker(str(save_dir / "epoch_marker")) == 0
    assert (save_dir / "model_epo0.ckpt").exists() and (save_dir / "optimizer_epo0.ckpt").exists()
    assert not os.path.exists(local_dir)


def test_pruning_keeps_the_checkpoint_that_the_epoch_marker_points_to(app, tmp_path, monkeypatch):
    save_dir = tmp_path / "save_dir"
    app.config["train"]["ckpt_per_epoch"] = 4  # epochs 0-2 are not retained

    def checkpoint_and_prune(epoch):
        # as in train(): pruning the previous checkpoint is queued after publishing the new one
        app.checkpoint(epoch, is_first_checkpoint=False)
        app.run_save_dir_task(f"pruning checkpoint of epoch {epoch - 1}", lambda: app.prune_checkpoint(epoch - 1))

    app.checkpoint(0, is_first_checkpoint=True)
    checkpoint_and_prune(1)
    app.close_background_writer()
    assert not (save_dir / "model_epo0.ckpt").exists() and not (save_dir / "optimizer_epo0.ckpt").exists()

    def failing_copy(src, dst):
        raise OSError("mount timed out")

    monkeypatch.setattr("src.utils.app.shutil.copyfile", failing_copy)
    checkpoint_and_prune(2)
    app.background_writer.submit("wait for the queued tasks", lambda: None).result()
    assert os.listdir(app.local_output_dir) == []
    app.close_background_writer()

    assert read_epoch_marker(str(save_dir / "epoch_marker")) == 1
    assert (save_dir / "model_epo1.ckpt").exists() and (save_dir / "optimizer_epo1.ckpt").exists()
