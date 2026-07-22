# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from src.datasets.image_dataset import ImageFolder
from src.models.common_model import CompressionModel
from src.transforms.functional import rgb2ycbcr
from src.utils.app import BaseTrainVideoApp


class TrainImageApp(BaseTrainVideoApp):
    def __init__(self):
        super().__init__()
        self.loss_func = None

    @property
    def model_config(self):
        return self.config["model"]["i_frame"]

    def create_model(self):
        from src.utils.model_factory import create_image_model

        return create_image_model(self.model_config)

    def create_train_dataset(self):
        config = self.get_config_by_path("train.dataset", expected_type=dict)

        patch_w, patch_h = config["patch_size"]
        return ImageFolder(self.resolve_path(config["path"]), patch_w, patch_h, crop_method="random", random_flip=True)

    def prepare_for_epoch(self, epoch):
        super().prepare_for_epoch(epoch)

        config = self.get_config_by_path("train", expected_type=dict)
        lr = config["learning_rate"]
        loss_type = config["loss_type"]

        assert self.optimizer is not None
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        if self.rank <= 0:
            self.metrics_writer.add_scalar("lr", lr, epoch)

        model = self.get_unwrapped_model(self.model)
        self.initialize_lambdas(model)

        from src.losses.codec import get_loss_func

        self.loss_func = get_loss_func(
            loss_type,
            is_yuv420=self.is_yuv420,
            auxiliary_models=self.auxiliary_models,
            bpp_weight=config.get("bpp_loss_weight"),
            perceptual_loss_weight=config.get("perceptual_loss_weight"),
            distortion_loss_weight=config.get("distortion_loss_weight"),
            roi_weight=config.get("roi_weight"),
            perceptual_rgb_weight=config.get("perceptual_rgb_weight"),
        )

    def train_step(self, epoch, step, batch):
        control_at_step = self.get_config_by_path("train.controller.at_step", default=False)
        if control_at_step:
            self.control_lambdas()

        assert self.optimizer is not None
        self.optimizer.zero_grad()
        assert self.training_lambdas is not None
        rate_num = batch.size(0)

        if self.is_yuv420:
            batch = rgb2ycbcr(batch)

        qp_num = self.training_lambdas.size(0)
        q_index = torch.randint(qp_num, size=(rate_num,))
        lambdas = self.training_lambdas[q_index]

        q_index = q_index.to(batch.device)
        lambdas = lambdas.to(batch.device)

        assert self.model is not None
        assert self.loss_func is not None
        rd = self.model(batch, q_index=q_index)
        ld = self.loss_func(rd, batch, lambdas=lambdas)
        loss = ld.pop("loss")

        metrics = {
            "loss": loss.detach(),
            "bpp_y": rd["bpp_y"].detach(),
            "bpp_z": rd["bpp_z"].detach(),
        }
        for n, v in ld.items():
            metrics[n] = v.detach()

        metric_names = {n: i for i, n in enumerate(metrics.keys())}
        metric_values = torch.stack(list(metrics.values()), dim=-1)
        q_index, metric_values = self.all_gather_tensors(q_index, metric_values)

        self.add_step_metric("loss", metric_values[:, metric_names["loss"]].mean())
        self.add_step_info(metric_names, metric_values, qp_num=qp_num, q_index=q_index)

        loss.mean().backward()
        self.optimizer_step()

    def test_model(self, epoch, model: torch.nn.Module):
        model.eval()
        # clear outdated PMF distributions
        assert isinstance(model, CompressionModel)
        model.reset_encoder_pmf()
        self.run_benchmark_test(epoch, i_frame_model=model, p_frame_model=None)


if __name__ == "__main__":
    TrainImageApp().main()
