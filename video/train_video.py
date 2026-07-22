# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import contextlib
import logging
import numbers
import random
from typing import Any, Callable, Optional, Sequence, Iterable

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel

from src.datasets.video_dataset import FastVideoFolder
from src.datasets.weighted_dataset import WeightedDataset
from src.transforms.functional import rgb2ycbcr
from src.utils.app import BaseTrainVideoApp
from src.utils.ane_mode import drift_simulation
from src.utils.stream_helper import get_state_dict


class ForwardPassWrapper(nn.Module):
    """
    Wrapper to call model and loss in DDP forward pass (required for find_unused_parameters = True)
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, forward: Callable, *args, **kwargs):
        return forward(self.module, *args, **kwargs)


class FrameGradClipper:
    def __init__(self, max_norm: float, device: torch.device):
        self._max_norm = torch.tensor(max_norm, device=device)
        self._frame_count = 0
        self._stats = torch.zeros(2, device=device)
        self._grad_norm, self._grad_scale = self._stats

    def clip(self, dpb):
        to_pack = list()
        split = list()
        offset = 0
        for n, v in dpb.items():
            if v.requires_grad:
                vf = v.flatten()
                to_pack.append(vf)
                size = len(vf)
                split.append((n, v.shape, offset, size))
                offset += size

        packed = torch.cat(to_pack)

        packed.register_hook(self._clip_grad)

        for n, shape, offset, size in split:
            dpb[n] = packed[offset : offset + size].view(shape)

        return dpb

    def _clip_grad(self, g: torch.Tensor):
        norm = torch.linalg.norm(g, 2)
        scale = self._max_norm / torch.maximum(norm, self._max_norm)

        self._frame_count += 1
        self._grad_norm += norm
        self._grad_scale += scale

        return g * scale

    @property
    def stats(self):
        return self._stats / max(1, self._frame_count)


class TrainVideoApp(BaseTrainVideoApp):
    def __init__(self):
        super().__init__()

        self.p_frame_offset = 1
        self.is_cascaded = False
        self.cascade_frame_ranges: Optional[tuple[int, ...]] = None
        self.is_ref_image_cascaded = False
        self.is_mv_feature_cascaded = False
        self.is_rand_rate = False
        self._per_frame_qp_prob = 0.0
        self._qp_transition_prob = 0.0
        self.loss_func = None
        self.only_me = False
        self.frame_index_map: Optional[tuple[int, ...]] = None
        self.i_qp_map = None
        self.distortion_weights = None
        self.clear_grad_for_y_q = False
        self._brightness_aug = None

        self._i_frame_model = None
        self._optic_flow_loss = None

    def close(self):
        super().close()
        self._i_frame_model = None
        self._optic_flow_loss = None

    @property
    def i_frame_model(self):
        model = self._i_frame_model
        if model is None:
            self._i_frame_model = model = self.load_i_frame_model()
        return model

    def load_i_frame_model(self):
        config = self.config["model"]["i_frame"]

        from src.utils.model_factory import create_image_model

        model = create_image_model(config)

        model_path = self.resolve_path(config["ckpt_path"], default_source="checkpoints_mount")
        logging.info(f"loading image model from: {model_path}")

        state_dict = get_state_dict(model_path)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)

        for p in model.parameters():
            p.requires_grad = False

        return model

    @property
    def model_config(self) -> dict[str, Any]:
        return self.config["model"]["p_frame"]

    def create_model(self):
        from src.utils.model_factory import create_video_model

        config = self._resolve_nested_ckpt_paths(self.model_config)
        assert isinstance(config, dict)
        return create_video_model(config)

    def _resolve_nested_ckpt_paths(self, config):
        """Recursively resolve every ckpt_path string in a config tree. Needed for MoE models."""
        if isinstance(config, dict):
            return {
                k: (
                    self.resolve_path(v, default_source="checkpoints_mount")
                    if k == "ckpt_path" and isinstance(v, str)
                    else self._resolve_nested_ckpt_paths(v)
                )
                for k, v in config.items()
            }
        if isinstance(config, list):
            return [self._resolve_nested_ckpt_paths(v) for v in config]
        return config

    def initialize_model(self, model):
        if self._initialize_pretrained_part(model, "movement estimator", "train.mv_pretrain_path", "mv_module_names"):
            return
        if self._initialize_pretrained_part(model, "optic flow", "train.me_pretrain_path", "me_module_names"):
            return

    def _initialize_pretrained_part(self, model, part_name, config_path, module_names_attr):
        pretrain_path = self.resolve_path(
            self.get_config_by_path(config_path, default=None), default_source="checkpoints_mount"
        )
        if pretrain_path is None:
            return False

        logging.info(f"loading {part_name} weigths from {pretrain_path}")
        state = get_state_dict(pretrain_path)
        for prefix in getattr(model, module_names_attr):
            module = getattr(model, prefix)
            if isinstance(module, nn.Parameter):
                with torch.no_grad():
                    module.copy_(state[prefix])
            else:
                prefix_states = dict()
                for n, p in state.items():
                    if len(n) > len(prefix) and n.startswith(prefix) and n[len(prefix)] == ".":
                        prefix_states[n[len(prefix) + 1 :]] = p

                getattr(model, prefix).load_state_dict(prefix_states)

        return True

    def make_train_model(self, model):
        # model will be wrapped into DDP in prepare_for_epoch
        return ForwardPassWrapper(model)

    @staticmethod
    def get_unwrapped_model(model: Optional[torch.nn.Module]):
        if isinstance(model, DistributedDataParallel):
            model = model.module
        if isinstance(model, ForwardPassWrapper):
            model = model.module
        return model

    def create_optimizer(self):
        assert self.model is not None
        model = self.get_unwrapped_model(self.model)
        assert model is not None

        kwargs = self.get_config_by_path("train.optimizer.params", default=None)
        if kwargs is None:
            kwargs = dict()
        else:
            kwargs = kwargs.copy()

        kwargs["lr"] = self.get_config_by_path("train.learning_rate", expected_type=numbers.Real)

        optimizer_type = self.get_config_by_path("train.optimizer.type", default="AdamW", expected_type=str)
        optimizer_type = getattr(torch.optim, optimizer_type)

        me_module_names = getattr(model, "me_module_names", None)
        if me_module_names is not None:
            me_params = list()
            remainder = list()
            for n, p in model.named_parameters():
                if self._has_prefix(n, me_module_names):
                    me_params.append(p)
                else:
                    remainder.append(p)

            return optimizer_type(
                [
                    {"params": me_params, "ratio": 0.01},
                    {"params": remainder, "ratio": 1.0},
                ],
                **kwargs,
            )
        else:
            return optimizer_type(model.parameters(), **kwargs)

    @staticmethod
    def _has_prefix(name: str, prefixes: Iterable[str]):
        ln = len(name)
        for prefix in prefixes:
            lp = len(prefix)
            if ln > lp:
                if name.startswith(prefix) and name[lp] == ".":
                    return True
            elif ln == lp:
                if name == prefix:
                    return True

        return False

    def create_video_dataset(self, config, patch_w, patch_h, default_config):
        target_resolution = config.get("target_resolution", default_config.get("target_resolution"))
        allow_tiling = config.get("allow_tiling", default_config.get("allow_tiling"))
        resampling_method = config.get("resampling_method", default_config.get("resampling_method"))
        frame_selection = config.get("frame_selection", default_config.get("frame_selection", "random"))
        max_frame_distance = config.get("max_frame_distance", default_config.get("max_frame_distance"))
        frame_distance = config.get("frame_distance", default_config.get("frame_distance"))
        random_flip = config.get("random_flip", default_config["random_flip"])
        n_frames = config.get("n_frames", default_config["n_frames"])
        thread_count = config.get("thread_count", default_config.get("thread_count"))
        precomputed_masks_path = config.get("precomputed_masks_path", default_config.get("precomputed_masks_path"))
        padding_simulation_prob = config.get(
            "padding_simulation_prob", default_config.get("padding_simulation_prob", 0.0)
        )
        padding_simulation_min = config.get("padding_simulation_min", default_config.get("padding_simulation_min", 0.0))
        padding_simulation_max = config.get(
            "padding_simulation_max", default_config.get("padding_simulation_max", 0.15)
        )
        padding_simulation_color = config.get(
            "padding_simulation_color", default_config.get("padding_simulation_color", "gray")
        )
        padding_simulation_alignment = config.get(
            "padding_simulation_alignment", default_config.get("padding_simulation_alignment", "bottom_right")
        )
        target_path = config.get("target_path")
        if target_path in (None, ""):
            target_description_path = None
        else:
            target_description_path = self.resolve_path(target_path)
        target_prob = config.get("target_prob", default_config.get("target_prob", 1.0))

        return FastVideoFolder(
            self.resolve_path(config["path"]),
            patch_w,
            patch_h,
            crop_method="random",
            target_resolution=target_resolution,
            allow_tiling=allow_tiling,
            resampling_method=resampling_method,
            frame_selection=frame_selection,
            max_frame_distance=max_frame_distance,
            frame_distance=frame_distance,
            random_flip=random_flip,
            frame_num=n_frames,
            thread_count=thread_count,
            precomputed_masks_path=precomputed_masks_path,
            padding_simulation_prob=padding_simulation_prob,
            padding_simulation_min=padding_simulation_min,
            padding_simulation_max=padding_simulation_max,
            padding_simulation_color=padding_simulation_color,
            padding_simulation_alignment=padding_simulation_alignment,
            target_description_path=target_description_path,
            target_prob=target_prob,
        )

    @staticmethod
    def _validate_data_sources_target_path(top_level_config, data_sources):
        """Validate `target_path` semantics when `train.dataset.data_sources` is used.

        Constraints enforced:
        - `train.dataset.target_path` must NOT be set alongside `data_sources`.
          The two are ambiguous together: a top-level value cannot be safely
          inherited (different sources may need different restored targets,
          or none at all), so we require the per-source declaration.
        - All sources must agree on target presence: either every source
          declares a non-empty real `target_path`, or none do. Mixed batches
          would emit a `target_video` key for some samples and not others,
          which crashes PyTorch's default_collate (KeyError on heterogeneous
          dict keys).
        """
        top_level_target_path = top_level_config.get("target_path")
        if top_level_target_path not in (None, ""):
            raise ValueError(
                "train.dataset.target_path must not be set when "
                "train.dataset.data_sources is used. Declare target_path "
                "inside each data source instead (it is strictly per-source)."
            )
        presences = {name: bool(ds.get("target_path")) for name, ds in data_sources.items()}
        if len(set(presences.values())) > 1:
            with_target = sorted(n for n, p in presences.items() if p)
            without_target = sorted(n for n, p in presences.items() if not p)
            raise ValueError(
                "Inconsistent target_path across data sources: "
                f"sources with target={with_target}, "
                f"sources without target={without_target}. "
                "All sources must agree on target presence "
                "(mixed batches would crash default_collate). "
            )

    def create_train_dataset(self):
        config = self.get_config_by_path("train.dataset", expected_type=dict)

        patch_w, patch_h = config["patch_size"]
        desired_length = config.get("desired_length")

        datasets = []
        weights = []

        raw_data_sources = config.get("data_sources")
        if raw_data_sources is None:
            data_sources = {"dataset": config}
        else:
            self._validate_data_sources_target_path(config, raw_data_sources)
            data_sources = raw_data_sources

        for ds in data_sources.values():
            datasets.append(self.create_video_dataset(ds, patch_w, patch_h, config))
            weights.append(ds.get("weight"))

        if len(datasets) == 0:
            raise ValueError("No data sources is defined")
        if desired_length is None and len(datasets) == 1:
            return datasets[0]

        if all(w is None for w in weights):
            weights = None
        elif any(w is None for w in weights):
            raise ValueError("Either all or no datasets must have weights specified")

        if weights is None and desired_length is None:
            return torch.utils.data.ConcatDataset(datasets)

        return WeightedDataset(datasets, weights, desired_length)

    def prepare_for_epoch(self, epoch):
        super().prepare_for_epoch(epoch)

        assert self.model is not None
        assert self.optimizer is not None

        config = self.get_config_by_path("train", expected_type=dict)

        lr = config["learning_rate"]
        param_group = config.get("param_group", None)
        loss_type = config["loss_type"]
        is_cascaded = config["cascade"]
        is_rand_rate = config["random_rate"]
        transfer_weights = config["transfer_weights"]

        for g in self.optimizer.param_groups:
            if is_cascaded:
                g["lr"] = lr * g.get("ratio", 1)
            else:
                g["lr"] = lr

        self._update_optimizer_params()

        if self.rank <= 0:
            self.metrics_writer.add_scalar("lr", lr, epoch)

        model = self.get_unwrapped_model(self.model)
        assert model is not None
        if transfer_weights != "none":
            logging.info(f"running weight transfer: {transfer_weights}")
            model.copy_feature_extractor_weight(transfer_weights)  # type: ignore[operator]

        only_me = False
        if param_group == "inter":
            only_me = True
            # train motion estimation and compensation blocks
            self._require_grad(model, model.mv_module_names, invert=False)  # type: ignore[arg-type]
        elif param_group == "residue":
            # train residue network
            self._require_grad(model, model.mv_module_names, invert=True)  # type: ignore[arg-type]
        elif param_group in (None, "both", "all"):
            modules_to_train = config.get("modules_to_train")
            if modules_to_train is not None:
                self._require_grad(model, modules_to_train, invert=False)
            else:
                # train all parameters
                model.requires_grad_()
        else:
            assert False

        self.only_me = only_me

        noise_level = config.get("noise_level", 0.5)
        model.set_noise_level(noise_level)  # type: ignore[operator]

        model.set_use_ckpt(bool(config.get("use_backprop_checkpoints", False)))  # type: ignore[operator]

        self.is_cascaded = is_cascaded
        self.is_rand_rate = is_rand_rate

        if is_rand_rate:
            self._per_frame_qp_prob = float(config.get("random_rate_per_frame", 0.0))
            self._qp_transition_prob = float(config.get("qp_transition_prob", 0.0))
            if not (0.0 <= self._per_frame_qp_prob <= 1.0):
                raise ValueError(f"train.random_rate_per_frame ({self._per_frame_qp_prob}) must be within [0, 1]")
            if not (0.0 <= self._qp_transition_prob <= 1.0):
                raise ValueError(f"train.qp_transition_prob ({self._qp_transition_prob}) must be within [0, 1]")
            if self._per_frame_qp_prob + self._qp_transition_prob > 1.0 + 1e-6:
                raise ValueError(
                    f"train.random_rate_per_frame ({self._per_frame_qp_prob}) + "
                    f"train.qp_transition_prob ({self._qp_transition_prob}) must be <= 1.0"
                )

        from src.losses.codec import get_loss_func

        self.loss_func = get_loss_func(
            loss_type,
            is_yuv420=self.is_yuv420,
            mse_lambda_exponent=config.get("mse_lambda_exponent"),
            bpp_weight=config.get("bpp_loss_weight"),
            mse_y_weight=config.get("mse_y_weight"),
            mse_yuv_420=config.get("mse_yuv_420"),
            mse_yuv_mean_in_psnr=config.get("mse_yuv_mean_in_psnr"),
            mse_rgb_weight=config.get("mse_rgb_weight"),
            create_optic_flow_loss=self._create_optic_flow_loss,
            auxiliary_models=self.auxiliary_models,
            perceptual_loss_weight=config.get("perceptual_loss_weight"),
            distortion_loss_weight=config.get("distortion_loss_weight"),
            roi_weight=config.get("roi_weight"),
            perceptual_rgb_weight=config.get("perceptual_rgb_weight"),
            roi_bg_ratio=config.get("roi_bg_ratio"),
            p_roi=config.get("p_roi"),
        )
        self.loss_func.to(self.device)

        self.clear_grad_for_y_q = config.get("clear_grad_for_y_q", False)

        # read brightness augmentation config
        ds_cfg = self.get_config_by_path("train.dataset", expected_type=dict)
        aug = ds_cfg.get("brightness_aug", None)
        if isinstance(aug, dict):
            self._brightness_aug = {
                "enabled": bool(aug.get("enabled", False)),
                "alpha_min": float(aug.get("alpha_min", 1.0)),
                "alpha_max": float(aug.get("alpha_max", 1.0)),
                "beta_min": float(aug.get("beta_min", 0.0)),
                "beta_max": float(aug.get("beta_max", 0.0)),
                "only_y_channel": bool(aug.get("only_y_channel", True)),
            }
        else:
            self._brightness_aug = None

        if (
            self.rank <= 0
            and self._brightness_aug is not None
            and self._brightness_aug["enabled"]
            and self._brightness_aug["only_y_channel"]
            and not self.is_yuv420
        ):
            logging.warning(
                "brightness_aug.only_y_channel=True has no effect when not using YUV420 mode; "
                "augmentation will be applied to all RGB channels"
            )

        n_frames = config["dataset"]["n_frames"]
        self.is_ref_image_cascaded = config.get("is_ref_image_cascaded", False)
        self.is_mv_feature_cascaded = config.get("is_mv_feature_cascaded", is_cascaded and n_frames < 8)

        frame_index_map: tuple[int, ...] | None = model.frame_index_map  # type: ignore[assignment]
        if frame_index_map is not None and n_frames < 15 and len(frame_index_map) > 4:
            frame_index_map = frame_index_map[:4]
        self.frame_index_map = frame_index_map

        self.initialize_lambdas(model)

        distortion_weights = config.get("distortion_weights")
        if distortion_weights is None:
            if "ssim" in loss_type:
                distortion_weights = [0.9, 1.1, 1.0]
            elif is_cascaded and n_frames >= 8:
                distortion_weights = [0.5, 1.2, 0.9]
            else:
                distortion_weights = [1.0, 2.0, 1.5]

        if len(distortion_weights) > 0:
            if frame_index_map is not None and config.get("reindex_distortion_weights", True):
                distortion_weights = [
                    distortion_weights[frame_index_map[idx % len(frame_index_map)]]
                    for idx in range(self.p_frame_offset, n_frames)
                ]

            if is_cascaded:
                first_frame_distortion_weight = config.get("first_frame_distortion_weight")
                if first_frame_distortion_weight is not None:
                    distortion_weights[0] = first_frame_distortion_weight

        if any(x != 1 for x in distortion_weights):
            self.distortion_weights = torch.tensor(distortion_weights, dtype=torch.float).to(self.device)
        else:
            self.distortion_weights = None

        frame_0_type = config.get("frame_0_type")
        if frame_0_type is None:
            frame_0_type = "i-frame" if n_frames > 3 else "pass-through"

        if frame_0_type == "i-frame":
            self.p_frame_offset = 1
            self.i_qp_map = self.get_i_frame_qp_map(self.qp_num)
        elif frame_0_type == "pass-through":
            self.p_frame_offset = 1
            self.i_qp_map = None
        elif frame_0_type == "none":
            self.p_frame_offset = 0
            self.i_qp_map = None
        else:
            raise ValueError(f"Unknown frame #0 type: {frame_0_type}")

        self.i_frame_dropout = config.get("i_frame_dropout", 0.0)

        self.drift_loss_weight = float(config.get("drift_loss_weight", 0.0))
        self.drift_feature_loss_weight = float(config.get("drift_feature_loss_weight", 0.0))
        self.use_drift_loss = self.drift_loss_weight > 0 or self.drift_feature_loss_weight > 0
        self.drift_mode = config.get("drift_mode", "ane")
        _bptt = config.get("drift_bptt", 0)
        self.drift_bptt = 10**9 if _bptt == "full" else int(_bptt)
        self.drift_y_noise_level = float(config.get("drift_y_noise_level", 0.0))
        self.drift_feature_noise_level = float(config.get("drift_feature_noise_level", 0.0))
        self.drift_memory_noise_level = float(config.get("drift_memory_noise_level", 0.0))
        self.drift_lambda_exponent = float(config.get("drift_lambda_exponent", 1.0))
        self.drift_noise_dist = config.get("drift_noise_dist", "uniform")
        if self.drift_noise_dist not in ("uniform", "gaussian"):
            raise ValueError(f"drift_noise_dist must be 'uniform' or 'gaussian', got {self.drift_noise_dist!r}")
        drift_ane_opts = config.get("drift_ane") or {}
        self.drift_ane_simulate_ops = drift_ane_opts.get("simulate_ops", None)
        self.drift_ane_exact_conv = bool(drift_ane_opts.get("exact_conv", False))
        self._dpb_drift = None
        self._drift_bptt_count = 0

        if not is_cascaded:
            self.cascade_frame_ranges = None
        else:
            cascade_partition_frames = config.get("cascade_partition_frames", [])
            cascade_partition_frames = sorted(
                idx for idx in cascade_partition_frames if self.p_frame_offset < idx < n_frames
            )
            cascade_partition_frames.append(n_frames)
            self.cascade_frame_ranges = tuple(cascade_partition_frames)

        if self.rank >= 0:
            find_unused = not is_cascaded or (
                self.cascade_frame_ranges is not None and len(self.cascade_frame_ranges) > 1
            )
            if not find_unused:
                unwrapped = self.get_unwrapped_model(self.model)
                expert_schedule = getattr(unwrapped, "expert_schedule", None)
                num_experts = getattr(unwrapped, "num_experts", 1)
                if expert_schedule is not None and num_experts > 1:
                    all_experts_in_schedule = len(set(expert_schedule)) == num_experts
                    if not all_experts_in_schedule:
                        find_unused = True
            self._update_ddp_model(find_unused)

    def get_i_frame_qp_map(self, qp_num, *, suppress_rescale_mapping=False):
        i_qp_range = self.get_config_by_path("model.p_frame.i_qp_range", default=None)
        if i_qp_range is None and suppress_rescale_mapping:
            # suppress mapping when full i-frame q-point range is used
            return None

        i_qp_num = self.i_frame_model.get_qp_num()
        if i_qp_range is None:
            i_qp_min, i_qp_max = 0, i_qp_num - 1
        else:
            if not isinstance(i_qp_range, Sequence) or len(i_qp_range) != 2:
                raise ValueError(f"Invalid i-frame q-point range: {i_qp_range}")

            i_qp_min, i_qp_max = i_qp_range
            i_qp_min, i_qp_max = int(i_qp_min), int(i_qp_max)
            if not (0 <= i_qp_min <= i_qp_max < i_qp_num):
                raise ValueError(f"Invalid i-frame q-point range: {i_qp_min}, {i_qp_max}")

        if suppress_rescale_mapping and i_qp_min == 0 and i_qp_max == i_qp_num - 1:
            # suppress mapping when full i-frame q-point range is used
            return None

        return torch.linspace(i_qp_min, i_qp_max, qp_num, dtype=torch.int64, device=self.device)

    def _require_grad(self, model: nn.Module, prefixes: Sequence[str], *, invert: bool):
        trainable_count = 0
        frozen_count = 0
        trainable_params = 0
        frozen_params = 0

        for n, p in model.named_parameters():
            require_grad = self._has_prefix(n, prefixes)
            if invert:
                require_grad = not require_grad

            p.requires_grad = require_grad

            num_params = p.numel()
            if require_grad:
                trainable_count += 1
                trainable_params += num_params
            else:
                frozen_count += 1
                frozen_params += num_params

        logging.info(
            f"Parameter gradients configured: "
            f"{trainable_count} trainable ({trainable_params:,} params), "
            f"{frozen_count} frozen ({frozen_params:,} params)"
        )

    def _update_ddp_model(self, find_unused_parameters):
        model = self.model
        if isinstance(model, DistributedDataParallel):
            if model.find_unused_parameters == find_unused_parameters:
                return

            model = model.module

        self.model = DistributedDataParallel(model, find_unused_parameters=find_unused_parameters)

    def _create_optic_flow_loss(self):
        optic_flow_loss = self._optic_flow_loss
        if optic_flow_loss is None:
            optic_flow_path = self.resolve_path(
                self.get_config_by_path("train.optic_flow_loss_path"), default_source="checkpoints_mount"
            )

            logging.info(f"loading optic flow loss weigths from {optic_flow_path}")

            from src.models.spynet import ME_Spynet

            optic_flow = ME_Spynet(3, 3, 0, 0, False)
            optic_flow.load_state_dict(get_state_dict(optic_flow_path))
            optic_flow.eval()

            from src.losses.optic_flow import OpticFlowLoss

            self._optic_flow_loss = optic_flow_loss = OpticFlowLoss(optic_flow)

        return optic_flow_loss

    def _update_optimizer_params(self):
        params = self.get_config_by_path("train.optimizer.params", default=None)
        if params is None:
            return

        assert self.optimizer is not None
        groups = self.optimizer.param_groups
        for n, v in params.items():
            changed = False
            for g in groups:
                if g[n] == v:
                    continue

                if not changed:
                    logging.info(f"Setting optimizer parameter {n} = {v}")
                    changed = True

                g[n] = v

    def train_step(self, epoch, step, batch):
        assert self.qp_num is not None
        assert self.training_lambdas is not None

        control_at_step = self.get_config_by_path("train.controller.at_step", default=False)
        if control_at_step:
            self.control_lambdas()

        videos = batch["video"]
        masks = batch.get("mask")
        target_videos = batch.get("target_video")

        # reshape batch to [batch x frame x channel x height x width]
        batch_size = videos.size(0)
        videos = videos.reshape(batch_size, -1, 3, *videos.shape[2:])
        n_frames = videos.size(1)
        assert n_frames > self.p_frame_offset

        if target_videos is not None:
            target_videos = target_videos.reshape(batch_size, n_frames, 3, *target_videos.shape[2:])

        if self.is_yuv420:
            # convert to yuv444
            videos = videos.flatten(0, 1)
            videos = rgb2ycbcr(videos)
            videos = videos.reshape(batch_size, n_frames, *videos.shape[1:])
            if target_videos is not None:
                target_videos = target_videos.flatten(0, 1)
                target_videos = rgb2ycbcr(target_videos)
                target_videos = target_videos.reshape(batch_size, n_frames, *target_videos.shape[1:])

        # apply brightness augmentation (identical draw on input and target)
        if self._brightness_aug is not None and self._brightness_aug["enabled"]:
            a, b = self._sample_alpha_beta((batch_size, 1, 1, 1, 1), videos.device)
            if self.is_yuv420:
                if self._brightness_aug.get("only_y_channel", True):
                    videos = self._apply_brightness_yuv_y(videos, a, b)
                    if target_videos is not None:
                        target_videos = self._apply_brightness_yuv_y(target_videos, a, b)
                else:
                    videos = self._apply_brightness_yuv_all(videos, a, b)
                    if target_videos is not None:
                        target_videos = self._apply_brightness_yuv_all(target_videos, a, b)
            else:
                videos = self._apply_brightness_rgb(videos, a, b)
                if target_videos is not None:
                    target_videos = self._apply_brightness_rgb(target_videos, a, b)

        # sample q-indices
        if self.is_rand_rate:
            per_frame_qp_prob = self._per_frame_qp_prob
            qp_transition_prob = self._qp_transition_prob
            if per_frame_qp_prob > 0.0 or qp_transition_prob > 0.0:
                q_index = torch.randint(self.qp_num, size=(batch_size, 1)).repeat(1, n_frames)
                for b in range(batch_size):
                    r = torch.rand(()).item()
                    if r < per_frame_qp_prob:
                        q_index[b] = torch.randint(self.qp_num, size=(n_frames,))
                    elif r < per_frame_qp_prob + qp_transition_prob and n_frames > 1:
                        n_trans = int(torch.randint(1, 3, size=()).item())
                        positions = torch.randperm(n_frames - 1)[: min(n_trans, n_frames - 1)] + 1
                        for tf in positions.sort().values.tolist():
                            q_index[b, tf:] = int(torch.randint(self.qp_num, size=()).item())
                    # else: keep the constant qp set above
            else:
                q_index = torch.randint(self.qp_num, size=(batch_size,))
        else:
            assert self.training_lambdas.size(0) == 64
            q_index = torch.randint(4, size=(batch_size,)) * 21

        if q_index.dim() == 1:
            q_index = q_index.unsqueeze(1).expand(-1, n_frames).contiguous()
        lambdas = self.training_lambdas[q_index]

        q_index = q_index.to(videos.device)
        lambdas = lambdas.to(videos.device)

        kwargs = dict()
        if n_frames <= self.p_frame_offset + 1 and self.only_me:
            kwargs["only_me"] = self.only_me

        dpb: dict[str, Any] = {
            "ref_frame": None,
            "ref_feature": None,
            "ref_mv_feature": None,
            "ref_y": None,
            "ref_mv_y": None,
        }

        if self.use_drift_loss:
            self._dpb_drift = {
                "ref_frame": None,
                "ref_feature": None,
                "ref_mv_feature": None,
                "ref_y": None,
                "ref_mv_y": None,
            }
            self._drift_bptt_count = 0

        # obtain i-frame
        if self.p_frame_offset > 0:
            ref_frame = videos[:, 0]
            if self.i_frame_dropout > 0 and random.random() < self.i_frame_dropout:
                ref_frame = torch.full_like(ref_frame, 0.5)
            elif self.i_qp_map is not None:
                q_index_i = self.i_qp_map[q_index[:, 0]]
                with torch.no_grad():
                    ref_frame = self.i_frame_model(ref_frame, q_index=q_index_i, recon_only=True)
                    # make sure i-frame model does not produce NaNs
                    torch.nan_to_num_(ref_frame, 0.0, 0.0, 1.0)

            dpb["ref_frame"] = ref_frame

        # step over batch
        if n_frames <= self.p_frame_offset + 1 or not self.is_cascaded:
            metric_names, metric_values = self._step_by_frame(
                videos, dpb=dpb, q_index=q_index, lambdas=lambdas, masks=masks, target_videos=target_videos, **kwargs
            )
        else:
            metric_names, metric_values = self._step_by_clip(
                videos, dpb=dpb, q_index=q_index, lambdas=lambdas, masks=masks, target_videos=target_videos, **kwargs
            )

        # average and reduce metrics
        if n_frames > self.p_frame_offset + 1:
            metric_values /= n_frames - self.p_frame_offset
        assert metric_values is not None
        # Bin each clip by its first-frame q-index (approximate when q-index varies per frame).
        q_index_for_bins = q_index[:, 0].contiguous()
        q_index_for_bins, metric_values = self.all_gather_tensors(q_index_for_bins, metric_values)

        # accumulate metrics
        assert metric_names is not None
        self.add_step_metric("loss", metric_values[:, metric_names["loss"]].mean())
        for name in ("drift_pixel", "drift_feature"):
            if name in metric_names:
                self.add_step_metric(name, metric_values[:, metric_names[name]].mean())
        self.add_step_info(metric_names, metric_values, qp_num=self.qp_num, q_index=q_index_for_bins)

    def _step_by_frame(
        self,
        videos: torch.Tensor,
        *,
        dpb,
        q_index: torch.Tensor,
        lambdas: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        target_videos: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Runs forward/backward pass over frames - optimizer takes a step after each frame and zeros gradients."""
        assert self.model is not None
        assert self.optimizer is not None
        assert self.loss_func is not None

        n_frames = videos.shape[1]
        clip_metric_names = None
        clip_metric_values = None
        for frame_idx in range(self.p_frame_offset, n_frames):
            x = videos[:, frame_idx]
            target_x = target_videos[:, frame_idx] if target_videos is not None else None
            mask = masks[:, frame_idx] if masks is not None else None

            self.optimizer.zero_grad()

            loss, dpb, metric_names, metric_values = self.model(
                self._frame_forward,
                x,
                dpb=dpb,
                q_index=q_index,
                frame_idx=frame_idx,
                lambdas=lambdas,
                is_last_frame=frame_idx == n_frames - 1,
                mask=mask,
                target_x=target_x,
                **kwargs,
            )

            loss.backward()
            if self.use_drift_loss:
                self._truncate_drift_bptt()
            if self.loss_func.bpp_weight == 0 and self.clear_grad_for_y_q:
                model = self.get_unwrapped_model(self.model)
                assert model is not None
                model.clear_grad_for_y_q()  # type: ignore[operator]

            self.optimizer_step()

            if clip_metric_names is None:
                clip_metric_names = metric_names
                clip_metric_values = metric_values
            else:
                assert clip_metric_names == metric_names
                assert clip_metric_values is not None
                clip_metric_values += metric_values

        return clip_metric_names, clip_metric_values

    def _base_frame_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        *,
        dpb,
        q_index: torch.Tensor,
        frame_idx: int,
        lambdas: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target_x: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Runs a forward pass over one frame, calculates frame loss and returns the metrics."""
        assert self.frame_index_map is not None
        assert self.loss_func is not None
        fa_idx = self.frame_index_map[frame_idx % len(self.frame_index_map)]

        frame_q_index = q_index[:, frame_idx]
        frame_lambdas = lambdas[:, frame_idx]

        rd = model(x, dpb, q_index=frame_q_index, fa_idx=fa_idx, **kwargs)

        if self.distortion_weights is not None:
            weight_idx = min(frame_idx - self.p_frame_offset, len(self.distortion_weights) - 1)
            frame_lambdas = frame_lambdas * self.distortion_weights[weight_idx]

        loss_target = target_x if target_x is not None else x
        ld = self.loss_func(rd, loss_target, lambdas=frame_lambdas, mask=mask)
        loss = ld.pop("loss")

        drift_pixel = drift_feature = None
        if self.use_drift_loss:
            drift_pixel, drift_feature = self._drift_step(model, rd, dpb, frame_q_index, fa_idx)
            drift_scale = frame_lambdas**self.drift_lambda_exponent
            loss = loss + drift_scale * (
                self.drift_loss_weight * drift_pixel + self.drift_feature_loss_weight * drift_feature
            )

        metrics = dict(loss=loss.detach())
        metrics.update((n, v.detach()) for n, v in rd.items() if n.startswith("bpp_"))
        metrics.update((n, v.detach()) for n, v in ld.items())
        if drift_pixel is not None:
            metrics["drift_pixel"] = drift_pixel.detach()
        if drift_feature is not None:
            metrics["drift_feature"] = drift_feature.detach()

        metric_names = {n: i for i, n in enumerate(metrics.keys())}
        metric_values = torch.stack(list(metrics.values()), dim=-1)
        return loss.mean(), rd["dpb"], metric_names, metric_values

    def _drift_step(self, model, rd, dpb, q_index, fa_idx):
        dpb_drift = self._dpb_drift
        if dpb_drift is not None and dpb_drift.get("ref_frame") is None and dpb.get("ref_frame") is not None:
            dpb_drift["ref_frame"] = dpb["ref_frame"].detach()

        dpb_out = rd["dpb"]
        y_res_q = rd["y_q"]
        if self.drift_y_noise_level > 0:
            y_res_q = tuple((self._drift_perturb(sym, self.drift_y_noise_level), scales) for sym, scales in y_res_q)

        with drift_simulation(
            self.drift_mode,
            device_type=dpb_out["ref_frame"].device.type,
            simu=self.drift_ane_simulate_ops,
            exact_conv=self.drift_ane_exact_conv,
        ):
            x_hat_drift, feat_drift, dpb_drift_out, _, _ = model.decode_core(
                dpb_drift, q_index, fa_idx, z_hat=rd["z_hat"], y_res_q=y_res_q
            )

        perturb_feature = self.drift_feature_noise_level > 0 or self.drift_memory_noise_level > 0
        if perturb_feature:
            dpb_drift_out, feat_drift = self._drift_perturb_dpb_feature(dpb_drift_out, feat_drift)

        self._dpb_drift = dpb_drift_out
        self._drift_bptt_count += 1
        if self._drift_bptt_count > self.drift_bptt:
            self._truncate_drift_bptt()

        drift_pixel = (dpb_out["ref_frame"] - x_hat_drift).pow(2).flatten(1).mean(dim=1)
        drift_feature = (dpb_out["ref_feature"] - feat_drift).pow(2).flatten(1).mean(dim=1)
        return drift_pixel, drift_feature

    def _drift_perturb_dpb_feature(self, dpb, feature):
        if dpb is None or feature is None:
            return dpb, feature

        if feature.shape[1] % 2 != 0:
            raise ValueError(f"drift feature noise expects an even channel count, got {feature.shape[1]}")

        feature_part, memory_part = feature.chunk(2, dim=1)
        if self.drift_feature_noise_level > 0:
            feature_part = self._drift_perturb(feature_part, self.drift_feature_noise_level)
        if self.drift_memory_noise_level > 0:
            memory_part = self._drift_perturb(memory_part, self.drift_memory_noise_level)
        feature = torch.cat((feature_part, memory_part), dim=1)
        dpb = dict(dpb)
        dpb["ref_feature"] = feature
        return dpb, feature

    def _drift_perturb(self, x, noise_level):
        if self.drift_noise_dist == "gaussian":
            noise = torch.randn_like(x) * (noise_level / 3**0.5)
        else:
            noise = torch.empty_like(x).uniform_(-noise_level, noise_level)
        return x + noise.detach()

    def _truncate_drift_bptt(self):
        if self._dpb_drift is not None:
            self._dpb_drift = {n: (v.detach() if torch.is_tensor(v) else v) for n, v in self._dpb_drift.items()}
        self._drift_bptt_count = 0

    def _frame_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        *,
        dpb,
        q_index: torch.Tensor,
        frame_idx: int,
        lambdas: torch.Tensor,
        is_last_frame: bool,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Runs a forward pass over one frame, updates the buffer and returns the metrics."""
        loss, dpb, metric_names, metric_values = self._base_frame_forward(
            model, x, dpb=dpb, q_index=q_index, frame_idx=frame_idx, lambdas=lambdas, mask=mask, **kwargs
        )
        if is_last_frame:
            dpb = None
        else:
            dpb = {n: v.detach() for n, v in dpb.items()}

        return loss, dpb, metric_names, metric_values

    def _step_by_clip(
        self,
        videos: torch.Tensor,
        *,
        dpb,
        q_index: torch.Tensor,
        lambdas: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        target_videos: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Runs forward/backward pass over clip - optimizer takes a step after each clip, gradients
        are accumulated over frames."""
        assert self.model is not None
        assert self.optimizer is not None
        assert self.cascade_frame_ranges is not None

        frame_grad_clipper = self._create_frame_grad_clipper()

        clip_metric_names = None
        clip_metric_values = None

        self.optimizer.zero_grad()

        start_idx = self.p_frame_offset
        for frame_idx in self.cascade_frame_ranges:
            is_last = frame_idx == self.cascade_frame_ranges[-1]
            if not is_last and self.rank >= 0:
                context = self.model.no_sync()  # type: ignore[operator]
            else:
                context = contextlib.nullcontext()

            with context:
                loss, dpb, metric_names, metric_values = self.model(
                    self._frame_range_forward,
                    videos,
                    start_idx,
                    frame_idx,
                    dpb=dpb,
                    q_index=q_index,
                    lambdas=lambdas,
                    return_dpb_frame=None if is_last else frame_idx - 1,
                    frame_grad_clipper=frame_grad_clipper,
                    masks=masks,
                    target_videos=target_videos,
                    **kwargs,
                )
                loss.backward()
                if self.use_drift_loss:
                    self._truncate_drift_bptt()

            del loss

            if clip_metric_names is None:
                clip_metric_names = metric_names
                clip_metric_values = metric_values
            else:
                assert clip_metric_names == metric_names
                assert clip_metric_values is not None
                clip_metric_values += metric_values

            start_idx = frame_idx

        self.optimizer_step()

        if frame_grad_clipper is not None:
            self._add_frame_grad_stats(frame_grad_clipper)

        return clip_metric_names, clip_metric_values

    def _frame_range_forward(
        self,
        model: nn.Module,
        videos: torch.Tensor,
        start,
        stop,
        *,
        dpb,
        return_dpb_frame,
        q_index: torch.Tensor,
        lambdas: torch.Tensor,
        frame_grad_clipper: Optional[FrameGradClipper] = None,
        masks: Optional[torch.Tensor] = None,
        target_videos: Optional[torch.Tensor] = None,
    ):
        """Runs a forward pass over a clip by frames and returns the average loss."""
        return_dpb = None
        clip_loss = None
        clip_metric_names = None
        clip_metric_values = None
        for frame_idx in range(start, stop):
            x = videos[:, frame_idx]
            target_x = target_videos[:, frame_idx] if target_videos is not None else None
            mask = masks[:, frame_idx] if masks is not None else None

            loss, dpb, metric_names, metric_values = self._base_frame_forward(
                model,
                x,
                dpb=dpb,
                q_index=q_index,
                frame_idx=frame_idx,
                lambdas=lambdas,
                mask=mask,
                target_x=target_x,
            )

            # Prepare dpb for next frame range
            if frame_idx == return_dpb_frame:
                # detach dpb tensors to avoid backpropagating through them
                return_dpb = {n: v.detach() for n, v in dpb.items()}

            # Prepare dpb for next frame
            if frame_idx != stop - 1:
                # If cascaded, detach only the tensors that are not cascaded
                if not self.is_ref_image_cascaded:
                    ref_frame = dpb.get("ref_frame")
                    if ref_frame is not None:
                        dpb["ref_frame"] = ref_frame.detach()
                if not self.is_mv_feature_cascaded:
                    ref_mv_feature = dpb.get("ref_mv_feature")
                    if ref_mv_feature is not None:
                        dpb["ref_mv_feature"] = ref_mv_feature.detach()

                if frame_grad_clipper is not None:
                    dpb = frame_grad_clipper.clip(dpb)

            if clip_loss is None:
                clip_loss = loss
            else:
                clip_loss += loss

            if clip_metric_names is None:
                clip_metric_names = metric_names
                clip_metric_values = metric_values
            else:
                assert clip_metric_names == metric_names
                assert clip_metric_values is not None
                clip_metric_values += metric_values

        average_frame_loss = self.get_config_by_path("train.average_frame_loss", default=True)
        if average_frame_loss:
            if (stop - start) > 1:
                clip_loss /= stop - start
        return clip_loss, return_dpb, clip_metric_names, clip_metric_values

    def _create_frame_grad_clipper(self):
        grad_limit = self.get_config_by_path("train.frame_grad_max_norm", expected_type=numbers.Real, default=0.0)
        if grad_limit <= 0:
            return None

        return FrameGradClipper(grad_limit, self.device)

    def _add_frame_grad_stats(self, frame_grad_clipper: FrameGradClipper):
        stats = frame_grad_clipper.stats
        self.all_reduce(stats)

        self.add_step_metric("frame_grad_norm", stats[0].item())
        self.add_step_metric("frame_grad_scale", stats[1].item())

    def test_model(self, epoch, model: Any):
        # Optional model optimizations for eval (may return a copy)
        eval_config = self.get_config_by_path("model.p_frame", expected_type=dict, default={})
        model = model.prepare_for_eval(eval_config)

        model.eval()
        # clear outdated PMF distributions
        model.reset_encoder_pmf()
        self.run_benchmark_test(
            epoch,
            i_frame_model=lambda: self.i_frame_model,
            p_frame_model=model,
            i_frame_qp_map=self.get_i_frame_qp_map(model.get_qp_num(), suppress_rescale_mapping=True),
        )
        if self._auxiliary_models is not None:
            self._auxiliary_models.offload_heavy_models()

    def _sample_alpha_beta(self, shape, device):
        assert self._brightness_aug is not None
        a = torch.empty(shape, device=device).uniform_(
            self._brightness_aug["alpha_min"], self._brightness_aug["alpha_max"]
        )
        b = torch.empty(shape, device=device).uniform_(
            self._brightness_aug["beta_min"], self._brightness_aug["beta_max"]
        )
        return a, b

    def _apply_brightness_rgb(self, videos: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # videos: [B, F, 3, H, W] in [0,1]; apply the given per-video a,b to all frames
        return (a * videos + b).clamp_(0.0, 1.0)

    def _apply_brightness_yuv_y(self, videos: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # videos: [B, F, 3, H, W] in [0,1]; apply the given per-video a,b to the Y channel
        y = videos[:, :, :1, :, :]  # [B, F, 1, H, W]
        videos[:, :, :1, :, :] = (a * y + b).clamp_(0.0, 1.0)
        return videos

    def _apply_brightness_yuv_all(self, videos: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # videos: [B, F, 3, H, W] in [0,1]; apply the given per-video a,b to all YUV channels
        return (a * videos + b).clamp_(0.0, 1.0)


if __name__ == "__main__":
    TrainVideoApp().main()
