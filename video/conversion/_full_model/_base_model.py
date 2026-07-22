# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import torch
import numpy as np
import dataclasses
from pathlib import Path
from ..utils import download_job_outputs
from ..types import ModelParams, ModelVersion, GaussianCoderPmf, BitEstimatorPmf
from src.utils.stream_helper import get_state_dict, get_padding_size
from src.models.entropy_models import GaussianEncoder, BitEstimator

from typing import Callable


class BaseFullModel(torch.nn.Module):
    gaussian_encoder: GaussianEncoder
    bit_estimator_z: BitEstimator
    z_channel: int
    total_qp_num: int
    get_qp_num: Callable[[], int]

    def __init__(
        self,
        *args,
        model_version: ModelVersion,
        fake_quantized: bool = False,
        disable_feature_reset: bool = False,
        model_params: ModelParams | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if model_params is not None:
            self._model_params = model_params
        else:
            # QP mapping from 0-51 range to the model's QP range
            qp_mapping = None
            if self.get_qp_num() == 64:
                qps = np.arange(52, dtype=float)
                qp_mapping = np.clip(np.round((44.0 - qps) / 0.3), 0, 63).astype(int).tolist()

            self._model_params = ModelParams(
                model_version=model_version,
                pixel_range=1.0,
                qp_num=self.get_qp_num(),
                total_qp_num=self.total_qp_num,
                frame_index_map=[0, 1, 0, 2, 0, 2, 0, 2],
                qp_shift=[0, 0, 0],
                feature_channels=256,
                latent_channels=128,
                hyperprior_channels=self.z_channel,
                downsample_feature=8,
                downsample_latent=16,
                downsample_hyperprior=4,
                y_scale_repeat=None,
                quantize_scale_decoder=False,
                weights_version=None,
                weights_path=None,
                iframe_period=None,
                reset_period=None,
                ltr_start_idx=0,
                ltr_period=None,
                fake_quantized=fake_quantized,
                disable_feature_reset=disable_feature_reset,
                qp_mapping=qp_mapping,
                extra_params=kwargs,
            )

    def load_weights(
        self,
        *,
        job_outputs_dir: Path | str,
        weights_version: str,
        weights_path: Path | str,
        iframe_period: int | None,
        reset_period: int | None,
        ltr_start_idx: int = 0,
        ltr_period: int | None = None,
    ) -> None:
        self._model_params = dataclasses.replace(
            self._model_params,
            weights_path=Path(weights_path).as_posix(),
            weights_version=weights_version,
            iframe_period=iframe_period,
            reset_period=reset_period,
            ltr_start_idx=ltr_start_idx,
            ltr_period=ltr_period,
        )

        abs_weights_path = str(download_job_outputs(weights_path, job_outputs_dir))
        state_dict = get_state_dict(abs_weights_path)
        state_dict = _fix_quantization_negative_scales(state_dict)
        self.load_state_dict(state_dict, strict=True)

        print(f"Loaded {weights_version} weights from {abs_weights_path} for: ")
        for t in type(self).mro():
            print(f" -> {t.__module__}.{t.__name__}")

    def save_pmf_tables(self, output_path: Path | str) -> None:
        with open(Path(output_path) / "gaussian_pmf.json", "w") as f:
            json.dump(dataclasses.asdict(self.gaussian_coder_pmf), f)
        with open(Path(output_path) / "bit_estimator_pmf.json", "w") as f:
            json.dump(dataclasses.asdict(self.bit_estimator_pmf), f)

    def save_auxiliary_data(self, output_path: Path | str) -> None:
        self.save_pmf_tables(output_path)

    @torch.no_grad()
    def _fuse_depthconvblock_alphas(self):
        import torch.nn as nn
        from src.models.dmc_6.layers import DMCConv2d, DepthConvBlock

        def _fold_alpha_into_last_conv(alpha: nn.Parameter, seq: nn.Sequential):
            if alpha is None:
                return
            s = float(alpha)
            if s == 1.0:
                return

            for m in reversed(seq):
                if isinstance(m, DMCConv2d):
                    m.weight.mul_(s)
                    if m.bias is not None:
                        m.bias.mul_(s)
                    break

        for m in self.modules():
            if isinstance(m, DepthConvBlock):
                if hasattr(m, "alpha1") and m.alpha1 is not None:
                    _fold_alpha_into_last_conv(m.alpha1, m.dc)
                    m.register_parameter("alpha1", None)
                if hasattr(m, "alpha2") and m.alpha2 is not None:
                    _fold_alpha_into_last_conv(m.alpha2, m.ffn)
                    m.register_parameter("alpha2", None)

    def optimize_structure(self):
        """Perform in-place structural optimizations on the model (e.g. layer fusion)."""
        self._fuse_depthconvblock_alphas()
        return self

    @staticmethod
    def unpack_with_mask(x, means, mask, chunks: int):
        # Fix for: OptimizeConcatViewCopies Pass failed :
        # Branches must have different inputs loc(fused<{name = "main", type = "Func"}>["main"]):
        # error: OptimizeConcatViewCopies Pass failed : Branches must have different inputs
        assert chunks == 2
        mask1, mask2 = mask.chunk(2, dim=1)
        means1, means2 = means.chunk(2, dim=1)
        x = torch.cat([(x + means1) * mask1, (x + means2) * mask2], dim=1)
        return x

    @property
    def model_params(self) -> ModelParams:
        return self._model_params

    @property
    def gaussian_coder_pmf(self) -> GaussianCoderPmf:
        coder = self.gaussian_encoder
        coder.build_pmf()
        pmf_lengths, pmf_offsets, pmf_table = coder.get_pmf()
        assert pmf_lengths is not None and pmf_offsets is not None and pmf_table is not None
        return GaussianCoderPmf(
            scale_min=coder.scale_min,
            scale_max=coder.scale_max,
            scale_levels=coder.scale_level,
            index_space=coder.scale_input_in_index_space,
            pmf_lengths=pmf_lengths.tolist(),
            pmf_offsets=pmf_offsets.tolist(),
            pmf_table=pmf_table.tolist(),
        )

    @property
    def bit_estimator_pmf(self) -> BitEstimatorPmf:
        coder = self.bit_estimator_z
        coder.build_pmf()
        pmf_lengths, pmf_offsets, pmf_table = coder.get_pmf()
        assert pmf_lengths is not None and pmf_offsets is not None and pmf_table is not None
        return BitEstimatorPmf(
            qp_num=coder.qp_num,
            channels=coder.channel,
            pmf_lengths=pmf_lengths.tolist(),
            pmf_offsets=pmf_offsets.tolist(),
            pmf_table=pmf_table.tolist(),
        )


class FxTraceableMixin:
    def _init_masks(self, size):
        # Note: This approach restricts the model to a single fixed resolution
        device = next(self.parameters()).device  # type: ignore
        dtype = next(self.parameters()).dtype  # type: ignore
        self._mask_0, self._mask_1 = get_mask_dual(size, dtype=dtype, device=device)

    def train(self, mode=True):
        # Force CoreML quantizer to trace model in eval mode
        if mode:
            print("Warning: Training is not supported for full models, keeping in eval mode")
            return
        super().train(mode)  # type: ignore

    def get_mask_dual(self, batch, channel, height, width, dtype, device):
        # Fix for torch.fx tracing error: remove dynamic mask generation
        return [self._mask_0, self._mask_1]

    @staticmethod
    def pack_with_mask(x, mask, chunks: int):
        # Fix for torch.fx tracing error: remove asserts
        x = x * mask
        while chunks > 1:
            chunks //= 2
            x1, x2 = x.chunk(2, dim=1)
            x = x1 + x2
        return x

    @staticmethod
    def separate_prior(params, *, is_video: bool = False):
        # Fix for torch.fx tracing error (apply_upper_lower_bound seems not to be traced correctly)
        if is_video:
            quant_step, scales, means = params.chunk(3, 1)
            quant_step = torch.max(quant_step, 0.5 * torch.ones_like(quant_step))
            return quant_step, scales, means
        else:
            raise NotImplementedError("Separate prior for image not implemented")

    def slice_to_y(self, param, slice_shape):
        # Fix for torch.fx tracing error: replicate_pad is not traced correctly
        left, right = -slice_shape[0], param.shape[-1] + slice_shape[1]
        top, bottom = -slice_shape[2], param.shape[-2] + slice_shape[3]
        return param[:, :, top:bottom, left:right]

    def pad_for_y(self, y, *, z_factor=4):
        # Fix for torch.fx tracing error: remove conditional training flow
        # Fix for torch.fx tracing error: replace replicate_pad with torch.nn.functional.pad
        _, _, H, W = y.size()
        padding = get_padding_size(H, W, z_factor)
        y = torch.nn.functional.pad(y, padding, mode="replicate")
        return y, tuple(-x for x in padding)


class ConfigOnlyFullModel(BaseFullModel):
    def __init__(
        self,
        model_params: ModelParams,
        gaussian_coder_pmf: GaussianCoderPmf,
        bit_estimator_pmf: BitEstimatorPmf,
    ):
        super().__init__(model_version=model_params.model_version, model_params=model_params)
        self._gaussian_coder_pmf = gaussian_coder_pmf
        self._bit_estimator_pmf = bit_estimator_pmf

    @property
    def gaussian_coder_pmf(self) -> GaussianCoderPmf:
        return self._gaussian_coder_pmf

    @property
    def bit_estimator_pmf(self) -> BitEstimatorPmf:
        return self._bit_estimator_pmf


def _fix_quantization_negative_scales(state_dict):
    quantized_module_names = []
    for k in state_dict.keys():
        if k.endswith("feature_scale1"):
            quantized_module_names.append(k[:-15])

    for module_name in quantized_module_names:
        weights = state_dict[f"{module_name}.weight"]
        scale2 = state_dict[f"{module_name}.weight_scale2"]
        if scale2.min() <= 0:
            print(f"Module {module_name} has negative scale2 values")
            weights *= -1
            scale2 *= -1
    return state_dict


def get_mask_dual(size, dtype, device):
    def get_one_channel_dual_mask(height, width, dtype, device):
        micro_mask_0 = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
        mask_0 = micro_mask_0.repeat((height + 1) // 2, (width + 1) // 2)
        mask_0 = mask_0[:height, :width]
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)

        micro_mask_1 = torch.tensor(((0, 1), (1, 0)), dtype=dtype, device=device)
        mask_1 = micro_mask_1.repeat((height + 1) // 2, (width + 1) // 2)
        mask_1 = mask_1[:height, :width]
        mask_1 = torch.unsqueeze(mask_1, 0)
        mask_1 = torch.unsqueeze(mask_1, 0)
        return mask_0, mask_1

    batch = 1
    channel, height, width = size
    assert channel % 2 == 0
    m = torch.ones((batch, channel // 2, height, width), dtype=dtype, device=device)
    m0, m1 = get_one_channel_dual_mask(height, width, dtype, device)
    mask_0 = torch.cat((m * m0, m * m1), dim=1)
    mask_1 = torch.cat((m * m1, m * m0), dim=1)
    return mask_0, mask_1
