# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import dataclasses
from ._base_model import BaseFullModel
from src.utils.conv_fusion import fuse_conv_chain, fuse_subpel_chain
from src.models.dmc_6.layers import SubpelConv2x, SubpelConvNx
from src.models.dmc_6.dmc_61sb import DMC


class TraceableMLVC(BaseFullModel, DMC):
    def __init__(self, *args, pixel_range: float = 1.0, round_output: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_params = dataclasses.replace(
            self._model_params,
            pixel_range=pixel_range,
            frame_index_map=list(self.frame_index_map),
            qp_shift=list(self.qp_shift),
            feature_channels=2 * self.feature_channels,
            latent_channels=self.y_channels,
            downsample_feature=8,
            downsample_latent=16,
            downsample_hyperprior=2**self.hyperprior_num_blocks,
            y_scale_repeat=self.y_scale_repeat,
        )
        self._round_output = round_output
        self._model_params.extra_params["round_output"] = round_output

    def apply_feature_adaptor(self, dpb):
        if self.model_params.disable_feature_reset:
            feature_ref_feature, feature_ref_memory = dpb["ref_feature"].chunk(2, dim=1)
            feature_ref_feature = self.feature_adaptor_p(feature_ref_feature)
            return feature_ref_feature, feature_ref_memory
        else:
            frame = self.shift_input(dpb["ref_frame"])
            ref_frame = torch.nn.functional.pixel_unshuffle(frame, self.pixel_shuffle_factor)
            frame_feature = self.feature_adaptor_i(ref_frame)
            frame_ref_feature, frame_ref_memory = frame_feature.chunk(2, dim=1)
            if self.chain_feature_adaptors:
                frame_ref_feature = self.feature_adaptor_p(frame_ref_feature)

            feature_ref_feature, feature_ref_memory = dpb["ref_feature"].chunk(2, dim=1)
            feature_ref_feature = self.feature_adaptor_p(feature_ref_feature)

            ref_exists = dpb["ref_exists"]
            ref_feature = ref_exists * feature_ref_feature + (1 - ref_exists) * frame_ref_feature
            ref_memory = ref_exists * feature_ref_memory + (1 - ref_exists) * frame_ref_memory
            return ref_feature, ref_memory

    def unshift_output(self, x):
        # Override clamping range to match pixel range and do optional rounding
        res = x if not self.input_offset else x - self.input_offset
        res = res.clamp_(0.0, self._model_params.pixel_range)
        if self._round_output:
            res = torch.round(res)
        return res

    def clamp_x_hat(self, x_hat):
        # Clamping is done in unshift_output
        return x_hat

    @torch.no_grad()
    def _fuse_offset(self):
        if not self.input_offset:
            return

        def _fold_input_shift_into_conv(conv: nn.Conv2d, k: float):
            assert conv.groups == 1, "Grouped conv not supported by bias-fusion"
            delta = conv.weight.view(conv.out_channels, -1).sum(dim=1) * k
            if conv.bias is None:
                conv.bias = nn.Parameter(delta.clone())
            else:
                conv.bias.data.add_(delta)

        def _fold_output_shift_into_conv(conv: nn.Conv2d, k: float):
            assert conv.groups == 1, "Grouped conv not supported by bias-fusion"
            if conv.bias is None:
                conv.bias = nn.Parameter(
                    torch.full((conv.out_channels,), k, dtype=conv.weight.dtype, device=conv.weight.device)
                )
            else:
                conv.bias.data.add_(k)

        c = float(self.input_offset)

        # encoder and feature_adaptor_i must compensate for the input shift
        _fold_input_shift_into_conv(self.encoder.conv1, c)
        if self.feature_adaptor_i.conv.adaptor is not None:
            _fold_input_shift_into_conv(self.feature_adaptor_i.conv.adaptor, c)

        # recon_generation_net must compensate for the unshift
        _fold_output_shift_into_conv(self.recon_generation_net.head, -c)

        # record that the offset has been fused
        self._model_params.extra_params["input_offset"] = None
        self._model_params.extra_params["input_offset_fused"] = True

        self.input_offset = None

    @torch.no_grad()
    def _fuse_pixel_range_scaling(self):
        if self._model_params.pixel_range == 1.0:
            return

        # Input scaling
        k1 = 1.0 / self._model_params.pixel_range
        self.encoder.conv1.weight.mul_(k1)
        if self.feature_adaptor_i.conv.adaptor is not None:
            self.feature_adaptor_i.conv.adaptor.weight.mul_(k1)

        # Output scaling
        k2 = self._model_params.pixel_range
        self.recon_generation_net.head.weight.mul_(k2)
        if self.recon_generation_net.head.bias is not None:
            self.recon_generation_net.head.bias.mul_(k2)

    @torch.no_grad()
    def _fuse_hyperencoder(self):
        if self.hyperprior_variant != "mini" or not isinstance(self.hyper_encoder.conv, nn.Sequential):
            return
        convs = list(self.hyper_encoder.conv)
        if len(convs) < 2 or not all(isinstance(c, nn.Conv2d) for c in convs):
            return

        print(f"Fusing {len(convs)} hyperencoder conv layers...")
        in_ch = convs[0].in_channels
        out_ch = convs[-1].out_channels
        fused_weight, fused_bias, stride = fuse_conv_chain(convs)
        new_model = nn.Conv2d(in_ch, out_ch, kernel_size=stride, stride=stride)  # type: ignore[arg-type]
        new_model.weight.data = fused_weight
        assert new_model.bias is not None
        new_model.bias.data = fused_bias
        self.hyper_encoder.conv = new_model  # type: ignore[assignment]

    @torch.no_grad()
    def _fuse_hyperdecoder(self):
        if self.hyperprior_variant != "mini" or not isinstance(self.hyper_decoder.conv, nn.Sequential):
            return
        subpels = list(self.hyper_decoder.conv)
        if len(subpels) < 2 or not all(isinstance(s, SubpelConv2x) for s in subpels):
            return

        print(f"Fusing {len(subpels)} hyperdecoder subpel conv layers...")
        in_ch = subpels[0].conv[0].in_channels  # type: ignore[index]
        out_ch = subpels[-1].conv[0].out_channels // 4  # type: ignore[index]
        fused_weight, fused_bias, factor = fuse_subpel_chain(subpels)
        new_model = SubpelConvNx(in_ch, out_ch, factor, 1)
        new_model.conv.weight.data = fused_weight
        if new_model.conv.bias is None:
            new_model.conv.bias = nn.Parameter(fused_bias.clone())
        else:
            new_model.conv.bias.data = fused_bias
        self.hyper_decoder.conv = new_model  # type: ignore[assignment]

    def optimize_structure(self):
        super().optimize_structure()
        self._fuse_offset()
        self._fuse_hyperencoder()
        self._fuse_hyperdecoder()
        self._fuse_pixel_range_scaling()
        return self
