# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import dataclasses
from ._base_model import BaseFullModel
from src.models.dmc_6.dmc_61s import DMC


class TraceableMLVC(BaseFullModel, DMC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_params = dataclasses.replace(
            self._model_params,
            frame_index_map=list(self.frame_index_map),
            qp_shift=list(self.qp_shift),
            feature_channels=self.feature_channels,
            latent_channels=self.y_channels,
            downsample_feature=8,
            downsample_latent=16,
            downsample_hyperprior=8,
            y_scale_repeat=2,
        )

    def apply_feature_adaptor(self, dpb):
        if self.model_params.disable_feature_reset:
            feature_feature = self.feature_adaptor_p(dpb["ref_feature"])
            return feature_feature
        else:
            frame = torch.nn.functional.pixel_unshuffle(dpb["ref_frame"], 8)
            frame_feature = self.feature_adaptor_i(frame)

            feature_feature = self.feature_adaptor_p(dpb["ref_feature"])
            ref_exists = dpb["ref_exists"]

            return ref_exists * feature_feature + (1 - ref_exists) * frame_feature
