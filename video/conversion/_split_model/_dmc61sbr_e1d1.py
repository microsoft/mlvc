# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from collections import namedtuple
from ..types import (
    ModelPartId,
    RuntimeParams,
)
from .._full_model import BaseFullModel
from .._model_wrapper import ModelWrapper
from ._base_split_model import BaseDecoderPart, prepare_model_parts
from ._dmc61sr_e1d1 import Encoder as Dmc61srEncoder
from ._dmc61sr_e1d1 import ReferenceManager as Dmc61srReferenceManager
from ._dmc61sr_e1d1 import SplitModel as Dmc61srSplitModel
from src.models.utils import apply_upper_lower_bound


class Encoder(Dmc61srEncoder):
    pass


class Decoder(BaseDecoderPart):
    InputType = namedtuple(
        "MLVCDecoderInput",
        [
            "z_raw",
            "y_raw_0",
            "y_raw_1",
            "ref_feature",
            "q_index_shifted",
        ],
    )
    OutputType = namedtuple("MLVCDecoderOutput", ["x_hat", "feature"])

    def forward(self, z_raw, y_raw_0, y_raw_1, ref_feature, q_index_shifted):
        dpb = {
            "ref_feature": ref_feature,
        }

        q_feature = self.model.q_feature[q_index_shifted]
        q_decoder = self.model.q_decoder[q_index_shifted]

        # Part 1
        ctx, ctx_t, memory = self.model.context_generation(dpb, q_feature)
        params = self.model.res_prior_param_decoder(z_raw, ctx_t, self.dims.y_slice_shape)
        quant_step, means_0 = params.chunk(2, 1)
        quant_step = apply_upper_lower_bound(quant_step, lower=0.5)

        # Part 2
        y_hat_0 = self.model.unpack_with_mask(y_raw_0, means_0, self.mask_0, chunks=2)
        means_1 = self.model.y_spatial_prior(torch.cat((y_hat_0, params), dim=1))

        # Part 3
        y_hat_1 = self.model.unpack_with_mask(y_raw_1, means_1, self.mask_1, 2)
        y_hat = (y_hat_0 + y_hat_1) * quant_step
        x_hat, feature, memory = self.model.get_recon_and_feature(y_hat, ctx, q_decoder, memory, q_index_shifted)
        x_hat = self.model.unshift_output(x_hat)
        feature = torch.cat((feature, memory), dim=1)

        return Decoder.OutputType(x_hat=x_hat, feature=feature)


class ReferenceManager(Dmc61srReferenceManager):
    pass


class SplitModel(Dmc61srSplitModel):
    def __init__(
        self,
        *,
        full_model: BaseFullModel,
        model_parts: dict[ModelPartId, ModelWrapper] | None,
        runtime_params: RuntimeParams,
        model_height: int,
        model_width: int,
        use_encoder: bool,
        use_decoder: bool,
        split_type: str = "dmc61sbr_e1d1",
        **kwargs,
    ):

        if model_parts is None:
            torch_model_parts = {}
            if use_encoder:
                torch_model_parts.update(
                    {
                        ModelPartId.ENCODER: Encoder(full_model, model_height, model_width),
                    }
                )
            if use_decoder:
                torch_model_parts.update(
                    {
                        ModelPartId.DECODER: Decoder(full_model, model_height, model_width),
                    }
                )
            model_parts = prepare_model_parts(torch_model_parts, runtime_params)

        super().__init__(
            split_type=split_type,
            full_model=full_model,
            model_parts=model_parts,
            runtime_params=runtime_params,
            model_height=model_height,
            model_width=model_width,
            use_encoder=use_encoder,
            use_decoder=use_decoder,
            **kwargs,
        )
