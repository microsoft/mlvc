# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import List
import logging
from dataclasses import dataclass

import torch
from torch import nn

from ..models import pretrained_networks as pn


@dataclass
class PerceptualModelConfig:
    base_model: str
    normalize_input: bool
    pretrained: bool = True
    normalize_feature_maps: bool = True
    use_lpips_aggregation: bool = False
    use_lpips_weights: bool = False
    pretrained_lpips_path: str = "lpips_vgg16.pth"


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # ImageNet normalization for for [-1, 1] inputs
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None], persistent=False)
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None], persistent=False)

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1):
        super().__init__()

        layers = [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def normalize_tensor(in_feat, eps=1e-10):
    sqrt_eps = 1.0e-6
    feat_sum = torch.sum(in_feat**2, dim=1, keepdim=True).clamp_min(sqrt_eps)
    norm_factor = torch.sqrt(feat_sum)
    return in_feat / (norm_factor + eps)


class PerceptualModel(nn.Module):
    def __init__(self, pretrained_models_path: str, config: dict):
        super().__init__()

        self.config = PerceptualModelConfig(**config)
        self.pretrained_models_path = Path(pretrained_models_path)

        self.base_model = self._init_base_model()
        self.no_maps = len(self.resolutions)

        self.scaling_layer = ScalingLayer()

        if self.config.use_lpips_aggregation:
            if self.config.use_lpips_weights:
                self._init_lpips_model()

        self.eval()

    def _init_base_model(self):
        base_model = self.config.base_model
        pretrained = self.config.pretrained

        if base_model == "vgg16":
            model = pn.vgg16(requires_grad=False, pretrained=pretrained)
        else:
            raise NotImplementedError(f"Base model {base_model} not implemented")

        self.resolutions = model.feature_resolutions
        return model

    def _convert_lins(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        _ = prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("lin", "").replace("model.1.weight", "model.0.weight")
            new_state_dict[new_key] = v

        state_dict.clear()
        state_dict.update(new_state_dict)

    def _init_lpips_model(self):
        full_path = self.pretrained_models_path / self.config.pretrained_lpips_path

        state_dict = torch.load(full_path, map_location="cpu", weights_only=True)

        channels = self.base_model.feature_channels
        self.lins = nn.ModuleList([NetLinLayer(ch) for ch in channels])

        self.lins._register_load_state_dict_pre_hook(self._convert_lins)
        self.lins.load_state_dict(state_dict, strict=True)
        logging.info("Initialised LPIPS model")

    def _normalize_input(self, x):
        return 2 * x - 1

    @staticmethod
    def _spatial_average(x, keepdim=True):
        return x.mean([2, 3], keepdim=keepdim)

    @staticmethod
    def upsample(in_tens, out_HW=(64, 64)):  # assumes scale factor is same for H and W
        return nn.Upsample(size=out_HW, mode="bilinear", align_corners=False)(in_tens)

    def extract_base_model_features(self, x) -> List[torch.Tensor]:
        if self.config.normalize_input is True:
            inp = self._normalize_input(x)
        else:
            inp = x

        inp = self.scaling_layer(inp)
        out = self.base_model(inp)
        return out

    def weight_features(
        self,
        out1,
        out2,
        *,
        normalize_feature_maps: bool = True,
        use_lpips_aggregation: bool = False,
        use_lpips_weights: bool = False,
        spatial: bool = False,
    ):

        total = 0
        for i in range(self.no_maps):
            if normalize_feature_maps:
                norm1, norm2 = normalize_tensor(out1[i]), normalize_tensor(out2[i])
            else:
                norm1, norm2 = out1[i], out2[i]

            if use_lpips_aggregation:
                distortion = (norm1 - norm2) ** 2
                if use_lpips_weights:
                    distortion = self.lins[i](distortion)
                else:
                    distortion = distortion.sum(dim=1, keepdim=True)
            else:
                raise NotImplementedError("Only LPIPS aggregation is implemented")

            # Distortion here is [B, 1, H, W]
            if spatial:
                distortion = self.upsample(distortion, out_HW=out1[0].shape[2:])
            else:
                distortion = self._spatial_average(distortion, keepdim=True)
                distortion = distortion.flatten()

            total += distortion

        return total

    def forward(self, x1, x2, *, spatial: bool = False):
        out1, out2 = self.extract_base_model_features(x1), self.extract_base_model_features(x2)

        normalize_feature_maps = self.config.normalize_feature_maps
        use_lpips_aggregation = self.config.use_lpips_aggregation
        use_lpips_weights = self.config.use_lpips_weights

        return self.weight_features(
            out1,
            out2,
            normalize_feature_maps=normalize_feature_maps,
            use_lpips_aggregation=use_lpips_aggregation,
            use_lpips_weights=use_lpips_weights,
            spatial=spatial,
        )


if __name__ == "__main__":
    torch.manual_seed(42)
    pretrained_path = "."
    image_size = (256, 256)
    x1 = torch.rand(1, 3, *image_size, requires_grad=True)
    x2 = torch.rand(1, 3, *image_size, requires_grad=True)

    config = {
        "base_model": "vgg16",
        "normalize_input": True,
        "use_lpips_aggregation": True,
        "use_lpips_weights": True,
    }

    loss_lpips_general = PerceptualModel(pretrained_models_path=pretrained_path, config=config)
    out_lpips_general = loss_lpips_general(x1, x2)
    out_lpips_general.backward(retain_graph=True)
    assert x1.grad is not None and x2.grad is not None
    grad_lpips_general_x1 = x1.grad.clone()
    grad_lpips_general_x2 = x2.grad.clone()
    x1.grad.zero_()
    x2.grad.zero_()

    import lpips  # type: ignore[import-not-found]

    loss_lpips = lpips.LPIPS(net="vgg", lpips=True)
    out_lpips = loss_lpips(x1, x2, normalize=True)
    out_lpips.backward(retain_graph=True)
    assert x1.grad is not None and x2.grad is not None
    grad_lpips_x1 = x1.grad.clone()
    grad_lpips_x2 = x2.grad.clone()

    assert out_lpips_general == out_lpips
    assert torch.allclose(grad_lpips_general_x1, grad_lpips_x1, atol=1e-6)
    assert torch.allclose(grad_lpips_general_x2, grad_lpips_x2, atol=1e-6)
    print("All outputs and gradients as expected")
