# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import warnings
import logging

import torch
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights  # type: ignore[reportMissingImports]

from src.transforms.functional import ycbcr2rgb


def normalize(tensor, mean, std):
    # Normalize per channel
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


class FaRLModel(nn.Module):
    def __init__(self, pretrained_path: str, device):
        super().__init__()
        from facer.face_detection import RetinaFaceDetector  # type: ignore[reportMissingImports]
        from facer.face_parsing import FaRLFaceParser  # type: ignore[reportMissingImports]

        detector_path = os.path.join(pretrained_path, "checkpoints", "mobilenet0.25_Final.pth")
        parser_path = os.path.join(pretrained_path, "checkpoints", "face_parsing.farl.lapa.main_ema_136500_jit191.pt")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are using `torch.load` with `weights_only=False`.*",
                category=FutureWarning,
            )
            self.face_detector = RetinaFaceDetector(conf_name="mobilenet", model_path=detector_path).to(device)
            self.face_parser = FaRLFaceParser(conf_name="lapa/448", model_path=parser_path, device=device)

        logging.info("FaRL face parser loaded.")

    def forward(self, x):
        x = (x * 255).to(torch.uint8)

        with torch.inference_mode():
            faces = self.face_detector(x)

        batch_size = x.shape[0]
        if len(faces["image_ids"]) == 0:
            empty_mask = torch.zeros(batch_size, 1, x.shape[2], x.shape[3], device=x.device)
            return empty_mask
        else:
            with torch.inference_mode():
                faces = self.face_parser(x, faces)

        seg_logits = faces["seg"]["logits"]
        seg_probs = 1 - seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w

        image_ids = set([f.item() for f in faces["image_ids"]])
        masks = []
        for iid in range(batch_size):
            if iid in image_ids:
                # Aggregate all faces for frame
                ind = torch.where(faces["image_ids"] == iid)
                # Sum faces over background class
                mask = seg_probs[ind][:, 0].sum(dim=0).clamp(0.0, 1.0).squeeze(0)
            else:
                mask = torch.zeros(x.shape[2:], device=x.device)
            masks.append(mask)
        return torch.stack(masks, dim=0).unsqueeze(1)


class DeepLabV3Model(nn.Module):
    def __init__(self, pretrained_path: str):
        super().__init__()
        self.model: nn.Module = torch.hub.load(  # type: ignore[reportAttributeAccessIssue]
            repo_or_dir=pretrained_path,
            source="local",
            model="deeplabv3_resnet50",
            weights=DeepLabV3_ResNet50_Weights.DEFAULT,
        )
        logging.info("DeepLabV3 segmentation model loaded.")

    def forward(self, x):
        # ImageNet [0, 1] normalisation
        x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        with torch.inference_mode():
            output = self.model(x)["out"]

        human_category = 15
        probas = torch.nn.functional.softmax(output, dim=1)
        mask = probas[:, human_category, :, :].unsqueeze(1)
        return mask


class SegmentationModel(nn.Module):
    def __init__(self, pretrained_path: str, config: dict, device):
        super().__init__()

        base_model = config.get("base_model")
        if base_model is None:
            raise ValueError("Segmentation model must be specified in the configuration.")

        if base_model == "deeplabv3":
            model = DeepLabV3Model(pretrained_path)
            model.to(device)
        elif base_model == "farl":
            model = FaRLModel(pretrained_path, device)
        else:
            raise ValueError(f"Unsupported base model: {base_model}")

        model.eval()
        self.model = model

    @torch.no_grad()
    def forward(self, x, is_yuv420, binarise=True):
        if is_yuv420:
            x = ycbcr2rgb(x)
        # Returns a [0, 1] mask of shape [B, 1, H, W]
        mask = self.model(x)

        if binarise:
            mask = (mask > 0.5).float()
        return mask
