# Copyright (c) Microsoft Corporation.
# Repository-specific modifications licensed under the MIT license.
# Adapted from https://github.com/sunwei925/SimpleVQA (Apache-2.0); see LICENSE.
# Modified for streaming feature extraction and repository integration.

import math
import os
from typing import Any, Mapping, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .model import resnet50
from .slowfast import slowfast_r50


def simple_vqa_sample_count(total_frames: int, frame_rate: float) -> int:
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(f"SimpleVQA requires a positive finite frame rate, got {frame_rate}")
    rounded_frame_rate = int(round(frame_rate))
    if rounded_frame_rate <= 0:
        raise ValueError(f"SimpleVQA requires a frame rate that rounds to at least one, got {frame_rate}")
    seconds = total_frames // rounded_frame_rate
    if seconds <= 0:
        raise ValueError(
            f"SimpleVQA requires at least one second of video, got {total_frames} frames at {frame_rate} fps"
        )
    return seconds


def pack_slowfast_pathways(frames, device):
    slow_indices = torch.linspace(0, frames.shape[2] - 1, frames.shape[2] // 4, device=frames.device).long()
    return [torch.index_select(frames, 2, slow_indices).to(device), frames.to(device)]


class _SlowFastFeatureExtractor:
    def __init__(self, checkpoint_path: str, device):
        model = slowfast_r50()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint.get("model_state", checkpoint))
        blocks = list(model.get_submodule("blocks").children())
        self.feature_extraction = torch.nn.Sequential(*blocks[:5]).to(device).eval()
        self.slow_avg_pool = model.get_submodule("blocks.5.pool.0").to(device).eval()
        self.fast_avg_pool = model.get_submodule("blocks.5.pool.1").to(device).eval()
        self.output_pool = model.get_submodule("blocks.6.output_pool").to(device).eval()

    def __call__(self, pathways):
        features = self.feature_extraction(pathways)
        return (
            self.output_pool(self.slow_avg_pool(features[0])),
            self.output_pool(self.fast_avg_pool(features[1])),
        )


class SimpleVQA:
    def __init__(self, pretrained_path: Optional[str], device, config: Mapping[str, Any]):
        self.device = device
        checkpoint_path = self._resolve_path(config.get("checkpoint", "SimpleVQA/UGC_BVQA_model.pth"), pretrained_path)
        slowfast_path = self._resolve_path(
            config.get("slowfast_checkpoint", "SimpleVQA/SLOWFAST_8x8_R50.pyth"), pretrained_path
        )
        for path in (checkpoint_path, slowfast_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"SimpleVQA checkpoint not found: {path}")

        self.model = resnet50().to(device)
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
        self.model.eval()
        self.motion_model = _SlowFastFeatureExtractor(slowfast_path, device)
        self.spatial_transform = transforms.Compose(
            [
                transforms.Resize(520),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.motion_transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.45] * 3, std=[0.225] * 3),
            ]
        )

    @staticmethod
    def _resolve_path(path, pretrained_path: Optional[str]) -> str:
        path = os.fspath(path)
        if os.path.isabs(path) or os.path.exists(path):
            return os.path.abspath(path)
        if pretrained_path is None:
            raise ValueError(f"SimpleVQA requires pretrained_path or an existing absolute asset path: {path}")
        return os.path.abspath(os.path.join(pretrained_path, path))

    def _extract_motion_feature(self, frames):
        tensor = torch.stack(frames).unsqueeze(0).permute(0, 2, 1, 3, 4)
        with torch.inference_mode():
            slow, fast = self.motion_model(pack_slowfast_pathways(tensor, self.device))
        return torch.cat([slow.flatten(), fast.flatten()])

    def start_video(self, *, total_frames: int, frame_rate: float):
        return SimpleVQAStream(self, total_frames=total_frames, frame_rate=frame_rate)


class SimpleVQAStream:
    def __init__(self, metric: SimpleVQA, *, total_frames: int, frame_rate: float):
        self.metric = metric
        self.total_frames = total_frames
        self.spatial_count = simple_vqa_sample_count(total_frames, frame_rate)
        self.rounded_frame_rate = int(round(frame_rate))
        self.frame_index = 0
        self.spatial_frames = []
        self.completed_motion = {}
        self.active_motion = {}
        self.frame_shape = None
        self.closed = False

    def add_frame(self, rgb):
        if self.closed or self.frame_index >= self.total_frames:
            raise ValueError("SimpleVQA stream has already received its complete video")
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("SimpleVQA frames must be HWC uint8 RGB arrays")
        if self.frame_shape is not None and rgb.shape != self.frame_shape:
            raise ValueError("SimpleVQA frame dimensions changed within a video")
        self.frame_shape = rgb.shape
        if self.frame_index % self.rounded_frame_rate == 0:
            second = self.frame_index // self.rounded_frame_rate
            if second < self.spatial_count:
                self.spatial_frames.append(self.metric.spatial_transform(Image.fromarray(rgb)))
                self.active_motion[second] = []
        motion_frame = None
        for second, clip in tuple(self.active_motion.items()):
            if motion_frame is None:
                motion_frame = self.metric.motion_transform(Image.fromarray(rgb))
            clip.append(motion_frame)
            if len(clip) == 32:
                self.completed_motion[second] = self.metric._extract_motion_feature(clip)
                del self.active_motion[second]
        self.frame_index += 1

    def finish(self) -> float:
        if self.closed:
            raise ValueError("SimpleVQA stream has already been finalized")
        self.closed = True
        try:
            if self.frame_index != self.total_frames:
                raise ValueError(f"SimpleVQA received only {self.frame_index} frames; expected {self.total_frames}")
            for second, clip in self.active_motion.items():
                clip += [clip[-1]] * (32 - len(clip))
                self.completed_motion[second] = self.metric._extract_motion_feature(clip)
            with torch.inference_mode():
                spatial = torch.stack(self.spatial_frames).unsqueeze(0).to(self.metric.device)
                motion = torch.stack([self.completed_motion[index] for index in range(self.spatial_count)]).unsqueeze(0)
                score = float(self.metric.model(spatial, motion).item())
            if not math.isfinite(score):
                raise ValueError(f"SimpleVQA returned a nonfinite score: {score}")
            return score
        finally:
            self.spatial_frames.clear()
            self.active_motion.clear()
            self.completed_motion.clear()
