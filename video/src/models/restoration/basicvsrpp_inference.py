# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import torch

from .basicvsrpp_arch import load_basicvsrpp


class BasicVSRPPInference:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        *,
        chunk_size: int,
        overlap: int,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if chunk_size - 2 * overlap <= 0:
            raise ValueError(f"chunk_size - 2 * overlap must be > 0, got chunk_size={chunk_size}, overlap={overlap}")

        self.model = model.to(device)
        self.device = device
        self.chunk_size = chunk_size
        self.overlap = overlap
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        weights_path: str,
        device: torch.device,
        *,
        mid_channels: int,
        num_blocks: int,
        chunk_size: int,
        overlap: int,
    ):
        model = load_basicvsrpp(
            weights_path,
            map_location="cpu",
            mid_channels=mid_channels,
            num_blocks=num_blocks,
            is_low_res_input=False,
        )
        return cls(model, device, chunk_size=chunk_size, overlap=overlap)

    @torch.inference_mode()
    def restore_chunk(self, rgb_frames: Sequence[np.ndarray]) -> tuple[list[np.ndarray], float]:
        if not rgb_frames:
            return [], 0.0

        frames = torch.stack([torch.from_numpy(frame) for frame in rgb_frames], dim=0)
        frames = frames.unsqueeze(0).to(self.device, non_blocking=True)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        restored = self.model(frames)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        restored = restored.clamp_(0.0, 1.0).squeeze(0).cpu().numpy().astype(np.float32)
        return [restored[index] for index in range(restored.shape[0])], elapsed

    def restore_sequence(self, rgb_frames: Sequence[np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
        n_frames = len(rgb_frames)
        if n_frames == 0:
            return [], []
        if n_frames <= self.chunk_size:
            restored, elapsed = self.restore_chunk(rgb_frames)
            return restored, [elapsed]

        usable = self.chunk_size - 2 * self.overlap
        restored_frames: list[np.ndarray | None] = [None] * n_frames
        chunk_times: list[float] = []

        output_start = 0
        while output_start < n_frames:
            chunk_start = max(0, output_start - self.overlap)
            chunk_end = min(n_frames, output_start + usable + self.overlap)
            chunk_output, elapsed = self.restore_chunk(rgb_frames[chunk_start:chunk_end])
            chunk_times.append(elapsed)

            keep_start = output_start - chunk_start
            halo_right = chunk_end - min(output_start + usable, n_frames)
            keep_end = len(chunk_output) - halo_right if halo_right > 0 else len(chunk_output)
            for chunk_index in range(keep_start, keep_end):
                frame_index = output_start + chunk_index - keep_start
                if frame_index < n_frames:
                    restored_frames[frame_index] = chunk_output[chunk_index]

            output_start += usable

        missing = [index for index, frame in enumerate(restored_frames) if frame is None]
        if missing:
            raise RuntimeError(f"BasicVSR++ chunk assembly did not produce frames: {missing}")

        return [frame for frame in restored_frames if frame is not None], chunk_times
