# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import dataclasses
from src.utils import rate_controller as _rc
from .types import FrameType, RateControlInfo


def _translate_frame_type(frame_type: FrameType) -> _rc.FrameType:
    return _rc.FrameType(frame_type.value)


def _translate_rate_control_info(info: _rc.RateControlInfo) -> RateControlInfo:
    return RateControlInfo(**dataclasses.asdict(info))


class RateController:
    def __init__(
        self,
        image_width: int,
        image_height: int,
        bitrate: float,
        fps: float,
        **kwargs,
    ):
        if "frame_weights" in kwargs:
            kwargs["frame_weights"] = {_translate_frame_type(k): v for k, v in kwargs["frame_weights"].items()}
        self._inner = _rc.RateController(image_width, image_height, bitrate, fps, **kwargs)

    def configure(self, bitrate: float | None = None, fps: float | None = None) -> None:
        self._inner.configure(bitrate=bitrate, fps=fps)

    def solve_q_index(self, presentation_time: float, frame_type: FrameType, **kwargs) -> int | None:
        return self._inner.solve_q_index(presentation_time, _translate_frame_type(frame_type), **kwargs)

    def update(self, header_bits: int, payload_bits: int) -> RateControlInfo:
        return _translate_rate_control_info(self._inner.update(header_bits, payload_bits))
