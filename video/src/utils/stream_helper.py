# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from collections.abc import Sequence
from enum import Enum
from typing import Any, Literal, Tuple, Optional, Union, overload

import numpy as np
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from src.transforms.functional import yuv_420_to_444

__all__ = [
    "get_padding_size",
    "get_downsampled_shape",
    "get_state_dict",
    "prepare_frame",
    "calc_psnr",
    "is_bytes_like",
    "open_encoder_streams",
    "flush_encoder_streams",
    "open_decoder_streams",
    "check_decoder_eof",
    "PaddingMode",
    "PaddingAlignment",
]


class PaddingMode(str, Enum):
    REPLICATE = "replicate"
    REFLECT = "reflect"
    CONSTANT_BLACK = "constant_black"
    CONSTANT_GRAY = "constant_gray"


class PaddingAlignment(str, Enum):
    BOTTOM_RIGHT = "bottom_right"
    TOP_LEFT = "top_left"
    BOTH = "both"


def get_padding_size(
    height: int,
    width: int,
    p: int = 64,
    alignment: Union[str, PaddingAlignment] = PaddingAlignment.BOTTOM_RIGHT,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    if isinstance(alignment, str):
        alignment = PaddingAlignment(alignment)

    if target_height is not None or target_width is not None:
        if target_height is None or target_width is None:
            raise ValueError("Both target_height and target_width must be provided together.")
        if target_height < height or target_width < width:
            raise ValueError(
                f"target_resolution ({target_width}x{target_height}) is smaller than input ({width}x{height})"
            )
        new_h = ((target_height + p - 1) // p) * p
        new_w = ((target_width + p - 1) // p) * p
    else:
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p

    total_pad_h = new_h - height
    total_pad_w = new_w - width

    if alignment == PaddingAlignment.BOTTOM_RIGHT:
        padding_left = 0
        padding_top = 0
    elif alignment == PaddingAlignment.TOP_LEFT:
        padding_left = total_pad_w
        padding_top = total_pad_h
    elif alignment == PaddingAlignment.BOTH:
        padding_left = total_pad_w // 2
        padding_top = total_pad_h // 2
    else:
        raise ValueError(f"Unknown alignment: {alignment}")

    padding_right = total_pad_w - padding_left
    padding_bottom = total_pad_h - padding_top

    return padding_left, padding_right, padding_top, padding_bottom


def get_downsampled_shape(height, width, p):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)


@overload
def get_state_dict(ckpt_path, *, return_epoch: Literal[False] = ...) -> dict: ...


@overload
def get_state_dict(ckpt_path, *, return_epoch: Literal[True]) -> tuple[Any, dict]: ...


def get_state_dict(ckpt_path, *, return_epoch=False) -> dict | tuple[Any, dict]:
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=True)

    state_dict = ckpt
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "net" in state_dict:
        state_dict = state_dict["net"]
    consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")

    if not return_epoch:
        return state_dict
    else:
        return ckpt.get("epoch"), state_dict


def prepare_frame(
    image: Tuple[np.ndarray, np.ndarray],
    is_yuv420: bool = True,
    precision: str = "fp16",
    device=None,
    padding_size: int = 16,
    padding_mode: PaddingMode = PaddingMode.REPLICATE,
    padding_alignment: PaddingAlignment = PaddingAlignment.BOTTOM_RIGHT,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor, Tuple[int, int, int, int]]:
    """Prepare a YUV444 frame for the model, and a YUV420 frame for evaluation"""

    if isinstance(padding_mode, str):
        padding_mode = PaddingMode(padding_mode)
    if isinstance(padding_alignment, str):
        padding_alignment = PaddingAlignment(padding_alignment)

    # noinspection PyShadowingNames
    def _to_tensor(x, device=None):
        ten = torch.from_numpy(x).float()
        if device is not None:
            ten = ten.to(device, non_blocking=True)

        ten = ten.unsqueeze(0)
        return ten

    if is_yuv420:
        y, uv = image
        u, v = _to_tensor(uv, device).chunk(2, dim=1)
        yuv_tensors = _to_tensor(y, device), u, v
        x = yuv_420_to_444(yuv_tensors, mode="nearest")
    else:
        yuv_tensors = _to_tensor(image, device).unsqueeze(0)
        x = yuv_tensors

    height = x.shape[2]
    width = x.shape[3]

    if target_height is not None and target_width is not None:
        input_is_portrait = height > width
        target_is_portrait = target_height > target_width
        if input_is_portrait != target_is_portrait:
            target_height, target_width = target_width, target_height

    padding = get_padding_size(
        height,
        width,
        padding_size,
        alignment=padding_alignment,
        target_height=target_height,
        target_width=target_width,
    )

    if any(v != 0 for v in padding):
        if padding_mode == PaddingMode.REPLICATE:
            x = torch.nn.functional.pad(x, padding, mode="replicate")
        elif padding_mode == PaddingMode.REFLECT:
            x = torch.nn.functional.pad(x, padding, mode="reflect")
        elif padding_mode == PaddingMode.CONSTANT_BLACK:
            if is_yuv420:
                y_ch = torch.nn.functional.pad(x[:, 0:1], padding, mode="constant", value=0.0)
                u_ch = torch.nn.functional.pad(x[:, 1:2], padding, mode="constant", value=0.5)
                v_ch = torch.nn.functional.pad(x[:, 2:3], padding, mode="constant", value=0.5)
                x = torch.cat([y_ch, u_ch, v_ch], dim=1)
            else:
                x = torch.nn.functional.pad(x, padding, mode="constant", value=0.0)
        elif padding_mode == PaddingMode.CONSTANT_GRAY:
            x = torch.nn.functional.pad(x, padding, mode="constant", value=0.5)
        else:
            raise ValueError(f"Unknown padding mode: {padding_mode}")

    if precision == "fp16":
        x = x.to(torch.float16)

    return x, yuv_tensors, padding


def calc_psnr(x1, x2):
    mse = torch.square(x1 - x2)
    mse = mse.sum().cpu() / math.prod(mse.shape[2:])

    if not torch.isfinite(mse):
        return -999.9
    if mse < 1e-10:
        return 999.9
    return -10 * torch.log10(mse).item()


def is_bytes_like(o):
    try:
        memoryview(o)
        return True
    except TypeError:
        return False


def open_encoder_streams(batch_size):
    from msrtc.rans import RansEncoderStream

    if batch_size == 1:
        return RansEncoderStream()
    else:
        return tuple(RansEncoderStream() for _ in range(batch_size))


def flush_encoder_streams(streams):
    if not isinstance(streams, Sequence):
        return bytes(streams.flush())
    else:
        return tuple(bytes(s.flush()) for s in streams)


def open_decoder_streams(bit_streams):
    if is_bytes_like(bit_streams):
        bit_streams = (bit_streams,)

    from msrtc.rans import RansDecoderStream

    return tuple(RansDecoderStream(s) for s in bit_streams)


def check_decoder_eof(bit_streams):
    for s in bit_streams:
        s.decodeEOF()
