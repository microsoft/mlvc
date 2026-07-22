# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum

from . import _msrtc_rans

__all__ = ["RansVariant"]


class RansVariant(enum.IntEnum):
    RansByte = _msrtc_rans.RansByte
    Rans64 = _msrtc_rans.Rans64
