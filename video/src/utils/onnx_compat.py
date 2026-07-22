# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Sequence

import torch


def replicate_pad(x: torch.Tensor, *, pad: Sequence[int]):
    def untrace(v):
        if isinstance(v, torch.Tensor):
            v = v.item()
        return v

    for i in range(0, len(pad) // 2):
        left = untrace(pad[2 * i])
        right = untrace(pad[2 * i + 1])

        if left == 0 and right == 0:
            continue

        tail = (slice(0, None),) * i

        if left < 0 or right < 0:
            if left < 0:
                start = -left
                left = 0
            else:
                start = 0

            if right < 0:
                end = right
                right = 0
            else:
                end = None

            x = x[(..., slice(start, end)) + tail]

        if left > 0 or right > 0:
            parts = list()

            if left > 0:
                p = x[(..., slice(0, 1)) + tail]
                for _ in range(int(left)):
                    parts.append(p)

            parts.append(x)

            if right > 0:
                p = x[(..., slice(-1, None)) + tail]
                for _ in range(int(right)):
                    parts.append(p)

            x = torch.cat(parts, dim=x.ndim - i - 1)

    return x
