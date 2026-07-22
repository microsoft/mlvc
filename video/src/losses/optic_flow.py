# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OpticFlowLoss"]


class OpticFlowLoss(nn.Module):
    def __init__(self, optic_flow: nn.Module, weight=0.01):
        super().__init__()
        self.optic_flow = optic_flow
        self.weight = weight

    def forward(self, rd):
        info = rd["sup_info"]
        flow4, flow8, flow16 = self._calc_flow(info["cur_frame"], info["ref_frame"])

        r = info["r"]

        loss = (
            self._flow_sup(r, info["enc_kernel2"], flow4)
            + self._flow_sup(r, info["enc_kernel3"], flow8)
            + self._flow_sup(r, info["dec_kernel2"], flow4)
            + self._flow_sup(r, info["dec_kernel3"], flow8)
            + self._flow_sup(r, info["hyper_enc_kernel"], flow16)
            + self._flow_sup(r, info["hyper_dec_kernel"], flow16)
        )

        return loss * self.weight

    @torch.no_grad()
    def _calc_flow(self, cur_frame, ref_frame):
        flow = self.optic_flow(cur_frame, ref_frame)
        flow2 = F.avg_pool2d(flow, kernel_size=2) / 2
        flow4 = F.avg_pool2d(flow2, kernel_size=2) / 2
        flow8 = F.avg_pool2d(flow4, kernel_size=2) / 2
        flow16 = F.avg_pool2d(flow8, kernel_size=2) / 2

        return flow4, flow8, flow16

    @staticmethod
    def _flow_sup(r, kernel, flow):
        d = 2 * r + 1
        flow_y, flow_x = flow[:, 0:1], flow[:, 1:2]

        grid = torch.linspace(-r, r, d, device=flow.device, dtype=flow.dtype).reshape(1, -1, 1, 1)

        distance_y = (flow_y.repeat(1, d, 1, 1) - grid) ** 2  # B,  d, h, w
        distance_y = distance_y.unsqueeze(1).repeat(1, d, 1, 1, 1).flatten(1, 2)  # B,d*d, h, w
        distance_x = (flow_x.repeat(1, d, 1, 1) - grid) ** 2  # B,  d, h, w
        distance_x = distance_x.unsqueeze(2).repeat(1, 1, d, 1, 1).flatten(1, 2)  # B,d*d, h, w

        distance = torch.softmax(-(distance_x + distance_y), dim=1)
        flow_sup = (kernel - distance).abs()
        return torch.mean(flow_sup, dim=(1, 2, 3))
