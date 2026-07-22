# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import dataclasses
import inspect
import math
from typing import Optional, Callable, Dict, Any

import torch
import torch.nn as nn

from ..transforms.functional import yuv_444_to_420, rgb2ycbcr, ycbcr2rgb
from ..metrics.msssim import MS_SSIM
from ..utils.common import downsample_mask, AuxiliaryModels

__all__ = ["get_loss_func"]


@dataclasses.dataclass(eq=False)
class FrameLossComponents:
    recon: torch.Tensor
    bpp: Optional[torch.Tensor] = None
    calibration: Optional[torch.Tensor] = None
    stats: Dict[str, Any] = dataclasses.field(default_factory=dict)


class BaseFrameLoss(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, rd, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None) -> FrameLossComponents:
        """
        Calc bits per pixel and reconsruction loss components
        Args:
            rd: model result dictionary
            target: target image
            mask: optional mask for ROI pixels, if None, all pixels are considered equally important

        Returns:
            frame loss components

        """
        raise NotImplementedError()


class CodecLoss(nn.Module):
    def __init__(self, *, frame_loss: BaseFrameLoss, mse_lambda_exponent: Optional[float], bpp_weight: Optional[float]):
        super().__init__()
        self.frame_loss = frame_loss
        if mse_lambda_exponent is not None:
            mse_lambda_exponent = float(mse_lambda_exponent)
        else:
            mse_lambda_exponent = 1.0
        self.mse_lambda_exponent = mse_lambda_exponent
        self.bpp_weight = float(bpp_weight) if bpp_weight is not None else 1.0

    def forward(self, rd, target: torch.Tensor, *, lambdas: torch.Tensor, mask: Optional[torch.Tensor] = None):
        components = self.frame_loss(rd, target, mask=mask)
        ld = components.stats
        ld["loss"] = self.combine_loss(components.recon, components.bpp, lambdas)
        return ld

    def combine_loss(self, recon_loss, bpp_loss, lambdas):
        mse_lambda_exponent = self.mse_lambda_exponent
        bpp_weight = self.bpp_weight

        if bpp_weight == 0:
            bpp_loss = None
        if mse_lambda_exponent == 1.0:
            loss = recon_loss * lambdas
            if bpp_loss is not None:
                loss += bpp_weight * bpp_loss
        elif mse_lambda_exponent == 0.0:
            loss = recon_loss
            if bpp_loss is not None:
                loss += bpp_weight * (bpp_loss / lambdas)
        else:
            loss = recon_loss * torch.pow(lambdas, mse_lambda_exponent)
            if bpp_loss is not None:
                loss += bpp_weight * (bpp_loss * torch.pow(lambdas, mse_lambda_exponent - 1))

        return loss


def get_roi_weights(roi_bg_ratio: float, p_roi: float):
    """
    Loss = (1 / N) ∑ eᵢ * weight_map_i
    where e_i = (x_i - x_hat_i) ** 2 and N is the number of pixels.

    We want these properties:
    * total "focus" of trained model should be k times more on RoI than on background, regardless of mask model
    * expected loss magnitude is similar to original

    When model is being trained, the expected weighted error over whole training set is given by
    E_C = E[(1/N)∑_{C} e_i * w_i] = (1/N) * N * p_C * w_C * E[e]
    where C is ROI or BG.

    Their ratio is given by:
    E_ROI/E_BG = p_ROI w_ROI E[e] / (1 - p_ROI) w_BG E[e] = p * w_ROI / (1-p) * w_bg

    Given k, we need to find w_bg and w_roi such that
    1. p * w_ROI / (1-p) * w_bg = k (E_ROI/E_BG = k)
    2. (1-p)*w_bg + p*w_roi = 1 (same magnitude as original)

    Solution to the system is given by
    w_ROI = k / (p * (1 + k) )
    w_bg = 1 / ( (1-p) * (1+k) )

    Benefits:
    * On average across the dataset, loss derivative for ROI pixels is k times stronger than for BG pxels. It does
    not depend on mask sizes.
    * Same magnitude of loss as original.
    * We don't need two function call passes (one for ROI and one for background) to calculate loss.
    """

    w_bg = 1 / ((1 + roi_bg_ratio) * (1 - p_roi))
    w_roi = roi_bg_ratio / (p_roi * (1 + roi_bg_ratio))
    return w_bg, w_roi


def get_roi_weights_normalized(roi_bg_ratio: float, r: torch.Tensor, N: int):
    """
    Normalize MSE for each frame, s.t.
    1) r * w_roi + (N - r) * w_bg = N
    2) w_roi = k * w_bg
    where r is the number of ROI pixels, N is the total number of pixels in the frame.
    Note that when r = 0, w_bg = 1, which means the original MSE.
    """
    w_bg = N / (N - r + roi_bg_ratio * r)
    w_roi = roi_bg_ratio * w_bg
    return w_bg, w_roi


def get_roi_weights_wrapper(
    roi_bg_ratio: float, x: torch.Tensor, r: torch.Tensor, N: int, p_roi: Optional[float] = None
):
    if p_roi is not None:
        w_bg, w_roi = get_roi_weights(roi_bg_ratio=roi_bg_ratio, p_roi=p_roi)
    else:
        w_bg, w_roi = get_roi_weights_normalized(roi_bg_ratio=roi_bg_ratio, r=r, N=N)

    batch_size = x.shape[0]
    w_bg = torch.ones((batch_size, 1, 1, 1), device=x.device) * w_bg
    w_roi = torch.ones((batch_size, 1, 1, 1), device=x.device) * w_roi
    return w_bg, w_roi


class BaseMSEFrameLoss(BaseFrameLoss, abc.ABC):
    def __init__(
        self,
        *,
        is_yuv420: bool,
        mse_y_weight: Optional[float],
        mse_yuv_420: Optional[float],
        mse_yuv_mean_in_psnr: Optional[bool],
        mse_rgb_weight: Optional[float],
    ):
        super().__init__()
        self.is_yuv420 = is_yuv420

        if mse_yuv_mean_in_psnr is None:
            mse_yuv_mean_in_psnr = False
        elif not isinstance(mse_yuv_mean_in_psnr, bool):
            raise ValueError(f"Invalid mse_yuv_mean_in_psnr value: {mse_yuv_mean_in_psnr}")
        self.mse_yuv_mean_in_psnr = mse_yuv_mean_in_psnr

        if mse_y_weight is not None:
            mse_y_weight = float(mse_y_weight)
            if mse_y_weight < 0:
                raise ValueError(f"Invalid mse_y_weight value: {mse_y_weight}")
        else:
            mse_y_weight = 8 if self.mse_yuv_mean_in_psnr else 4
        self.mse_y_weight = mse_y_weight

        if mse_yuv_420 is None:
            mse_yuv_420 = True
        elif not isinstance(mse_yuv_420, bool):
            raise ValueError(f"Invalid mse_yuv_420 value: {mse_yuv_420}")
        self.mse_yuv_420 = mse_yuv_420

        if mse_rgb_weight is not None:
            mse_rgb_weight = float(mse_rgb_weight)
            if not (0 <= mse_rgb_weight <= 1):
                raise ValueError(f"Invalid mse_rgb_weight value: {mse_rgb_weight}")
        else:
            mse_rgb_weight = 0.0
        self.mse_rgb_weight = mse_rgb_weight

    @staticmethod
    def mean_vector_mse(
        x, x_hat, *, flatten_dim=1, mask=None, roi_bg_ratio: Optional[float] = None, p_roi: Optional[float] = None
    ):
        """
        Args:
            roi_bg_ratio: ratio of weights for ROI and background pixels.
            p_roi: probability of a pixel belonging to ROI. This can be used to normalise between different mask
            models. For example, in Vimeo dataset, faces are much rarer than bodies.
        """

        pixel_num = math.prod(x.shape[2:])
        mse = torch.nn.functional.mse_loss(x, x_hat, reduction="none")

        if mask is not None:
            if roi_bg_ratio is None:
                raise ValueError("roi_bg_ratio must be set when mask is provided")

            w_bg, w_roi = get_roi_weights_wrapper(roi_bg_ratio, x, r=mask.sum(dim=(1, 2, 3)), N=pixel_num, p_roi=p_roi)
            weight_map = mask * w_roi + (1.0 - mask) * w_bg
            mse = mse * weight_map.view_as(mse)

        mse = mse.flatten(flatten_dim).sum(dim=-1) / pixel_num
        return mse

    def _prepare_yuv_rgb(self, x, x_hat, rgb_weight: float):
        if self.is_yuv420:
            org_yuv = x
            rec_yuv = x_hat

            if rgb_weight == 0:
                org_rgb = None
                rec_rgb = None
            else:
                # rgb2ycbcr contains torch.clamp, should be replaced by apply_lower_upper_bound
                org_rgb = ycbcr2rgb(org_yuv)
                rec_rgb = ycbcr2rgb(rec_yuv)
        else:
            org_rgb = x
            rec_rgb = x_hat

            if rgb_weight == 1:
                org_yuv = None
                rec_yuv = None
            else:
                # rgb2ycbcr contains torch.clamp, should be replaced by apply_lower_upper_bound
                org_yuv = rgb2ycbcr(org_rgb)
                rec_yuv = rgb2ycbcr(rec_rgb)

        return org_yuv, rec_yuv, org_rgb, rec_rgb

    def _calc_frame_mse(self, org_yuv, rec_yuv, org_rgb, rec_rgb, *, mask=None, roi_bg_ratio=None, p_roi=None):
        if self.mse_rgb_weight != 0:
            mse_rgb = self.mean_vector_mse(org_rgb, rec_rgb, mask=mask, roi_bg_ratio=roi_bg_ratio, p_roi=p_roi)
        else:
            mse_rgb = None

        if self.mse_rgb_weight != 1:
            if self.mse_yuv_420:
                org_y, org_u, org_v = yuv_444_to_420(org_yuv)
                rec_y, rec_u, rec_v = yuv_444_to_420(rec_yuv)
                mse_y = self.mean_vector_mse(org_y, rec_y, mask=mask, roi_bg_ratio=roi_bg_ratio, p_roi=p_roi)

                mask_2x = downsample_mask(mask, factor=2) if mask is not None else None
                mse_u = self.mean_vector_mse(org_u, rec_u, mask=mask_2x, roi_bg_ratio=roi_bg_ratio, p_roi=p_roi)
                mse_v = self.mean_vector_mse(org_v, rec_v, mask=mask_2x, roi_bg_ratio=roi_bg_ratio, p_roi=p_roi)
            else:
                if mask is not None:
                    raise NotImplementedError("Masked MSE only implemented for YUV420")

                mse_y, mse_u, mse_v = self.mean_vector_mse(org_yuv, rec_yuv, mask=mask, flatten_dim=2).chunk(3, 1)

            if self.mse_yuv_mean_in_psnr:
                scale = 1 / (2 + self.mse_y_weight)
                mse_yuv = 3 * torch.exp(
                    scale
                    * (self.mse_y_weight * torch.log(mse_y + 1e-6) + torch.log(mse_u + 1e-6) + torch.log(mse_v + 1e-6))
                )
            else:
                scale = 3 / (2 + self.mse_y_weight)
                mse_yuv = scale * (self.mse_y_weight * mse_y + mse_u + mse_v)
        else:
            mse_yuv = None

        if self.mse_rgb_weight == 0:
            mse = mse_yuv
        elif self.mse_rgb_weight == 1:
            mse = mse_rgb
        else:
            assert mse_rgb is not None and mse_yuv is not None
            mse = self.mse_rgb_weight * mse_rgb + (1 - self.mse_rgb_weight) * mse_yuv

        assert mse is not None
        return mse

    def calc_frame_mse(self, x, x_hat, *, mask=None, roi_bg_ratio=None, p_roi=None):
        org_yuv, rec_yuv, org_rgb, rec_rgb = self._prepare_yuv_rgb(x, x_hat, self.mse_rgb_weight)
        mse = self._calc_frame_mse(
            org_yuv, rec_yuv, org_rgb, rec_rgb, mask=mask, roi_bg_ratio=roi_bg_ratio, p_roi=p_roi
        )
        return mse

    @staticmethod
    def calc_psnr(mse):
        return torch.nan_to_num(-10 * torch.log10(torch.clamp_min(mse, 1.0e-10)), 0.0)


class loss_me_mse(BaseMSEFrameLoss):
    def forward(self, rd, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None):
        me_mse = self.calc_frame_mse(rd["warp_frame"], target)
        return FrameLossComponents(
            recon=me_mse,
            stats=dict(
                me_mse=me_mse,
            ),
        )


class loss_me_rdc_mse(BaseMSEFrameLoss):
    def forward(self, rd, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None):
        me_mse = self.calc_frame_mse(rd["warp_frame"], target)
        bpp = rd["bpp_mv_y"] + rd["bpp_mv_z"]
        return FrameLossComponents(
            recon=me_mse,
            bpp=bpp,
            stats=dict(
                me_mse=me_mse,
            ),
        )


class loss_recon_mse(BaseMSEFrameLoss):
    def forward(self, rd, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None):
        mse = self.calc_frame_mse(rd["x_hat"], target)
        return FrameLossComponents(
            recon=mse,
            stats=dict(
                mse=mse,
            ),
        )


class loss_recon_rdc_mse(BaseMSEFrameLoss):
    def forward(self, rd, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None):
        mse = self.calc_frame_mse(rd["x_hat"], target)
        bpp = rd["bpp_y"] + rd["bpp_z"]
        return FrameLossComponents(
            recon=mse,
            bpp=bpp,
            stats=dict(
                mse=mse,
            ),
        )


class loss_rdc_mse_flow_sup(BaseMSEFrameLoss):
    def __init__(
        self,
        *,
        is_yuv420: bool,
        mse_y_weight: Optional[float],
        mse_yuv_420: Optional[float],
        mse_yuv_mean_in_psnr: Optional[bool],
        mse_rgb_weight: Optional[float],
        optic_flow_loss: nn.Module,
    ):
        super().__init__(
            is_yuv420=is_yuv420,
            mse_y_weight=mse_y_weight,
            mse_yuv_420=mse_yuv_420,
            mse_yuv_mean_in_psnr=mse_yuv_mean_in_psnr,
            mse_rgb_weight=mse_rgb_weight,
        )
        self.optic_flow_loss = optic_flow_loss

    def forward(self, rd, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None):
        mse = self.calc_frame_mse(rd["x_hat"], target)
        # MSRA applies per frame distortion weights only to MSE part
        # Here distortion weights are already premultiplied into lambdas
        flow_sup = self.optic_flow_loss(rd)
        bpp = rd["bpp"]
        return FrameLossComponents(
            recon=mse + flow_sup,
            bpp=bpp,
            stats=dict(
                mse=mse,
                flow_sup=flow_sup,
            ),
        )


class loss_total_rdc_mse(BaseMSEFrameLoss):
    def forward(self, rd, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None):
        mse = self.calc_frame_mse(rd["x_hat"], target)
        bpp = rd["bpp"]
        return FrameLossComponents(
            recon=mse,
            bpp=bpp,
            stats=dict(
                mse=mse,
                psnr=self.calc_psnr(mse),
            ),
        )


class loss_total_rdc_psnr(BaseMSEFrameLoss):
    def __init__(
        self,
        *,
        is_yuv420: bool,
        mse_y_weight: Optional[float],
        mse_yuv_420: Optional[float],
        mse_rgb_weight: Optional[float],
    ):
        super().__init__(
            is_yuv420=is_yuv420,
            mse_y_weight=mse_y_weight,
            mse_yuv_420=mse_yuv_420,
            mse_yuv_mean_in_psnr=True,
            mse_rgb_weight=mse_rgb_weight,
        )

    def forward(self, rd, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None):
        mse = self.calc_frame_mse(rd["x_hat"], target)
        bpp = rd["bpp"]
        return FrameLossComponents(
            recon=mse,
            bpp=bpp,
            stats=dict(
                mse=mse,
                psnr=self.calc_psnr(mse),
            ),
        )


class BasePerceptualFrameLoss(BaseMSEFrameLoss, abc.ABC):
    def __init__(
        self,
        *,
        is_yuv420: bool,
        mse_y_weight: Optional[float],
        mse_yuv_420: Optional[float],
        mse_yuv_mean_in_psnr: Optional[bool],
        mse_rgb_weight: Optional[float],
        perceptual_loss: nn.Module,
        distortion_loss_weight: float,
        perceptual_loss_weight: float,
        perceptual_rgb_weight: Optional[float],
    ):
        super().__init__(
            is_yuv420=is_yuv420,
            mse_y_weight=mse_y_weight,
            mse_yuv_420=mse_yuv_420,
            mse_yuv_mean_in_psnr=mse_yuv_mean_in_psnr,
            mse_rgb_weight=mse_rgb_weight,
        )
        self.perceptual_loss = perceptual_loss

        if distortion_loss_weight is None:
            raise ValueError("distortion_loss_weight must be set")
        if perceptual_loss_weight is None:
            raise ValueError("perceptual_loss_weight must be set")

        self.distortion_loss_weight = distortion_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight

        perceptual_rgb_weight = float(perceptual_rgb_weight) if perceptual_rgb_weight is not None else 0.0
        if not 0 <= perceptual_rgb_weight <= 1:
            raise ValueError(f"Invalid perceptual_rgb_weight value: {perceptual_rgb_weight}")

        self.perceptual_rgb_weight = perceptual_rgb_weight
        self.is_yuv420 = is_yuv420

    def _calc_frame_perceptual(
        self,
        org_yuv,
        rec_yuv,
        org_rgb,
        rec_rgb,
        *,
        mask,
        flatten_dim=1,
        roi_bg_ratio: Optional[float] = None,
        p_roi: Optional[float] = None,
    ) -> torch.Tensor:
        """Get spatial loss and weight it by mask if provided."""

        perceptual_rgb = (
            self.perceptual_loss(rec_rgb, org_rgb, spatial=True) if self.perceptual_rgb_weight != 0 else None
        )
        perceptual_yuv = (
            self.perceptual_loss(rec_yuv, org_yuv, spatial=True) if self.perceptual_rgb_weight != 1 else None
        )

        if self.perceptual_rgb_weight == 0:
            perceptual = perceptual_yuv
        elif self.perceptual_rgb_weight == 1:
            perceptual = perceptual_rgb
        else:
            assert perceptual_rgb is not None and perceptual_yuv is not None
            w = self.perceptual_rgb_weight
            perceptual = w * perceptual_rgb + (1 - w) * perceptual_yuv

        assert perceptual is not None
        pixel_num = math.prod(perceptual.shape[2:])
        if mask is not None:
            if roi_bg_ratio is None or p_roi is None:
                raise ValueError("roi_bg_ratio and p_roi must be set when mask is provided")

            w_bg, w_roi = get_roi_weights_wrapper(
                roi_bg_ratio, x=perceptual, r=mask.sum(dim=(1, 2, 3)), N=pixel_num, p_roi=p_roi
            )
            weight_map = mask * w_roi + (1.0 - mask) * w_bg

            perc = perceptual * weight_map.view_as(perceptual)
        else:
            perc = perceptual

        perc = perc.flatten(flatten_dim).sum(dim=-1) / pixel_num
        return perc


class loss_total_rdc_perceptual(BasePerceptualFrameLoss):
    def forward(self, rd, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None):
        # Does either MSE or perceptual require RGB?
        rgb_weight = (self.mse_rgb_weight + self.perceptual_rgb_weight) / 2
        org_yuv, rec_yuv, org_rgb, rec_rgb = self._prepare_yuv_rgb(target, rd["x_hat"], rgb_weight=rgb_weight)

        mse = self._calc_frame_mse(org_yuv, rec_yuv, org_rgb, rec_rgb)
        perceptual = self._calc_frame_perceptual(org_yuv, rec_yuv, org_rgb, rec_rgb, mask=None)

        total = self.distortion_loss_weight * mse + self.perceptual_loss_weight * perceptual

        return FrameLossComponents(
            recon=total,
            bpp=rd["bpp"],
            stats=dict(
                mse=mse,
                perceptual=perceptual,
                psnr=self.calc_psnr(mse),
            ),
        )


class loss_total_rdc_psnr_roi_per_pixel(BaseMSEFrameLoss):
    def __init__(
        self,
        *,
        is_yuv420: bool,
        mse_y_weight: Optional[float],
        mse_yuv_420: Optional[float],
        mse_rgb_weight: Optional[float],
        roi_bg_ratio: float,
        p_roi: Optional[float] = None,
        segmentation_model: nn.Module,
    ):
        super().__init__(
            is_yuv420=is_yuv420,
            mse_y_weight=mse_y_weight,
            mse_yuv_420=mse_yuv_420,
            mse_yuv_mean_in_psnr=True,
            mse_rgb_weight=mse_rgb_weight,
        )
        self.segmentation_model = segmentation_model

        assert roi_bg_ratio is not None, "roi_bg_ratio must be set"
        self.roi_bg_ratio = float(roi_bg_ratio)

        if p_roi is not None:
            assert 0 < p_roi < 1, f"ROI weight must be in (0, 1), got {p_roi}"
        self.p_roi = p_roi

    def forward(self, rd, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None):
        bpp = rd["bpp"]
        if mask is None:
            mask = self.segmentation_model(target, is_yuv420=self.is_yuv420)
        else:
            mask = mask.unsqueeze(1)
        assert isinstance(mask, torch.Tensor)
        mask_proba = mask.sum(dim=[1, 2, 3]) / math.prod(target.shape[2:])

        recon_total = self.calc_frame_mse(
            rd["x_hat"], target, mask=mask, roi_bg_ratio=self.roi_bg_ratio, p_roi=self.p_roi
        )
        return FrameLossComponents(
            recon=recon_total,
            bpp=bpp,
            stats=dict(
                recon_total=recon_total,
                mask_proba=mask_proba,
            ),
        )


class loss_total_rdc_perceptual_roi_per_pixel(BasePerceptualFrameLoss):
    def __init__(
        self,
        *,
        is_yuv420: bool,
        mse_y_weight: Optional[float],
        mse_yuv_420: Optional[float],
        mse_yuv_mean_in_psnr: Optional[bool],
        mse_rgb_weight: Optional[float],
        perceptual_loss: nn.Module,
        distortion_loss_weight: float,
        perceptual_loss_weight: float,
        perceptual_rgb_weight: Optional[float],
        roi_bg_ratio: float,
        p_roi: Optional[float] = None,
        segmentation_model: nn.Module,
    ):
        super().__init__(
            is_yuv420=is_yuv420,
            mse_y_weight=mse_y_weight,
            mse_yuv_420=mse_yuv_420,
            mse_yuv_mean_in_psnr=mse_yuv_mean_in_psnr,
            mse_rgb_weight=mse_rgb_weight,
            perceptual_loss=perceptual_loss,
            distortion_loss_weight=distortion_loss_weight,
            perceptual_loss_weight=perceptual_loss_weight,
            perceptual_rgb_weight=perceptual_rgb_weight,
        )

        if roi_bg_ratio is None:
            raise ValueError("roi_bg_ratio must be set")
        if p_roi is not None:
            if not 0 <= p_roi <= 1:
                raise ValueError(f"ROI weight must be in [0, 1], got {p_roi}")

        self.roi_bg_ratio = roi_bg_ratio
        self.p_roi = p_roi

        self.segmentation_model = segmentation_model

    def forward(self, rd, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None):
        bpp = rd["bpp"]
        if mask is None:
            mask = self.segmentation_model(target, is_yuv420=self.is_yuv420)
        else:
            mask = mask.unsqueeze(1)
        assert isinstance(mask, torch.Tensor)
        mask_proba = mask.sum(dim=[1, 2, 3]) / math.prod(target.shape[2:])

        # Does either MSE or perceptual require RGB?
        rgb_weight = (self.mse_rgb_weight + self.perceptual_rgb_weight) / 2
        org_yuv, rec_yuv, org_rgb, rec_rgb = self._prepare_yuv_rgb(target, rd["x_hat"], rgb_weight=rgb_weight)

        mse_masked = self.calc_frame_mse(
            target, rd["x_hat"], mask=mask, roi_bg_ratio=self.roi_bg_ratio, p_roi=self.p_roi
        )

        perceptual_masked = self._calc_frame_perceptual(
            org_yuv, rec_yuv, org_rgb, rec_rgb, mask=mask, roi_bg_ratio=self.roi_bg_ratio, p_roi=self.p_roi
        )

        final = self.distortion_loss_weight * mse_masked + self.perceptual_loss_weight * perceptual_masked

        return FrameLossComponents(
            recon=final,
            bpp=bpp,
            stats=dict(
                final=final,
                mask_proba=mask_proba,
                mse_masked=mse_masked,
                perceptual_masked=perceptual_masked,
            ),
        )


class BaseMSSSIMFrameLoss(BaseFrameLoss, abc.ABC):
    def __init__(self, *, is_yuv420: bool):
        super().__init__()
        self.is_yuv420 = is_yuv420
        self.msssim = MS_SSIM(channels=1 if is_yuv420 else 3)

    def calc_frame_msssim(self, x, x_hat):
        assert self.msssim is not None

        if not self.is_yuv420:
            msssim = self.msssim(x, x_hat)
        else:
            org_y, org_u, org_v = yuv_444_to_420(x)
            rec_y, rec_u, rec_v = yuv_444_to_420(x_hat)
            msssim_y = self.msssim(org_y, rec_y)
            msssim_u = self.msssim(org_u, rec_u)
            msssim_v = self.msssim(org_v, rec_v)

            msssim = (6 * msssim_y + msssim_u + msssim_v) / 8

        return msssim


class loss_total_rdc_ms_ssim(BaseMSSSIMFrameLoss):
    def forward(self, rd, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None):
        ssim = self.calc_frame_msssim(rd["x_hat"], target)
        bpp = rd["bpp"]
        return FrameLossComponents(
            recon=ssim / 17,
            bpp=bpp,
            stats=dict(
                ssim=ssim,
            ),
        )


def get_loss_func(
    loss_type: str,
    *,
    is_yuv420: bool,
    mse_lambda_exponent: Optional[float] = None,
    bpp_weight: Optional[float] = None,
    mse_y_weight: Optional[float] = None,
    mse_yuv_420: Optional[float] = None,
    mse_yuv_mean_in_psnr: Optional[bool] = None,
    mse_rgb_weight: Optional[float] = None,
    create_optic_flow_loss: Optional[Callable[[], nn.Module]] = None,
    auxiliary_models: Optional[AuxiliaryModels] = None,
    perceptual_loss_weight: Optional[float] = None,
    distortion_loss_weight: Optional[float] = None,
    roi_weight: Optional[float] = None,
    perceptual_rgb_weight: Optional[float] = None,
    roi_bg_ratio: Optional[float] = None,
    p_roi: Optional[float] = None,
):
    frame_loss_name = f"loss_{loss_type}"
    frame_loss_type = globals().get(frame_loss_name)
    if not isinstance(frame_loss_type, type):
        raise ValueError(f"loss {loss_type} is not defined")

    arg_names = set(inspect.signature(frame_loss_type).parameters.keys())

    def get_init_args(**kwargs):
        return {name: value for name, value in kwargs.items() if name in arg_names}

    args = get_init_args(
        is_yuv420=is_yuv420,
        mse_y_weight=mse_y_weight,
        mse_yuv_420=mse_yuv_420,
        mse_yuv_mean_in_psnr=mse_yuv_mean_in_psnr,
        mse_rgb_weight=mse_rgb_weight,
        perceptual_loss_weight=perceptual_loss_weight,
        distortion_loss_weight=distortion_loss_weight,
        roi_weight=roi_weight,
        perceptual_rgb_weight=perceptual_rgb_weight,
        roi_bg_ratio=roi_bg_ratio,
        p_roi=p_roi,
    )

    if "optic_flow_loss" in arg_names:
        assert create_optic_flow_loss is not None
        args["optic_flow_loss"] = create_optic_flow_loss()

    if "perceptual_loss" in arg_names:
        assert auxiliary_models is not None
        args["perceptual_loss"] = auxiliary_models.perceptual_model

    if "segmentation_model" in arg_names:
        assert auxiliary_models is not None
        args["segmentation_model"] = auxiliary_models.segmentation_model

    frame_loss = frame_loss_type(**args)
    assert isinstance(frame_loss, BaseFrameLoss)
    return CodecLoss(frame_loss=frame_loss, mse_lambda_exponent=mse_lambda_exponent, bpp_weight=bpp_weight)
