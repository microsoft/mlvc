# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from enum import Enum
from dataclasses import dataclass


class FrameType(str, Enum):
    I_FRAME = "i_frame"
    P_FRAME = "p_frame"
    LTR_RECOVERY = "ltr_recovery"


@dataclass(frozen=True)
class RateControlInfo:
    # Bucket
    target_bucket_level: float
    effective_bucket_target_level: float
    estimated_bucket_level: float
    actual_bucket_level: float
    # Frame bits
    nominal_frame_bits: int
    allocated_frame_bits: int
    actual_frame_bits: int
    # Q-index
    raw_q_index: int
    q_index: int


# --------------------------------------------------------------------------------------------------------------------
# Rate allocation
# --------------------------------------------------------------------------------------------------------------------


class LeakyBucket:
    def __init__(
        self,
        bitrate: float = 500e3,  # bits per sec
        fps: float = 30.0,
        bucket_size: float = 1.0,  # seconds
        initial_level: float = 0.1,
    ):
        self._bitrate = bitrate
        self._fps = fps
        self._bucket_size = bucket_size
        self._initial_level = initial_level

        # Initialize bucket state
        self._capacity_bits = int(self._bitrate * self._bucket_size)
        self._fill_bits: int = int(initial_level * self._capacity_bits)
        self._last_drain_timestamp: float | None = None

    def configure(self, bitrate: float | None = None, fps: float | None = None) -> None:
        if bitrate is not None:
            # Preserve fill deviation from target level across bitrate change
            excess_bits = self._fill_bits - int(self._initial_level * self._capacity_bits)
            new_capacity_bits = int(bitrate * self._bucket_size)
            new_fill_bits = np.clip(int(self._initial_level * new_capacity_bits) + excess_bits, 0, new_capacity_bits)

            self._bitrate = bitrate
            self._capacity_bits = new_capacity_bits
            self._fill_bits = new_fill_bits

        if fps is not None:
            self._fps = fps

    def calc_drain_secs(self, presentation_time: float) -> float:
        if self._last_drain_timestamp is None:
            return 1.0 / self._fps
        return presentation_time - self._last_drain_timestamp

    def calc_fill_bits(self, presentation_time: float) -> int:
        return self._fill_bits - self._calc_drain_bits(presentation_time)

    def update(self, presentation_time: float, frame_bits: int) -> None:
        self._fill_bits = self.calc_fill_bits(presentation_time) + frame_bits
        self._last_drain_timestamp = presentation_time

    @property
    def bitrate(self) -> float:
        return self._bitrate

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def bucket_size(self) -> float:
        return self._bucket_size

    @property
    def fill_bits(self) -> int:
        return self._fill_bits

    @property
    def capacity_bits(self) -> int:
        return self._capacity_bits

    @property
    def level(self) -> float:
        return self._fill_bits / self._capacity_bits

    def _calc_drain_bits(self, presentation_time: float) -> int:
        return min(self._fill_bits, int(self.calc_drain_secs(presentation_time) * self._bitrate))


@dataclass(frozen=True)
class RateAllocatorResult:
    nominal_bits: int
    allocated_bits: int
    effective_target_level: float
    estimated_level: float


class RateAllocator:
    def __init__(
        self,
        bitrate: float = 500e3,  # bits per sec
        fps: float = 30.0,
        target_level: float = 0.1,  # [0, 1]
        overshoot_tau: float = 0.25,  # seconds
        undershoot_tau: float = 1.0,  # seconds
        planned_excess_tau: float = 0.5,  # seconds
    ):
        self._target_level = target_level
        self._overshoot_tau = overshoot_tau
        self._undershoot_tau = undershoot_tau
        self._planned_excess_tau = planned_excess_tau

        self._bucket = LeakyBucket(bitrate=bitrate, fps=fps, initial_level=target_level)
        self._accumulated_excess_bits: int = 0

    def configure(
        self,
        bitrate: float | None = None,
        fps: float | None = None,
    ) -> None:
        self._bucket.configure(bitrate=bitrate, fps=fps)

    def allocate(
        self,
        presentation_time: float,
        frame_weight: float,
        undershoot_tau: float | None = None,
        overshoot_tau: float | None = None,
    ) -> RateAllocatorResult:
        # Nominal frame budget (weighted equal share bits)
        nominal_bits = int(frame_weight * (self._bucket.bitrate / self._bucket.fps))

        # Target bucket fill bits (excess bits corrected)
        planned_excess_bits = self._calc_planned_excess_bits(presentation_time, frame_weight)
        target_fill_bits = int(self._target_level * self._bucket.capacity_bits) + planned_excess_bits

        # Calculate correction bits
        current_fill_bits = self._bucket.calc_fill_bits(presentation_time)
        error_bits = target_fill_bits - (current_fill_bits + nominal_bits)
        correction_tau = (
            (undershoot_tau if undershoot_tau is not None else self._undershoot_tau)
            if error_bits >= 0
            else (overshoot_tau if overshoot_tau is not None else self._overshoot_tau)
        )
        correction_bits = int((1.0 / (correction_tau * self._bucket.fps)) * error_bits)

        # Final allocation: clamp to [0.33x, 2x] of nominal and cap at bucket headroom
        bucket_max_fill_bits = int(0.9 * self._bucket.capacity_bits)
        bucket_headroom_bits = bucket_max_fill_bits - current_fill_bits
        allocated_bits = np.clip(
            min(nominal_bits + correction_bits, bucket_headroom_bits),
            a_min=int(0.33 * nominal_bits),
            a_max=int(2.0 * nominal_bits),
        )

        # Drop frame if it would overflow the bucket
        if current_fill_bits + allocated_bits > bucket_max_fill_bits:
            allocated_bits = 0

        return RateAllocatorResult(
            nominal_bits=nominal_bits,
            allocated_bits=allocated_bits,
            effective_target_level=target_fill_bits / self._bucket.capacity_bits,
            estimated_level=(current_fill_bits + allocated_bits) / self._bucket.capacity_bits,
        )

    def update(self, presentation_time: float, frame_weight: float, frame_bits: int) -> None:
        self._accumulated_excess_bits = self._calc_planned_excess_bits(presentation_time, frame_weight)
        self._bucket.update(presentation_time, frame_bits)

    @property
    def target_level(self) -> float:
        return self._target_level

    @property
    def bucket_level(self) -> float:
        return self._bucket.level

    def _calc_planned_excess_bits(self, presentation_time: float, frame_weight: float) -> int:
        drain_secs = self._bucket.calc_drain_secs(presentation_time)
        decay_bits = int((drain_secs / self._planned_excess_tau) * self._accumulated_excess_bits)
        excess_bits = max(0, int((frame_weight - 1.0) * (self._bucket.bitrate / self._bucket.fps)))

        max_excess_bits = int(0.5 * self._bucket.capacity_bits)
        return np.clip(self._accumulated_excess_bits - decay_bits + excess_bits, 0, max_excess_bits)


# --------------------------------------------------------------------------------------------------------------------
# Rate implementation
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RateModelParams:
    initial_alpha: float
    beta: float
    alpha_tau: float  # number of updates
    seed_alpha_tau: float = 2.0  # number of updates
    beta_ramp_target: float | None = None
    beta_ramp_duration: float | None = None
    min_q_index: int = 0
    max_q_index: int = 63


class RateModel:
    def __init__(self, params: RateModelParams) -> None:
        self._params = params

        self._seed_alpha = self._params.initial_alpha
        self.reset()

    def reset(self):
        self._alpha = self._seed_alpha
        self._beta = self._params.beta
        self._num_updates = 0

    def solve_q_index(self, bpp: float) -> int:
        q = -np.log(max(1e-9, bpp) / self._alpha) / self._beta
        return int(np.clip(np.round(q), self._params.min_q_index, self._params.max_q_index))

    def predict_bpp(self, q_index: int) -> float:
        return self._alpha * np.exp(-self._beta * q_index)

    def update(self, q_index: int, bpp: float) -> None:
        # Adapt alpha
        last_observed_alpha = bpp * np.exp(self._beta * q_index)
        self._alpha += (1.0 / self._params.alpha_tau) * (last_observed_alpha - self._alpha)

        # Update seed alpha on first update
        if self._num_updates == 0:
            self._seed_alpha += (1.0 / self._params.seed_alpha_tau) * (last_observed_alpha - self._seed_alpha)

        # Increment update count
        self._num_updates += 1

        # Update beta with optional ramping if configured
        if self._params.beta_ramp_target is not None and self._params.beta_ramp_duration is not None:
            ramp_progress = min(1.0, self._num_updates / max(1.0, self._params.beta_ramp_duration))
            new_beta = self._params.beta + ramp_progress * (self._params.beta_ramp_target - self._params.beta)

            # Preserve predicted bpp at q_index: alpha * exp(-beta*q) must be invariant
            self._alpha *= np.exp((new_beta - self._beta) * q_index)
            self._beta = new_beta

    @property
    def alpha(self) -> float:
        return self._alpha


class RateImplementation:
    def __init__(
        self,
        iframe_model_params=RateModelParams(initial_alpha=0.04969, beta=-0.03626, alpha_tau=2.0),
        ltr_recovery_model_params=RateModelParams(initial_alpha=0.01047, beta=-0.05306, alpha_tau=2.0),
        p_frame_after_idr_model_params=RateModelParams(
            initial_alpha=0.02156,
            beta=-0.03173,
            alpha_tau=2.0,
            beta_ramp_target=-0.07654,
            beta_ramp_duration=9.0,
        ),
        p_frame_after_ltr_model_params=RateModelParams(
            initial_alpha=0.00315,
            beta=-0.05925,
            alpha_tau=2.0,
            beta_ramp_target=-0.08307,
            beta_ramp_duration=9.0,
        ),
    ):
        self._iframe_model = RateModel(iframe_model_params)
        self._ltr_recovery_model = RateModel(ltr_recovery_model_params)
        self._pframe_after_idr_model = RateModel(p_frame_after_idr_model_params)
        self._pframe_after_ltr_model = RateModel(p_frame_after_ltr_model_params)
        self._last_recovery_type = FrameType.I_FRAME

    def solve_q_index(self, frame_type: FrameType, bpp: float) -> int:
        # NB: bpp should be calculated on the payload only (excluding headers and other overhead)
        return self._get_model(frame_type).solve_q_index(bpp)

    def update(self, frame_type: FrameType, q_index: int, bpp: float) -> None:
        # NB: bpp should be calculated on the payload only (excluding headers and other overhead)

        # Update rate model
        self._get_model(frame_type).update(q_index, bpp)

        # After an I-frame or LTR recovery frame, reset the P-frame model
        if frame_type in (FrameType.I_FRAME, FrameType.LTR_RECOVERY):
            self._pframe_after_idr_model.reset()
            self._pframe_after_ltr_model.reset()
            self._last_recovery_type = frame_type

    def predict_bpp(self, frame_type: FrameType, q_index: int) -> float:
        return self._get_model(frame_type).predict_bpp(q_index)

    def _get_model(self, frame_type: FrameType) -> RateModel:
        if frame_type == FrameType.I_FRAME:
            return self._iframe_model
        elif frame_type == FrameType.P_FRAME and self._last_recovery_type == FrameType.I_FRAME:
            return self._pframe_after_idr_model
        elif frame_type == FrameType.P_FRAME and self._last_recovery_type == FrameType.LTR_RECOVERY:
            return self._pframe_after_ltr_model
        elif frame_type == FrameType.LTR_RECOVERY:
            return self._ltr_recovery_model
        else:
            raise ValueError(f"Unsupported frame type: {frame_type}")


# --------------------------------------------------------------------------------------------------------------------
# Rate controller
# --------------------------------------------------------------------------------------------------------------------


class RateController:
    def __init__(
        self,
        image_width: int,
        image_height: int,
        bitrate: float,
        fps: float,
        *,
        max_q_index_increase: int | None = None,
        max_q_index_decrease: int | None = None,
        frame_weights: dict[FrameType, float] = {
            FrameType.I_FRAME: 10.0,
            FrameType.LTR_RECOVERY: 6.0,
            FrameType.P_FRAME: 1.0,
        },
        frame_weight_tau: float = 2.0,
    ):
        self._image_width = image_width
        self._image_height = image_height
        self._rate_alloc = RateAllocator(bitrate=bitrate, fps=fps)
        self._rate_impl = RateImplementation()
        self._max_q_index_increase = max_q_index_increase
        self._max_q_index_decrease = max_q_index_decrease
        self._frame_weights = frame_weights
        self._frame_weight_tau = frame_weight_tau

        # Current state
        self._presentation_time: float = 0.0
        self._frame_type = FrameType.I_FRAME
        self._frame_weight: float = 1.0
        self._alloc_result: RateAllocatorResult | None = None
        self._raw_q_index: int | None = None
        self._res_q_index: int | None = None

        # Previous frame state
        self._prev_frame_weight: float = 1.0
        self._prev_res_q_index: int | None = None

    def configure(self, bitrate: float | None = None, fps: float | None = None) -> None:
        self._rate_alloc.configure(bitrate=bitrate, fps=fps)

    def solve_q_index(
        self,
        presentation_time: float,
        frame_type: FrameType,
        *,
        reserved_overhead_bits: int = 0,
    ) -> int | None:
        # Frame weight (decay + update)
        frame_weight = self._prev_frame_weight + (1.0 / self._frame_weight_tau) * (1.0 - self._prev_frame_weight)
        if self._frame_weights[frame_type] > frame_weight:
            frame_weight = self._frame_weights[frame_type]

        # Disable undershoot correction for intra and LTR-recovery frames
        undershoot_tau = 100.0 if frame_type in [FrameType.I_FRAME, FrameType.LTR_RECOVERY] else None

        # Rate allocation
        alloc_result = self._rate_alloc.allocate(
            presentation_time=presentation_time,
            frame_weight=frame_weight,
            undershoot_tau=undershoot_tau,
        )

        # Rate implementation
        target_bpp = max(0, alloc_result.allocated_bits - reserved_overhead_bits) / (
            self._image_width * self._image_height
        )
        model_q_index = self._rate_impl.solve_q_index(frame_type, target_bpp)

        # QP smoothing
        if self._prev_res_q_index is not None and (
            self._max_q_index_decrease is not None or self._max_q_index_increase is not None
        ):
            min_q = (
                self._prev_res_q_index - self._max_q_index_decrease if self._max_q_index_decrease is not None else None
            )
            max_q = (
                self._prev_res_q_index + self._max_q_index_increase if self._max_q_index_increase is not None else None
            )
            smoothed_q_index = int(np.clip(model_q_index, min_q, max_q))
        else:
            smoothed_q_index = model_q_index

        # Save state
        self._presentation_time = presentation_time
        self._frame_type = frame_type
        self._frame_weight = frame_weight
        self._alloc_result = alloc_result
        self._raw_q_index = model_q_index
        self._res_q_index = smoothed_q_index

        # Return None if frame should be dropped
        if alloc_result.allocated_bits <= 0:
            return None
        return self._res_q_index

    def update(
        self,
        header_bits: int,
        payload_bits: int,
    ) -> RateControlInfo:
        if self._res_q_index is None:
            raise ValueError("solve_q_index(...) must be called before calling update(...)")

        assert self._alloc_result is not None
        assert self._raw_q_index is not None

        # Update rate allocator
        actual_frame_bits = header_bits + payload_bits
        self._rate_alloc.update(self._presentation_time, self._frame_weight, actual_frame_bits)

        # Update implementation model
        actual_payload_bpp = payload_bits / (self._image_width * self._image_height)
        self._rate_impl.update(self._frame_type, self._res_q_index, actual_payload_bpp)

        # Update prev state
        self._prev_frame_weight = self._frame_weight
        self._prev_res_q_index = self._res_q_index

        # Return info
        return RateControlInfo(
            # Bucket
            target_bucket_level=self._rate_alloc.target_level,
            effective_bucket_target_level=self._alloc_result.effective_target_level,
            estimated_bucket_level=self._alloc_result.estimated_level,
            actual_bucket_level=self._rate_alloc.bucket_level,
            # Frame bit budget
            nominal_frame_bits=self._alloc_result.nominal_bits,
            allocated_frame_bits=self._alloc_result.allocated_bits,
            actual_frame_bits=actual_frame_bits,
            # Q-index
            raw_q_index=self._raw_q_index,
            q_index=self._res_q_index,
        )
