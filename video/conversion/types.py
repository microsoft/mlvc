# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import torch
import platform
import numpy as np
import dataclasses
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from typing import Any


# -------------------------------------------------------------------------------------
# Full model
# -------------------------------------------------------------------------------------


ModelVersion = str


@dataclass(frozen=True)
class ModelParams:
    model_version: ModelVersion
    pixel_range: float
    qp_num: int
    total_qp_num: int
    frame_index_map: list[int]
    qp_shift: list[int]
    feature_channels: int
    latent_channels: int
    hyperprior_channels: int
    downsample_feature: int
    downsample_latent: int
    downsample_hyperprior: int
    y_scale_repeat: int | None
    quantize_scale_decoder: bool
    weights_version: str | None
    weights_path: str | None
    iframe_period: int | None
    reset_period: int | None
    ltr_start_idx: int
    ltr_period: int | None
    fake_quantized: bool
    disable_feature_reset: bool
    qp_mapping: list[int] | None
    extra_params: dict[str, Any]


@dataclass(frozen=True)
class BasePmf:
    pmf_lengths: list[int]
    pmf_offsets: list[int]
    pmf_table: list[int]


@dataclass(frozen=True)
class GaussianCoderPmf(BasePmf):
    scale_min: float
    scale_max: float
    scale_levels: int
    index_space: bool

    @classmethod
    def load(cls, model_path: str | Path, filename: str | None = None) -> "GaussianCoderPmf":
        with open(Path(model_path) / (filename or "gaussian_pmf.json"), "r") as f:
            data = json.load(f)
        return cls(**data)


@dataclass(frozen=True)
class BitEstimatorPmf(BasePmf):
    qp_num: int
    channels: int

    @classmethod
    def load(cls, model_path: str | Path, filename: str | None = None) -> "BitEstimatorPmf":
        with open(Path(model_path) / (filename or "bit_estimator_pmf.json"), "r") as f:
            data = json.load(f)
        return cls(**data)


# -------------------------------------------------------------------------------------
# Model wrapper
# -------------------------------------------------------------------------------------


class ModelType(str, Enum):
    COREML = "coreml"
    ONNX = "onnx"
    OPENVINO = "openvino"
    TORCH = "torch"


class ModelPrecision(str, Enum):
    FP16 = "fp16"
    FP32 = "fp32"

    @classmethod
    def to_torch_dtype(cls, precision: "ModelPrecision") -> torch.dtype:
        if precision == cls.FP16:
            return torch.float16
        elif precision == cls.FP32:
            return torch.float32
        else:
            raise ValueError(f"Precision {precision} not supported")

    @classmethod
    def to_numpy_dtype(cls, precision: "ModelPrecision"):
        if precision == cls.FP16:
            return np.float16
        elif precision == cls.FP32:
            return np.float32
        else:
            raise ValueError(f"Precision {precision} not supported")


class TorchDevice(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class CoremlComputeUnits(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    ALL = "all"


class OnnxExecutionProvider(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    DIRECTML = "directml"
    OPENVINO = "openvino"
    QNN = "qnn"
    COREML = "coreml"


class OpenvinoDevice(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"


@dataclass(frozen=True)
class RuntimeParams(DataClassJsonMixin):
    # Torch
    torch_device: TorchDevice = TorchDevice.CPU

    # CoreML
    coreml_compute_units: CoremlComputeUnits = CoremlComputeUnits.NPU
    coreml_fast_prediction: bool = True

    # ONNX
    onnx_execution_provider: OnnxExecutionProvider = (
        OnnxExecutionProvider.QNN
        if platform.system() == "Windows" and platform.machine() == "ARM64"
        else OnnxExecutionProvider.CPU
    )
    onnx_provider_options: dict[str, str] = dataclasses.field(default_factory=dict)

    # OpenVINO
    openvino_device: OpenvinoDevice = OpenvinoDevice.NPU
    openvino_enable_profiling: bool = False
    openvino_use_onnx: bool = False


# -------------------------------------------------------------------------------------
# Split model
# -------------------------------------------------------------------------------------


class ModelPartId(str, Enum):
    FULL_MODEL = "FullModel"
    ENCODER = "MLVCEncoder"
    ENCODER_PART1 = "MLVCEncoderPart1"
    ENCODER_PART2 = "MLVCEncoderPart2"
    DECODER = "MLVCDecoder"
    DECODER_PART1 = "MLVCDecoderPart1"
    DECODER_PART2 = "MLVCDecoderPart2"
    DECODER_PART3 = "MLVCDecoderPart3"


ModelSplitType = str


@dataclass(frozen=True)
class SplitModelParams:
    split_type: ModelSplitType
    model_width: int
    model_height: int
    extra_params: dict[str, Any]


@dataclass(frozen=True)
class ModelPartMetadata:
    precision: ModelPrecision
    input_fields: list[str]
    output_fields: list[str]


@dataclass(frozen=True)
class ModelData:
    outputs: Any
    inputs: Any


@dataclass(frozen=True)
class ModelOpProfile:
    operator_type: str
    cost: float
    preferred_compute_device: str

    # OpenVINO
    operator_name: str | None = None
    status: str | None = None
    exec_type: str | None = None
    real_time: float | None = None
    cpu_time: float | None = None

    # CoreML
    output_name: str | None = None
    supported_compute_devices: list[str] | None = None


@dataclass(frozen=True)
class EncoderOutput:
    model_data: dict[ModelPartId, ModelData]
    padding: tuple[int, int, int, int, bool]  # the boolean indicates if the image was transposed
    original_frame: tuple[np.ndarray, np.ndarray, np.ndarray]  # YUV420
    reconstructed_frame: tuple[np.ndarray, np.ndarray, np.ndarray] | None  # YUV420
    bitstream: bytes
    timers: dict[str, float]


@dataclass(frozen=True)
class DecoderOutput:
    model_data: dict[ModelPartId, ModelData]
    reconstructed_frame: tuple[np.ndarray, np.ndarray, np.ndarray]  # YUV420
    timers: dict[str, float]


# -------------------------------------------------------------------------------------
# Frame loop types
# -------------------------------------------------------------------------------------


class PaddingMode(str, Enum):
    EDGE = "edge"
    REFLECT = "reflect"
    CONSTANT_GRAY = "constant_gray"
    CONSTANT_BLACK = "constant_black"
    CONSTANT_WHITE = "constant_white"
    HYBRID = "hybrid"


class PaddingDirection(str, Enum):
    BOTTOM_RIGHT = "bottom_right"
    UPPER_LEFT = "upper_left"
    BOTH = "both"


class FrameType(str, Enum):
    I_FRAME = "i_frame"
    P_FRAME = "p_frame"
    LTR_RECOVERY = "ltr_recovery"


@dataclass(frozen=True)
class FrameLoopParams:
    video_path: str
    image_width: int
    image_height: int
    frame_count: int
    fps: float
    q_index: int | None
    bitrate: float | None
    q_index_overrides: dict[int, int | None]
    iframe_period: int | None
    reset_period: int | None
    ltr_start_idx: int | None
    ltr_period: int | None
    proactive_ltr_recovery: bool
    padding_mode: PaddingMode
    padding_direction: PaddingDirection
    include_bitstream_overhead: bool
    metrics_bit_depth: int
    use_encoder: bool
    use_decoder: bool


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


@dataclass(frozen=True)
class FrameMetrics:
    psnr: float
    psnr_y: float
    psnr_u: float
    psnr_v: float
    bpp: float


@dataclass(frozen=True)
class FrameLoopFrameResult:
    presentation_time: float
    frame_type: FrameType
    frame_idx: int
    ref_frame_idx: int | None
    q_index: int
    feature_reset: bool
    marked_as_ltr: bool
    encoder: EncoderOutput | None
    decoder: DecoderOutput | None
    rate_control_info: RateControlInfo | None
    metrics: FrameMetrics
    timers: dict[str, float]
    op_profiles: dict[ModelPartId, list[ModelOpProfile]]

    @property
    def all_model_data(self) -> dict[ModelPartId, ModelData]:
        return {
            **(self.encoder.model_data if self.encoder else {}),
            **(self.decoder.model_data if self.decoder else {}),
        }

    @property
    def all_timers(self) -> dict[str, float]:
        return {
            **(self.encoder.timers if self.encoder else {}),
            **(self.decoder.timers if self.decoder else {}),
            **self.timers,
        }


@dataclass(frozen=True)
class FrameLoopResults:
    params: FrameLoopParams
    frames: list[FrameLoopFrameResult]


@dataclass(frozen=True)
class AggregatedMetric:
    mean: float
    count: int
    std: float
    median: float
    min: float
    max: float
    values: list[float | int] = dataclasses.field(default_factory=list)

    @classmethod
    def from_array(
        cls,
        arr: np.ndarray | list,
        digits: int | None = None,
        save_values: bool = False,
    ) -> "AggregatedMetric":
        def _round(x: float | int) -> float | int:
            if isinstance(x, (int, np.integer)):
                return int(x)
            if digits is None:
                return float(x)
            return float(round(x, digits))

        if len(arr) == 0:
            return cls(mean=np.nan, std=np.nan, median=np.nan, min=np.nan, max=np.nan, count=0)

        values = np.array(arr)
        return cls(
            mean=_round(np.mean(values).item()),
            std=_round(np.std(values).item()),
            median=_round(np.median(values).item()),
            min=_round(np.min(values)),
            max=_round(np.max(values)),
            count=len(values),
            values=[_round(v) for v in values] if save_values else [],
        )


@dataclass(frozen=True)
class FrameLoopSummary:
    q_index: AggregatedMetric
    psnr: AggregatedMetric
    psnr_y: AggregatedMetric
    psnr_u: AggregatedMetric
    psnr_v: AggregatedMetric
    bpp: AggregatedMetric
    timers: dict[str, AggregatedMetric]


# -------------------------------------------------------------------------------------
# Exporter
# -------------------------------------------------------------------------------------


class TargetDevice(str, Enum):
    GENERIC = "generic"
    APPLE = "apple"
    INTEL = "intel"
    QUALCOMM = "qualcomm"


class ScaleDecoderType(str, Enum):
    BUILTIN = "builtin"
    UPSAMPLE = "upsample"


@dataclass(frozen=True)
class ExporterParams:
    model_type: ModelType
    target_device: TargetDevice
    precision: ModelPrecision
    scale_decoder_type: str
    test_video_path: str
    image_width: int
    image_height: int
    frame_count: int
    output_path: str
    output_name: str | None
    extra_params: dict[str, Any]


@dataclass(frozen=True)
class ConversionParams:
    full_model_params: ModelParams
    split_model_params: SplitModelParams
    runtime_params: RuntimeParams
    exporter_params: ExporterParams


@dataclass
class ConversionMetadata(DataClassJsonMixin):
    name: str
    timestamp: str
    git_hash: str
    platform: str
    platform_version: str
    params: ConversionParams
    model_parts_metadata: dict[ModelPartId, ModelPartMetadata]
    conversion_loop_params: list[FrameLoopParams]
    conversion_loop_results: list[FrameLoopSummary]

    def save(self, model_path: Path | str):
        dest_file = Path(model_path) / "metadata.json"
        print(f"Saving metadata to {dest_file}")
        with open(dest_file, "w") as f:
            json.dump(dataclasses.asdict(self), f)

    @classmethod
    def load(cls, model_path: str | Path, filename: str | None = None) -> "ConversionMetadata":
        with open(Path(model_path) / (filename or "metadata.json"), "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# -------------------------------------------------------------------------------------
# Model bundle
# -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MlvcVersion:
    major: int
    minor: int

    def __str__(self):
        return f"v{self.major}.{self.minor}"


@dataclass(frozen=True)
class RegistryEntry:
    model_path: str
    weights_path: str | None
    sha256: str


class EncoderInterfaceType(str, Enum):
    FP16_SCALE_SENDING_NO_RESET_1P = "fp16_scale_sending_no_reset_1p"


class DecoderInterfaceType(str, Enum):
    FP16_SCALE_SENDING_NO_RESET_1P = "fp16_scale_sending_no_reset_1p"


@dataclass(frozen=True)
class ModelPartManifest:
    registry_id: str
    function_name: str | None


@dataclass(frozen=True)
class ModelManifest:
    metadata_path: str
    scale_decoder_model_path: str | None
    scale_decoder_weights_path: str | None
    gaussian_pmf_path: str
    bit_estimator_pmf_path: str
    model_parts: dict[ModelPartId, ModelPartManifest]  # model_part_id -> ModelPartManifest
    # TODO: support other interface types
    encoder_interface_type: EncoderInterfaceType = EncoderInterfaceType.FP16_SCALE_SENDING_NO_RESET_1P
    decoder_interface_type: DecoderInterfaceType = DecoderInterfaceType.FP16_SCALE_SENDING_NO_RESET_1P


@dataclass(frozen=True)
class ModelMetadata:
    model_width: int
    model_height: int
    pixel_range: float
    qp_num: int
    total_qp_num: int
    frame_index_map: list[int]
    qp_shift: list[int]
    feature_channels: int
    latent_channels: int
    hyperprior_channels: int
    downsample_feature: int
    downsample_latent: int
    downsample_hyperprior: int
    scale_decoder_type: str
    y_scale_repeat: int | None
    iframe_period: int | None
    reset_period: int | None
    ltr_start_idx: int
    ltr_period: int | None
    qp_mapping: list[int] | None


@dataclass(frozen=True)
class BundleManifest(DataClassJsonMixin):
    mlvc_version: MlvcVersion
    bundle_name: str
    timestamp: str
    model_type: ModelType
    target_device: TargetDevice
    model_registry: dict[str, RegistryEntry]  # registry_id -> RegistryEntry
    model_manifests: dict[str, ModelManifest]  # model_id -> ModelManifest
    model_metadata: dict[str, ModelMetadata]  # model_id -> ModelMetadata
    extra_params: dict[str, Any]

    def save(self, model_path: Path | str):
        dest_file = Path(model_path) / "bundle_manifest.json"
        print(f"Saving metadata to {dest_file}")
        with open(dest_file, "w") as f:
            json.dump(dataclasses.asdict(self), f)

    @classmethod
    def load(cls, model_path: str | Path) -> "BundleManifest":
        with open(Path(model_path) / "bundle_manifest.json", "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# -------------------------------------------------------------------------------------
# Powermetrics types
# -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PowermetricsSample:
    timestamp: float
    context: str
    gpu_power: float
    npu_power: float
    cpu_power: float
    combined_power: float
    raw_data: dict[str, Any]


@dataclass(frozen=True)
class PowermetricsStats:
    gpu_power: AggregatedMetric
    npu_power: AggregatedMetric
    cpu_power: AggregatedMetric
    combined_power: AggregatedMetric

    @classmethod
    def from_samples(cls, context: str, samples: list[PowermetricsSample]) -> "PowermetricsStats":
        filtered_samples = [row for row in samples if row.context == context]

        # Discard first measurement
        if len(filtered_samples) > 1:
            filtered_samples = filtered_samples[1:]

        gpu_power = AggregatedMetric.from_array([row.gpu_power for row in filtered_samples], digits=3)
        npu_power = AggregatedMetric.from_array([row.npu_power for row in filtered_samples], digits=3)
        cpu_power = AggregatedMetric.from_array([row.cpu_power for row in filtered_samples], digits=3)
        combined_power = AggregatedMetric.from_array([row.combined_power for row in filtered_samples], digits=3)
        return cls(
            gpu_power=gpu_power,
            npu_power=npu_power,
            cpu_power=cpu_power,
            combined_power=combined_power,
        )


@dataclass(frozen=True)
class PowermetricsData:
    samples: list[PowermetricsSample]
    stats: dict[str, PowermetricsStats]


# -------------------------------------------------------------------------------------
# Model Profile
# -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelOpStats:
    num_ops_cpu: int
    num_ops_gpu: int
    num_ops_npu: int
    cost_cpu: float
    cost_gpu: float
    cost_npu: float

    @property
    def total_ops(self):
        return self.num_ops_cpu + self.num_ops_gpu + self.num_ops_npu

    @property
    def total_cost(self):
        return self.cost_cpu + self.cost_gpu + self.cost_npu


@dataclass(frozen=True)
class ModelProfile:
    operations: list[ModelOpProfile]
    op_stats: ModelOpStats


# -------------------------------------------------------------------------------------
# Model tester
# -------------------------------------------------------------------------------------


@dataclass
class ValidateConversionResults(DataClassJsonMixin):
    params: list[FrameLoopParams]
    azureml_results: list[FrameLoopSummary]
    conversion_results: list[FrameLoopSummary]
    validation_results: list[FrameLoopSummary]


@dataclass
class ProfileResults(DataClassJsonMixin):
    profiles: dict[ModelPartId, ModelProfile]


@dataclass
class BenchmarkResults(DataClassJsonMixin):
    frame_loop_params: FrameLoopParams
    frame_loop_summary: FrameLoopSummary
    powermetrics_data: PowermetricsData


@dataclass
class ValidationTestResults(DataClassJsonMixin):
    params: list[FrameLoopParams]
    results: list[FrameLoopSummary]
