# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import shutil
import platform
import datetime

import dataclasses
from pathlib import Path
from abc import ABC, abstractmethod
from ..const import (
    DEFAULT_EXPORT_DIR,
    DEFAULT_TEST_DATA_DIR,
    DEFAULT_CONVERT_FRAME_COUNT,
    get_default_test_video,
)
from ..types import (
    ModelPartId,
    ModelData,
    ExporterParams,
    ConversionParams,
    ConversionMetadata,
    ModelType,
    TargetDevice,
    ModelPrecision,
    ScaleDecoderType,
    FrameLoopParams,
    FrameLoopSummary,
)
from .._frame_loop import FrameLoop, aggregate_frame_loop_results
from .._split_model import BaseSplitModel
from ..utils import get_git_revision_hash


class BaseExporter(ABC):
    def __init__(
        self,
        model_type: ModelType,
        target_device: TargetDevice,
        split_model: BaseSplitModel,
        precision: ModelPrecision = ModelPrecision.FP16,
        scale_decoder_type: str | None = None,
        test_data_dir: str = DEFAULT_TEST_DATA_DIR,
        test_video_path: Path | str | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
        frame_count: int = DEFAULT_CONVERT_FRAME_COUNT,
        output_path: Path | str = DEFAULT_EXPORT_DIR,
        output_name: str | None = None,
        skip_if_exists: bool = False,
        q_index_list: list[int] = [21, 63],
    ) -> None:
        print(
            f"Exporting model type: {model_type.value}, target device: {target_device.value}, precision: "
            f"{precision.value}"
        )

        self._model_type = model_type
        self._target_device = target_device
        self._split_model = split_model
        self._precision = precision
        self._scale_decoder_type = scale_decoder_type
        self._test_data_dir = Path(test_data_dir)

        if test_video_path is not None:
            if image_width is None or image_height is None:
                raise ValueError("Image width and height must be specified if test video path is provided")
            self._test_video_path = Path(test_video_path)
            self._image_width = image_width
            self._image_height = image_height
        else:
            self._test_video_path, self._image_width, self._image_height = get_default_test_video(
                split_model.split_model_params.model_width,
                split_model.split_model_params.model_height,
            )

        self._frame_count = frame_count
        self._output_path = Path(output_path)
        self._output_name = output_name
        self._skip_if_exists = skip_if_exists
        self._q_index_list = q_index_list

    def run(self) -> Path:
        # Output name
        if self._output_name is not None:
            output_name = self._output_name
        else:
            model_params = self._split_model.full_model.model_params
            model_version = model_params.model_version
            weights_version = model_params.weights_version
            output_name = f"{model_version}-{weights_version}"
            if self._precision != ModelPrecision.FP16:
                output_name += f"-{self._precision.value}"

        # Model output path
        split_model_params = self._split_model.split_model_params
        model_path = (
            Path(self._output_path)
            / output_name
            / f"{self._model_type.value}-{self._target_device.value}"
            / f"{split_model_params.model_width}x{split_model_params.model_height}"
        )
        if self._skip_if_exists and (model_path / "metadata.json").exists():
            print(f"Model already exists at {model_path}, skipping export")
            return model_path

        if model_path.exists():
            print(f"Deleting existing model at {model_path}")
            shutil.rmtree(model_path, ignore_errors=True)
        model_path.mkdir(parents=True, exist_ok=True)

        # Run conversion loop
        conversion_loop_params: list[FrameLoopParams] = []
        conversion_loop_results: list[FrameLoopSummary] = []
        example_model_data: dict[ModelPartId, list[ModelData]] = {}
        for q_index in self._q_index_list:
            conversion_loop = FrameLoop(
                split_model=self._split_model,
                video_path=self._test_video_path,
                image_width=self._image_width,
                image_height=self._image_height,
                frame_count=self._frame_count,
                test_data_dir=self._test_data_dir,
            )
            loop_results = conversion_loop.run(q_index=q_index, output_model_data=True)
            conversion_loop_params.append(loop_results.params)
            conversion_loop_results.append(aggregate_frame_loop_results(loop_results))

            for frame in loop_results.frames:
                for model_part_id, model_data in frame.all_model_data.items():
                    if model_part_id not in example_model_data:
                        example_model_data[model_part_id] = []
                    example_model_data[model_part_id].append(model_data)

        # Export model parts
        full_model = self._split_model.full_model
        for model_part_id, model_part in self._split_model.model_parts.items():
            print(f"Exporting {model_part_id.value}...")
            self._export(
                model_name=model_part_id.value,
                model=model_part.torch_model,
                example_model_data=example_model_data[model_part_id],
                output_path=model_path,
                fake_quantized=full_model.model_params.fake_quantized,
            )

        # Export scale decoder
        self._export_scale_decoder(model_path)

        # Save auxiliary data (PMF tables, etc.)
        full_model.save_auxiliary_data(model_path)

        # Save metadata
        conversion_metadata = self._compose_metadata(
            output_name=output_name,
            conversion_loop_params=conversion_loop_params,
            conversion_loop_results=conversion_loop_results,
        )
        conversion_metadata.save(model_path)
        return model_path

    @abstractmethod
    def _export(
        self,
        model_name: str,
        model: torch.nn.Module,
        example_model_data: list[ModelData],
        output_path: Path,
        fake_quantized: bool = False,
    ) -> None:
        pass

    def _export_scale_decoder(self, model_path: Path) -> None:
        if self._split_model.scale_decoder is None:
            # Model without scale decoder
            if self._scale_decoder_type not in (None, ScaleDecoderType.BUILTIN):
                raise ValueError(f"Scale decoder specified ({self._scale_decoder_type}) but no scale decoder")
            self._scale_decoder_type = ScaleDecoderType.BUILTIN
            return
        elif self._split_model.scale_decoder.type == ScaleDecoderType.UPSAMPLE:
            # Upsample scale decoder
            if self._scale_decoder_type is None:
                self._scale_decoder_type = ScaleDecoderType.UPSAMPLE
            if self._scale_decoder_type != ScaleDecoderType.UPSAMPLE:
                raise ValueError(
                    f"Scale decoder type specified ({self._scale_decoder_type}) does not match actual scale "
                    f"decoder type (upsample)"
                )
        else:
            # Other scale decoder types are handled by an optional extension
            try:
                from .._scale_decoder_ext import export_scale_decoder_ext
            except ImportError as e:
                raise ValueError(f"Unsupported scale decoder type: {self._split_model.scale_decoder.type}") from e

            self._scale_decoder_type = export_scale_decoder_ext(self, model_path)

    def _compose_metadata(
        self,
        output_name: str,
        conversion_loop_params: list[FrameLoopParams] = [],
        conversion_loop_results: list[FrameLoopSummary] = [],
    ) -> ConversionMetadata:
        split_model = self._split_model

        model_parts_metadata = {}
        for model_part_id, model_part in split_model.model_parts.items():
            model_parts_metadata[model_part_id] = dataclasses.replace(model_part.metadata, precision=self._precision)

        assert self._scale_decoder_type is not None, "Scale decoder type must be resolved before composing metadata"

        return ConversionMetadata(
            name=output_name,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            git_hash=get_git_revision_hash(),
            platform=platform.platform(),
            platform_version=platform.version(),
            params=ConversionParams(
                full_model_params=split_model.full_model.model_params,
                split_model_params=split_model.split_model_params,
                runtime_params=split_model.runtime_params,
                exporter_params=ExporterParams(
                    model_type=self._model_type,
                    target_device=self._target_device,
                    precision=self._precision,
                    scale_decoder_type=self._scale_decoder_type,
                    test_video_path=self._test_video_path.as_posix(),
                    image_width=self._image_width,
                    image_height=self._image_height,
                    frame_count=self._frame_count,
                    output_path=self._output_path.as_posix(),
                    output_name=self._output_name,
                    extra_params={},
                ),
            ),
            model_parts_metadata=model_parts_metadata,
            conversion_loop_params=conversion_loop_params,
            conversion_loop_results=conversion_loop_results,
        )
