# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
from pathlib import Path
from ..types import (
    RuntimeParams,
    BundleManifest,
    ModelManifest,
    ConversionMetadata,
    GaussianCoderPmf,
    BitEstimatorPmf,
    ScaleDecoderType,
)
from .._full_model import BaseFullModel, ConfigOnlyFullModel
from .._full_model._model_factory import get_model_config
from ._base_split_model import BaseSplitModel
from .._model_wrapper import ModelWrapper
from .._scale_decoder import scale_decoder_factory
from ..utils import get_model_type_extension
from src.utils.stream_helper import get_downsampled_shape


def split_full_model(
    full_model: BaseFullModel,
    split_type: str | None = None,
    runtime_params: RuntimeParams = RuntimeParams(),
    model_width: int = 640,
    model_height: int = 368,
    use_encoder: bool = True,
    use_decoder: bool = True,
    conversion_metadata: ConversionMetadata | None = None,
    **kwargs,
) -> BaseSplitModel:

    downsample_latent = full_model.model_params.downsample_latent
    if model_width % downsample_latent != 0:
        raise ValueError(f"Input width {model_width} must be a multiple of {downsample_latent}")
    if model_height % downsample_latent != 0:
        raise ValueError(f"Input height {model_height} must be a multiple of {downsample_latent}")

    if split_type is None:
        model_version = full_model.model_params.model_version
        try:
            config = get_model_config(model_version)
            split_type = config.get("split_type")
            if split_type is None:
                raise ValueError(f"No split_type defined in config for model version '{model_version}'")
        except (ValueError, FileNotFoundError) as e:
            raise ValueError(f"Cannot determine split type for model version '{model_version}': {e}")

    try:
        split_module = importlib.import_module(f"._{split_type}", package=__package__)
        SplitModel = split_module.SplitModel
    except ImportError as e:
        raise NotImplementedError(
            f"Split type '{split_type}' is not available. The module '_{split_type}' could not be imported: {e}"
        )

    model_parts = kwargs.pop("model_parts", None)
    scale_decoder = kwargs.pop("scale_decoder", None)
    return SplitModel(
        full_model=full_model,
        model_parts=model_parts,
        scale_decoder=scale_decoder,
        runtime_params=runtime_params,
        model_width=model_width,
        model_height=model_height,
        use_encoder=use_encoder,
        use_decoder=use_decoder,
        conversion_metadata=conversion_metadata,
        **kwargs,
    )


def load_split_model(
    model_path: Path | str,
    runtime_params: RuntimeParams = RuntimeParams(),
    use_encoder: bool = True,
    use_decoder: bool = True,
    model_id: str | None = None,
    **kwargs,
) -> BaseSplitModel:

    manifest: BundleManifest | None = None
    model_manifest: ModelManifest | None = None
    if (Path(model_path) / "bundle_manifest.json").exists():
        if model_id is None:
            raise ValueError("model_id must be specified when loading from a bundle")

        manifest = BundleManifest.load(model_path)
        print(f"Available bundled models: {', '.join(list(manifest.model_manifests.keys()))}")
        if model_id not in manifest.model_manifests:
            raise ValueError(f"Bundled model ID {model_id} not found")
        model_manifest = manifest.model_manifests[model_id]

    conversion_metadata = ConversionMetadata.load(
        model_path,
        filename=model_manifest.metadata_path if model_manifest else None,
    )
    gaussian_coder_pmf = GaussianCoderPmf.load(
        model_path,
        filename=model_manifest.gaussian_pmf_path if model_manifest else None,
    )
    bit_estimator_pmf = BitEstimatorPmf.load(
        model_path,
        filename=model_manifest.bit_estimator_pmf_path if model_manifest else None,
    )

    full_model = ConfigOnlyFullModel(
        model_params=conversion_metadata.params.full_model_params,
        gaussian_coder_pmf=gaussian_coder_pmf,
        bit_estimator_pmf=bit_estimator_pmf,
    )

    model_type = conversion_metadata.params.exporter_params.model_type
    extension = get_model_type_extension(model_type)

    model_parts = {}
    for model_part_id, model_part_metadata in conversion_metadata.model_parts_metadata.items():
        if not use_encoder and "encoder" in model_part_id.value.lower():
            continue
        if not use_decoder and "decoder" in model_part_id.value.lower():
            continue

        if manifest is not None and model_manifest is not None:
            if model_part_id not in model_manifest.model_parts:
                raise ValueError(f"Model part {model_part_id} not found in bundled model")
            bundled_model_part = model_manifest.model_parts[model_part_id]
            registry_entry = manifest.model_registry[bundled_model_part.registry_id]
            model_part_path = Path(model_path) / registry_entry.model_path
            function_name = bundled_model_part.function_name
        else:
            model_part_path = Path(model_path) / f"{model_part_id.value}.{extension}"
            function_name = None

        model_parts[model_part_id] = ModelWrapper.load_converted_model(
            model_type,
            model_part_path,
            metadata=model_part_metadata,
            runtime_params=runtime_params,
            function_name=function_name,
        )

    model_params = conversion_metadata.params.full_model_params
    split_model_params = conversion_metadata.params.split_model_params

    if conversion_metadata.params.exporter_params.scale_decoder_type != ScaleDecoderType.BUILTIN:
        y_height, y_width = get_downsampled_shape(
            split_model_params.model_height,
            split_model_params.model_width,
            model_params.downsample_latent,
        )
        scale_decoder = scale_decoder_factory(
            type=conversion_metadata.params.exporter_params.scale_decoder_type,
            model_path=model_path,
            y_shape=(model_params.latent_channels, y_height, y_width),
            index_space=full_model.gaussian_coder_pmf.index_space,
            scale_max_idx=full_model.gaussian_coder_pmf.scale_levels - 1,
            channel_repeat=model_params.y_scale_repeat,
            spatial_repeat=model_params.downsample_hyperprior,
            filename=model_manifest.scale_decoder_model_path if model_manifest else None,
        )
    else:
        scale_decoder = None

    return split_full_model(
        full_model=full_model,
        split_type=split_model_params.split_type,
        model_parts=model_parts,
        scale_decoder=scale_decoder,
        runtime_params=runtime_params,
        model_width=split_model_params.model_width,
        model_height=split_model_params.model_height,
        use_encoder=use_encoder,
        use_decoder=use_decoder,
        conversion_metadata=conversion_metadata,
        **split_model_params.extra_params,
        **kwargs,
    )
