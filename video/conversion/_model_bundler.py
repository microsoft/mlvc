# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import onnx
import shutil
import hashlib
import datetime
import tarfile
import coremltools as ct
from abc import ABC, abstractmethod
from pathlib import Path
from .const import DEFAULT_BUNDLE_DIR, get_default_target_device
from ._scale_decoder import SCALE_DECODER_FILENAMES
from .types import (
    ConversionMetadata,
    ModelType,
    TargetDevice,
    ModelPartId,
    MlvcVersion,
    RegistryEntry,
    ModelManifest,
    ModelMetadata,
    ModelPartManifest,
    BundleManifest,
)
from .utils import get_model_type_extension


class BaseBundler(ABC):
    def __init__(
        self,
        bundle_name: str,
        mlvc_version: MlvcVersion,
        model_type: ModelType,
        target_device: TargetDevice | None,
        output_path: Path | str = DEFAULT_BUNDLE_DIR,
        create_archive: bool = True,
        gzip_compress: bool = False,
    ) -> None:
        self._bundle_name = bundle_name
        self._mlvc_version = mlvc_version
        self._model_type = model_type
        self._target_device = target_device or get_default_target_device(model_type)
        self._output_path = Path(output_path)
        self._create_archive = create_archive
        self._gzip_compress = gzip_compress

    def bundle(self, model_paths: list[Path | str]) -> BundleManifest:
        print(f"Bundling {len(model_paths)} models...")

        # Prepare output directory
        bundle_output_dir = (
            self._output_path
            / f"{self._bundle_name}-{self._model_type.value}-{self._target_device.value}-{self._mlvc_version}"
        )
        if bundle_output_dir.exists():
            print(f"Clearing existing output directory: {bundle_output_dir}")
            shutil.rmtree(bundle_output_dir, ignore_errors=True)
        bundle_output_dir.mkdir(parents=True, exist_ok=True)

        # Remove any existing archive
        for suffix in [".tar", ".tar.gz"]:
            archive_path = Path(self._output_path) / f"{bundle_output_dir.name}{suffix}"
            if archive_path.exists():
                print(f"Removing existing archive: {archive_path}")
                archive_path.unlink()

        # Load all models and metadata
        conversion_metadata: dict[str, ConversionMetadata] = {}
        metadata_files: dict[str, str] | dict[str, Path] = {}
        gaussian_pmf_files: dict[str, str] | dict[str, Path] = {}
        bit_estimator_pmf_files: dict[str, str] | dict[str, Path] = {}
        model_files: dict[tuple[str, ModelPartId], str | Path] = {}
        scale_decoder_model_files: dict[str, str] | dict[str, Path] = {}
        scale_decoder_weights_files: dict[str, str] | dict[str, Path] = {}
        extension = get_model_type_extension(self._model_type)

        for model_path in model_paths:
            metadata = ConversionMetadata.load(model_path)
            if self._model_type != metadata.params.exporter_params.model_type:
                raise ValueError("Unexpected model type")
            if self._target_device != metadata.params.exporter_params.target_device:
                raise ValueError("Unexpected target device")

            model_width = metadata.params.split_model_params.model_width
            model_height = metadata.params.split_model_params.model_height
            model_id = f"{model_width}x{model_height}"

            conversion_metadata[model_id] = metadata
            metadata_files[model_id] = Path(model_path) / "metadata.json"
            gaussian_pmf_files[model_id] = Path(model_path) / "gaussian_pmf.json"
            bit_estimator_pmf_files[model_id] = Path(model_path) / "bit_estimator_pmf.json"
            for model_part_id in metadata.model_parts_metadata.keys():
                model_files[model_id, model_part_id] = Path(model_path) / f"{model_part_id.value}.{extension}"

            scale_decoder_file = self._find_scale_decoder_file(Path(model_path))
            if scale_decoder_file is not None:
                scale_decoder_model_files[model_id] = scale_decoder_file
                weights_file = scale_decoder_file.with_suffix(".bin")
                if weights_file.exists():
                    scale_decoder_weights_files[model_id] = weights_file

        # Bundle all files
        metadata_bundle = self._bundle_json(metadata_files, bundle_output_dir)
        gaussian_pmf_bundle = self._bundle_json(gaussian_pmf_files, bundle_output_dir)
        bit_estimator_pmf_bundle = self._bundle_json(bit_estimator_pmf_files, bundle_output_dir)
        scale_decoder_model_bundle = (
            self._bundle_json(scale_decoder_model_files, bundle_output_dir) if scale_decoder_model_files else {}
        )
        scale_decoder_weights_bundle = (
            self._bundle_json(scale_decoder_weights_files, bundle_output_dir) if scale_decoder_weights_files else {}
        )
        model_registry, model_part_manifests = self._bundle_models(model_files, bundle_output_dir)

        # Construct bundled metadata
        model_manifests: dict[str, ModelManifest] = {}
        model_metadata: dict[str, ModelMetadata] = {}
        for model_id, metadata in conversion_metadata.items():
            assert model_id in metadata_bundle
            assert model_id in gaussian_pmf_bundle
            assert model_id in bit_estimator_pmf_bundle

            model_params = metadata.params.full_model_params
            split_model_params = metadata.params.split_model_params

            model_manifests[model_id] = ModelManifest(
                metadata_path=metadata_bundle[model_id].as_posix(),
                scale_decoder_model_path=(
                    scale_decoder_model_bundle[model_id].as_posix() if model_id in scale_decoder_model_bundle else None
                ),
                scale_decoder_weights_path=(
                    scale_decoder_weights_bundle[model_id].as_posix()
                    if model_id in scale_decoder_weights_bundle
                    else None
                ),
                gaussian_pmf_path=gaussian_pmf_bundle[model_id].as_posix(),
                bit_estimator_pmf_path=bit_estimator_pmf_bundle[model_id].as_posix(),
                model_parts={
                    model_part_id: bundled_model_part
                    for (mid, model_part_id), bundled_model_part in model_part_manifests.items()
                    if mid == model_id
                },
            )

            model_metadata[model_id] = ModelMetadata(
                model_width=split_model_params.model_width,
                model_height=split_model_params.model_height,
                pixel_range=model_params.pixel_range,
                qp_num=model_params.qp_num,
                total_qp_num=model_params.total_qp_num,
                frame_index_map=model_params.frame_index_map,
                qp_shift=model_params.qp_shift,
                feature_channels=model_params.feature_channels,
                latent_channels=model_params.latent_channels,
                hyperprior_channels=model_params.hyperprior_channels,
                downsample_feature=model_params.downsample_feature,
                downsample_latent=model_params.downsample_latent,
                downsample_hyperprior=model_params.downsample_hyperprior,
                scale_decoder_type=metadata.params.exporter_params.scale_decoder_type,
                y_scale_repeat=model_params.y_scale_repeat,
                iframe_period=model_params.iframe_period,
                reset_period=model_params.reset_period,
                ltr_start_idx=model_params.ltr_start_idx,
                ltr_period=model_params.ltr_period,
                qp_mapping=model_params.qp_mapping,
            )

        bundle_manifest = self._compose_manifest(model_registry, model_manifests, model_metadata)
        bundle_manifest.save(bundle_output_dir)

        if self._create_archive:
            _create_tar(
                bundle_output_dir,
                gzip_compress=self._gzip_compress,
            )
        return bundle_manifest

    def _compose_manifest(
        self,
        model_registry: dict[str, RegistryEntry],
        model_manifests: dict[str, ModelManifest],
        model_metadata: dict[str, ModelMetadata],
    ) -> BundleManifest:
        return BundleManifest(
            bundle_name=self._bundle_name,
            mlvc_version=self._mlvc_version,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            model_type=self._model_type,
            target_device=self._target_device,
            model_registry=model_registry,
            model_manifests=model_manifests,
            model_metadata=model_metadata,
            extra_params={},
        )

    @staticmethod
    def _find_scale_decoder_file(model_dir: Path) -> Path | None:
        for name in SCALE_DECODER_FILENAMES.values():
            path = model_dir / name
            if path.exists():
                return path
        return None

    def _bundle_json(self, input_files: dict[str, str] | dict[str, Path], output_path: Path | str) -> dict[str, Path]:

        # Check all input files have the same suffix
        suffixes = set([Path(p).suffix for p in input_files.values()])
        if len(suffixes) != 1:
            raise ValueError("All input files must have the same suffix")
        suffix = suffixes.pop()

        # Read and hash all input files
        data_by_hash = {}
        file_data_hashes = {}
        for name, input_file in input_files.items():
            with open(input_file, "rb") as f:
                data_bytes = f.read()
            sha256 = hashlib.sha256(data_bytes).hexdigest()
            if sha256 in data_by_hash and data_by_hash[sha256] != data_bytes:
                raise ValueError(f"Hash collision for {input_file}")
            data_by_hash[sha256] = data_bytes
            file_data_hashes[name] = sha256

        # Save all unique files
        output_file_by_hash = {}

        def _save_file(sha256: str, path: Path) -> None:
            output_file_by_hash[sha256] = path.relative_to(Path(output_path))
            with open(path, "wb") as f:
                f.write(data_by_hash[sha256])

        if len(data_by_hash) == 0:
            pass
        elif len(data_by_hash) == 1:
            stem = Path(list(input_files.values())[0]).stem
            sha256 = list(data_by_hash.keys())[0]
            output_file = Path(output_path) / f"{stem}{suffix}"
            _save_file(sha256, output_file)
        elif len(data_by_hash) == len(input_files):
            for name, input_file in input_files.items():
                sha256 = file_data_hashes[name]
                output_file = Path(output_path) / f"{Path(input_file).stem}_{name}{suffix}"
                _save_file(sha256, output_file)
        else:
            stem = Path(list(input_files.values())[0]).stem
            for sha256, data_bytes in data_by_hash.items():
                output_file = Path(output_path) / f"{stem}{sha256}{suffix}"
                _save_file(sha256, output_file)

        return {name: output_file_by_hash[sha256] for name, sha256 in file_data_hashes.items()}

    @abstractmethod
    def _bundle_models(
        self,
        model_files: dict[tuple[str, ModelPartId], str | Path],
        output_path: Path | str,
    ) -> tuple[dict[str, RegistryEntry], dict[tuple[str, ModelPartId], ModelPartManifest]]:
        pass


class CoreMlBundler(BaseBundler):
    def __init__(
        self,
        *args,
        create_coreml_archive: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._create_coreml_archive = create_coreml_archive

    def _bundle_models(
        self,
        model_files: dict[tuple[str, ModelPartId], str | Path],
        output_path: Path | str,
    ) -> tuple[dict[str, RegistryEntry], dict[tuple[str, ModelPartId], ModelPartManifest]]:

        model_part_manifests = {}
        model_desc = ct.utils.MultiFunctionDescriptor()

        output_model_filename = "bundled_model.mlpackage"
        registry_id = f"{self._bundle_name}-{self._model_type.value}-{self._target_device.value}-{self._mlvc_version}"

        for (model_id, model_part_id), model_path in model_files.items():
            function_name = f"{model_part_id.value}_{model_id}"
            model_desc.add_function(
                str(model_path),
                src_function_name="main",
                target_function_name=function_name,
            )
            model_desc.default_function_name = function_name
            model_part_manifests[model_id, model_part_id] = ModelPartManifest(
                registry_id=registry_id,
                function_name=function_name,
            )

        mlpackage_path = Path(output_path) / output_model_filename
        ct.utils.save_multifunction(model_desc, str(mlpackage_path))

        if self._create_coreml_archive:
            _create_tar(mlpackage_path, include_root_dir=False)
            output_model_filename = output_model_filename + ".tar"
            shutil.rmtree(mlpackage_path, ignore_errors=True)

        model_registry: dict[str, RegistryEntry] = {
            registry_id: RegistryEntry(
                model_path=output_model_filename,
                weights_path=None,
                sha256=_calc_sha256_checksum([Path(output_path) / output_model_filename]),
            )
        }

        return model_registry, model_part_manifests


class OnnxBundler(BaseBundler):
    def __init__(
        self,
        *args,
        size_threshold: int = 64,
        weights_path: str = "weights.bin",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._size_threshold = size_threshold
        self._weights_path = weights_path

    def _bundle_models(
        self,
        model_files: dict[tuple[str, ModelPartId], str | Path],
        output_path: Path | str,
    ) -> tuple[dict[str, RegistryEntry], dict[tuple[str, ModelPartId], ModelPartManifest]]:
        from onnx.external_data_helper import _get_all_tensors, set_external_data

        # Load ONNX models
        onnx_models = {}
        for (model_id, model_part_id), model_path in model_files.items():
            onnx_models[model_id, model_part_id] = onnx.load(model_path)

        # Find shared large tensors
        tensors_by_hash = {}
        for model in onnx_models.values():
            tensors = _get_all_tensors(model)
            for tensor in tensors:
                if not tensor.HasField("raw_data"):
                    continue
                if len(tensor.raw_data) < self._size_threshold:
                    continue

                sha256 = hashlib.sha256(tensor.raw_data).hexdigest()
                if sha256 in tensors_by_hash and tensors_by_hash[sha256][0].raw_data != tensor.raw_data:
                    raise ValueError("Hash collision detected")

                if sha256 not in tensors_by_hash:
                    tensors_by_hash[sha256] = []
                tensors_by_hash[sha256].append(tensor)

        # Save shared tensors to external files
        weights_filepath = Path(output_path) / self._weights_path
        with open(weights_filepath, "wb") as data_file:
            for sha256, tensors in tensors_by_hash.items():
                if len(tensors) == 1:
                    continue
                raw_data = tensors[0].raw_data
                offset = data_file.tell()
                data_file.write(raw_data)

                for tensor in tensors:
                    set_external_data(tensor, self._weights_path, offset, data_file.tell() - offset)
                    tensor.ClearField("raw_data")

        model_registry = {}
        model_part_manifests = {}
        for (model_id, model_part_id), model in onnx_models.items():
            model_filename = f"{model_part_id.value}_{model_id}.onnx"
            model_filepath = Path(output_path) / model_filename
            print(f"Saving model with external data: {model_filepath}")
            onnx.save_model(model, model_filepath)

            registry_id = (
                f"{self._bundle_name}-{self._model_type.value}-{self._target_device.value}-"
                f"{self._mlvc_version}-{model_part_id.value}-{model_id}"
            )
            model_registry[registry_id] = RegistryEntry(
                model_path=model_filename,
                weights_path=self._weights_path,
                sha256=_calc_sha256_checksum([model_filepath, weights_filepath]),
            )
            model_part_manifests[model_id, model_part_id] = ModelPartManifest(
                registry_id=registry_id,
                function_name=None,
            )
        return model_registry, model_part_manifests

    def _compose_manifest(self, *args, **kwargs) -> BundleManifest:
        res = super()._compose_manifest(*args, **kwargs)
        res.extra_params.update(
            {
                "size_threshold": self._size_threshold,
                "weights_path": self._weights_path,
            }
        )
        return res


def _create_tar(
    directory_path: Path | str,
    gzip_compress: bool = False,
    include_root_dir: bool = False,
) -> Path:
    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        raise ValueError(f"Not a directory: {directory_path}")

    suffix = ".tar.gz" if gzip_compress else ".tar"
    mode = "w:gz" if gzip_compress else "w"
    archive_path = directory_path.parent / f"{directory_path.name}{suffix}"
    with tarfile.open(archive_path, mode, format=tarfile.USTAR_FORMAT) as tar:
        if include_root_dir:
            tar.add(directory_path, arcname=directory_path.name, recursive=True)
        else:
            for item in directory_path.rglob("*"):
                arcname = item.relative_to(directory_path)
                tar.add(item, arcname=arcname, recursive=False)
    print(f"Archive created: {archive_path}")
    return archive_path


def _calc_sha256_checksum(file_paths: list[Path | str]) -> str:
    sha256 = hashlib.sha256()
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            sha256.update(f.read())
    return sha256.hexdigest()


def model_bundler_factory(
    model_type: ModelType,
    **kwargs,
) -> BaseBundler:
    if model_type == ModelType.COREML:
        return CoreMlBundler(model_type=model_type, **kwargs)
    elif model_type == ModelType.ONNX:
        return OnnxBundler(model_type=model_type, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
