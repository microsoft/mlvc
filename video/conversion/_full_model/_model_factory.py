# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import re
import yaml
from pathlib import Path
from ..const import DEFAULT_JOB_OUTPUTS_DIR
from .._azure import get_azureml_job
from ._base_model import BaseFullModel

_CONFIG_PATH = Path(__file__).parent / "model_configs.yaml"

_config_cache: dict | None = None


def _load_configs() -> dict:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"model_configs.yaml not found at {_CONFIG_PATH}.\n"
            f"Copy model_configs_example.yaml to model_configs.yaml and configure your weights paths."
        )

    with open(_CONFIG_PATH) as f:
        _config_cache = yaml.safe_load(f) or {}

    assert _config_cache is not None
    return _config_cache


def get_model_config(model_version: str) -> dict:
    configs = _load_configs()
    if model_version not in configs:
        available = ", ".join(sorted(configs.keys()))
        raise ValueError(f"Unknown model version: {model_version}. Available: {available}")
    return configs[model_version]


def get_available_models() -> list[str]:
    return sorted(_load_configs().keys())


def get_available_split_types() -> list[str]:
    split_types = set()
    for config in _load_configs().values():
        if st := config.get("split_type"):
            split_types.add(st)
    return sorted(split_types)


def full_model_factory(
    model_version: str,
    weights_version: str | None = None,
    weights_path: str | Path | None = None,
    iframe_period: int | None = None,
    reset_period: int | None = None,
    ltr_start_idx: int | None = None,
    ltr_period: int | None = None,
    job_outputs_path: str | Path = DEFAULT_JOB_OUTPUTS_DIR,
    skip_load_weights: bool = False,
    fx_traceable: bool = False,
    **kwargs,
) -> BaseFullModel:
    config = get_model_config(model_version)

    # Handle fx_traceable special case (config must define fx_traceable_class)
    if fx_traceable:
        fx_class = config.get("fx_traceable_class")
        if fx_class is None:
            raise ValueError(f"fx_traceable is not supported for {model_version}")
        model_class_name = fx_class
    else:
        model_class_name = config["class"]

    module = importlib.import_module(config["module"], package=__package__)
    model_class = getattr(module, model_class_name)

    model_params = config.get("params", {}).copy()
    model_params.update(kwargs)

    # Instantiate model
    model = model_class(model_version=model_version, **model_params)

    # Load weights
    if not skip_load_weights:
        final_weights_path = weights_path or config.get("weights_path")
        final_weights_version = weights_version or config.get("weights_version")
        final_iframe_period = iframe_period if iframe_period is not None else config.get("iframe_period")
        final_reset_period = reset_period if reset_period is not None else config.get("reset_period")
        final_ltr_start_idx = ltr_start_idx if ltr_start_idx is not None else config.get("ltr_start_idx", 0)
        final_ltr_period = ltr_period if ltr_period is not None else config.get("ltr_period")

        if final_weights_path is not None and final_weights_version is None:
            match = re.match(
                r"checkpoints/dcvc-dc/(dcvc-dc_.*)/model_epo(\d+)\.ckpt",
                Path(final_weights_path).as_posix(),
            )
            if match:
                job_name, epoch = match.groups()
                job = get_azureml_job(job_name)
                final_weights_version = f"{job.display_name}_epo{epoch}"
                print(f"Generated weights version: {final_weights_version}")
            else:
                print(f"Warn: Could not extract job name and epoch from {final_weights_path}")

        if final_weights_path is None:
            raise ValueError(
                f"weights_path required for {model_version}. "
                "Either pass it explicitly or configure in model_configs.yaml"
            )

        if final_weights_version is None:
            raise ValueError(
                f"weights_version required for {model_version}. "
                "Either pass it explicitly or configure in model_configs.yaml"
            )

        model.load_weights(
            job_outputs_dir=job_outputs_path,
            weights_version=final_weights_version,
            weights_path=Path(final_weights_path),
            iframe_period=final_iframe_period,
            reset_period=final_reset_period,
            ltr_start_idx=final_ltr_start_idx,
            ltr_period=final_ltr_period,
        )

    # Optimize pytorch model
    model.optimize_structure()

    # Set model to eval mode
    model.eval()

    return model
