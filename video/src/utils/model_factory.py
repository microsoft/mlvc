# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Any

__all__ = ["create_image_model", "create_video_model"]


def create_image_model(config: Dict[str, Any]):
    model_type = config["type"]
    if model_type == "DMCI-6.0":
        from src.models.dmc_6.dmci_60 import DMCI

        model = DMCI(
            N=config["N"],
            z_channel=config["z_channel"],
            activation=config.get("activation", "WSiLU"),
        )
    else:
        try:
            from . import model_factory_ext

            return model_factory_ext.create_image_model(config)
        except ImportError:
            raise ValueError(f"Unknown model type: {model_type}")

    return model


def create_video_model(config: Dict[str, Any]):
    model_type = config["type"]
    if model_type == "DMC-6.1":
        from src.models.dmc_6.dmc_61 import DMC

        model = DMC(
            qp_num=config.get("qp_num", 64),
            activation=config.get("activation", "WSiLU"),
        )
    elif model_type == "DMC-6.1s":
        from src.models.dmc_6.dmc_61s import DMC

        model = DMC(
            qp_num=config.get("qp_num", 64),
            hidden_channels=config.get("hidden_channels", 256),
            feature_channels=config.get("feature_channels", 256),
            recon_channels=config.get("recon_channels", 256),
            z_channels=config.get("z_channels", 128),
            y_channels=config.get("y_channels", 128),
            spatial_prior_channels=config.get("spatial_prior_channels", 128 * 3),
            depth_conv_block_params={
                "activation": config.get("activation", "WSiLU"),
                "zero_init_residual": config.get("zero_init_residual", False),
            },
            input_offset=config.get("input_offset"),
            network_mode=config.get("network_mode", "fp32"),
        )
    elif model_type == "DMC-6.1sb":
        from src.models.dmc_6.dmc_61sb import DMC

        model = DMC(
            qp_num=config.get("qp_num", 64),
            qp_shift=config.get("qp_shift", (0, 8, 4)),
            hidden_channels=config.get("hidden_channels", 256),
            feature_channels=config.get("feature_channels", 256),
            recon_channels=config.get("recon_channels", 256),
            z_channels=config.get("z_channels", 128),
            y_channels=config.get("y_channels", 128),
            y_scale_repeat=config.get("y_scale_repeat", 2),
            spatial_prior_channels=config.get("spatial_prior_channels", 128 * 3),
            prior_fusion_channels=config.get("prior_fusion_channels"),
            hyperprior_num_blocks=config.get("hyperprior_num_blocks", 3),
            pixel_shuffle_factor=config.get("pixel_shuffle_factor", 8),
            depth_conv_block_params={
                "activation": config.get("activation", "WSiLU"),
                "zero_init_residual": config.get("zero_init_residual", False),
                "chunk_mode": config.get("chunk_mode", "split"),
                "ffn_gate_activation": config.get("ffn_gate_activation"),
                "ffn_channel_multiplier": config.get("ffn_channel_multiplier", 4),
                "dc_channel_multiplier": config.get("dc_channel_multiplier", 1.0),
            },
            input_offset=config.get("input_offset"),
            memory_activation=config.get("memory_activation", "tanh"),
            gate_activation=config.get("gate_activation", "sigmoid"),
            chain_feature_adaptors=config.get("chain_feature_adaptors", False),
            hyperprior_variant=config.get("hyperprior_variant", "full"),
            feature_extractor_num_conv1_layers=config.get("feature_extractor_num_conv1_layers", 2),
            feature_extractor_num_conv2_layers=config.get("feature_extractor_num_conv2_layers", 4),
        )
    else:
        try:
            from . import model_factory_ext

            return model_factory_ext.create_video_model(config)
        except ImportError:
            raise ValueError(f"Unknown model type: {model_type}")

    return model
