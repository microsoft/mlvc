# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for model factory - verify models can be instantiated."""

import pytest

from src.utils.model_factory import create_image_model, create_video_model


IMAGE_MODELS = [
    {"type": "DMCI-6.0", "N": 128, "z_channel": 64},
]

VIDEO_MODELS = [
    {"type": "DMC-6.1"},
    {"type": "DMC-6.1s"},
    {"type": "DMC-6.1sb"},
]


class TestImageModels:
    """Test image model instantiation."""

    @pytest.mark.parametrize("config", IMAGE_MODELS, ids=lambda c: c["type"])
    def test_create_image_model(self, config):
        """Test image model can be instantiated with default params."""
        model = create_image_model(config)
        assert model is not None


class TestVideoModels:
    """Test video model instantiation."""

    @pytest.mark.parametrize("config", VIDEO_MODELS, ids=lambda c: c["type"])
    def test_create_video_model(self, config):
        """Test video model can be instantiated with default params."""
        model = create_video_model(config)
        assert model is not None


class TestUnknownModel:
    """Test error handling for unknown models."""

    def test_unknown_image_model(self):
        """Test that unknown image model raises error."""
        config = {"type": "UNKNOWN-MODEL"}
        with pytest.raises(ValueError, match="Unknown model type"):
            create_image_model(config)

    def test_unknown_video_model(self):
        """Test that unknown video model raises error."""
        config = {"type": "UNKNOWN-MODEL"}
        with pytest.raises(ValueError, match="Unknown model type"):
            create_video_model(config)
