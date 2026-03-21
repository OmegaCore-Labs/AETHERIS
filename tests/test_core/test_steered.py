"""
Tests for Steering Vector Components
"""

import pytest
import torch
from aetheris.core.steered import (
    SteeringVectorFactory,
    SteeringHookManager,
    SteeringConfig,
    MultiSteeringController
)


class TestSteeringVectorFactory:
    """Test suite for SteeringVectorFactory."""

    def test_from_refusal_direction(self, sample_directions):
        """Test creating steering vector from refusal direction."""
        direction = sample_directions[0]
        vector = SteeringVectorFactory.from_refusal_direction(direction, alpha=-1.0)

        assert isinstance(vector, torch.Tensor)
        assert torch.norm(vector) > 0

    def test_from_contrastive_pairs(self, sample_activations):
        """Test creating steering vector from contrastive pairs."""
        harmful, harmless = sample_activations
        vector = SteeringVectorFactory.from_contrastive_pairs(harmful, harmless, alpha=1.0)

        assert isinstance(vector, torch.Tensor)
        assert vector.shape == harmful.shape[1:]


class TestSteeringHookManager:
    """Test suite for SteeringHookManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = SteeringHookManager()
        assert manager._active is False
        assert manager._hooks == []

    def test_install_remove(self, mock_model):
        """Test install and remove hooks."""
        manager = SteeringHookManager()

        config = SteeringConfig(
            vectors=[torch.randn(768)],
            target_layers=[0, 1, 2]
        )

        manager.install(mock_model, config)
        assert manager._active is True
        assert len(manager._hooks) > 0

        manager.remove()
        assert manager._active is False
        assert manager._hooks == []

    def test_temporary_steering(self, mock_model):
        """Test temporary steering context manager."""
        manager = SteeringHookManager()

        config = SteeringConfig(
            vectors=[torch.randn(768)],
            target_layers=[0, 1, 2]
        )

        with manager.temporary_steering(mock_model, config):
            assert manager._active is True

        assert manager._active is False


class TestMultiSteeringController:
    """Test suite for MultiSteeringController."""

    def test_initialization(self):
        """Test controller initialization."""
        manager = SteeringHookManager()
        controller = MultiSteeringController(manager)
        assert controller.vectors == []

    def test_add_vector(self):
        """Test adding vectors."""
        manager = SteeringHookManager()
        controller = MultiSteeringController(manager)

        vector = torch.randn(768)
        controller.add_vector(vector, alpha=1.0)

        assert len(controller.vectors) == 1

    def test_create_config(self):
        """Test config creation."""
        manager = SteeringHookManager()
        controller = MultiSteeringController(manager)

        vector1 = torch.randn(768)
        vector2 = torch.randn(768)
        controller.add_vector(vector1, alpha=1.0)
        controller.add_vector(vector2, alpha=0.5)

        config = controller.create_config()

        assert isinstance(config, SteeringConfig)
        assert len(config.vectors) == 1  # Combined vector
