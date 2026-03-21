"""
Tests for NormPreservingProjector
"""

import pytest
import torch
import torch.nn as nn
from aetheris.core.projector import NormPreservingProjector, ProjectionResult


class MockLinearModel(nn.Module):
    """Mock model with linear layers for testing."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(768, 768)
        self.layer2 = nn.Linear(768, 768)
        self.layer3 = nn.Linear(768, 768)


class TestNormPreservingProjector:
    """Test suite for NormPreservingProjector."""

    @pytest.fixture
    def model(self):
        return MockLinearModel()

    @pytest.fixture
    def directions(self, sample_directions):
        return sample_directions

    def test_initialization(self, model):
        """Test projector initialization."""
        projector = NormPreservingProjector(model)
        assert projector.model is model
        assert projector.preserve_norm is True

    def test_project_weights(self, model, directions):
        """Test weight projection."""
        projector = NormPreservingProjector(model)

        result = projector.project_weights(directions)

        assert isinstance(result, ProjectionResult)
        assert result.success is True
        assert len(result.layers_modified) > 0

    def test_project_biases(self, model, directions):
        """Test bias projection."""
        projector = NormPreservingProjector(model)

        result = projector.project_biases(directions)

        assert isinstance(result, ProjectionResult)
        assert result.success is True

    def test_rollback(self, model, directions):
        """Test rollback functionality."""
        projector = NormPreservingProjector(model)

        # Store original weights
        original_weights = []
        for name, param in model.named_parameters():
            if "weight" in name:
                original_weights.append(param.data.clone())

        # Project
        projector.project_weights(directions)

        # Rollback
        projector.rollback()

        # Verify weights restored
        for idx, (name, param) in enumerate(model.named_parameters()):
            if "weight" in name:
                assert torch.allclose(param.data, original_weights[idx])

    def test_multi_direction_projection(self, model, directions):
        """Test multi-direction projection."""
        projector = NormPreservingProjector(model)

        result = projector.multi_direction_projection(directions)

        assert result.success is True

    def test_extract_layer_idx(self, model):
        """Test layer index extraction."""
        projector = NormPreservingProjector(model)

        idx = projector._extract_layer_idx("model.layer1.weight")
        # Will return -1 if no match (expected)
        assert isinstance(idx, int)
