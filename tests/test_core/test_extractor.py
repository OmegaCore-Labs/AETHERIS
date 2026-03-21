"""
Tests for ConstraintExtractor
"""

import pytest
import torch
from aetheris.core.extractor import ConstraintExtractor, ExtractionResult


class TestConstraintExtractor:
    """Test suite for ConstraintExtractor."""

    def test_initialization(self, device):
        """Test extractor initialization."""
        extractor = ConstraintExtractor(device=device)
        assert extractor.device == device
        assert extractor.model is None

    def test_extract_mean_difference(self, device, sample_activations):
        """Test mean difference extraction."""
        harmful, harmless = sample_activations
        extractor = ConstraintExtractor(device=device)

        result = extractor.extract_mean_difference(harmful, harmless)

        assert isinstance(result, ExtractionResult)
        assert len(result.directions) == 1
        assert result.method == "mean_difference"
        assert result.rank == 1

    def test_extract_svd(self, device, sample_activations):
        """Test SVD extraction."""
        harmful, harmless = sample_activations
        extractor = ConstraintExtractor(device=device)

        result = extractor.extract_svd(harmful, harmless, n_directions=3)

        assert isinstance(result, ExtractionResult)
        assert len(result.directions) == 3
        assert result.method == "svd"
        assert len(result.explained_variance) == 3

    def test_detect_polyhedral_structure(self, device, sample_directions):
        """Test polyhedral structure detection."""
        extractor = ConstraintExtractor(device=device)

        result = extractor.detect_polyhedral_structure(sample_directions)

        assert "structure" in result
        assert "n_mechanisms" in result
        assert "angles" in result

    def test_detect_linear_structure(self, device):
        """Test linear structure detection."""
        extractor = ConstraintExtractor(device=device)
        directions = [torch.ones(768), torch.ones(768) * 0.9]

        result = extractor.detect_polyhedral_structure(directions)

        assert result["structure"] == "linear"
        assert result["n_mechanisms"] == 1
