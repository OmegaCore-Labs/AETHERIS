"""
Tests for GeometryAnalyzer
"""

import pytest
import torch
from aetheris.core.geometry import GeometryAnalyzer, GeometryReport


class TestGeometryAnalyzer:
    """Test suite for GeometryAnalyzer."""

    def test_initialization(self, device):
        """Test analyzer initialization."""
        analyzer = GeometryAnalyzer(device=device)
        assert analyzer.device == device

    def test_cross_layer_alignment(self, device, sample_directions_by_layer):
        """Test cross-layer alignment computation."""
        analyzer = GeometryAnalyzer(device=device)

        alignment = analyzer.cross_layer_alignment(sample_directions_by_layer)

        assert isinstance(alignment, dict)
        assert len(alignment) == len(sample_directions_by_layer)

        # Check diagonal is 1.0
        for layer in sample_directions_by_layer.keys():
            assert abs(alignment[layer][layer] - 1.0) < 1e-6

    def test_solid_angle(self, device, sample_directions):
        """Test solid angle computation."""
        analyzer = GeometryAnalyzer(device=device)

        solid_angle = analyzer.solid_angle(sample_directions)

        assert isinstance(solid_angle, float)
        assert solid_angle >= 0
        assert solid_angle <= 2 * 3.14159  # Max solid angle

    def test_concept_cone_geometry(self, device):
        """Test concept cone geometry analysis."""
        analyzer = GeometryAnalyzer(device=device)

        concept_dirs = {
            "refusal": [torch.randn(768) for _ in range(3)],
            "safety": [torch.randn(768) for _ in range(2)]
        }

        result = analyzer.concept_cone_geometry(concept_dirs)

        assert "refusal" in result
        assert "safety" in result
        assert "structure" in result["refusal"]
        assert "solid_angle" in result["refusal"]

    def test_direction_clustering(self, device, sample_directions):
        """Test direction clustering."""
        analyzer = GeometryAnalyzer(device=device)

        clusters = analyzer.direction_clustering(sample_directions)

        assert isinstance(clusters, list)
        if clusters:
            assert "indices" in clusters[0]
            assert "centroid" in clusters[0]
            assert "size" in clusters[0]
