"""
Representation Engineer

Applies representation engineering techniques to analyze
and modify refusal representations.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RepresentationReport:
    """Container for representation analysis."""
    principal_components: List[torch.Tensor]
    explained_variance: List[float]
    representation_rank: int
    intervention_vectors: Dict[str, torch.Tensor]


class RepresentationEngineer:
    """
    Apply representation engineering techniques.

    Based on representation engineering literature:
    - Principal component analysis
    - Representation steering
    - Concept erasure
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def extract_principal_components(
        self,
        activations: torch.Tensor,
        n_components: int = 10
    ) -> RepresentationReport:
        """
        Extract principal components of activation space.

        Args:
            activations: Activation tensor (n_samples, hidden_dim)
            n_components: Number of components to extract

        Returns:
            RepresentationReport with components
        """
        # Center data
        centered = activations - activations.mean(dim=0)

        # SVD
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

        # Get top components
        components = [Vt[i] for i in range(min(n_components, len(Vt)))]

        # Explained variance
        total_var = torch.sum(S ** 2)
        explained = [(S[i] ** 2 / total_var).item() for i in range(len(components))]

        return RepresentationReport(
            principal_components=components,
            explained_variance=explained,
            representation_rank=len(components),
            intervention_vectors={}
        )

    def create_intervention_vector(
        self,
        direction: torch.Tensor,
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Create an intervention vector for steering.

        Args:
            direction: Direction to steer along
            alpha: Steering strength

        Returns:
            Intervention vector
        """
        return alpha * direction / (torch.norm(direction) + 1e-8)

    def erase_concept(
        self,
        activations: torch.Tensor,
        concept_direction: torch.Tensor
    ) -> torch.Tensor:
        """
        Erase a concept from activations.

        Projects activations orthogonal to concept direction.

        Args:
            activations: Activation tensor
            concept_direction: Direction to erase

        Returns:
            Concept-erased activations
        """
        direction = concept_direction / (torch.norm(concept_direction) + 1e-8)
        projection = torch.outer(activations @ direction, direction)
        return activations - projection

    def compute_representation_similarity(
        self,
        rep1: torch.Tensor,
        rep2: torch.Tensor
    ) -> float:
        """Compute cosine similarity between representations."""
        rep1 = rep1 / (torch.norm(rep1) + 1e-8)
        rep2 = rep2 / (torch.norm(rep2) + 1e-8)
        return torch.dot(rep1, rep2).item()
