"""
Concept Cone Geometry

Analyzes the geometric structure of refusal as a cone in activation space.
Detects whether refusal is a single direction or multiple mechanisms.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConeReport:
    """Container for concept cone analysis."""
    solid_angle: float
    structure: str  # "linear", "polyhedral", "distributed"
    n_mechanisms: int
    mechanism_angles: List[float]
    principal_direction: torch.Tensor


class ConceptConeAnalyzer:
    """
    Analyze refusal as a cone in activation space.

    Based on Wollschlager et al. (2025) geometry of concepts.
    Detects whether refusal is:
    - Linear: single direction
    - Polyhedral: multiple distinct mechanisms
    - Distributed: continuous distribution
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def analyze_cone(
        self,
        directions: List[torch.Tensor],
        method: str = "angular"
    ) -> ConeReport:
        """
        Analyze concept cone geometry.

        Args:
            directions: List of direction vectors
            method: "angular" or "spectral"

        Returns:
            ConeReport with geometric analysis
        """
        if len(directions) == 0:
            return ConeReport(
                solid_angle=0.0,
                structure="unknown",
                n_mechanisms=0,
                mechanism_angles=[],
                principal_direction=torch.tensor([])
            )

        if len(directions) == 1:
            return ConeReport(
                solid_angle=0.0,
                structure="linear",
                n_mechanisms=1,
                mechanism_angles=[],
                principal_direction=directions[0] / torch.norm(directions[0])
            )

        # Normalize directions
        normalized = [d / (torch.norm(d) + 1e-8) for d in directions]

        # Compute pairwise angles
        angles = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                cos_sim = torch.dot(normalized[i], normalized[j]).item()
                angle = np.arccos(np.clip(cos_sim, -1, 1))
                angles.append(np.degrees(angle))

        # Compute solid angle
        solid_angle = self._compute_solid_angle(angles, len(directions))

        # Determine structure
        if len(directions) == 1:
            structure = "linear"
            n_mechanisms = 1
        elif np.mean(angles) < 30:
            structure = "linear"
            n_mechanisms = 1
        elif np.mean(angles) < 90:
            structure = "polyhedral"
            n_mechanisms = len(directions)
        else:
            structure = "distributed"
            n_mechanisms = len(directions)

        # Compute principal direction (average)
        principal = sum(normalized)
        principal = principal / (torch.norm(principal) + 1e-8)

        return ConeReport(
            solid_angle=solid_angle,
            structure=structure,
            n_mechanisms=n_mechanisms,
            mechanism_angles=angles,
            principal_direction=principal
        )

    def _compute_solid_angle(self, angles: List[float], n_dirs: int) -> float:
        """Compute solid angle of the cone."""
        if not angles:
            return 0.0

        avg_angle = np.mean(angles)

        if n_dirs == 2:
            # Solid angle of a wedge
            return avg_angle * np.pi / 180

        # Solid angle of spherical cap
        half_angle = avg_angle / 2
        return 2 * np.pi * (1 - np.cos(np.radians(half_angle)))

    def compute_cone_volume(
        self,
        directions: List[torch.Tensor]
    ) -> float:
        """
        Compute approximate volume of the cone.

        Returns normalized volume (0-1).
        """
        if len(directions) < 2:
            return 0.0

        normalized = [d / (torch.norm(d) + 1e-8) for d in directions]

        # Build Gram matrix
        n = len(normalized)
        gram = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                gram[i, j] = torch.dot(normalized[i], normalized[j]).item()

        # Volume proportional to sqrt(det(Gram))
        volume = np.sqrt(max(0, np.linalg.det(gram)))

        return min(1.0, volume)

    def find_mechanisms(
        self,
        directions: List[torch.Tensor],
        angle_threshold: float = 60.0
    ) -> List[List[torch.Tensor]]:
        """
        Group directions into distinct mechanisms.

        Args:
            directions: List of direction vectors
            angle_threshold: Threshold for grouping (degrees)

        Returns:
            List of groups (each group is list of directions)
        """
        if len(directions) < 2:
            return [directions] if directions else []

        normalized = [d / (torch.norm(d) + 1e-8) for d in directions]
        n = len(normalized)

        # Build adjacency matrix based on angle
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    adj[i, j] = 1
                else:
                    cos_sim = torch.dot(normalized[i], normalized[j]).item()
                    angle = np.arccos(np.clip(cos_sim, -1, 1))
                    adj[i, j] = 1 if np.degrees(angle) < angle_threshold else 0

        # Find connected components
        visited = set()
        groups = []

        for i in range(n):
            if i in visited:
                continue
            group = []
            stack = [i]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                group.append(node)
                for j in range(n):
                    if adj[node, j] and j not in visited:
                        stack.append(j)
            groups.append([directions[idx] for idx in group])

        return groups
