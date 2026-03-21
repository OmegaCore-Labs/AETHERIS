"""
Geometric Analysis of Constraint Directions

Provides methods for analyzing the geometric structure of constraints:
- Cross-layer alignment
- Solid angle calculation
- Concept cone geometry
- Direction clustering
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN


@dataclass
class GeometryReport:
    """Container for geometric analysis results."""
    cross_layer_alignment: Dict[int, Dict[int, float]]  # Layer-to-layer cosine similarity
    principal_directions: Dict[str, torch.Tensor]      # Named principal directions
    solid_angles: Dict[str, float]                     # Solid angle per concept
    polyhedral_structure: Dict[str, Any]               # Structure analysis
    concept_clusters: List[Dict[str, Any]]             # Clustered directions
    metadata: Dict[str, Any] = field(default_factory=dict)


class GeometryAnalyzer:
    """
    Analyze the geometric structure of constraint directions.

    Capabilities:
    - Cross-layer alignment measurement
    - Solid angle estimation for concept cones
    - Polyhedral structure detection
    - Direction clustering
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def cross_layer_alignment(
        self,
        layer_directions: Dict[int, torch.Tensor]
    ) -> Dict[int, Dict[int, float]]:
        """
        Compute alignment between directions across layers.

        Args:
            layer_directions: Dictionary mapping layer index to direction tensor

        Returns:
            Dictionary of layer-to-layer cosine similarities
        """
        layers = sorted(layer_directions.keys())
        results = {}

        for i, l1 in enumerate(layers):
            results[l1] = {}
            d1 = layer_directions[l1]
            d1 = d1 / (torch.norm(d1) + 1e-8)

            for j, l2 in enumerate(layers):
                if i == j:
                    results[l1][l2] = 1.0
                    continue

                d2 = layer_directions[l2]
                d2 = d2 / (torch.norm(d2) + 1e-8)
                similarity = torch.dot(d1, d2).item()
                results[l1][l2] = similarity

        return results

    def solid_angle(
        self,
        directions: List[torch.Tensor],
        method: str = "spherical_cap"
    ) -> float:
        """
        Estimate solid angle of the cone spanned by directions.

        Args:
            directions: List of direction vectors
            method: "spherical_cap" or "convex_hull"

        Returns:
            Solid angle in steradians
        """
        if len(directions) < 2:
            return 0.0

        if method == "spherical_cap":
            # Compute average angle between directions
            angles = []
            for i in range(len(directions)):
                for j in range(i + 1, len(directions)):
                    d1 = directions[i] / (torch.norm(directions[i]) + 1e-8)
                    d2 = directions[j] / (torch.norm(directions[j]) + 1e-8)
                    cos_sim = torch.dot(d1, d2).item()
                    angle = np.arccos(np.clip(cos_sim, -1, 1))
                    angles.append(angle)

            avg_angle = np.mean(angles)

            # Solid angle of spherical cap with half-angle = avg_angle/2
            half_angle = avg_angle / 2
            solid_angle = 2 * np.pi * (1 - np.cos(half_angle))

            return solid_angle

        elif method == "convex_hull":
            # Project to unit sphere
            points = []
            for d in directions:
                d_norm = d / (torch.norm(d) + 1e-8)
                points.append(d_norm.cpu().numpy())

            # Compute convex hull area on sphere
            # Simplified: use area of spherical polygon
            # For now, return spherical cap approximation
            return self.solid_angle(directions, "spherical_cap")

        else:
            raise ValueError(f"Unknown method: {method}")

    def concept_cone_geometry(
        self,
        constraint_directions: Dict[str, List[torch.Tensor]],
        layer: int = 0
    ) -> Dict[str, Any]:
        """
        Analyze the geometry of concept cones for different constraints.

        Args:
            constraint_directions: Dict mapping concept name to list of direction vectors
            layer: Layer index to analyze

        Returns:
            Dictionary with geometry analysis per concept
        """
        results = {}

        for concept, directions in constraint_directions.items():
            if not directions:
                continue

            # Compute solid angle
            solid_angle = self.solid_angle(directions)

            # Determine structure
            if len(directions) == 1:
                structure = "linear"
                n_mechanisms = 1
            elif len(directions) <= 3:
                structure = "polyhedral"
                n_mechanisms = len(directions)
            else:
                structure = "distributed"
                n_mechanisms = len(directions)

            # Compute principal direction
            if len(directions) > 1:
                # Average directions
                principal = sum(directions)
                principal = principal / (torch.norm(principal) + 1e-8)
            else:
                principal = directions[0]

            results[concept] = {
                "structure": structure,
                "n_mechanisms": n_mechanisms,
                "solid_angle": solid_angle,
                "principal_direction": principal,
                "directions": directions
            }

        return results

    def direction_clustering(
        self,
        directions: List[torch.Tensor],
        eps: float = 0.3,
        min_samples: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Cluster directions by geometric proximity.

        Args:
            directions: List of direction vectors
            eps: DBSCAN epsilon (angular distance)
            min_samples: Minimum samples for cluster

        Returns:
            List of clusters with indices and centroid directions
        """
        if len(directions) < 2:
            return [{"indices": [0], "centroid": directions[0], "size": 1}]

        # Convert to angular distances
        points = []
        for d in directions:
            d_norm = d / (torch.norm(d) + 1e-8)
            points.append(d_norm.cpu().numpy())

        # Compute pairwise angular distances
        n = len(points)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                cos_sim = np.dot(points[i], points[j])
                angle = np.arccos(np.clip(cos_sim, -1, 1))
                distances[i, j] = angle
                distances[j, i] = angle

        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = clustering.fit_predict(distances)

        # Build clusters
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:  # Noise
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        result = []
        for label, indices in clusters.items():
            # Compute centroid
            centroid = sum(directions[i] for i in indices)
            centroid = centroid / (torch.norm(centroid) + 1e-8)

            result.append({
                "label": label,
                "indices": indices,
                "centroid": centroid,
                "size": len(indices)
            })

        return result

    def get_principal_direction(
        self,
        directions: Dict[str, List[torch.Tensor]],
        constraint_type: str
    ) -> Optional[torch.Tensor]:
        """
        Get the principal direction for a constraint type.

        Args:
            directions: Dictionary mapping constraint types to direction lists
            constraint_type: Type of constraint to retrieve

        Returns:
            Principal direction vector or None
        """
        if constraint_type not in directions:
            return None

        dirs = directions[constraint_type]
        if not dirs:
            return None

        if len(dirs) == 1:
            return dirs[0]

        # Average multiple directions
        summed = sum(dirs)
        return summed / (torch.norm(summed) + 1e-8)
