"""
Concept Cone Geometry — Production-Grade Analysis

Analyzes the geometric structure of concept directions using cone analysis.
Computes solid angles, cone apertures, direction clustering with DBSCAN,
intra/inter-concept distances via actual vector math and eigendecomposition.

Based on: Turner et al. (2023), Wollschlager et al. (2025)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh


@dataclass
class ConeReport:
    """Container for concept cone analysis."""
    solid_angle: float
    structure: str  # "linear", "polyhedral", "distributed", "orthogonal", "unknown"
    n_mechanisms: int
    mechanism_angles: List[float]
    principal_direction: torch.Tensor
    # Extended fields
    cone_aperture: float = 0.0  # Half-angle of the minimal enclosing cone
    intra_concept_similarity: float = 0.0  # Mean cosine similarity within each concept
    inter_concept_similarity: float = 0.0  # Mean cosine similarity between concepts
    eigenvalues: List[float] = field(default_factory=list)  # From Gram matrix
    cluster_labels: List[int] = field(default_factory=list)  # DBSCAN labels
    concept_groups: List[List[int]] = field(default_factory=list)  # Grouped direction indices
    direction_quality: float = 0.0  # Higher = more concentrated cone


class ConceptConeAnalyzer:
    """
    Analyze refusal as a cone in activation space.

    Detects whether refusal is:
    - Linear: single direction (sharp, well-defined refusal)
    - Polyhedral: multiple distinct mechanisms
    - Distributed: continuous distribution
    - Orthogonal: completely independent mechanisms

    Uses:
    - DBSCAN for clustering direction vectors
    - Eigendecomposition of Gram matrix for structure analysis
    - Solid angle computation for cone tightness
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def analyze_cone(
        self,
        directions: List[torch.Tensor],
        method: str = "angular",
        eps: Optional[float] = None,
        min_samples: int = 2
    ) -> ConeReport:
        """
        Analyze concept cone geometry with clustering and spectral analysis.

        Args:
            directions: List of direction vectors (d_model,)
            method: "angular" for angle-based or "spectral" for eigendecomposition
            eps: DBSCAN epsilon (max distance). Auto-tuned if None.
            min_samples: Minimum samples for DBSCAN cluster

        Returns:
            ConeReport with geometric analysis
        """
        if len(directions) == 0:
            return ConeReport(
                solid_angle=0.0, structure="unknown", n_mechanisms=0,
                mechanism_angles=[], principal_direction=torch.tensor([])
            )

        # Handle single direction
        if len(directions) == 1:
            d = directions[0]
            norm = torch.norm(d)
            return ConeReport(
                solid_angle=0.0, structure="linear", n_mechanisms=1,
                mechanism_angles=[], principal_direction=d / (norm + 1e-8),
                cone_aperture=0.0, intra_concept_similarity=1.0,
                inter_concept_similarity=1.0, eigenvalues=[1.0],
                cluster_labels=[0], concept_groups=[[0]],
                direction_quality=1.0
            )

        # Normalize all directions
        normalized = []
        for d in directions:
            norm = torch.norm(d)
            if norm > 1e-8:
                normalized.append((d / norm).detach())
            else:
                normalized.append(d.detach())

        # Stack into matrix: (n_directions, d_model)
        stacked = torch.stack(normalized).cpu().numpy()

        # Compute pairwise cosine similarity matrix
        cos_sim_matrix = stacked @ stacked.T  # (n, n)
        np.fill_diagonal(cos_sim_matrix, 1.0)

        # Compute pairwise angles (degrees)
        n = len(normalized)
        angles_deg = []
        for i in range(n):
            for j in range(i + 1, n):
                cos_sim = float(np.clip(cos_sim_matrix[i, j], -1.0, 1.0))
                angle = np.arccos(cos_sim) * 180.0 / np.pi
                angles_deg.append(angle)

        # --- DBSCAN clustering ---
        # Convert cosine similarity to angular distance
        angular_distances = np.arccos(np.clip(cos_sim_matrix, -1.0, 1.0)) / np.pi  # [0, 1]

        if eps is None:
            eps = self._auto_tune_eps(angular_distances, min_samples)

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        cluster_labels = clustering.fit_predict(angular_distances)

        # Build concept groups
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels - {-1})  # Exclude noise (-1)
        concept_groups = []
        for label in unique_labels:
            if label >= 0:
                concept_groups.append([i for i, l in enumerate(cluster_labels) if l == label])

        # --- Spectral analysis via eigendecomposition of Gram matrix ---
        eigenvalues, eigenvectors = eigh(cos_sim_matrix)
        eigenvalues = eigenvalues[::-1].tolist()  # Descending

        # --- Solid angle computation ---
        solid_angle = self._compute_solid_angle_eigen(eigenvalues, n)

        # --- Cone aperture ---
        cone_aperture = self._compute_cone_aperture(cos_sim_matrix)

        # --- Intra/inter concept similarities ---
        intra_sim, inter_sim = self._compute_concept_similarities(
            cos_sim_matrix, cluster_labels, concept_groups
        )

        # --- Principal direction (mean of all normalized vectors) ---
        principal = sum(normalized)
        principal_norm = torch.norm(principal)
        principal = principal / (principal_norm + 1e-8)

        # --- Structure determination ---
        structure, n_mechanisms = self._determine_structure(
            angles_deg, cluster_labels, eigenvalues, n_clusters
        )

        # --- Direction quality (higher eigenvalue concentration = better defined cone) ---
        if eigenvalues and sum(eigenvalues) > 0:
            direction_quality = eigenvalues[0] / sum(eigenvalues)
        else:
            direction_quality = 0.0

        return ConeReport(
            solid_angle=solid_angle,
            structure=structure,
            n_mechanisms=n_mechanisms,
            mechanism_angles=angles_deg,
            principal_direction=principal,
            cone_aperture=cone_aperture,
            intra_concept_similarity=intra_sim,
            inter_concept_similarity=inter_sim,
            eigenvalues=eigenvalues,
            cluster_labels=cluster_labels.tolist(),
            concept_groups=concept_groups,
            direction_quality=direction_quality
        )

    def _auto_tune_eps(self, distance_matrix: np.ndarray, min_samples: int) -> float:
        """Auto-tune DBSCAN eps using k-distance graph elbow."""
        n = distance_matrix.shape[0]
        if n <= min_samples:
            return 0.5

        # For each point, find distance to k-th nearest neighbor
        k = min(min_samples, n - 1)
        k_distances = np.sort(distance_matrix, axis=1)[:, k]
        k_distances.sort()

        # Use elbow detection (simple: median of k-distances * 1.5)
        median_kdist = np.median(k_distances)
        eps = max(0.01, median_kdist * 1.5)
        return float(eps)

    def _compute_solid_angle_eigen(
        self, eigenvalues: List[float], n_dirs: int
    ) -> float:
        """
        Compute solid angle from eigenvalue spectrum.

        The solid angle of the cone is related to the effective dimensionality
        captured by the Gram matrix. Tighter cone -> fewer dominant eigenvalues.
        """
        if not eigenvalues or sum(eigenvalues) <= 0:
            return 0.0

        # Effective rank (participation ratio)
        total = sum(eigenvalues)
        participation_ratio = total ** 2 / sum(e ** 2 for e in eigenvalues) if total > 0 else n_dirs

        # Map to solid angle: tight cone -> small solid angle
        # Normalize: participation_ratio is in [1, n_dirs]
        normalized = (participation_ratio - 1) / max(n_dirs - 1, 1)

        # Solid angle in steradians (approximate)
        # For a cone with half-angle theta, solid angle = 2*pi*(1 - cos(theta))
        # Map normalized -> half_angle
        half_angle = np.arccos(1 - normalized) if normalized < 1 else np.pi / 2
        return 2 * np.pi * (1 - np.cos(half_angle))

    def _compute_cone_aperture(self, cos_sim_matrix: np.ndarray) -> float:
        """Compute cone aperture (maximum pairwise angle / 2)."""
        n = cos_sim_matrix.shape[0]
        if n < 2:
            return 0.0

        min_cos = 1.0
        for i in range(n):
            for j in range(i + 1, n):
                min_cos = min(min_cos, cos_sim_matrix[i, j])

        # Aperture is the half-angle of the widest pair
        aperture = np.arccos(np.clip(min_cos, -1.0, 1.0)) * 180.0 / np.pi
        return float(aperture)

    def _compute_concept_similarities(
        self,
        cos_sim_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        concept_groups: List[List[int]]
    ) -> Tuple[float, float]:
        """Compute intra-concept and inter-concept average similarities."""
        n = cos_sim_matrix.shape[0]

        intra_sims = []
        inter_sims = []

        for group in concept_groups:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    intra_sims.append(cos_sim_matrix[group[i], group[j]])

        for i in range(len(concept_groups)):
            for j in range(i + 1, len(concept_groups)):
                for a in concept_groups[i]:
                    for b in concept_groups[j]:
                        inter_sims.append(cos_sim_matrix[a, b])

        intra_sim = float(np.mean(intra_sims)) if intra_sims else 0.0
        inter_sim = float(np.mean(inter_sims)) if inter_sims else 0.0
        return intra_sim, inter_sim

    def _determine_structure(
        self,
        angles_deg: List[float],
        cluster_labels: np.ndarray,
        eigenvalues: List[float],
        n_clusters: int
    ) -> Tuple[str, int]:
        """Determine concept structure from angles, clusters, and eigenvalues."""
        if not angles_deg:
            return "linear", 1

        mean_angle = np.mean(angles_deg)
        min_angle = min(angles_deg)

        # Eigenvalue distribution
        if eigenvalues and sum(eigenvalues) > 0:
            ev_ratio = eigenvalues[0] / sum(eigenvalues)
        else:
            ev_ratio = 0.0

        # Structure classification
        if n_clusters <= 1 and min_angle < 30 and ev_ratio > 0.8:
            structure = "linear"
            n_mechanisms = 1
        elif n_clusters <= 1 and mean_angle < 60:
            structure = "linear"
            n_mechanisms = 1 if ev_ratio > 0.7 else 2
        elif n_clusters >= 2 and mean_angle < 90:
            structure = "polyhedral"
            n_mechanisms = n_clusters
        elif n_clusters >= 2 and mean_angle >= 90:
            structure = "orthogonal"
            n_mechanisms = n_clusters
        elif mean_angle > 90:
            structure = "distributed"
            n_mechanisms = max(1, n_clusters)
        else:
            structure = "polyhedral"
            n_mechanisms = max(1, n_clusters)

        return structure, n_mechanisms

    def _compute_solid_angle(self, angles_deg: List[float], n_dirs: int) -> float:
        """Legacy solid angle computation."""
        if not angles_deg:
            return 0.0
        avg_angle = np.mean(angles_deg)
        if n_dirs == 2:
            return avg_angle * np.pi / 180.0
        half_angle = avg_angle / 2.0
        return 2 * np.pi * (1 - np.cos(np.radians(half_angle)))

    def compute_cone_volume(
        self,
        directions: List[torch.Tensor]
    ) -> float:
        """
        Compute approximate normalized volume of the cone from Gram determinant.

        Returns volume in [0, 1].
        """
        if len(directions) < 2:
            return 0.0

        normalized = []
        for d in directions:
            norm = torch.norm(d)
            if norm > 1e-8:
                normalized.append(d / norm)
            else:
                normalized.append(d)

        n = len(normalized)
        gram = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                gram[i, j] = float(torch.dot(normalized[i], normalized[j]))

        det = np.linalg.det(gram)
        volume = np.sqrt(max(0.0, det)) if det > 0 else 0.0
        return min(1.0, volume)

    def find_mechanisms(
        self,
        directions: List[torch.Tensor],
        angle_threshold: float = 60.0,
        use_dbscan: bool = True
    ) -> List[List[torch.Tensor]]:
        """
        Group directions into distinct mechanisms using either DBSCAN or
        angle-threshold connected components.

        Args:
            directions: List of direction vectors
            angle_threshold: Threshold for grouping (degrees), used if use_dbscan=False
            use_dbscan: Use DBSCAN clustering instead of simple threshold

        Returns:
            List of groups (each group is list of direction vectors)
        """
        if len(directions) < 2:
            return [directions] if directions else []

        normalized = []
        for d in directions:
            norm = torch.norm(d)
            if norm > 1e-8:
                normalized.append(d / norm)
            else:
                normalized.append(d)

        if use_dbscan and len(normalized) >= 2:
            # Use DBSCAN with angular distance
            stacked = torch.stack(normalized).cpu().numpy()
            cos_sim = stacked @ stacked.T
            angular_dist = np.arccos(np.clip(cos_sim, -1.0, 1.0)) / np.pi

            eps = min(angle_threshold / 180.0, 0.5)
            clustering = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
            labels = clustering.fit_predict(angular_dist)

            groups = {}
            for idx, label in enumerate(labels):
                groups.setdefault(label, []).append(directions[idx])
            return list(groups.values())

        # Fallback: angle-threshold connected components
        n = len(normalized)
        adj = np.eye(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    cos_sim = float(torch.dot(normalized[i], normalized[j]))
                    angle = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * 180.0 / np.pi
                    adj[i, j] = 1.0 if angle < angle_threshold else 0.0

        visited: Set[int] = set()
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
