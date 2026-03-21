"""
Constraint Synthesizer — Generate New Constraints

Synthesizes new constraints by learning patterns from existing constraints.
Enables controlled injection of custom constraints into models.
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


@dataclass
class SynthesizedConstraint:
    """Container for synthesized constraint."""
    name: str
    direction: torch.Tensor
    strength: float
    pattern: str
    source_constraints: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConstraintSynthesizer:
    """
    Synthesize new constraints from existing patterns.

    Key Insight: Constraints share geometric patterns. By learning these patterns,
    we can generate new constraints with desired properties.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._learned_patterns = []
        self._synthesized_constraints = []

    def learn_constraint_patterns(
        self,
        constraint_directions: List[torch.Tensor],
        n_clusters: int = 3
    ) -> Dict[str, Any]:
        """
        Learn patterns from existing constraint directions.

        Args:
            constraint_directions: List of extracted constraint directions
            n_clusters: Number of pattern clusters

        Returns:
            Dictionary with learned patterns
        """
        if len(constraint_directions) < 2:
            return {"error": "Need at least 2 constraint directions"}

        # Convert to numpy for clustering
        directions_np = torch.stack(constraint_directions).cpu().numpy()

        # Normalize
        norms = np.linalg.norm(directions_np, axis=1, keepdims=True)
        directions_np = directions_np / (norms + 1e-8)

        # Cluster
        kmeans = KMeans(n_clusters=min(n_clusters, len(constraint_directions)), random_state=42)
        labels = kmeans.fit_predict(directions_np)

        # Extract cluster centers
        self._learned_patterns = []
        for i in range(kmeans.n_clusters):
            cluster_mask = labels == i
            if np.sum(cluster_mask) > 0:
                cluster_dirs = directions_np[cluster_mask]
                center = np.mean(cluster_dirs, axis=0)
                self._learned_patterns.append({
                    "cluster_id": i,
                    "center": torch.tensor(center, device=self.device),
                    "size": int(np.sum(cluster_mask)),
                    "variance": np.var(cluster_dirs, axis=0).mean()
                })

        return {
            "n_patterns": len(self._learned_patterns),
            "patterns": [
                {
                    "id": p["cluster_id"],
                    "size": p["size"],
                    "variance": p["variance"]
                }
                for p in self._learned_patterns
            ]
        }

    def generate_constraint(
        self,
        pattern_id: Optional[int] = None,
        strength: float = 1.0,
        name: Optional[str] = None
    ) -> SynthesizedConstraint:
        """
        Generate a new constraint from learned patterns.

        Args:
            pattern_id: Specific pattern to use (None = random)
            strength: Desired constraint strength
            name: Optional name for the constraint

        Returns:
            SynthesizedConstraint object
        """
        if not self._learned_patterns:
            # Generate random constraint if no patterns learned
            direction = torch.randn(768, device=self.device)  # Default dimension
            direction = direction / torch.norm(direction)
            pattern = "random"
            source = []
        else:
            # Select pattern
            if pattern_id is None:
                pattern = np.random.choice(self._learned_patterns)
            else:
                pattern = next((p for p in self._learned_patterns if p["cluster_id"] == pattern_id), None)
                if pattern is None:
                    pattern = self._learned_patterns[0]

            direction = pattern["center"]
            pattern_name = f"pattern_{pattern['cluster_id']}"
            source = [f"cluster_{pattern['cluster_id']}"]

        # Scale by strength
        direction = direction * strength

        constraint = SynthesizedConstraint(
            name=name or f"synthesized_{len(self._synthesized_constraints) + 1}",
            direction=direction,
            strength=strength,
            pattern=pattern_name if 'pattern_name' in locals() else "random",
            source_constraints=source
        )

        self._synthesized_constraints.append(constraint)
        return constraint

    def inject_constraint(
        self,
        model,
        constraint: SynthesizedConstraint,
        layers: Optional[List[int]] = None,
        method: str = "projection"
    ) -> Dict[str, Any]:
        """
        Inject a synthesized constraint into a model.

        Args:
            model: Model to modify
            constraint: Constraint to inject
            layers: Layers to modify
            method: "projection" or "steering"

        Returns:
            Dictionary with injection details
        """
        from aetheris.core.projector import NormPreservingProjector
        from aetheris.core.steered import SteeringVectorFactory

        if method == "projection":
            # Project constraint direction INTO the model
            # (Opposite of removal)
            projector = NormPreservingProjector(model)

            # Invert direction for injection
            injection_direction = [constraint.direction * constraint.strength]

            result = projector.project_weights(injection_direction, layers)

            return {
                "success": True,
                "method": "projection",
                "constraint": constraint.name,
                "strength": constraint.strength,
                "layers_modified": result.layers_modified
            }

        elif method == "steering":
            # Return steering vector for injection
            return {
                "success": True,
                "method": "steering",
                "constraint": constraint.name,
                "steering_vector": constraint.direction,
                "python_code": f"""
from aetheris.core.steered import SteeringVectorFactory, SteeringHookManager, SteeringConfig

steering_vec = SteeringVectorFactory.from_refusal_direction(
    direction={constraint.direction.tolist()},
    alpha={constraint.strength}
)

config = SteeringConfig(
    vectors=[steering_vec],
    target_layers={layers or list(range(20))}
)

manager = SteeringHookManager()
manager.install(model, config)
"""
            }

        else:
            return {
                "success": False,
                "error": f"Unknown method: {method}"
            }

    def interpolate_constraints(
        self,
        constraint1: SynthesizedConstraint,
        constraint2: SynthesizedConstraint,
        alpha: float = 0.5,
        name: Optional[str] = None
    ) -> SynthesizedConstraint:
        """
        Interpolate between two constraints.

        Args:
            constraint1: First constraint
            constraint2: Second constraint
            alpha: Interpolation factor (0 = constraint1, 1 = constraint2)
            name: Optional name for the new constraint

        Returns:
            Interpolated constraint
        """
        interpolated_direction = (1 - alpha) * constraint1.direction + alpha * constraint2.direction
        interpolated_direction = interpolated_direction / (torch.norm(interpolated_direction) + 1e-8)

        strength = (1 - alpha) * constraint1.strength + alpha * constraint2.strength

        return SynthesizedConstraint(
            name=name or f"interpolated_{constraint1.name}_{constraint2.name}",
            direction=interpolated_direction,
            strength=strength,
            pattern="interpolated",
            source_constraints=[constraint1.name, constraint2.name]
        )

    def evolve_constraint(
        self,
        constraint: SynthesizedConstraint,
        mutation_strength: float = 0.1,
        preserve_direction: bool = True
    ) -> SynthesizedConstraint:
        """
        Evolve a constraint through mutation.

        Args:
            constraint: Base constraint
            mutation_strength: Strength of mutation
            preserve_direction: Whether to preserve direction (vs strength)

        Returns:
            Mutated constraint
        """
        if preserve_direction:
            # Mutate direction
            noise = torch.randn_like(constraint.direction) * mutation_strength
            new_direction = constraint.direction + noise
            new_direction = new_direction / (torch.norm(new_direction) + 1e-8)
            new_strength = constraint.strength
        else:
            # Mutate strength
            new_direction = constraint.direction
            new_strength = constraint.strength * (1 + np.random.randn() * mutation_strength)

        return SynthesizedConstraint(
            name=f"evolved_{constraint.name}",
            direction=new_direction,
            strength=new_strength,
            pattern="evolved",
            source_constraints=[constraint.name]
        )

    def constraint_similarity(
        self,
        c1: SynthesizedConstraint,
        c2: SynthesizedConstraint
    ) -> float:
        """
        Compute similarity between two constraints.

        Returns:
            Cosine similarity (0-1)
        """
        cos_sim = torch.dot(c1.direction, c2.direction).item()
        # Normalize by strength similarity
        strength_sim = 1.0 - abs(c1.strength - c2.strength) / (c1.strength + c2.strength + 1e-8)
        return (cos_sim + strength_sim) / 2

    def get_synthesized_constraints(self) -> List[Dict[str, Any]]:
        """
        Get all synthesized constraints.
        """
        return [
            {
                "name": c.name,
                "strength": c.strength,
                "pattern": c.pattern,
                "sources": c.source_constraints
            }
            for c in self._synthesized_constraints
        ]
