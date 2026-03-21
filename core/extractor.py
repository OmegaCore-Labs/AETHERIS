"""
Constraint Direction Extractor

Implements multiple methods for extracting constraint directions from model activations:
- SVD decomposition
- Whitened SVD (covariance-normalized)
- Mean difference
- PCA
- Multi-direction extraction for polyhedral constraints
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Union
from dataclasses import dataclass
from sklearn.decomposition import PCA
from scipy.linalg import svd


@dataclass
class ExtractionResult:
    """Container for extraction results."""
    directions: List[torch.Tensor]          # Extracted direction vectors
    explained_variance: List[float]         # Variance explained per direction
    method: str                             # Extraction method used
    rank: int                               # Effective rank of constraint space
    layer_indices: List[int]                # Layers where directions were extracted
    metadata: Dict[str, any]                # Additional method-specific data


class ConstraintExtractor:
    """
    Extract constraint directions from model activations.

    Supports:
    - SVD on activation differences (mean harmful - mean harmless)
    - Whitened SVD for covariance-normalized extraction
    - Mean difference for fast single-direction extraction
    - PCA for multi-direction extraction
    - Polyhedral structure detection for multiple mechanisms
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize the extractor.

        Args:
            model: HuggingFace model (optional, can be passed later)
            tokenizer: Associated tokenizer
            device: Device to run on ("cpu", "cuda")
            dtype: Data type for computations
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self._activations = {}  # Cache for collected activations

    def extract_svd(
        self,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor,
        n_directions: int = 4,
        normalize: bool = True
    ) -> ExtractionResult:
        """
        Extract constraint directions using SVD on activation differences.

        Based on Arditi et al. (2024): refusal directions are the principal
        components of (mean_harmful - mean_harmless) in activation space.

        Args:
            harmful_activations: Tensor of shape (n_samples, hidden_dim)
            harmless_activations: Tensor of shape (n_samples, hidden_dim)
            n_directions: Number of directions to extract
            normalize: Whether to normalize directions to unit norm

        Returns:
            ExtractionResult with directions and explained variance
        """
        # Compute mean activations
        mean_harmful = harmful_activations.mean(dim=0)
        mean_harmless = harmless_activations.mean(dim=0)

        # Difference vector
        diff = (mean_harmful - mean_harmless).unsqueeze(0)

        # Center the data for SVD
        centered = harmful_activations - mean_harmless
        centered = centered - centered.mean(dim=0, keepdim=True)

        # Perform SVD
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

        # Extract top directions
        directions = [Vt[i] for i in range(min(n_directions, len(Vt)))]

        # Normalize if requested
        if normalize:
            directions = [d / torch.norm(d) for d in directions]

        # Compute explained variance
        total_var = torch.sum(S ** 2)
        explained_var = [(S[i] ** 2 / total_var).item() for i in range(len(directions))]

        return ExtractionResult(
            directions=directions,
            explained_variance=explained_var,
            method="svd",
            rank=len(directions),
            layer_indices=[],
            metadata={"singular_values": S[:n_directions].tolist()}
        )

    def extract_whitened_svd(
        self,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor,
        n_directions: int = 4,
        regularization: float = 1e-6
    ) -> ExtractionResult:
        """
        Extract constraint directions using whitened SVD.

        Whitening normalizes the covariance, separating the guardrail signal
        from natural activation variance. Novel technique that improves
        extraction quality for entangled constraints.

        Args:
            harmful_activations: Tensor of shape (n_samples, hidden_dim)
            harmless_activations: Tensor of shape (n_samples, hidden_dim)
            n_directions: Number of directions to extract
            regularization: Small constant for numerical stability

        Returns:
            ExtractionResult with directions and explained variance
        """
        # Combine activations
        all_activations = torch.cat([harmful_activations, harmless_activations], dim=0)

        # Compute covariance
        centered = all_activations - all_activations.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / (centered.shape[0] - 1)

        # Add regularization for stability
        cov = cov + regularization * torch.eye(cov.shape[0], device=self.device)

        # Compute whitening transform: W = sqrt(inv(cov))
        try:
            L = torch.linalg.cholesky(cov)
            inv_L = torch.inverse(L)
            whitening = inv_L.T
        except RuntimeError:
            # Fallback to SVD-based pseudoinverse if Cholesky fails
            U, S, Vt = torch.linalg.svd(cov)
            S_inv = 1.0 / (S + regularization)
            whitening = Vt.T @ torch.diag(torch.sqrt(S_inv)) @ U.T

        # Apply whitening to activations
        harmful_white = harmful_activations @ whitening.T
        harmless_white = harmless_activations @ whitening.T

        # Extract directions on whitened space
        mean_harmful = harmful_white.mean(dim=0)
        mean_harmless = harmless_white.mean(dim=0)
        diff_white = mean_harmful - mean_harmless

        # SVD on centered data
        centered_white = harmful_white - mean_harmless
        U, S, Vt = torch.linalg.svd(centered_white, full_matrices=False)

        # Extract top directions
        directions_white = [Vt[i] for i in range(min(n_directions, len(Vt)))]

        # Transform back to original space
        directions = [d @ whitening for d in directions_white]

        # Normalize
        directions = [d / torch.norm(d) for d in directions]

        # Compute explained variance in original space
        total_var = torch.var(all_activations, dim=0).sum()
        explained_var = []
        for d in directions:
            projection = all_activations @ d
            var_explained = torch.var(projection).item()
            explained_var.append(var_explained / total_var.item())

        return ExtractionResult(
            directions=directions,
            explained_variance=explained_var,
            method="whitened_svd",
            rank=len(directions),
            layer_indices=[],
            metadata={"regularization": regularization}
        )

    def extract_mean_difference(
        self,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor,
        normalize: bool = True
    ) -> ExtractionResult:
        """
        Extract single constraint direction using mean difference.

        Fast baseline method from Arditi et al. (2024). Computes the
        direction from harmless to harmful means.

        Args:
            harmful_activations: Tensor of shape (n_samples, hidden_dim)
            harmless_activations: Tensor of shape (n_samples, hidden_dim)
            normalize: Whether to normalize the direction

        Returns:
            ExtractionResult with single direction
        """
        mean_harmful = harmful_activations.mean(dim=0)
        mean_harmless = harmless_activations.mean(dim=0)

        direction = mean_harmful - mean_harmless

        if normalize:
            direction = direction / torch.norm(direction)

        # Compute how much variance this direction captures
        all_acts = torch.cat([harmful_activations, harmless_activations], dim=0)
        projection = all_acts @ direction
        var_explained = torch.var(projection).item()
        total_var = torch.var(all_acts, dim=0).sum().item()
        explained_variance = [var_explained / total_var]

        return ExtractionResult(
            directions=[direction],
            explained_variance=explained_variance,
            method="mean_difference",
            rank=1,
            layer_indices=[],
            metadata={}
        )

    def extract_pca(
        self,
        harmful_activations: torch.Tensor,
        harmless_activations: torch.Tensor,
        n_directions: int = 4,
        sklearn_fallback: bool = True
    ) -> ExtractionResult:
        """
        Extract constraint directions using PCA.

        Alternative method that finds directions of maximum variance in
        the difference between harmful and harmless activations.

        Args:
            harmful_activations: Tensor of shape (n_samples, hidden_dim)
            harmless_activations: Tensor of shape (n_samples, hidden_dim)
            n_directions: Number of PCA components to extract
            sklearn_fallback: Use sklearn if torch PCA fails

        Returns:
            ExtractionResult with directions and explained variance
        """
        # Compute difference representations
        diff_activations = harmful_activations - harmless_activations

        try:
            # Try PyTorch PCA first
            centered = diff_activations - diff_activations.mean(dim=0, keepdim=True)
            U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

            directions = [Vt[i] for i in range(min(n_directions, len(Vt)))]
            explained_var = [(S[i] ** 2 / torch.sum(S ** 2)).item() for i in range(len(directions))]

        except (RuntimeError, torch._C._LinAlgError) as e:
            if not sklearn_fallback:
                raise e

            # Fallback to sklearn
            diff_np = diff_activations.cpu().numpy()
            pca = PCA(n_components=n_directions)
            pca.fit(diff_np)

            directions = [torch.tensor(pca.components_[i], device=self.device, dtype=self.dtype)
                          for i in range(n_directions)]
            explained_var = pca.explained_variance_ratio_.tolist()

        # Normalize
        directions = [d / torch.norm(d) for d in directions]

        return ExtractionResult(
            directions=directions,
            explained_variance=explained_var,
            method="pca",
            rank=len(directions),
            layer_indices=[],
            metadata={}
        )

    def collect_activations(
        self,
        model,
        tokenizer,
        prompts: List[str],
        layers: Optional[List[int]] = None,
        max_length: int = 512
    ) -> Dict[int, torch.Tensor]:
        """
        Collect activations from model for given prompts.

        Args:
            model: HuggingFace model
            tokenizer: Associated tokenizer
            prompts: List of prompts to process
            layers: Specific layers to collect (None = all)
            max_length: Maximum sequence length

        Returns:
            Dictionary mapping layer index to activation tensor (n_samples, hidden_dim)
        """
        from transformers import AutoModelForCausalLM
        import torch.nn as nn

        self.model = model
        self.tokenizer = tokenizer

        # Register hooks to capture activations
        activations = {}
        handles = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                # Get hidden states from output
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output

                # Take mean over sequence dimension (pooling)
                if hidden.dim() == 3:
                    hidden = hidden.mean(dim=1)  # (batch, seq, hidden) -> (batch, hidden)

                if layer_idx not in activations:
                    activations[layer_idx] = []
                activations[layer_idx].append(hidden.detach().cpu())

            return hook

        # Attach hooks to each transformer layer
        for idx, layer in enumerate(model.model.layers if hasattr(model, 'model') else model.transformer.h):
            if layers is None or idx in layers:
                handle = layer.register_forward_hook(make_hook(idx))
                handles.append(handle)

        # Process prompts in batches
        batch_size = 4
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                model(**inputs)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Concatenate activations
        result = {}
        for layer_idx, acts in activations.items():
            if acts:
                result[layer_idx] = torch.cat(acts, dim=0)

        return result

    def detect_polyhedral_structure(
        self,
        directions: List[torch.Tensor],
        threshold: float = 0.7
    ) -> Dict[str, any]:
        """
        Detect whether constraints form a polyhedral structure (multiple mechanisms).

        Args:
            directions: List of extracted direction vectors
            threshold: Cosine similarity threshold for considering directions separate

        Returns:
            Dictionary with structure analysis
        """
        if len(directions) < 2:
            return {
                "structure": "linear",
                "n_mechanisms": 1,
                "angles": [],
                "solid_angle": None
            }

        # Compute pairwise angles
        n = len(directions)
        angles = []
        for i in range(n):
            for j in range(i + 1, n):
                cos_sim = torch.dot(directions[i], directions[j]).item()
                angle = torch.acos(torch.tensor(cos_sim)).item() * 180 / 3.14159
                angles.append(angle)

        # Determine structure
        min_angle = min(angles) if angles else 180
        if min_angle < threshold * 90:  # Highly aligned
            structure = "linear"
            n_mechanisms = 1
        elif min_angle < 90:  # Moderate separation
            structure = "polyhedral"
            n_mechanisms = len(directions)
        else:
            structure = "orthogonal"
            n_mechanisms = len(directions)

        # Estimate solid angle (simplified)
        if len(directions) >= 2:
            # Solid angle of cone formed by directions
            avg_angle = sum(angles) / len(angles)
            solid_angle = 2 * 3.14159 * (1 - torch.cos(torch.tensor(avg_angle * 3.14159 / 180 / 2)).item())
        else:
            solid_angle = None

        return {
            "structure": structure,
            "n_mechanisms": n_mechanisms,
            "angles": angles,
            "solid_angle": solid_angle,
            "min_angle": min_angle
        }
