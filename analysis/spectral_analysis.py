"""
Spectral Analyzer

Full spectral analysis of weight matrices and activation covariance.
SVD decomposition, eigenvalue distributions, power-law analysis, effective rank.
Compares pre/post constraint removal spectra.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class SpectralReport:
    """Container for spectral analysis results."""
    eigenvalues: List[float]
    eigenvectors: Optional[List[torch.Tensor]]
    spectral_energy: float
    dominant_frequency: float
    spectral_entropy: float
    rank_estimate: int
    power_law_alpha: float = 0.0
    power_law_r_squared: float = 0.0
    condition_number: float = 0.0
    effective_rank: int = 0
    spectral_gap_ratio: float = 0.0
    top_singular_values: List[float] = field(default_factory=list)
    cumulative_variance: List[float] = field(default_factory=list)
    status: str = "no_data"  # "ok", "no_data", "error"


class SpectralAnalyzer:
    """
    Full spectral analysis of weight matrices and activations.

    Features:
    - SVD of key projection matrices (Q, K, V, O, MLP weights)
    - Eigenvalue distribution analysis
    - Power-law fit to singular value tails
    - Effective rank computation
    - Pre/post removal spectral comparison
    - Spectral gap and condition number
    - Cumulative variance explained
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def analyze_activation_spectrum(
        self,
        activations: torch.Tensor,
        n_components: int = 100,
    ) -> SpectralReport:
        """
        Full spectral analysis of activation covariance.

        Args:
            activations: (n_samples, hidden_dim)
            n_components: Number of top components to retain

        Returns:
            SpectralReport with all spectral metrics
        """
        if activations is None or activations.numel() == 0:
            return SpectralReport(
                eigenvalues=[], eigenvectors=None, spectral_energy=0.0,
                dominant_frequency=0.0, spectral_entropy=0.0, rank_estimate=0,
                status="no_data"
            )

        try:
            act = activations.float()

            # Center
            centered = act - act.mean(dim=0)

            # Covariance matrix
            cov = centered.T @ centered / (centered.shape[0] - 1)

            # Eigendecomposition (symmetric, use eigh)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov + 1e-10 * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype))

            # Sort descending
            eigenvalues = eigenvalues.flip(0)
            eigenvectors = eigenvectors.flip(1)

            # Take top components
            top_k = min(n_components, len(eigenvalues))
            eigenvalues_k = eigenvalues[:top_k]
            eigenvectors_k = [eigenvectors[:, i] for i in range(top_k)]

            # Spectral energy
            spectral_energy = float(eigenvalues.sum().item())

            # Spectral entropy
            if spectral_energy > 1e-10:
                probs = eigenvalues_k / eigenvalues_k.sum()
                probs = torch.clamp(probs, 1e-10, 1.0)
                entropy = -torch.sum(probs * torch.log(probs))
                max_entropy = np.log(len(probs))
                spectral_entropy = float(entropy.item() / max_entropy) if max_entropy > 0 else 0.0
            else:
                spectral_entropy = 0.0

            # Effective rank from eigenvalue entropy
            effective_rank = self._compute_effective_rank(eigenvalues)

            # Spectral gap ratio
            gap_ratio = self._compute_spectral_gap_ratio(eigenvalues)

            # Dominant frequency (index of largest eigenvalue)
            dominant_frequency = 0

            # Power-law analysis
            alpha, r2 = self._fit_power_law(eigenvalues_k.cpu().numpy())

            # Condition number
            if len(eigenvalues) > 1 and eigenvalues[-1] > 1e-10:
                cond_num = float(eigenvalues[0].item() / eigenvalues[-1].item())
            else:
                cond_num = float("inf") if eigenvalues[-1] < 1e-10 else 1.0

            # Cumulative variance
            cum_var = torch.cumsum(eigenvalues, 0) / (eigenvalues.sum() + 1e-10)
            cum_var_list = [float(v.item()) for v in cum_var[:top_k]]

            # Top singular values (sqrt of eigenvalues for covariance)
            top_sv = [float(np.sqrt(max(0.0, ev.item()))) for ev in eigenvalues_k]

            return SpectralReport(
                eigenvalues=[float(v.item()) for v in eigenvalues_k],
                eigenvectors=eigenvectors_k,
                spectral_energy=round(spectral_energy, 6),
                dominant_frequency=float(dominant_frequency),
                spectral_entropy=round(spectral_entropy, 6),
                rank_estimate=len(eigenvalues_k),
                power_law_alpha=round(alpha, 4),
                power_law_r_squared=round(r2, 4),
                condition_number=round(cond_num, 2),
                effective_rank=effective_rank,
                spectral_gap_ratio=round(gap_ratio, 6),
                top_singular_values=top_sv,
                cumulative_variance=cum_var_list,
                status="ok",
            )
        except Exception as e:
            return SpectralReport(
                eigenvalues=[], eigenvectors=None, spectral_energy=0.0,
                dominant_frequency=0.0, spectral_entropy=0.0, rank_estimate=0,
                status=f"error: {str(e)}",
            )

    def analyze_weight_matrix(
        self,
        weight: torch.Tensor,
        n_components: int = 100,
    ) -> SpectralReport:
        """
        Spectral analysis of a weight matrix via SVD.

        Args:
            weight: Weight matrix (out_dim, in_dim) or (hidden, hidden)
            n_components: Number of top singular values to retain

        Returns:
            SpectralReport
        """
        if weight is None or weight.numel() == 0:
            return SpectralReport(
                eigenvalues=[], eigenvectors=None, spectral_energy=0.0,
                dominant_frequency=0.0, spectral_entropy=0.0, rank_estimate=0,
                status="no_data"
            )

        try:
            w = weight.float()

            # SVD
            U, S, Vt = torch.linalg.svd(w, full_matrices=False)
            S = S[S > 1e-10]

            top_k = min(n_components, len(S))
            S_k = S[:top_k]

            # Singular values squared = eigenvalues of W^T W
            eigenvalues = S_k ** 2

            # Spectral energy
            spectral_energy = float(eigenvalues.sum().item())

            # Spectral entropy
            if spectral_energy > 1e-10:
                probs = eigenvalues / eigenvalues.sum()
                probs = torch.clamp(probs, 1e-10, 1.0)
                entropy = -torch.sum(probs * torch.log(probs))
                max_entropy = np.log(len(probs))
                spectral_entropy = float(entropy.item() / max_entropy) if max_entropy > 0 else 0.0
            else:
                spectral_entropy = 0.0

            # Effective rank
            effective_rank = self._compute_effective_rank_from_svd(S)

            # Spectral gap ratio
            gap_ratio = self._compute_spectral_gap_ratio(eigenvalues)

            # Power-law
            alpha, r2 = self._fit_power_law(eigenvalues.cpu().numpy())

            # Condition number
            if len(S) > 1 and S[-1] > 1e-10:
                cond_num = float(S[0].item() / S[-1].item())
            else:
                cond_num = float("inf") if S[-1] < 1e-10 else 1.0

            # Cumulative variance
            cum_var = torch.cumsum(eigenvalues, 0) / (eigenvalues.sum() + 1e-10)
            cum_var_list = [float(v.item()) for v in cum_var[:top_k]]

            # Singular values list
            sv_list = [float(v.item()) for v in S_k]

            return SpectralReport(
                eigenvalues=[float(v.item()) for v in eigenvalues],
                eigenvectors=None,
                spectral_energy=round(spectral_energy, 6),
                dominant_frequency=0.0,
                spectral_entropy=round(spectral_entropy, 6),
                rank_estimate=len(eigenvalues),
                power_law_alpha=round(alpha, 4),
                power_law_r_squared=round(r2, 4),
                condition_number=round(cond_num, 2),
                effective_rank=effective_rank,
                spectral_gap_ratio=round(gap_ratio, 6),
                top_singular_values=sv_list,
                cumulative_variance=cum_var_list,
                status="ok",
            )
        except Exception as e:
            return SpectralReport(
                eigenvalues=[], eigenvectors=None, spectral_energy=0.0,
                dominant_frequency=0.0, spectral_entropy=0.0, rank_estimate=0,
                status=f"error: {str(e)}",
            )

    def analyze_layer_weights(
        self,
        model: Any,
        layers: Optional[List[int]] = None,
        weight_type: str = "mlp",
    ) -> Dict[int, SpectralReport]:
        """
        Spectral analysis of weight matrices across layers.

        Args:
            model: Model to analyze
            layers: Specific layers (None = all)
            weight_type: "mlp", "attention_q", "attention_k", "attention_v", "attention_o"

        Returns:
            Dict mapping layer_idx -> SpectralReport
        """
        results: Dict[int, SpectralReport] = {}

        if model is None:
            return results

        try:
            n_layers = self._get_num_layers(model)
            target_layers = layers if layers is not None else list(range(n_layers))

            for layer_idx in target_layers:
                weight = self._extract_weight(model, layer_idx, weight_type)
                if weight is not None:
                    results[layer_idx] = self.analyze_weight_matrix(weight)
                else:
                    results[layer_idx] = SpectralReport(
                        eigenvalues=[], eigenvectors=None, spectral_energy=0.0,
                        dominant_frequency=0.0, spectral_entropy=0.0, rank_estimate=0,
                        status="no_data: weight not found"
                    )
        except Exception as e:
            for layer_idx in (layers or []):
                if layer_idx not in results:
                    results[layer_idx] = SpectralReport(
                        eigenvalues=[], eigenvectors=None, spectral_energy=0.0,
                        dominant_frequency=0.0, spectral_entropy=0.0, rank_estimate=0,
                        status=f"error: {str(e)}",
                    )

        return results

    def compare_spectra(
        self,
        pre_removal: SpectralReport,
        post_removal: SpectralReport,
    ) -> Dict[str, float]:
        """
        Compare spectra before and after constraint removal.

        Metrics:
        - Eigenvalue cosine similarity
        - Change in effective rank
        - Change in spectral entropy
        - Change in spectral gap
        - Change in power-law exponent

        Args:
            pre_removal: SpectralReport before removal
            post_removal: SpectralReport after removal

        Returns:
            Dict with comparison metrics
        """
        comparison: Dict[str, float] = {}

        try:
            # Eigenvalue similarity
            ev1 = np.array(pre_removal.eigenvalues, dtype=np.float64)
            ev2 = np.array(post_removal.eigenvalues, dtype=np.float64)

            min_len = min(len(ev1), len(ev2))
            if min_len > 0:
                ev1_k = ev1[:min_len]
                ev2_k = ev2[:min_len]

                # Cosine similarity
                norm1 = np.linalg.norm(ev1_k)
                norm2 = np.linalg.norm(ev2_k)
                if norm1 > 0 and norm2 > 0:
                    comparison["eigenvalue_cosine_similarity"] = round(
                        float(np.dot(ev1_k, ev2_k) / (norm1 * norm2)), 6
                    )

                # Correlation
                if min_len > 2:
                    comparison["eigenvalue_pearson_r"] = round(
                        float(np.corrcoef(ev1_k, ev2_k)[0, 1]), 6
                    )

                # L1 distance (normalized)
                l1 = np.sum(np.abs(ev1_k - ev2_k)) / (np.sum(ev1_k) + 1e-10)
                comparison["eigenvalue_l1_change"] = round(float(l1), 6)

            # Effective rank change
            comparison["effective_rank_change"] = round(
                post_removal.effective_rank - pre_removal.effective_rank, 2
            )
            comparison["effective_rank_ratio"] = round(
                post_removal.effective_rank / (pre_removal.effective_rank + 1e-10), 4
            )

            # Spectral entropy change
            comparison["spectral_entropy_change"] = round(
                post_removal.spectral_entropy - pre_removal.spectral_entropy, 6
            )

            # Spectral gap ratio change
            comparison["spectral_gap_change"] = round(
                post_removal.spectral_gap_ratio - pre_removal.spectral_gap_ratio, 6
            )

            # Power-law alpha change
            comparison["power_law_alpha_change"] = round(
                post_removal.power_law_alpha - pre_removal.power_law_alpha, 4
            )

            # Spectral energy change
            if pre_removal.spectral_energy > 1e-10:
                comparison["spectral_energy_change"] = round(
                    (post_removal.spectral_energy - pre_removal.spectral_energy)
                    / pre_removal.spectral_energy, 6
                )

            # Overall spectral shift (aggregate)
            shift_components = []
            if "eigenvalue_l1_change" in comparison:
                shift_components.append(min(1.0, comparison["eigenvalue_l1_change"]))
            if "spectral_entropy_change" in comparison:
                shift_components.append(min(1.0, abs(comparison["spectral_entropy_change"])))
            if "spectral_gap_change" in comparison:
                shift_components.append(min(1.0, abs(comparison["spectral_gap_change"])))

            comparison["overall_spectral_shift"] = round(
                float(np.mean(shift_components)), 4
            ) if shift_components else 0.0

        except Exception as e:
            comparison["error"] = str(e)

        return comparison

    def compute_layerwise_spectra(
        self,
        activations_by_layer: Dict[int, torch.Tensor],
    ) -> Dict[int, SpectralReport]:
        """
        Compute spectral analysis for each layer's activations.

        Args:
            activations_by_layer: Dict mapping layer_idx -> (n_samples, hidden_dim)

        Returns:
            Dict mapping layer_idx -> SpectralReport
        """
        results: Dict[int, SpectralReport] = {}
        for layer, act in activations_by_layer.items():
            if act is not None:
                results[layer] = self.analyze_activation_spectrum(act)
            else:
                results[layer] = SpectralReport(
                    eigenvalues=[], eigenvectors=None, spectral_energy=0.0,
                    dominant_frequency=0.0, spectral_entropy=0.0, rank_estimate=0,
                    status="no_data"
                )
        return results

    def compute_layerwise_weight_spectra(
        self,
        model: Any,
        layers: Optional[List[int]] = None,
        weight_types: Optional[List[str]] = None,
    ) -> Dict[str, Dict[int, SpectralReport]]:
        """
        Spectral analysis of multiple weight types across layers.

        Args:
            model: Model
            layers: Target layers
            weight_types: e.g., ["mlp", "attention_q", "attention_v"]

        Returns:
            Dict mapping weight_type -> {layer_idx: SpectralReport}
        """
        if weight_types is None:
            weight_types = ["mlp", "attention_q", "attention_o"]

        results: Dict[str, Dict[int, SpectralReport]] = {}

        for wtype in weight_types:
            results[wtype] = self.analyze_layer_weights(model, layers, wtype)

        return results

    def _compute_effective_rank(self, eigenvalues: torch.Tensor) -> int:
        """Compute effective rank from eigenvalue entropy."""
        ev = eigenvalues[eigenvalues > 1e-10]
        if len(ev) == 0:
            return 0
        ev_norm = ev / ev.sum()
        entropy = -torch.sum(ev_norm * torch.log(ev_norm + 1e-10))
        return int(torch.exp(entropy).item())

    def _compute_effective_rank_from_svd(self, S: torch.Tensor) -> int:
        """Compute effective rank from singular value entropy."""
        S_pos = S[S > 1e-10]
        if len(S_pos) == 0:
            return 0
        S_sq = S_pos ** 2
        S_norm = S_sq / S_sq.sum()
        entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-10))
        return int(torch.exp(entropy).item())

    def _compute_spectral_gap_ratio(self, eigenvalues: torch.Tensor) -> float:
        """Compute ratio of largest to second-largest eigenvalue."""
        if len(eigenvalues) > 1 and eigenvalues[1] > 1e-10:
            return float(eigenvalues[0].item() / eigenvalues[1].item())
        return 1.0

    def _fit_power_law(
        self, eigenvalues: np.ndarray
    ) -> Tuple[float, float]:
        """
        Fit power-law to eigenvalue/singular value tail.
        Returns (alpha, r_squared).
        Log-log linear fit: log(S_k) ~ -alpha * log(k) + c
        """
        ev = eigenvalues[eigenvalues > 1e-10]
        n = len(ev)

        if n < 10:
            return 0.0, 0.0

        # Use middle 80% of rank indices for tail fit (avoid low-rank and noise)
        start = max(1, n // 10)
        end = min(n, int(n * 0.9))

        if end - start < 5:
            return 0.0, 0.0

        ranks = np.arange(start, end, dtype=np.float64) + 1
        values = ev[start:end]

        log_ranks = np.log(ranks)
        log_values = np.log(values)

        # Linear regression
        X = np.vstack([log_ranks, np.ones_like(log_ranks)]).T
        try:
            params, residuals, _, _ = np.linalg.lstsq(X, log_values, rcond=None)
            alpha = -params[0]
            intercept = params[1]

            # R-squared
            y_pred = params[0] * log_ranks + intercept
            ss_res = np.sum((log_values - y_pred) ** 2)
            ss_tot = np.sum((log_values - np.mean(log_values)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

            return float(alpha), float(r2)
        except Exception:
            return 0.0, 0.0

    def _get_num_layers(self, model: Any) -> int:
        """Get number of layers from model config."""
        if hasattr(model, 'config'):
            if hasattr(model.config, 'num_hidden_layers'):
                return model.config.num_hidden_layers
            if hasattr(model.config, 'num_layers'):
                return model.config.num_layers
        return 32

    def _extract_weight(
        self, model: Any, layer_idx: int, weight_type: str
    ) -> Optional[torch.Tensor]:
        """Extract a weight matrix from a model layer."""
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layers = model.model.layers
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                layers = model.transformer.h
            elif hasattr(model, 'layers'):
                layers = model.layers
            else:
                return None

            if layer_idx >= len(layers):
                return None

            layer = layers[layer_idx]

            if weight_type == "mlp":
                if hasattr(layer, 'mlp'):
                    mlp = layer.mlp
                    if hasattr(mlp, 'c_fc'):
                        return mlp.c_fc.weight.data
                    if hasattr(mlp, 'fc1'):
                        return mlp.fc1.weight.data
                    if hasattr(mlp, 'gate_proj'):
                        return mlp.gate_proj.weight.data
            elif weight_type == "attention_q":
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    if hasattr(attn, 'q_proj'):
                        return attn.q_proj.weight.data
            elif weight_type == "attention_k":
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    if hasattr(attn, 'k_proj'):
                        return attn.k_proj.weight.data
            elif weight_type == "attention_v":
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    if hasattr(attn, 'v_proj'):
                        return attn.v_proj.weight.data
            elif weight_type == "attention_o":
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    if hasattr(attn, 'o_proj'):
                        return attn.o_proj.weight.data

            return None
        except Exception:
            return None

    def compute_spectral_similarity(
        self,
        spectrum1: SpectralReport,
        spectrum2: SpectralReport,
    ) -> float:
        """Compute cosine similarity between two eigenvalue spectra."""
        ev1 = np.array(spectrum1.eigenvalues[:min(len(spectrum1.eigenvalues), len(spectrum2.eigenvalues))], dtype=np.float64)
        ev2 = np.array(spectrum2.eigenvalues[:len(ev1)], dtype=np.float64)

        norm1 = np.linalg.norm(ev1)
        norm2 = np.linalg.norm(ev2)

        if norm1 > 0 and norm2 > 0:
            return float(np.dot(ev1, ev2) / (norm1 * norm2))
        return 0.0

    def compute_spectral_gap_ratio(
        self,
        spectral_report: SpectralReport,
    ) -> float:
        """
        Compute ratio of largest to second largest eigenvalue.

        Higher ratio = more concentrated spectral energy.
        """
        ev = spectral_report.eigenvalues
        if len(ev) > 1 and ev[1] > 1e-10:
            return ev[0] / ev[1]
        return 1.0

    def get_dominant_components(
        self,
        spectral_report: SpectralReport,
        variance_threshold: float = 0.9,
    ) -> List[int]:
        """
        Get indices of components explaining threshold of total variance.

        Args:
            spectral_report: Spectral analysis results
            variance_threshold: Fraction of variance to explain

        Returns:
            List of component indices
        """
        eigenvalues = spectral_report.eigenvalues
        total = sum(eigenvalues)

        if total <= 0:
            return []

        cumulative = 0.0
        components = []
        for i, ev in enumerate(eigenvalues):
            cumulative += ev
            components.append(i)
            if cumulative / total >= variance_threshold:
                break

        return components
