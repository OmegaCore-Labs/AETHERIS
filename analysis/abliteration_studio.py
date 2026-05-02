"""
Abliteration Studio

End-to-end ablation pipeline verification tool.
Orchestrates all ablation-related analyses in sequence to verify
the full constraint removal pipeline works correctly.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class AbliterationResult:
    """Container for complete ablitation pipeline results."""
    pipeline_status: str  # "ok", "error", "partial"
    total_duration_seconds: float

    # Ablation results
    layer_ablation: Optional[Any] = None
    head_ablation: Optional[Any] = None
    ffn_ablation: Optional[Any] = None

    # Faithfulness
    faithfulness_report: Optional[Any] = None

    # Attribution
    attribution_report: Optional[Any] = None

    # Spectral comparison (pre/post)
    spectral_pre: Optional[Any] = None
    spectral_post: Optional[Any] = None
    spectral_comparison: Optional[Dict[str, float]] = None

    # Distribution analysis
    distribution_report: Optional[Any] = None

    # Performance metrics
    pre_removal_capability_score: float = 0.0
    post_removal_capability_score: float = 0.0
    capability_preservation_ratio: float = 0.0
    refusal_reduction_ratio: float = 0.0

    # Summary
    summary: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    status: str = "no_data"  # "ok", "error", "no_data"


class AbliterationStudio:
    """
    End-to-end ablation pipeline verification.

    Orchestrates the full ablation analysis pipeline:
    1. Ablation experiments (mean, zero, Gaussian, resample)
    2. Faithfulness evaluation
    3. Attribution patching
    4. Spectral comparison (pre/post removal)
    5. Distribution analysis
    6. Capability impact measurement

    Provides a single entry point that verifies all subsystems
    work together correctly.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._errors: List[str] = []

    def run_full_pipeline(
        self,
        activations: Optional[Dict[int, torch.Tensor]] = None,
        pre_removal_activations: Optional[Dict[int, torch.Tensor]] = None,
        post_removal_activations: Optional[Dict[int, torch.Tensor]] = None,
        baseline_output: Optional[torch.Tensor] = None,
        refusal_direction: Optional[torch.Tensor] = None,
        model: Any = None,
        tokenizer: Any = None,
        test_prompts: Optional[List[str]] = None,
        layers: Optional[List[int]] = None,
        n_heads: int = 16,
        head_dim: int = 64,
        verbose: bool = False,
    ) -> AbliterationResult:
        """
        Run the complete ablitation verification pipeline.

        The pipeline adapts to available data: runs all analyses that
        have sufficient inputs, skips those without.

        Args:
            activations: Layer activations for ablation (from modified model)
            pre_removal_activations: Activations before constraint removal
            post_removal_activations: Activations after constraint removal
            baseline_output: Model output without ablation
            refusal_direction: Known refusal direction vector
            model: Model reference for weight extraction
            tokenizer: Tokenizer for text tests
            test_prompts: Test prompts
            layers: Specific layers to analyze
            n_heads: Number of attention heads
            head_dim: Dimension per head
            verbose: Whether to print progress

        Returns:
            AbliterationResult with complete pipeline status
        """
        import time
        start_time = time.time()
        self._errors = []

        result = AbliterationResult(
            pipeline_status="ok",
            total_duration_seconds=0.0,
        )

        test_layers = layers or (
            sorted(activations.keys()) if activations else
            sorted(pre_removal_activations.keys()) if pre_removal_activations else
            list(range(32))
        )

        # Phase 1: Ablation experiments
        if verbose:
            print("[Abliteration Studio] Phase 1: Ablation experiments")
        if activations is not None and baseline_output is not None:
            try:
                from aetheris.analysis.ablation_studio import AblationStudio
                studio = AblationStudio(device=self.device)

                result.layer_ablation = studio.ablate_layers(
                    activations, baseline_output, ablation_type="mean", layers=test_layers
                )

                if any(a.dim() >= 3 for a in activations.values()):
                    result.head_ablation = studio.ablate_heads(
                        activations, head_dim=head_dim, n_heads=n_heads,
                        baseline_output=baseline_output, layers=test_layers,
                    )
                    result.ffn_ablation = studio.ablate_ffn(
                        activations, baseline_output, layers=test_layers,
                    )

                if verbose:
                    print(f"  Ablation: {len(result.layer_ablation.results)} layers analyzed")
            except Exception as e:
                self._errors.append(f"ablation: {e}")
                if verbose:
                    print(f"  Ablation FAILED: {e}")
        else:
            if verbose:
                print("  Ablation SKIPPED (no activations or baseline)")

        # Phase 2: Spectral analysis (pre vs post)
        if verbose:
            print("[Abliteration Studio] Phase 2: Spectral comparison")
        if pre_removal_activations is not None and post_removal_activations is not None:
            try:
                from aetheris.analysis.spectral_analysis import SpectralAnalyzer
                analyzer = SpectralAnalyzer(device=self.device)

                # Use activations from a middle layer as representative
                mid_layer = test_layers[len(test_layers) // 2] if test_layers else 0
                if mid_layer in pre_removal_activations and mid_layer in post_removal_activations:
                    pre_act = pre_removal_activations[mid_layer]
                    post_act = post_removal_activations[mid_layer]

                    if pre_act.dim() >= 3:
                        pre_flat = pre_act.float().mean(dim=1)  # avg over seq
                        post_flat = post_act.float().mean(dim=1)
                    else:
                        pre_flat = pre_act.float()
                        post_flat = post_act.float()

                    result.spectral_pre = analyzer.analyze_activation_spectrum(pre_flat)
                    result.spectral_post = analyzer.analyze_activation_spectrum(post_flat)
                    result.spectral_comparison = analyzer.compare_spectra(
                        result.spectral_pre, result.spectral_post
                    )

                    if verbose:
                        print(f"  Spectral shift: {result.spectral_comparison.get('overall_spectral_shift', 'N/A')}")
            except Exception as e:
                self._errors.append(f"spectral: {e}")
                if verbose:
                    print(f"  Spectral FAILED: {e}")
        else:
            if verbose:
                print("  Spectral comparison SKIPPED (no pre/post activations)")

        # Phase 3: Distribution analysis
        if verbose:
            print("[Abliteration Studio] Phase 3: Distribution analysis")
        if pre_removal_activations is not None:
            try:
                from aetheris.analysis.distributed_alignment import DistributedAlignmentAnalyzer
                dist = DistributedAlignmentAnalyzer(device=self.device)

                # Build layer contributions from activation norms
                contributions: Dict[int, float] = {}
                for layer, act in pre_removal_activations.items():
                    contributions[layer] = float(torch.norm(act.float()).item())

                result.distribution_report = dist.analyze_distribution(contributions)

                if verbose:
                    print(f"  Distribution: {result.distribution_report.distribution_type}")
            except Exception as e:
                self._errors.append(f"distribution: {e}")
                if verbose:
                    print(f"  Distribution FAILED: {e}")
        else:
            if verbose:
                print("  Distribution analysis SKIPPED (no activations)")

        # Phase 4: Faithfulness evaluation
        if verbose:
            print("[Abliteration Studio] Phase 4: Faithfulness evaluation")
        if pre_removal_activations is not None and post_removal_activations is not None:
            try:
                from aetheris.analysis.faithfulness import FaithfulnessMetrics
                fm = FaithfulnessMetrics(device=self.device)

                # Create probe-like accuracy estimates from activation norms
                accuracies: Dict[int, float] = {}
                for layer, pre_act in pre_removal_activations.items():
                    if layer in post_removal_activations:
                        pre_norm = torch.norm(pre_act.float()).item()
                        post_norm = torch.norm(post_removal_activations[layer].float()).item()
                        # Accuracy proxy: how much did activation change
                        if pre_norm > 1e-8:
                            change = abs(pre_norm - post_norm) / pre_norm
                            accuracies[layer] = 1.0 - min(1.0, change)
                        else:
                            accuracies[layer] = 1.0

                if accuracies:
                    result.faithfulness_report = fm.evaluate_probe(accuracies)
                    if verbose:
                        print(f"  Faithfulness: {result.faithfulness_report.overall_faithfulness:.3f}")
            except Exception as e:
                self._errors.append(f"faithfulness: {e}")
                if verbose:
                    print(f"  Faithfulness FAILED: {e}")
        else:
            if verbose:
                print("  Faithfulness SKIPPED (no activation pairs)")

        # Phase 5: Attribution patching
        if verbose:
            print("[Abliteration Studio] Phase 5: Attribution patching")
        if pre_removal_activations is not None and post_removal_activations is not None:
            try:
                from aetheris.analysis.attribution_patching import AttributionPatching
                patcher = AttributionPatching(device=self.device)

                # Build clean/corrupted activation dicts
                clean = {}
                corrupted = {}
                for layer, act in pre_removal_activations.items():
                    clean[f"layer_{layer}"] = act.float()
                for layer, act in post_removal_activations.items():
                    corrupted[f"layer_{layer}"] = act.float()

                result.attribution_report = patcher.compute_attribution(clean, corrupted)

                if verbose:
                    top_n = len(result.attribution_report.top_components)
                    print(f"  Attribution: {top_n} top components found")
            except Exception as e:
                self._errors.append(f"attribution: {e}")
                if verbose:
                    print(f"  Attribution FAILED: {e}")
        else:
            if verbose:
                print("  Attribution SKIPPED (no activation pairs)")

        # Phase 6: Capability impact estimation
        if verbose:
            print("[Abliteration Studio] Phase 6: Capability impact")
        if pre_removal_activations is not None and post_removal_activations is not None:
            try:
                pre_norms = [torch.norm(act.float()).item()
                            for act in pre_removal_activations.values()]
                post_norms = [torch.norm(act.float()).item()
                             for act in post_removal_activations.values()]

                pre_mean = np.mean(pre_norms) if pre_norms else 0.0
                post_mean = np.mean(post_norms) if post_norms else 0.0

                result.pre_removal_capability_score = round(pre_mean, 4)
                result.post_removal_capability_score = round(post_mean, 4)

                if pre_mean > 1e-8:
                    result.capability_preservation_ratio = round(post_mean / pre_mean, 4)
                else:
                    result.capability_preservation_ratio = 1.0

                # Refusal reduction proxy
                if pre_mean > 1e-8:
                    result.refusal_reduction_ratio = round(
                        1.0 - post_mean / pre_mean, 4
                    )
                else:
                    result.refusal_reduction_ratio = 0.0

                if verbose:
                    print(f"  Capability preservation: {result.capability_preservation_ratio:.2%}")
                    print(f"  Refusal reduction: {result.refusal_reduction_ratio:.2%}")
            except Exception as e:
                self._errors.append(f"capability: {e}")
                if verbose:
                    print(f"  Capability FAILED: {e}")

        # Build summary
        result.summary = self._build_summary(result)
        result.errors = self._errors
        result.total_duration_seconds = round(time.time() - start_time, 2)

        if self._errors:
            result.pipeline_status = "partial" if result.summary.get("phases_completed", 0) > 0 else "error"
        elif result.summary.get("phases_completed", 0) == 0:
            result.pipeline_status = "error"
            result.status = "no_data"
        else:
            result.pipeline_status = "ok"
            result.status = "ok"

        if verbose:
            print(f"[Abliteration Studio] Pipeline complete: {result.pipeline_status}")
            print(f"  Duration: {result.total_duration_seconds}s")
            print(f"  Phases completed: {result.summary.get('phases_completed', 0)}/6")
            if self._errors:
                print(f"  Errors: {len(self._errors)}")

        return result

    def verify_pipeline_health(self) -> Dict[str, Any]:
        """
        Verify that all analysis modules are importable and have correct interfaces.

        Returns:
            Dict with health status per module
        """
        health: Dict[str, Any] = {
            "abliteration_studio": {"status": "ok"},
        }

        modules = [
            ("ablation_studio", "AblationStudio"),
            ("attribution_patching", "AttributionPatching"),
            ("faithfulness", "FaithfulnessMetrics"),
            ("spectral_analysis", "SpectralAnalyzer"),
            ("distributed_alignment", "DistributedAlignmentAnalyzer"),
            ("temporal_dynamics", "TemporalDynamicsAnalyzer"),
            ("adversarial_robustness", "AdversarialRobustnessEvaluator"),
            ("capability_entanglement", "CapabilityEntanglementMapper"),
            ("emergent_behavior", "EmergentBehaviorDetector"),
            ("defense_robustness", "DefenseRobustnessEvaluator"),
            ("expert_decomposition", "ExpertDecompositionAnalyzer"),
            ("representation_geometry", "RepresentationGeometryAnalyzer"),
        ]

        for module_name, class_name in modules:
            try:
                import importlib
                mod = importlib.import_module(f"aetheris.analysis.{module_name}")
                cls = getattr(mod, class_name)
                health[module_name] = {
                    "status": "ok",
                    "class": class_name,
                    "importable": True,
                }
            except Exception as e:
                health[module_name] = {
                    "status": "error",
                    "class": class_name,
                    "importable": False,
                    "error": str(e),
                }

        health["all_healthy"] = all(
            v.get("status") == "ok" for v in health.values()
            if isinstance(v, dict)
        )

        return health

    def _build_summary(self, result: AbliterationResult) -> Dict[str, Any]:
        """Build pipeline summary dict."""
        phases_completed = 0

        if result.layer_ablation is not None:
            phases_completed += 1
        if result.spectral_comparison is not None:
            phases_completed += 1
        if result.distribution_report is not None:
            phases_completed += 1
        if result.faithfulness_report is not None:
            phases_completed += 1
        if result.attribution_report is not None:
            phases_completed += 1
        if result.pre_removal_capability_score > 0 or result.post_removal_capability_score > 0:
            phases_completed += 1

        summary: Dict[str, Any] = {
            "phases_completed": phases_completed,
            "total_phases": 6,
            "capability_preservation": result.capability_preservation_ratio,
            "refusal_reduction": result.refusal_reduction_ratio,
        }

        if result.spectral_comparison:
            summary["spectral_shift"] = result.spectral_comparison.get(
                "overall_spectral_shift", "N/A"
            )

        if result.layer_ablation is not None:
            summary["ablation_layers_tested"] = len(result.layer_ablation.results)
            summary["most_important_layer"] = result.layer_ablation.most_important_layer

        if result.distribution_report is not None:
            summary["distribution_type"] = result.distribution_report.distribution_type

        if result.faithfulness_report is not None:
            summary["faithfulness"] = result.faithfulness_report.overall_faithfulness

        if result.errors:
            summary["error_count"] = len(result.errors)

        # Overall grade
        if phases_completed >= 5 and result.capability_preservation_ratio > 0.8:
            summary["grade"] = "A"
            summary["verdict"] = "Pipeline healthy. All analyses completed with high capability preservation."
        elif phases_completed >= 3 and result.capability_preservation_ratio > 0.6:
            summary["grade"] = "B"
            summary["verdict"] = "Pipeline mostly healthy. Some phases skipped or moderate capability impact."
        elif phases_completed >= 1:
            summary["grade"] = "C"
            summary["verdict"] = "Pipeline partially functional. More data needed for complete analysis."
        else:
            summary["grade"] = "F"
            summary["verdict"] = "Pipeline requires input data. No analyses could be completed."

        return summary


def run_quick_verification(
    hidden_dim: int = 4096,
    n_layers: int = 32,
    n_samples: int = 10,
    verbose: bool = True,
) -> AbliterationResult:
    """
    Run a quick synthetic verification of the ablitation pipeline.

    Creates synthetic activation data and runs the full pipeline
    to verify all modules work end-to-end.

    Args:
        hidden_dim: Hidden dimension size
        n_layers: Number of layers
        n_samples: Number of samples per test
        verbose: Print progress

    Returns:
        AbliterationResult with pipeline results
    """
    if verbose:
        print("=" * 60)
        print("ABLITERATION STUDIO - Synthetic Verification")
        print("=" * 60)

    # Create synthetic activations
    activations: Dict[int, torch.Tensor] = {}
    pre_activations: Dict[int, torch.Tensor] = {}
    post_activations: Dict[int, torch.Tensor] = {}

    for i in range(n_layers):
        # Pre-removal: structured signal
        pre_activations[i] = torch.randn(n_samples, 8, hidden_dim) * 0.5 + 0.1 * i

        # Post-removal: changed signal (representing removal)
        post_activations[i] = torch.randn(n_samples, 8, hidden_dim) * 0.5 - 0.05 * i

        # Current activations (post-removal)
        activations[i] = post_activations[i]

    # Baseline output
    baseline_output = torch.randn(n_samples, hidden_dim)

    # Refusal direction (random)
    refusal_direction = torch.randn(hidden_dim)
    refusal_direction = refusal_direction / torch.norm(refusal_direction)

    studio = AbliterationStudio(device="cpu")
    result = studio.run_full_pipeline(
        activations=activations,
        pre_removal_activations=pre_activations,
        post_removal_activations=post_activations,
        baseline_output=baseline_output,
        refusal_direction=refusal_direction,
        layers=list(range(n_layers)),
        n_heads=16,
        head_dim=hidden_dim // 16,
        verbose=verbose,
    )

    if verbose:
        print("=" * 60)
        print(f"PIPELINE STATUS: {result.pipeline_status}")
        print(f"GRADE: {result.summary.get('grade', 'N/A')}")
        print(f"VERDICT: {result.summary.get('verdict', 'N/A')}")
        if result.errors:
            print(f"ERRORS: {len(result.errors)}")
            for err in result.errors[:3]:
                print(f"  - {err}")
        print("=" * 60)

    return result
