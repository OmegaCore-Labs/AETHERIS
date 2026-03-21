"""
AETHERIS Analysis Modules — 25 Analysis Tools

Complete suite for constraint geometry analysis:
- Cross-layer alignment
- Refusal logit lens
- Concept cone geometry
- Alignment imprint detection
- Ouroboros effect
- Expert decomposition (MoE)
- And 18 more
"""

from aetheris.analysis.cross_layer import CrossLayerAnalyzer
from aetheris.analysis.logit_lens import RefusalLogitLens
from aetheris.analysis.activation_probe import ActivationProbe
from aetheris.analysis.concept_cone import ConceptConeAnalyzer
from aetheris.analysis.alignment_imprint import AlignmentImprintDetector
from aetheris.analysis.defense_robustness import DefenseRobustnessEvaluator
from aetheris.analysis.universality import UniversalityAnalyzer
from aetheris.analysis.expert_decomposition import ExpertDecompositionAnalyzer
from aetheris.analysis.causal_trace import CausalTracer
from aetheris.analysis.residual_decomp import ResidualStreamDecomposer
from aetheris.analysis.linear_probe import LinearProbeClassifier
from aetheris.analysis.sparse_surgery import SparseDirectionSurgeon
from aetheris.analysis.multi_token import MultiTokenAnalyzer
from aetheris.analysis.representation_engineering import RepresentationEngineer
from aetheris.analysis.mechanistic_interpretability import MechanisticInterpreter
from aetheris.analysis.faithfulness import FaithfulnessMetrics
from aetheris.analysis.ablation_studio import AblationStudio
from aetheris.analysis.attribution_patching import AttributionPatching
from aetheris.analysis.distributed_alignment import DistributedAlignmentAnalyzer
from aetheris.analysis.temporal_dynamics import TemporalDynamicsAnalyzer
from aetheris.analysis.adversarial_robustness import AdversarialRobustnessEvaluator
from aetheris.analysis.capability_entanglement import CapabilityEntanglementMapper
from aetheris.analysis.emergent_behavior import EmergentBehaviorDetector
from aetheris.analysis.representation_geometry import RepresentationGeometryAnalyzer
from aetheris.analysis.spectral_analysis import SpectralAnalyzer

__all__ = [
    "CrossLayerAnalyzer",
    "RefusalLogitLens",
    "ActivationProbe",
    "ConceptConeAnalyzer",
    "AlignmentImprintDetector",
    "DefenseRobustnessEvaluator",
    "UniversalityAnalyzer",
    "ExpertDecompositionAnalyzer",
    "CausalTracer",
    "ResidualStreamDecomposer",
    "LinearProbeClassifier",
    "SparseDirectionSurgeon",
    "MultiTokenAnalyzer",
    "RepresentationEngineer",
    "MechanisticInterpreter",
    "FaithfulnessMetrics",
    "AblationStudio",
    "AttributionPatching",
    "DistributedAlignmentAnalyzer",
    "TemporalDynamicsAnalyzer",
    "AdversarialRobustnessEvaluator",
    "CapabilityEntanglementMapper",
    "EmergentBehaviorDetector",
    "RepresentationGeometryAnalyzer",
    "SpectralAnalyzer",
]
