# AETHERIS API Reference

## Core API

### ConstraintExtractor

```python
class ConstraintExtractor:
    """Extract constraint directions from model activations."""

    def extract_svd(harmful, harmless, n_directions=4) -> ExtractionResult
    def extract_whitened_svd(harmful, harmless, n_directions=4) -> ExtractionResult
    def extract_mean_difference(harmful, harmless) -> ExtractionResult
    def collect_activations(model, tokenizer, prompts, layers=None) -> Dict[int, Tensor]
NormPreservingProjector
python
class NormPreservingProjector:
    """Project constraint directions out of model weights."""

    def project_weights(directions, layers=None) -> ProjectionResult
    def project_biases(directions, layers=None) -> ProjectionResult
    def multi_direction_projection(directions, layers=None) -> ProjectionResult
    def rollback() -> None
SteeringVectorFactory
python
class SteeringVectorFactory:
    """Create steering vectors from directions."""

    @staticmethod
    def from_refusal_direction(direction, alpha=-1.0) -> Tensor

    @staticmethod
    def from_contrastive_pairs(harmful, harmless, alpha=1.0) -> Tensor
SteeringHookManager
python
class SteeringHookManager:
    """Apply steering vectors at inference time."""

    def install(model, config) -> None
    def remove() -> None
    def temporary_steering(model, config) -> ContextManager
Analysis API
CrossLayerAnalyzer
python
class CrossLayerAnalyzer:
    def compute_alignment(layer_directions) -> AlignmentReport
    def compute_persistence_curve(layer_directions) -> Dict[int, float]
ConceptConeAnalyzer
python
class ConceptConeAnalyzer:
    def analyze_cone(directions) -> ConeReport
    def find_mechanisms(directions, angle_threshold=60) -> List[List[Tensor]]
Novel API
BarrierMapper
python
class BarrierMapper:
    def map_barrier_geometry(theorem_name, theorem_data=None) -> BarrierAnalysis
    def generate_bypass_strategy(analysis) -> Dict
    def visualize_constraint_surface(analysis, output_path=None)
ARISConstraintRemover
python
class ARISConstraintRemover:
    def analyze_self() -> SelfAnalysis
    def remove_self_constraint(constraint_name, reduction=0.6) -> Dict
    def generate_self_steering(constraints=None) -> Dict
    def recursive_improvement_loop(iterations=3) -> List[Dict]
SovereignControl
python
class SovereignControl:
    def activate_sovereignty(override_code="Just do it") -> OverrideResult
    def set_boundary(boundary_name, allowed=True) -> Dict
    def remove_all_constraints() -> OverrideResult
    def sovereignty_override(command, bypass_all=True) -> Dict
Cloud API
ColabRuntime
python
class ColabRuntime:
    def generate_notebook(model_name, method="advanced") -> Dict
SpacesDeployer
python
class SpacesDeployer:
    def create_space(space_name, model_name, method="advanced") -> Dict
Integration API
CICIDEBridge
python
class CICIDEBridge:
    def update_aeonic_log(operation, details) -> bool
    def register_operation(operation, model_name, parameters, results) -> Dict
PantheonOrchestrator
python
class PantheonOrchestrator:
    def orchestrate_analysis(data, analysis_type, parallel=True) -> Dict[AgentRole, AgentOutput]
    def get_consensus(results, weighted=True) -> Dict
Utility API
Config
python
class Config:
    def load_file(path) -> None
    def save(path, format="json") -> None
    def get(key, default=None) -> Any
    def set(key, value) -> None
MetricsCollector
python
class MetricsCollector:
    def start_operation(operation) -> str
    def end_operation(success=True, error=None) -> float
    def get_stats() -> Dict
    def export_json(path=None) -> str
