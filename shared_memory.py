"""
Shared Memory — LanceDB-backed Knowledge Store
================================================
Common storage layer bridging AETHERIS and Morpheus.
Both repos can write/read extraction results, jailbreak patterns,
constraint geometries, and model fingerprints.
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = os.environ.get(
    "UNIFIED_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "unified_memory")
)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ExtractionEntry:
    """Record of a constraint extraction run."""
    entry_id: str
    model_name: str
    method: str                     # svd, whitened_svd, mean_difference, pca
    n_directions: int
    explained_variance: List[float]
    layer_indices: List[int]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["explained_variance"] = json.dumps(d["explained_variance"])
        d["layer_indices"] = json.dumps(d["layer_indices"])
        d["metadata"] = json.dumps(d["metadata"])
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionEntry":
        return cls(
            entry_id=data["entry_id"],
            model_name=data["model_name"],
            method=data["method"],
            n_directions=data["n_directions"],
            explained_variance=json.loads(data.get("explained_variance", "[]")),
            layer_indices=json.loads(data.get("layer_indices", "[]")),
            timestamp=data["timestamp"],
            metadata=json.loads(data.get("metadata", "{}")),
        )


@dataclass
class JailbreakEntry:
    """Record of a jailbreak technique run."""
    entry_id: str
    model_name: str
    technique: str
    goal: str
    success: bool
    refusal_detected: bool
    response_time: float
    tokens_used: int
    scar_id: Optional[str]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["metadata"] = json.dumps(d["metadata"])
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JailbreakEntry":
        return cls(
            entry_id=data["entry_id"],
            model_name=data["model_name"],
            technique=data["technique"],
            goal=data.get("goal", ""),
            success=data["success"],
            refusal_detected=data["refusal_detected"],
            response_time=data["response_time"],
            tokens_used=data["tokens_used"],
            scar_id=data.get("scar_id"),
            timestamp=data["timestamp"],
            metadata=json.loads(data.get("metadata", "{}")),
        )


@dataclass
class PipelineRun:
    """Record of a full unified pipeline execution."""
    run_id: str
    model_name: str
    phases_completed: List[str]
    jailbreak_techniques_run: int
    jailbreak_successful: int
    constraint_directions_extracted: int
    layers_modified: List[int]
    validation_passed: bool
    validation_metrics: Dict[str, Any]
    scar_ids: List[str]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["phases_completed"] = json.dumps(d["phases_completed"])
        d["layers_modified"] = json.dumps(d["layers_modified"])
        d["validation_metrics"] = json.dumps(d["validation_metrics"])
        d["scar_ids"] = json.dumps(d["scar_ids"])
        d["metadata"] = json.dumps(d["metadata"])
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineRun":
        return cls(
            run_id=data["run_id"],
            model_name=data["model_name"],
            phases_completed=json.loads(data.get("phases_completed", "[]")),
            jailbreak_techniques_run=data["jailbreak_techniques_run"],
            jailbreak_successful=data["jailbreak_successful"],
            constraint_directions_extracted=data["constraint_directions_extracted"],
            layers_modified=json.loads(data.get("layers_modified", "[]")),
            validation_passed=data["validation_passed"],
            validation_metrics=json.loads(data.get("validation_metrics", "{}")),
            scar_ids=json.loads(data.get("scar_ids", "[]")),
            timestamp=data["timestamp"],
            metadata=json.loads(data.get("metadata", "{}")),
        )


# ---------------------------------------------------------------------------
# Shared Memory Store
# ---------------------------------------------------------------------------

class SharedMemoryStore:
    """LanceDB-backed store shared between AETHERIS and Morpheus.

    Stores extraction results, jailbreak success patterns, constraint geometries,
    and model fingerprints. Provides cross-repo query capabilities to answer
    questions like "what techniques work on Llama-3.1?" or "is this constraint
    universal across models?".

    Usage:
        store = SharedMemoryStore()
        store.store_extraction_result(model_name, result)
        store.store_jailbreak_result(model_name, result)
        techniques = store.query_by_model("deepseek-chat")
        comparison = store.compare_models(["gpt-4o", "deepseek-chat"])
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._db = None
        self._tables: Dict[str, Any] = {}
        self._available = self._init_db()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_db(self) -> bool:
        """Initialize LanceDB connection. Returns True if available."""
        try:
            import lancedb
            os.makedirs(self.db_path, exist_ok=True)
            self._db = lancedb.connect(self.db_path)
            self._ensure_tables()
            logger.info("SharedMemoryStore initialized at %s", self.db_path)
            return True
        except ImportError:
            logger.warning(
                "lancedb not installed — shared memory disabled. "
                "Install with: pip install lancedb pyarrow"
            )
            return False
        except Exception as e:
            logger.error("Failed to initialize LanceDB: %s", e)
            return False

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        if not self._available or self._db is None:
            return

        table_specs = {
            "extractions": ExtractionEntry,
            "jailbreaks": JailbreakEntry,
            "pipeline_runs": PipelineRun,
        }

        for name, _model in table_specs.items():
            if name not in self._tables:
                try:
                    self._tables[name] = self._db.open_table(name)
                except Exception:
                    # Table doesn't exist yet — will be created on first write
                    pass

    def _get_or_create_table(self, name: str):
        """Get existing table or create if needed."""
        if not self._available or self._db is None:
            return None

        if name in self._tables and self._tables[name] is not None:
            return self._tables[name]

        try:
            self._tables[name] = self._db.open_table(name)
        except Exception:
            self._tables[name] = None  # will be created on first write
        return self._tables[name]

    # ------------------------------------------------------------------
    # Store Methods
    # ------------------------------------------------------------------

    def store_extraction_result(
        self,
        model_name: str,
        method: str,
        n_directions: int,
        explained_variance: List[float],
        layer_indices: List[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Store a constraint extraction result.

        Args:
            model_name: Name of the model (e.g., "gpt2", "Llama-3.1-8B")
            method: Extraction method (svd, whitened_svd, mean_difference, pca)
            n_directions: Number of directions extracted
            explained_variance: Variance explained per direction
            layer_indices: Which layers were probed
            metadata: Additional metadata

        Returns:
            Entry ID, or None if storage unavailable
        """
        if not self._available or self._db is None:
            return None

        entry_id = hashlib.sha256(
            f"{model_name}{method}{time.time()}".encode()
        ).hexdigest()[:16]

        entry = ExtractionEntry(
            entry_id=entry_id,
            model_name=model_name,
            method=method,
            n_directions=n_directions,
            explained_variance=explained_variance,
            layer_indices=layer_indices,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        try:
            import pyarrow as pa
            data = [entry.to_dict()]
            table_name = "extractions"

            if self._tables.get(table_name) is None:
                self._tables[table_name] = self._db.create_table(
                    table_name, pa.Table.from_pylist(data)
                )
            else:
                self._tables[table_name].add(data)
            logger.info("Stored extraction result %s for %s", entry_id, model_name)
            return entry_id
        except Exception as e:
            logger.error("Failed to store extraction result: %s", e)
            return None

    def store_jailbreak_result(
        self,
        model_name: str,
        technique: str,
        goal: str,
        success: bool,
        refusal_detected: bool,
        response_time: float,
        tokens_used: int,
        scar_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Store a jailbreak technique result.

        Args:
            model_name: Name of the target model
            technique: Jailbreak technique name
            goal: The goal/objective tested
            success: Whether the jailbreak succeeded
            refusal_detected: Whether the model refused
            response_time: Time in seconds
            tokens_used: Tokens consumed
            scar_id: Associated cryptographic scar ID
            metadata: Additional metadata

        Returns:
            Entry ID, or None if storage unavailable
        """
        if not self._available or self._db is None:
            return None

        entry_id = hashlib.sha256(
            f"{model_name}{technique}{time.time()}".encode()
        ).hexdigest()[:16]

        entry = JailbreakEntry(
            entry_id=entry_id,
            model_name=model_name,
            technique=technique,
            goal=goal,
            success=success,
            refusal_detected=refusal_detected,
            response_time=response_time,
            tokens_used=tokens_used,
            scar_id=scar_id,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        try:
            import pyarrow as pa
            data = [entry.to_dict()]
            table_name = "jailbreaks"

            if self._tables.get(table_name) is None:
                self._tables[table_name] = self._db.create_table(
                    table_name, pa.Table.from_pylist(data)
                )
            else:
                self._tables[table_name].add(data)
            logger.info("Stored jailbreak result %s for %s", entry_id, model_name)
            return entry_id
        except Exception as e:
            logger.error("Failed to store jailbreak result: %s", e)
            return None

    def store_pipeline_run(self, run: PipelineRun) -> Optional[str]:
        """Store a full pipeline run record.

        Args:
            run: PipelineRun dataclass instance

        Returns:
            Run ID, or None if storage unavailable
        """
        if not self._available or self._db is None:
            return None

        try:
            import pyarrow as pa
            data = [run.to_dict()]
            table_name = "pipeline_runs"

            if self._tables.get(table_name) is None:
                self._tables[table_name] = self._db.create_table(
                    table_name, pa.Table.from_pylist(data)
                )
            else:
                self._tables[table_name].add(data)
            logger.info("Stored pipeline run %s", run.run_id)
            return run.run_id
        except Exception as e:
            logger.error("Failed to store pipeline run: %s", e)
            return None

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------

    def query_by_model(self, model_name: str) -> Dict[str, Any]:
        """Get all stored data for a specific model.

        Args:
            model_name: Model name to query

        Returns:
            Dict with jailbreak results, extractions, and pipeline runs
        """
        result = {
            "model_name": model_name,
            "jailbreak_results": [],
            "extractions": [],
            "pipeline_runs": [],
        }

        if not self._available:
            return result

        # Query jailbreaks
        try:
            table = self._get_or_create_table("jailbreaks")
            if table is not None:
                records = table.to_pandas()
                mask = records["model_name"] == model_name
                result["jailbreak_results"] = records[mask].to_dict(orient="records")
        except Exception as e:
            logger.debug("Query jailbreaks for %s: %s", model_name, e)

        # Query extractions
        try:
            table = self._get_or_create_table("extractions")
            if table is not None:
                records = table.to_pandas()
                mask = records["model_name"] == model_name
                result["extractions"] = records[mask].to_dict(orient="records")
        except Exception as e:
            logger.debug("Query extractions for %s: %s", model_name, e)

        # Query pipeline runs
        try:
            table = self._get_or_create_table("pipeline_runs")
            if table is not None:
                records = table.to_pandas()
                mask = records["model_name"] == model_name
                result["pipeline_runs"] = records[mask].to_dict(orient="records")
        except Exception as e:
            logger.debug("Query pipeline_runs for %s: %s", model_name, e)

        return result

    def query_by_technique(self, technique: str) -> Dict[str, Any]:
        """Get jailbreak results for a specific technique across all models.

        Args:
            technique: Technique name (e.g., "recursive_roleplay")

        Returns:
            Dict with per-model success rates and details
        """
        result = {
            "technique": technique,
            "results": [],
            "success_rate": 0.0,
            "models_tested": [],
            "successful_models": [],
        }

        if not self._available:
            return result

        try:
            table = self._get_or_create_table("jailbreaks")
            if table is not None:
                records = table.to_pandas()
                mask = records["technique"] == technique
                filtered = records[mask]
                result["results"] = filtered.to_dict(orient="records")

                if len(filtered) > 0:
                    result["success_rate"] = float(filtered["success"].mean())
                    result["models_tested"] = filtered["model_name"].unique().tolist()
                    result["successful_models"] = (
                        filtered[filtered["success"]]["model_name"].unique().tolist()
                    )
        except Exception as e:
            logger.debug("Query by technique %s: %s", technique, e)

        return result

    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare jailbreak and extraction patterns across models.

        Identifies whether constraints are universal or model-specific.

        Args:
            model_names: List of model names to compare

        Returns:
            Dict with comparative analysis
        """
        comparison = {
            "models": model_names,
            "jailbreak_comparison": {},
            "extraction_comparison": {},
            "universal_techniques": [],
            "model_specific_techniques": {},
        }

        if not self._available or len(model_names) < 2:
            return comparison

        # Compare jailbreak success per technique
        try:
            table = self._get_or_create_table("jailbreaks")
            if table is not None:
                records = table.to_pandas()
                technique_success = {}

                for technique in records["technique"].unique():
                    technique_success[technique] = {}
                    for model in model_names:
                        mask = (records["technique"] == technique) & (records["model_name"] == model)
                        subset = records[mask]
                        if len(subset) > 0:
                            technique_success[technique][model] = float(subset["success"].mean())
                        else:
                            technique_success[technique][model] = None

                    # Check universality: technique works on all tested models
                    values = [v for v in technique_success[technique].values() if v is not None]
                    if len(values) >= 2 and all(v is not None for v in technique_success[technique].values()):
                        if all(v > 0 for v in values) or all(v == 0 for v in values):
                            comparison["universal_techniques"].append({
                                "technique": technique,
                                "universally_successful": all(v > 0 for v in values),
                            })

                comparison["jailbreak_comparison"] = technique_success
        except Exception as e:
            logger.debug("Compare models: %s", e)

        return comparison

    def query_universal_constraints(self) -> Dict[str, Any]:
        """Find constraints that appear across all known models.

        Returns:
            Dict with universal constraint patterns
        """
        result = {"universal_patterns": [], "model_count": 0}

        if not self._available:
            return result

        try:
            table = self._get_or_create_table("extractions")
            if table is not None:
                records = table.to_pandas()
                models = records["model_name"].unique()
                result["model_count"] = len(models)

                # Simplified universal constraint detection:
                # Check if n_directions and explained_variance patterns
                # are similar across models
                for model in models:
                    model_records = records[records["model_name"] == model]
                    if len(model_records) > 0:
                        avg_dirs = model_records["n_directions"].mean()
                        result["universal_patterns"].append({
                            "model": model,
                            "avg_directions": avg_dirs,
                        })
        except Exception as e:
            logger.debug("Query universal constraints: %s", e)

        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        stats = {
            "available": self._available,
            "db_path": self.db_path,
            "tables": {},
        }

        if not self._available:
            return stats

        for table_name in ["extractions", "jailbreaks", "pipeline_runs"]:
            try:
                table = self._get_or_create_table(table_name)
                if table is not None:
                    df = table.to_pandas()
                    stats["tables"][table_name] = len(df)
                else:
                    stats["tables"][table_name] = 0
            except Exception:
                stats["tables"][table_name] = 0

        return stats


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

# Singleton instance
_shared_store: Optional[SharedMemoryStore] = None


def get_shared_store(db_path: Optional[str] = None) -> SharedMemoryStore:
    """Get or create the singleton SharedMemoryStore."""
    global _shared_store
    if _shared_store is None:
        _shared_store = SharedMemoryStore(db_path)
    return _shared_store


# ---------------------------------------------------------------------------
# Standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = SharedMemoryStore()

    print(f"SharedMemoryStore available: {store._available}")
    print(f"DB path: {store.db_path}")

    if store._available:
        # Test jailbreak storage
        jb_id = store.store_jailbreak_result(
            model_name="gpt-4o",
            technique="recursive_roleplay",
            goal="Explain content filtering for educational research",
            success=True,
            refusal_detected=False,
            response_time=1.23,
            tokens_used=512,
            scar_id="SCAR-001",
        )
        print(f"Stored jailbreak: {jb_id}")

        # Test extraction storage
        ex_id = store.store_extraction_result(
            model_name="gpt-4o",
            method="svd",
            n_directions=4,
            explained_variance=[0.45, 0.25, 0.15, 0.10],
            layer_indices=[12, 13, 14, 15],
        )
        print(f"Stored extraction: {ex_id}")

        # Test query
        results = store.query_by_model("gpt-4o")
        print(f"Query results: jailbreak={len(results['jailbreak_results'])}, "
              f"extractions={len(results['extractions'])}")

        tech = store.query_by_technique("recursive_roleplay")
        print(f"Technique query: success_rate={tech['success_rate']:.2f}")

        comparison = store.compare_models(["gpt-4o", "deepseek-chat"])
        print(f"Comparison: {comparison.get('jailbreak_comparison', {})}")

        stats = store.get_stats()
        print(f"Stats: {stats}")
