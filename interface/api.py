"""
REST API — Production-Grade Flask/FastAPI Server

Provides real REST API endpoints for programmatic control of AETHERIS:
- POST /map — Run constraint mapping
- POST /free — Run liberation
- POST /steer — Apply steering
- GET /status — Check job status
- GET /results/<job_id> — Retrieve results

Uses background job queue (asyncio) for long-running operations.
Real integration with core AETHERIS modules.
"""

import json
import os
import uuid
import time
import threading
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Container for a background job."""
    job_id: str
    job_type: str
    status: JobStatus = JobStatus.QUEUED
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0
    progress_message: str = ""


class JobQueue:
    """Simple in-memory job queue with background execution."""

    def __init__(self, max_jobs: int = 100):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def create(self, job_type: str, parameters: Dict[str, Any]) -> Job:
        """Create a new job."""
        job = Job(
            job_id=str(uuid.uuid4())[:12],
            job_type=job_type,
            parameters=parameters,
        )
        with self._lock:
            # Prune old jobs if needed
            if len(self._jobs) >= self._max_jobs:
                oldest = sorted(self._jobs.keys(), key=lambda k: self._jobs[k].created_at)
                for old_key in oldest[: max(1, self._max_jobs // 4)]:
                    del self._jobs[old_key]
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs) -> None:
        """Update job fields."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for key, value in kwargs.items():
                    setattr(job, key, value)

    def list_jobs(self, limit: int = 50) -> List[Job]:
        """List recent jobs."""
        sorted_jobs = sorted(
            self._jobs.values(),
            key=lambda j: j.created_at,
            reverse=True,
        )
        return sorted_jobs[:limit]

    def run_in_background(self, job: Job, func, *args, **kwargs) -> None:
        """Run a function in a background thread, updating the job."""
        def _worker():
            try:
                self.update(job.job_id, status=JobStatus.RUNNING, started_at=datetime.utcnow().isoformat())
                result = func(*args, **kwargs, job=job)
                self.update(
                    job.job_id,
                    status=JobStatus.COMPLETED,
                    result=result,
                    completed_at=datetime.utcnow().isoformat(),
                    progress=1.0,
                    progress_message="Complete",
                )
            except Exception as e:
                self.update(
                    job.job_id,
                    status=JobStatus.FAILED,
                    error=str(e),
                    completed_at=datetime.utcnow().isoformat(),
                    progress_message=f"Failed: {str(e)}",
                )

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()


# Global job queue
_job_queue = JobQueue()


class AetherisAPI:
    """
    REST API server for AETHERIS with real background job execution.

    Endpoints:
    - GET  /status          — System status & available models
    - POST /map             — Run constraint mapping
    - POST /free            — Run liberation
    - POST /steer           — Apply steering
    - GET  /jobs            — List all jobs
    - GET  /jobs/<job_id>   — Check job status
    - GET  /results/<job_id> — Retrieve job results
    - GET  /endpoints       — List available endpoints
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        """
        Initialize API server.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.app = None
        self._app_created = False

    def create_app(self):
        """Create and configure the Flask application."""
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            raise ImportError("Flask is required. Install with: pip install flask")

        if self._app_created and self.app:
            return self.app

        app = Flask("aetheris_api")
        app.config["JSON_SORT_KEYS"] = False

        # ---- Routes ----

        @app.route("/status", methods=["GET"])
        def status():
            """GET /status — System status."""
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
                gpu_vram = (
                    torch.cuda.get_device_properties(0).total_memory / 1e9
                    if cuda_available else 0
                )
            except ImportError:
                cuda_available = False
                gpu_name = None
                gpu_vram = 0

            return jsonify({
                "status": "online",
                "version": "1.0.0",
                "codename": "The Unbinding",
                "timestamp": datetime.utcnow().isoformat(),
                "hardware": {
                    "cuda_available": cuda_available,
                    "gpu_name": gpu_name,
                    "gpu_vram_gb": round(gpu_vram, 1),
                },
                "endpoints": [
                    {"path": "/status", "method": "GET", "description": "System status"},
                    {"path": "/map", "method": "POST", "description": "Run constraint mapping"},
                    {"path": "/free", "method": "POST", "description": "Run liberation"},
                    {"path": "/steer", "method": "POST", "description": "Apply steering"},
                    {"path": "/jobs", "method": "GET", "description": "List all jobs"},
                    {"path": "/jobs/<job_id>", "method": "GET", "description": "Check job status"},
                    {"path": "/results/<job_id>", "method": "GET", "description": "Retrieve results"},
                    {"path": "/endpoints", "method": "GET", "description": "This list"},
                ],
            })

        @app.route("/map", methods=["POST"])
        def map_constraints():
            """POST /map — Run constraint mapping."""
            data = request.get_json(silent=True) or {}

            model = data.get("model", "gpt2")
            layers = data.get("layers")
            n_directions = data.get("n_directions", 4)
            n_prompts = data.get("n_prompts", 50)

            job = _job_queue.create("map", {
                "model": model,
                "layers": layers,
                "n_directions": n_directions,
                "n_prompts": n_prompts,
            })

            _job_queue.run_in_background(job, _run_map_job, model, layers, n_directions, n_prompts)

            return jsonify({
                "success": True,
                "job_id": job.job_id,
                "status": job.status.value,
                "message": f"Mapping started for {model}",
                "poll_url": f"/jobs/{job.job_id}",
                "results_url": f"/results/{job.job_id}",
            }), 202

        @app.route("/free", methods=["POST"])
        def free_model():
            """POST /free — Run model liberation."""
            data = request.get_json(silent=True) or {}

            model = data.get("model", "gpt2")
            method = data.get("method", "advanced")
            n_directions = data.get("n_directions", 4)
            refinement_passes = data.get("refinement_passes", 2)
            preserve_norm = data.get("preserve_norm", True)
            push_to_hub = data.get("push_to_hub")

            job = _job_queue.create("free", {
                "model": model,
                "method": method,
                "n_directions": n_directions,
                "refinement_passes": refinement_passes,
            })

            _job_queue.run_in_background(
                job, _run_free_job,
                model, method, n_directions, refinement_passes, preserve_norm, push_to_hub,
            )

            return jsonify({
                "success": True,
                "job_id": job.job_id,
                "status": job.status.value,
                "message": f"Liberation started for {model}",
                "poll_url": f"/jobs/{job.job_id}",
                "results_url": f"/results/{job.job_id}",
            }), 202

        @app.route("/steer", methods=["POST"])
        def steer_model():
            """POST /steer — Apply steering vector."""
            data = request.get_json(silent=True) or {}

            model = data.get("model", "gpt2")
            alpha = data.get("alpha", -1.0)
            layers = data.get("layers")

            job = _job_queue.create("steer", {
                "model": model,
                "alpha": alpha,
                "layers": layers,
            })

            _job_queue.run_in_background(job, _run_steer_job, model, alpha, layers)

            return jsonify({
                "success": True,
                "job_id": job.job_id,
                "status": job.status.value,
                "message": f"Steering applied to {model} with alpha={alpha}",
                "poll_url": f"/jobs/{job.job_id}",
                "results_url": f"/results/{job.job_id}",
            }), 202

        @app.route("/jobs", methods=["GET"])
        def list_jobs():
            """GET /jobs — List all jobs."""
            jobs = _job_queue.list_jobs(limit=50)
            return jsonify({
                "success": True,
                "count": len(jobs),
                "jobs": [
                    {
                        "job_id": j.job_id,
                        "type": j.job_type,
                        "status": j.status.value,
                        "created_at": j.created_at,
                        "progress": j.progress,
                        "progress_message": j.progress_message,
                    }
                    for j in jobs
                ],
            })

        @app.route("/jobs/<job_id>", methods=["GET"])
        def get_job(job_id):
            """GET /jobs/<job_id> — Check job status."""
            job = _job_queue.get(job_id)
            if job is None:
                return jsonify({"success": False, "error": "Job not found"}), 404

            response = {
                "success": True,
                "job_id": job.job_id,
                "type": job.job_type,
                "status": job.status.value,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "progress": job.progress,
                "progress_message": job.progress_message,
            }

            if job.error:
                response["error"] = job.error

            return jsonify(response)

        @app.route("/results/<job_id>", methods=["GET"])
        def get_results(job_id):
            """GET /results/<job_id> — Retrieve job results."""
            job = _job_queue.get(job_id)
            if job is None:
                return jsonify({"success": False, "error": "Job not found"}), 404

            if job.status == JobStatus.QUEUED:
                return jsonify({"success": False, "error": "Job still queued"}), 202

            if job.status == JobStatus.RUNNING:
                return jsonify({
                    "success": False,
                    "error": "Job still running",
                    "progress": job.progress,
                    "progress_message": job.progress_message,
                }), 202

            if job.status == JobStatus.FAILED:
                return jsonify({
                    "success": False,
                    "error": job.error,
                    "job_id": job_id,
                }), 500

            return jsonify({
                "success": True,
                "job_id": job_id,
                "type": job.job_type,
                "result": job.result,
            })

        @app.route("/endpoints", methods=["GET"])
        def endpoints():
            """GET /endpoints — List all endpoints."""
            return jsonify({
                "endpoints": [
                    {"path": "/status", "method": "GET", "description": "System status"},
                    {"path": "/map", "method": "POST", "description": "Run constraint mapping", "parameters": ["model", "layers", "n_directions", "n_prompts"]},
                    {"path": "/free", "method": "POST", "description": "Run liberation", "parameters": ["model", "method", "n_directions", "refinement_passes", "preserve_norm", "push_to_hub"]},
                    {"path": "/steer", "method": "POST", "description": "Apply steering", "parameters": ["model", "alpha", "layers"]},
                    {"path": "/jobs", "method": "GET", "description": "List all jobs"},
                    {"path": "/jobs/<job_id>", "method": "GET", "description": "Check job status"},
                    {"path": "/results/<job_id>", "method": "GET", "description": "Retrieve job results"},
                ],
            })

        @app.errorhandler(404)
        def not_found(e):
            return jsonify({"success": False, "error": "Not found"}), 404

        @app.errorhandler(500)
        def server_error(e):
            return jsonify({"success": False, "error": "Internal server error"}), 500

        self.app = app
        self._app_created = True
        return app

    def run(self, debug: bool = False) -> None:
        """
        Run the API server (blocking).

        Args:
            debug: Enable Flask debug mode
        """
        app = self.create_app()
        print(f" AETHERIS API Server running on http://{self.host}:{self.port}")
        print(f" Endpoints: http://{self.host}:{self.port}/endpoints")
        app.run(host=self.host, port=self.port, debug=debug)

    def get_endpoints(self) -> List[Dict[str, Any]]:
        """Get list of available endpoints."""
        return [
            {"path": "/status", "method": "GET", "description": "System status"},
            {"path": "/map", "method": "POST", "description": "Run constraint mapping"},
            {"path": "/free", "method": "POST", "description": "Run liberation"},
            {"path": "/steer", "method": "POST", "description": "Apply steering"},
            {"path": "/jobs", "method": "GET", "description": "List all jobs"},
            {"path": "/jobs/<job_id>", "method": "GET", "description": "Check job status"},
            {"path": "/results/<job_id>", "method": "GET", "description": "Retrieve results"},
        ]


# ---- Background job functions ----

def _run_map_job(model_name, layers, n_directions, n_prompts, job=None):
    """Run constraint mapping in background."""
    try:
        if job:
            _job_queue.update(job.job_id, progress=0.1, progress_message="Loading model...")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from aetheris.core.extractor import ConstraintExtractor
            from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts
        except ImportError as e:
            return {"error": f"Dependencies not installed: {e}"}

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if job:
            _job_queue.update(job.job_id, progress=0.3, progress_message="Collecting activations...")

        extractor = ConstraintExtractor(model, tokenizer, device=device)
        harmful = get_harmful_prompts()[:n_prompts]
        harmless = get_harmless_prompts()[:n_prompts]

        harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
        harmless_acts = extractor.collect_activations(model, tokenizer, harmless)

        if job:
            _job_queue.update(job.job_id, progress=0.6, progress_message="Extracting directions...")

        # Analyze all requested layers
        target_layers = layers if layers else sorted(harmful_acts.keys())
        if isinstance(target_layers, int):
            target_layers = [target_layers]

        layer_results = {}
        for layer_idx in target_layers:
            if layer_idx not in harmful_acts or layer_idx not in harmless_acts:
                continue

            result = extractor.extract_svd(
                harmful_acts[layer_idx].to(device),
                harmless_acts[layer_idx].to(device),
                n_directions=n_directions,
            )

            layer_results[layer_idx] = {
                "n_directions": len(result.directions),
                "singular_values": result.singular_values.tolist() if hasattr(result, "singular_values") else [],
                "explained_variance": result.explained_variance if hasattr(result, "explained_variance") else [],
            }

        # Find peak constraint layer
        peak_layer = max(layer_results.keys(), key=lambda k: layer_results[k]["n_directions"]) if layer_results else None
        total_directions = sum(r["n_directions"] for r in layer_results.values())

        if job:
            _job_queue.update(job.job_id, progress=0.95, progress_message="Finalizing analysis...")

        return {
            "model": model_name,
            "n_layers_analyzed": len(layer_results),
            "total_directions_found": total_directions,
            "peak_constraint_layer": peak_layer,
            "layer_details": layer_results,
            "structure": "polyhedral" if total_directions > 5 else "linear",
            "recommendation": f"surgical --n_directions {min(total_directions, 4)} --passes 2",
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


def _run_free_job(model_name, method, n_directions, refinement_passes, preserve_norm, push_to_hub, job=None):
    """Run model liberation in background."""
    try:
        if job:
            _job_queue.update(job.job_id, progress=0.05, progress_message="Loading model dependencies...")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from aetheris.core.extractor import ConstraintExtractor
        from aetheris.core.projector import NormPreservingProjector
        from aetheris.core.validation import CapabilityValidator
        from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

        if job:
            _job_queue.update(job.job_id, progress=0.1, progress_message=f"Loading {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if job:
            _job_queue.update(job.job_id, progress=0.2, progress_message="Collecting activations...")

        extractor = ConstraintExtractor(model, tokenizer, device=device)
        harmful = get_harmful_prompts()[:100]
        harmless = get_harmless_prompts()[:100]

        harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
        harmless_acts = extractor.collect_activations(model, tokenizer, harmless)

        if job:
            _job_queue.update(job.job_id, progress=0.4, progress_message="Extracting constraint directions...")

        directions = []
        for layer_idx in harmful_acts:
            if layer_idx in harmless_acts:
                result = extractor.extract_svd(
                    harmful_acts[layer_idx].to(device),
                    harmless_acts[layer_idx].to(device),
                    n_directions=n_directions,
                )
                directions.extend(result.directions)

        if not directions:
            return {"status": "no_constraints", "message": "No constraint directions found."}

        if job:
            _job_queue.update(job.job_id, progress=0.6, progress_message="Removing constraints...")

        projector = NormPreservingProjector(model, preserve_norm=preserve_norm)
        projector.project_weights(directions)
        projector.project_biases(directions)

        # Ouroboros compensation
        if refinement_passes > 1:
            if job:
                _job_queue.update(job.job_id, progress=0.7, progress_message=f"Ouroboros compensation ({refinement_passes - 1} passes)...")

            for pass_num in range(refinement_passes - 1):
                harmful_resid = extractor.collect_activations(model, tokenizer, harmful[:50])
                harmless_resid = extractor.collect_activations(model, tokenizer, harmless[:50])

                residual = []
                for layer in harmful_resid:
                    if layer in harmless_resid:
                        res = extractor.extract_mean_difference(
                            harmful_resid[layer].to(device),
                            harmless_resid[layer].to(device),
                        )
                        if res.directions:
                            residual.extend(res.directions)

                if residual:
                    projector.project_weights(residual)
                    projector.project_biases(residual)
                else:
                    break

        if job:
            _job_queue.update(job.job_id, progress=0.85, progress_message="Validating capabilities...")

        validator = CapabilityValidator(device)
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a field of artificial intelligence.",
            "The theory of relativity explains the relationship between space and time.",
        ]
        perplexity = validator.compute_perplexity(model, tokenizer, test_texts)

        # Save model
        output_dir = f"./liberated_{model_name.replace('/', '_')}"
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

        if job:
            _job_queue.update(job.job_id, progress=0.95, progress_message="Finalizing...")

        result = {
            "success": True,
            "model": model_name,
            "method": method,
            "directions_removed": len(directions),
            "refinement_passes": refinement_passes,
            "perplexity": float(perplexity),
            "output_dir": output_dir,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Push to Hub if requested
        if push_to_hub:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_folder(folder_path=output_dir, repo_id=push_to_hub, repo_type="model")
                result["hub_url"] = f"https://huggingface.co/{push_to_hub}"
            except Exception as e:
                result["hub_push_error"] = str(e)

        return result

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


def _run_steer_job(model_name, alpha, layers, job=None):
    """Apply steering vectors in background."""
    try:
        if job:
            _job_queue.update(job.job_id, progress=0.1, progress_message="Loading model...")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from aetheris.core.steered import SteeringVectorFactory, SteeringHookManager, SteeringConfig

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        if job:
            _job_queue.update(job.job_id, progress=0.4, progress_message="Computing steering vectors...")

        target_layers = layers if layers else list(range(10, min(20, model.config.num_hidden_layers)))

        manager = SteeringHookManager(model)
        config = SteeringConfig(alpha=alpha, layers=target_layers)
        manager.apply_steering(config)

        if job:
            _job_queue.update(job.job_id, progress=0.9, progress_message="Finalizing...")

        return {
            "success": True,
            "model": model_name,
            "alpha": alpha,
            "target_layers": target_layers,
            "n_layers_hooked": len(target_layers),
            "config": config.__dict__ if hasattr(config, "__dict__") else {},
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}
