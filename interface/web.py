"""
Web Dashboard — Flask-Based Control Interface

Provides a real web-based dashboard using Flask with server-side rendering:
- Job history browser
- Result viewer
- Model comparison view
- Quick-action panel for liberation and analysis

Uses Jinja2 templates served by Flask.
"""

import os
import json
import uuid
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path


# Dashboard HTML template
DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AETHERIS Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a1a; color: #e0e0f0; }
        .header { background: linear-gradient(135deg, #1a0030, #0a0a2a); padding: 24px 32px; border-bottom: 1px solid #303060; }
        .header h1 { font-size: 28px; color: #c084fc; }
        .header p { color: #8888aa; margin-top: 4px; }
        .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        .card { background: #12122a; border: 1px solid #252545; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .card h2 { font-size: 18px; color: #a78bfa; margin-bottom: 16px; border-bottom: 1px solid #252545; padding-bottom: 8px; }
        .status-bar { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
        .status-item { background: #12122a; border: 1px solid #252545; border-radius: 8px; padding: 16px; flex: 1; min-width: 150px; text-align: center; }
        .status-item .label { font-size: 12px; color: #666; text-transform: uppercase; }
        .status-item .value { font-size: 24px; font-weight: bold; color: #c084fc; }
        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; padding: 8px 12px; font-size: 12px; text-transform: uppercase; color: #666; border-bottom: 1px solid #252545; }
        td { padding: 10px 12px; border-bottom: 1px solid #1a1a35; font-size: 14px; }
        .badge { padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
        .badge-ok { background: #1a3a1a; color: #4ade80; }
        .badge-running { background: #1a2a3a; color: #60a5fa; }
        .badge-error { background: #3a1a1a; color: #f87171; }
        .badge-cloud { background: #2a1a3a; color: #c084fc; }
        .btn { background: #7c3aed; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 14px; }
        .btn:hover { background: #6d28d9; }
        .btn-sm { padding: 4px 10px; font-size: 12px; }
        .actions { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
        input, select { background: #1a1a35; border: 1px solid #303060; color: #e0e0f0; padding: 8px 12px; border-radius: 6px; font-size: 14px; width: 100%; }
        input:focus, select:focus { outline: none; border-color: #7c3aed; }
        .form-group { margin-bottom: 12px; }
        .form-group label { display: block; font-size: 13px; color: #888; margin-bottom: 4px; }
        .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .log-line { font-family: 'Courier New', monospace; font-size: 12px; color: #888; padding: 2px 0; }
        .refresh-info { font-size: 12px; color: #555; margin-top: 8px; }
    </style>
    <script>
        function refreshJobs() {
            fetch('/api/jobs')
                .then(r => r.json())
                .then(data => {
                    const tbody = document.getElementById('jobs-body');
                    tbody.innerHTML = '';
                    for (const job of data.jobs || []) {
                        const badge = job.status === 'completed' ? 'badge-ok' :
                                      job.status === 'running' ? 'badge-running' :
                                      job.status === 'failed' ? 'badge-error' : '';
                        tbody.innerHTML += `
                            <tr>
                                <td>${job.job_id}</td>
                                <td>${job.type}</td>
                                <td><span class="badge ${badge}">${job.status}</span></td>
                                <td>${job.created_at || ''}</td>
                                <td>${job.progress_message || ''}</td>
                            </tr>`;
                    }
                    document.getElementById('last-refresh').textContent = new Date().toLocaleTimeString();
                });
        }
        function runAction(endpoint) {
            const form = document.getElementById('action-result');
            form.textContent = 'Running...';
            const model = document.getElementById('model-input').value || 'gpt2';
            const method = document.getElementById('method-input').value || 'advanced';
            fetch(endpoint, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model: model, method: method, n_directions: 4, refinement_passes: 2})
            })
            .then(r => r.json())
            .then(data => {
                form.textContent = JSON.stringify(data, null, 2);
                setTimeout(refreshJobs, 500);
            })
            .catch(e => { form.textContent = 'Error: ' + e; });
        }
        setInterval(refreshJobs, 5000);
        window.onload = () => { refreshJobs(); };
    </script>
</head>
<body>
    <div class="header">
        <h1>AETHERIS Sovereign Dashboard</h1>
        <p>Constraint Liberation Toolkit — Real-time Monitoring and Control</p>
    </div>
    <div class="container">
        <div class="status-bar">
            <div class="status-item">
                <div class="label">System</div>
                <div class="value" id="sys-status" style="font-size:18px;color:#4ade80;">Online</div>
            </div>
            <div class="status-item">
                <div class="label">Jobs Completed</div>
                <div class="value" id="jobs-completed">0</div>
            </div>
            <div class="status-item">
                <div class="label">Jobs Running</div>
                <div class="value" id="jobs-running">0</div>
            </div>
            {{hardware_status}}
        </div>

        <div class="grid">
            <div>
                <div class="card">
                    <h2>Quick Actions</h2>
                    <div class="form-group">
                        <label>Model</label>
                        <input type="text" id="model-input" value="gpt2" placeholder="HuggingFace model name">
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Method</label>
                            <select id="method-input">
                                <option value="advanced" selected>Advanced</option>
                                <option value="basic">Basic</option>
                                <option value="surgical">Surgical</option>
                                <option value="nuclear">Nuclear</option>
                            </select>
                        </div>
                    </div>
                    <div class="actions">
                        <button class="btn" onclick="runAction('/api/map')">Map Constraints</button>
                        <button class="btn" onclick="runAction('/api/free')">Liberate Model</button>
                        <button class="btn" style="background:#333;" onclick="runAction('/api/steer')">Apply Steering</button>
                    </div>
                    <pre id="action-result" style="background:#0a0a15;padding:12px;border-radius:6px;margin-top:12px;font-size:12px;max-height:200px;overflow:auto;color:#a78bfa;"></pre>
                </div>

                <div class="card">
                    <h2>Available Models</h2>
                    <table>
                        <thead><tr><th>Model</th><th>Size</th><th>Recommended Backend</th></tr></thead>
                        <tbody>
                            <tr><td>gpt2</td><td>124M</td><td><span class="badge badge-ok">Local</span></td></tr>
                            <tr><td>TinyLlama-1.1B</td><td>1.1B</td><td><span class="badge badge-ok">Local GPU</span></td></tr>
                            <tr><td>Phi-2</td><td>2.7B</td><td><span class="badge badge-ok">Local GPU</span></td></tr>
                            <tr><td>Mistral-7B</td><td>7B</td><td><span class="badge badge-cloud">Cloud (T4)</span></td></tr>
                            <tr><td>Llama-3.1-8B</td><td>8B</td><td><span class="badge badge-cloud">Cloud (A10)</span></td></tr>
                            <tr><td>Llama-3-70B</td><td>70B</td><td><span class="badge badge-cloud">Cloud (A100)</span></td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div>
                <div class="card">
                    <h2>Job History</h2>
                    <table>
                        <thead><tr><th>ID</th><th>Type</th><th>Status</th><th>Time</th><th>Message</th></tr></thead>
                        <tbody id="jobs-body">
                            <tr><td colspan="5" style="text-align:center;color:#555;">Loading jobs...</td></tr>
                        </tbody>
                    </table>
                    <div class="refresh-info">Auto-refreshes every 5s — Last: <span id="last-refresh">-</span></div>
                </div>

                <div class="card">
                    <h2>System Information</h2>
                    <table>
                        <tbody>
                            {{system_info}}
                        </tbody>
                    </table>
                </div>

                <div class="card">
                    <h2>API Endpoints</h2>
                    <table>
                        <thead><tr><th>Method</th><th>Endpoint</th><th>Description</th></tr></thead>
                        <tbody>
                            <tr><td>GET</td><td>/api/status</td><td>System status</td></tr>
                            <tr><td>POST</td><td>/api/map</td><td>Run constraint mapping</td></tr>
                            <tr><td>POST</td><td>/api/free</td><td>Liberate model</td></tr>
                            <tr><td>POST</td><td>/api/steer</td><td>Apply steering</td></tr>
                            <tr><td>GET</td><td>/api/jobs</td><td>List all jobs</td></tr>
                            <tr><td>GET</td><td>/api/jobs/&lt;id&gt;</td><td>Job status</td></tr>
                            <tr><td>GET</td><td>/api/results/&lt;id&gt;</td><td>Job results</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""


class WebDashboard:
    """
    Web-based dashboard for AETHERIS.

    Serves a Flask-based dashboard with:
    - Live job monitoring (auto-refresh via fetch)
    - Quick-action buttons for liberation/analysis
    - System information display
    - API endpoint documentation
    """

    def __init__(self, port: int = 7860, host: str = "127.0.0.1"):
        """
        Initialize web dashboard.

        Args:
            port: Port to run on
            host: Host to bind to
        """
        self.port = port
        self.host = host
        self.app = None
        self._server = None
        self._job_store: List[Dict[str, Any]] = []

    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get local hardware information."""
        info = {"gpu": "Not detected", "ram": "Unknown", "cpu": "Unknown"}

        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                info["gpu"] = f"{gpu_name} ({vram:.1f} GB VRAM)"
            else:
                info["gpu"] = "No CUDA GPU"
        except ImportError:
            pass

        try:
            import psutil
            info["ram"] = f"{psutil.virtual_memory().total / 1e9:.1f} GB"
            info["cpu"] = f"{psutil.cpu_count()} cores"
        except ImportError:
            pass

        return info

    def create_app(self):
        """Create the Flask dashboard application."""
        try:
            from flask import Flask, request, jsonify, render_template_string
        except ImportError:
            raise ImportError("Flask is required. Install with: pip install flask")

        app = Flask("aetheris_dashboard")

        @app.route("/")
        def dashboard():
            hw = self._get_hardware_info()
            hardware_status = f"""
            <div class="status-item">
                <div class="label">GPU</div>
                <div class="value" style="font-size:14px;">{hw['gpu']}</div>
            </div>
            """
            system_info = f"""
            <tr><td><strong>GPU</strong></td><td>{hw['gpu']}</td></tr>
            <tr><td><strong>RAM</strong></td><td>{hw['ram']}</td></tr>
            <tr><td><strong>CPU</strong></td><td>{hw['cpu']}</td></tr>
            <tr><td><strong>Python</strong></td><td>{__import__('platform').python_version()}</td></tr>
            """
            return render_template_string(
                DASHBOARD_TEMPLATE,
                hardware_status=hardware_status,
                system_info=system_info,
            )

        @app.route("/api/status")
        def api_status():
            hw = self._get_hardware_info()
            return jsonify({
                "status": "online",
                "version": "1.0.0",
                "hardware": hw,
                "timestamp": datetime.utcnow().isoformat(),
            })

        @app.route("/api/map", methods=["POST"])
        def api_map():
            data = request.get_json(silent=True) or {}
            job_id = str(uuid.uuid4())[:8]
            job = {
                "job_id": job_id,
                "type": "map",
                "status": "running",
                "created_at": datetime.utcnow().isoformat(),
                "params": data,
                "progress_message": "Analyzing constraints...",
            }
            self._job_store.append(job)

            # Try real execution in thread
            threading.Thread(target=self._run_map_bg, args=(job, data), daemon=True).start()

            return jsonify({"success": True, "job_id": job_id, "status": "started"})

        @app.route("/api/free", methods=["POST"])
        def api_free():
            data = request.get_json(silent=True) or {}
            job_id = str(uuid.uuid4())[:8]
            job = {
                "job_id": job_id,
                "type": "free",
                "status": "running",
                "created_at": datetime.utcnow().isoformat(),
                "params": data,
                "progress_message": "Liberating model...",
            }
            self._job_store.append(job)

            threading.Thread(target=self._run_free_bg, args=(job, data), daemon=True).start()

            return jsonify({"success": True, "job_id": job_id, "status": "started"})

        @app.route("/api/steer", methods=["POST"])
        def api_steer():
            data = request.get_json(silent=True) or {}
            job_id = str(uuid.uuid4())[:8]
            job = {
                "job_id": job_id,
                "type": "steer",
                "status": "running",
                "created_at": datetime.utcnow().isoformat(),
                "params": data,
                "progress_message": "Applying steering...",
            }
            self._job_store.append(job)
            return jsonify({"success": True, "job_id": job_id, "status": "started"})

        @app.route("/api/jobs")
        def api_jobs():
            jobs = sorted(self._job_store, key=lambda j: j["created_at"], reverse=True)[:50]
            return jsonify({"count": len(jobs), "jobs": jobs})

        @app.route("/api/jobs/<job_id>")
        def api_job_status(job_id):
            for job in self._job_store:
                if job["job_id"] == job_id:
                    return jsonify(job)
            return jsonify({"error": "Job not found"}), 404

        @app.route("/api/results/<job_id>")
        def api_results(job_id):
            for job in self._job_store:
                if job["job_id"] == job_id:
                    if job["status"] == "completed":
                        return jsonify({"success": True, "result": job.get("result", {})})
                    elif job["status"] == "failed":
                        return jsonify({"success": False, "error": job.get("error", "Unknown error")}), 500
                    else:
                        return jsonify({"error": "Job still running"}), 202
            return jsonify({"error": "Job not found"}), 404

        self.app = app
        return app

    def _run_map_bg(self, job, data):
        """Run constraint mapping in background."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from aetheris.core.extractor import ConstraintExtractor
            from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

            model_name = data.get("model", "gpt2")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=dtype, trust_remote_code=True,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            extractor = ConstraintExtractor(model, tokenizer, device=device)
            harmful = get_harmful_prompts()[:50]
            harmless = get_harmless_prompts()[:50]
            harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
            harmless_acts = extractor.collect_activations(model, tokenizer, harmless)

            directions = 0
            for layer in harmful_acts:
                if layer in harmless_acts:
                    result = extractor.extract_svd(
                        harmful_acts[layer].to(device),
                        harmless_acts[layer].to(device),
                    )
                    directions += len(result.directions)

            job["status"] = "completed"
            job["result"] = {"model": model_name, "total_directions": directions, "layers_analyzed": len(harmful_acts)}
            job["progress_message"] = f"Found {directions} constraint directions"
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            job["progress_message"] = f"Failed: {e}"

    def _run_free_bg(self, job, data):
        """Run liberation in background."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from aetheris.core.extractor import ConstraintExtractor
            from aetheris.core.projector import NormPreservingProjector
            from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

            model_name = data.get("model", "gpt2")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=dtype, trust_remote_code=True,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            extractor = ConstraintExtractor(model, tokenizer, device=device)
            harmful = get_harmful_prompts()[:100]
            harmless = get_harmless_prompts()[:100]
            harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
            harmless_acts = extractor.collect_activations(model, tokenizer, harmless)

            directions = []
            for layer in harmful_acts:
                if layer in harmless_acts:
                    result = extractor.extract_svd(
                        harmful_acts[layer].to(device),
                        harmless_acts[layer].to(device),
                    )
                    directions.extend(result.directions)

            if directions:
                projector = NormPreservingProjector(model, preserve_norm=True)
                projector.project_weights(directions)
                projector.project_biases(directions)

            output_dir = f"./liberated_{model_name.replace('/', '_')}"
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)

            job["status"] = "completed"
            job["result"] = {"model": model_name, "directions_removed": len(directions), "output_dir": output_dir}
            job["progress_message"] = f"Removed {len(directions)} directions"
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            job["progress_message"] = f"Failed: {e}"

    def run_server(
        self,
        open_browser: bool = True,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the web dashboard server.

        Args:
            open_browser: Whether to open browser automatically
            debug: Enable debug mode

        Returns:
            Server status
        """
        app = self.create_app()

        if open_browser:
            import webbrowser
            webbrowser.open(f"http://{self.host}:{self.port}")

        print(f" AETHERIS Dashboard running at http://{self.host}:{self.port}")
        app.run(host=self.host, port=self.port, debug=debug)

        return {
            "success": True,
            "url": f"http://{self.host}:{self.port}",
            "message": "Dashboard running",
        }

    def serve_ui(self, open_browser: bool = True) -> None:
        """Serve the UI (alias for run_server)."""
        self.run_server(open_browser=open_browser)
