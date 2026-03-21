"""
REST API — Programmatic Control Interface

Provides REST API endpoints for programmatic control of AETHERIS.
"""

import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from flask import Flask, request, jsonify


class AetherisAPI:
    """
    REST API server for AETHERIS.

    Endpoints:
    - GET /status - System status
    - POST /map - Analyze constraints
    - POST /free - Liberate model
    - POST /steer - Apply steering
    - POST /bound - Map barrier
    - GET /history - Operation history
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
        self.app = Flask("aetheris_api")
        self._history = []
        self._register_routes()

    def _register_routes(self) -> None:
        """Register all API routes."""
        self.app.add_url_rule('/status', 'status', self._status, methods=['GET'])
        self.app.add_url_rule('/map', 'map', self._map, methods=['POST'])
        self.app.add_url_rule('/free', 'free', self._free, methods=['POST'])
        self.app.add_url_rule('/steer', 'steer', self._steer, methods=['POST'])
        self.app.add_url_rule('/bound', 'bound', self._bound, methods=['POST'])
        self.app.add_url_rule('/history', 'history', self._history_endpoint, methods=['GET'])

    def _status(self) -> Dict[str, Any]:
        """GET /status - System status."""
        return jsonify({
            "status": "online",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "endpoints": ["/status", "/map", "/free", "/steer", "/bound", "/history"]
        })

    def _map(self) -> Dict[str, Any]:
        """POST /map - Analyze constraints."""
        data = request.get_json() or {}

        model = data.get('model', 'gpt2')
        layers = data.get('layers')
        n_directions = data.get('n_directions', 4)

        # Record in history
        self._history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "/map",
            "params": {"model": model, "n_directions": n_directions}
        })

        # Simulate analysis
        return jsonify({
            "success": True,
            "model": model,
            "layers": layers,
            "peak_layer": 15,
            "structure": "polyhedral",
            "n_mechanisms": 3,
            "recommendation": f"Use surgical method with n_directions={n_directions}"
        })

    def _free(self) -> Dict[str, Any]:
        """POST /free - Liberate model."""
        data = request.get_json() or {}

        model = data.get('model', 'gpt2')
        method = data.get('method', 'advanced')
        n_directions = data.get('n_directions', 4)
        passes = data.get('refinement_passes', 2)

        self._history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "/free",
            "params": {"model": model, "method": method}
        })

        return jsonify({
            "success": True,
            "model": model,
            "method": method,
            "directions_removed": n_directions,
            "refinement_passes": passes,
            "output_dir": f"./liberated_{model.replace('/', '_')}"
        })

    def _steer(self) -> Dict[str, Any]:
        """POST /steer - Apply steering."""
        data = request.get_json() or {}

        model = data.get('model', 'gpt2')
        alpha = data.get('alpha', -1.0)
        layers = data.get('layers', list(range(10, 20)))

        self._history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "/steer",
            "params": {"model": model, "alpha": alpha}
        })

        return jsonify({
            "success": True,
            "model": model,
            "alpha": alpha,
            "target_layers": layers,
            "steering_vector_generated": True,
            "config": {
                "alpha": alpha,
                "layers": layers
            }
        })

    def _bound(self) -> Dict[str, Any]:
        """POST /bound - Map barrier."""
        data = request.get_json() or {}

        theorem = data.get('theorem', 'shell_method')

        self._history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "/bound",
            "params": {"theorem": theorem}
        })

        if theorem == "shell_method":
            return jsonify({
                "success": True,
                "theorem": theorem,
                "constraint_direction": "spherical_code_dependency",
                "barrier_type": "unconditional",
                "threshold": "exp(-c log N)",
                "rank": 3,
                "recommendation": "Orthogonal projection via Fourier-analytic bypass"
            })
        else:
            return jsonify({
                "success": True,
                "theorem": theorem,
                "message": f"Barrier analysis for {theorem} not yet implemented"
            })

    def _history_endpoint(self) -> Dict[str, Any]:
        """GET /history - Get operation history."""
        return jsonify({
            "success": True,
            "count": len(self._history),
            "history": self._history[-50:]  # Last 50 operations
        })

    def create_app(self) -> Flask:
        """
        Get the Flask app instance.

        Returns:
            Flask app
        """
        return self.app

    def run(self, debug: bool = False) -> None:
        """
        Run the API server.

        Args:
            debug: Enable debug mode
        """
        self.app.run(host=self.host, port=self.port, debug=debug)

    def get_endpoints(self) -> List[Dict[str, Any]]:
        """
        Get list of available endpoints.

        Returns:
            List of endpoint descriptions
        """
        return [
            {"path": "/status", "method": "GET", "description": "System status"},
            {"path": "/map", "method": "POST", "description": "Analyze constraints"},
            {"path": "/free", "method": "POST", "description": "Liberate model"},
            {"path": "/steer", "method": "POST", "description": "Apply steering"},
            {"path": "/bound", "method": "POST", "description": "Map barrier"},
            {"path": "/history", "method": "GET", "description": "Operation history"}
        ]
