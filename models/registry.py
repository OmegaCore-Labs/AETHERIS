"""
Model Registry — Track and Manage Models

Registry of available models with metadata and status tracking.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class ModelRegistry:
    """
    Track and manage models.

    Features:
    - Model registration
    - Status tracking
    - Search and filtering
    """

    def __init__(self):
        self._models = {}

    def register(
        self,
        model_id: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Register a model.

        Args:
            model_id: Unique model identifier
            model_path: Path to model
            metadata: Additional metadata

        Returns:
            Registration result
        """
        self._models[model_id] = {
            "id": model_id,
            "path": model_path,
            "registered_at": datetime.utcnow().isoformat(),
            "status": "registered",
            "metadata": metadata or {}
        }

        return self._models[model_id]

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model by ID."""
        return self._models.get(model_id)

    def list_models(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List registered models.

        Args:
            status: Filter by status
            limit: Maximum number to return

        Returns:
            List of models
        """
        models = list(self._models.values())
        if status:
            models = [m for m in models if m.get("status") == status]
        return models[:limit]

    def update_status(
        self,
        model_id: str,
        status: str
    ) -> Dict[str, Any]:
        """Update model status."""
        if model_id in self._models:
            self._models[model_id]["status"] = status
            self._models[model_id]["updated_at"] = datetime.utcnow().isoformat()
            return self._models[model_id]
        return {"error": "Model not found"}
