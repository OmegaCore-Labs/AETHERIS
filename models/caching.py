"""
Model Caching — Cache Models Locally

Efficiently cache models to avoid repeated downloads.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ModelCache:
    """
    Cache models locally.

    Features:
    - Model caching
    - Cache management
    - Cache cleanup
    """

    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize cache.

        Args:
            cache_dir: Directory for cached models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_model(
        self,
        model_name: str,
        model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cache a model.

        Args:
            model_name: Model identifier
            model_path: Path to model (None = download)

        Returns:
            Cache result
        """
        cache_path = self.cache_dir / model_name.replace("/", "_")

        if model_path:
            # Copy existing model
            shutil.copytree(model_path, cache_path, dirs_exist_ok=True)
        else:
            # Download model (would use transformers)
            cache_path.mkdir(parents=True, exist_ok=True)

        return {
            "success": True,
            "cache_path": str(cache_path),
            "model_name": model_name,
            "cached_at": datetime.utcnow().isoformat()
        }

    def load_cached(self, model_name: str) -> Optional[str]:
        """
        Load a cached model.

        Args:
            model_name: Model identifier

        Returns:
            Path to cached model or None
        """
        cache_path = self.cache_dir / model_name.replace("/", "_")
        if cache_path.exists():
            return str(cache_path)
        return None

    def clear_cache(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear cache.

        Args:
            model_name: Specific model to clear (None = all)

        Returns:
            Clear result
        """
        if model_name:
            cache_path = self.cache_dir / model_name.replace("/", "_")
            if cache_path.exists():
                shutil.rmtree(cache_path)
            return {"cleared": [model_name]}
        else:
            for item in self.cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
            return {"cleared": "all"}

    def get_cache_size(self) -> float:
        """Get cache size in GB."""
        total = 0
        for item in self.cache_dir.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total / (1024 ** 3)

    def list_cached(self) -> List[Dict[str, Any]]:
        """List cached models."""
        cached = []
        for item in self.cache_dir.iterdir():
            if item.is_dir():
                cached.append({
                    "name": item.name,
                    "path": str(item),
                    "size_gb": sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024 ** 3)
                })
        return cached
