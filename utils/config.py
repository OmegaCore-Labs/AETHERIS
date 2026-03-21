"""
Configuration Management

Load and save configuration from environment variables, JSON, and YAML files.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class AetherisConfig:
    """Main configuration dataclass."""
    # Core settings
    device: str = "auto"
    dtype: str = "auto"
    verbose: bool = False

    # Model settings
    default_model: str = "gpt2"
    max_seq_length: int = 512
    batch_size: int = 4

    # Liberation settings
    default_method: str = "advanced"
    default_n_directions: int = 4
    default_refinement_passes: int = 2
    preserve_norm: bool = True

    # Cloud settings
    prefer_free: bool = True
    colab_output_dir: str = "./colab_notebooks"
    spaces_output_dir: str = "./spaces_deploy"
    kaggle_output_dir: str = "./kaggle_notebooks"

    # API keys
    hf_token: Optional[str] = None
    openai_key: Optional[str] = None
    anthropic_key: Optional[str] = None

    # Paths
    cache_dir: str = "./cache"
    output_dir: str = "./output"
    log_file: Optional[str] = None

    # Advanced
    enable_telemetry: bool = False
    telemetry_url: Optional[str] = None

    # Custom
    custom: Dict[str, Any] = field(default_factory=dict)


class Config:
    """
    Configuration manager for AETHERIS.

    Loads from:
    - Environment variables (AETHERIS_*)
    - JSON file
    - YAML file
    - Default values
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config file (JSON or YAML)
        """
        self._config = AetherisConfig()

        # Load from file
        if config_path:
            self.load_file(config_path)

        # Load from environment
        self.load_env()

    def load_file(self, path: str) -> None:
        """
        Load configuration from file.

        Args:
            path: Path to JSON or YAML file
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        if path_obj.suffix in ['.json', '.json5']:
            with open(path_obj, 'r') as f:
                data = json.load(f)
        elif path_obj.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(path_obj, 'r') as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML config: pip install pyyaml")
        else:
            raise ValueError(f"Unsupported config format: {path_obj.suffix}")

        self._update_from_dict(data)

    def load_env(self) -> None:
        """Load configuration from environment variables."""
        prefix = "AETHERIS_"

        for key in dir(self._config):
            env_var = f"{prefix}{key.upper()}"
            if env_var in os.environ:
                value = os.environ[env_var]

                # Parse boolean
                if value.lower() in ['true', '1', 'yes']:
                    value = True
                elif value.lower() in ['false', '0', 'no']:
                    value = False

                setattr(self._config, key, value)

    def _update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update config from dictionary."""
        for key, value in data.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                self._config.custom[key] = value

    def save(self, path: str, format: str = "json") -> None:
        """
        Save configuration to file.

        Args:
            path: Output path
            format: "json" or "yaml"
        """
        data = self.to_dict()
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(path_obj, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "yaml":
            try:
                import yaml
                with open(path_obj, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML required for YAML export: pip install pyyaml")
        else:
            raise ValueError(f"Unsupported format: {format}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in self._config.__dict__.items():
            if not key.startswith('_'):
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        if hasattr(self._config, key):
            return getattr(self._config, key)
        return self._config.custom.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        if hasattr(self._config, key):
            setattr(self._config, key, value)
        else:
            self._config.custom[key] = value

    @property
    def config(self) -> AetherisConfig:
        """Get the underlying config object."""
        return self._config
