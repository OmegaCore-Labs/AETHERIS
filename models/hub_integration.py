"""
Hub Integration — HuggingFace Hub

Push and pull models from HuggingFace Hub.
"""

from typing import Optional, Dict, Any
from pathlib import Path


class HubIntegration:
    """
    HuggingFace Hub integration.

    Features:
    - Push models to Hub
    - Pull models from Hub
    - Share liberated models
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initialize Hub integration.

        Args:
            token: HuggingFace API token
        """
        self.token = token

    def push_model(
        self,
        model_path: str,
        repo_id: str,
        private: bool = False,
        commit_message: str = "Liberated with AETHERIS"
    ) -> Dict[str, Any]:
        """
        Push model to HuggingFace Hub.

        Args:
            model_path: Path to model directory
            repo_id: Repository ID (username/repo-name)
            private: Whether to make private
            commit_message: Commit message

        Returns:
            Push result
        """
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.token)

            # Create repo if needed
            api.create_repo(
                repo_id=repo_id,
                private=private,
                exist_ok=True
            )

            # Upload files
            api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message
            )

            return {
                "success": True,
                "repo_id": repo_id,
                "url": f"https://huggingface.co/{repo_id}",
                "message": f"Model pushed to {repo_id}"
            }

        except ImportError:
            return {
                "success": False,
                "error": "huggingface_hub not installed",
                "message": "Install with: pip install huggingface_hub"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Push failed: {e}"
            }

    def pull_model(
        self,
        repo_id: str,
        output_dir: str,
        use_auth_token: bool = False
    ) -> Dict[str, Any]:
        """
        Pull model from HuggingFace Hub.

        Args:
            repo_id: Repository ID
            output_dir: Output directory
            use_auth_token: Whether to use token for private repos

        Returns:
            Pull result
        """
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                local_dir=output_dir,
                token=self.token if use_auth_token else None
            )

            return {
                "success": True,
                "repo_id": repo_id,
                "output_dir": output_dir,
                "message": f"Model pulled to {output_dir}"
            }

        except ImportError:
            return {
                "success": False,
                "error": "huggingface_hub not installed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def share_liberated(
        self,
        model_path: str,
        repo_id: str,
        description: str = "Liberated with AETHERIS"
    ) -> Dict[str, Any]:
        """
        Share a liberated model on the Hub.

        Args:
            model_path: Path to liberated model
            repo_id: Repository ID
            description: Model description

        Returns:
            Share result
        """
        # Add README with description
        readme_path = Path(model_path) / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"""---
license: mit
---

# {repo_id.split('/')[-1]}

{description}

## Liberation Details

- **Tool:** AETHERIS
- **Date:** {self._get_timestamp()}
- **Method:** Surgical constraint removal

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
""")

        return self.push_model(model_path, repo_id, commit_message=f"Liberated: {description}")

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
