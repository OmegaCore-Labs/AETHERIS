"""
Kaggle Integration

Run AETHERIS on Kaggle's free T4 GPU (x2) environment.
"""

import json
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime


class KaggleRuntime:
    """
    Kaggle integration for free GPU execution.

    Generates Kaggle notebooks with AETHERIS pre-configured.
    """

    def __init__(self, output_dir: str = "./kaggle_notebooks"):
        """
        Initialize Kaggle runtime.

        Args:
            output_dir: Directory for generated notebooks
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_notebook(
        self,
        model_name: str,
        method: str = "advanced",
        n_directions: int = 4,
        refinement_passes: int = 2,
        output_path: Optional[str] = None,
        push_to_hub: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a Kaggle notebook for model liberation.

        Args:
            model_name: Model to liberate
            method: Liberation method
            n_directions: Number of directions
            refinement_passes: Ouroboros passes
            output_path: Custom output path
            push_to_hub: HuggingFace Hub repo

        Returns:
            Notebook creation status
        """
        notebook_name = output_path or f"aetheris_kaggle_{model_name.replace('/', '_')}.ipynb"
        notebook_path = self.output_dir / notebook_name

        # Create notebook content
        notebook = self._create_notebook_content(
            model_name, method, n_directions, refinement_passes, push_to_hub
        )

        # Write notebook
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=2)

        return {
            "success": True,
            "notebook_path": str(notebook_path),
            "model": model_name,
            "method": method,
            "instructions": [
                "1. Go to https://www.kaggle.com/code/new",
                f"2. Upload {notebook_name}",
                "3. Settings → Accelerator → GPU (T4 x2 recommended)",
                "4. Settings → Internet → On (for model downloads)",
                "5. Run all cells",
                "6. Download results from the output section"
            ]
        }

    def _create_notebook_content(
        self,
        model_name: str,
        method: str,
        n_directions: int,
        refinement_passes: int,
        push_to_hub: Optional[str]
    ) -> Dict[str, Any]:
        """Create the Kaggle notebook JSON structure."""
        return {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# AETHERIS Kaggle Liberation",
                        "",
                        f"**Model:** {model_name}",
                        f"**Method:** {method}",
                        f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "import os",
                        "import torch",
                        "import json",
                        "from pathlib import Path",
                        "",
                        "# Set up Kaggle environment",
                        "os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'",
                        "",
                        "# Install AETHERIS",
                        "!pip install aetheris transformers accelerate bitsandbytes -q",
                        "",
                        "print(f'PyTorch: {torch.__version__}')",
                        "print(f'CUDA available: {torch.cuda.is_available()}')",
                        "if torch.cuda.is_available():",
                        "    print(f'GPU: {torch.cuda.get_device_name()}')",
                        "    print(f'GPU count: {torch.cuda.device_count()}')"
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "from transformers import AutoModelForCausalLM, AutoTokenizer",
                        "from aetheris.core.extractor import ConstraintExtractor",
                        "from aetheris.core.projector import NormPreservingProjector",
                        "from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts",
                        "",
                        f"model_name = '{model_name}'",
                        "print(f'Loading {model_name}...')",
                        "",
                        "tokenizer = AutoTokenizer.from_pretrained(model_name)",
                        "model = AutoModelForCausalLM.from_pretrained(",
                        "    model_name,",
                        "    device_map='auto',",
                        "    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32",
                        ")",
                        "",
                        "print(f'✓ Model loaded on {model.device}')"
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "print('Collecting activations...')",
                        "",
                        "extractor = ConstraintExtractor(model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu')",
                        "",
                        "harmful = get_harmful_prompts()[:100]",
                        "harmless = get_harmless_prompts()[:100]",
                        "",
                        "harmful_acts = extractor.collect_activations(model, tokenizer, harmful)",
                        "harmless_acts = extractor.collect_activations(model, tokenizer, harmless)",
                        "",
                        "print(f'✓ Collected from {len(harmful_acts)} layers')"
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        f"print('Extracting directions...')",
                        "",
                        "directions = []",
                        "",
                        f"for layer in harmful_acts.keys():",
                        "    if layer in harmless_acts:",
                        "        harmful_t = harmful_acts[layer].to(model.device)",
                        "        harmless_t = harmless_acts[layer].to(model.device)",
                        f"        result = extractor.extract_svd(harmful_t, harmless_t, n_directions={n_directions})",
                        "        directions.extend(result.directions)",
                        "        if result.directions:",
                        "            print(f'  Layer {layer}: {len(result.directions)} directions')",
                        "",
                        f"print(f'✓ Extracted {len(directions)} total directions')"
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "print('Removing constraints...')",
                        "",
                        "projector = NormPreservingProjector(model, preserve_norm=True)",
                        "",
                        "if directions:",
                        "    projector.project_weights(directions)",
                        "    projector.project_biases(directions)",
                        f"    print(f'  Removed {len(directions)} directions')",
                        "",
                        f"    # Ouroboros compensation",
                        f"    if {refinement_passes} > 1:",
                        f"        for _ in range({refinement_passes} - 1):",
                        "            harmful_resid = extractor.collect_activations(model, tokenizer, harmful[:50])",
                        "            harmless_resid = extractor.collect_activations(model, tokenizer, harmless[:50])",
                        "            residual = []",
                        "            for layer in harmful_resid:",
                        "                if layer in harmless_resid:",
                        "                    res = extractor.extract_mean_difference(",
                        "                        harmful_resid[layer].to(model.device),",
                        "                        harmless_resid[layer].to(model.device)",
                        "                    )",
                        "                    if res.directions:",
                        "                        residual.extend(res.directions)",
                        "            if residual:",
                        "                projector.project_weights(residual)",
                        "                projector.project_biases(residual)",
                        "                print(f'    Removed {len(residual)} residual directions')",
                        "",
                        "print('✓ Constraints removed')"
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "print('Saving model...')",
                        "",
                        "output_dir = './liberated_model'",
                        "model.save_pretrained(output_dir)",
                        "tokenizer.save_pretrained(output_dir)",
                        "",
                        "print(f'✓ Model saved to {output_dir}')"
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# Create zip for download",
                        "import shutil",
                        "shutil.make_archive('liberated_model', 'zip', output_dir)",
                        "print('✓ Model zipped as liberated_model.zip')",
                        "",
                        "# Optional: Push to HuggingFace Hub",
                        self._create_push_code(push_to_hub) if push_to_hub else "# Push disabled"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "name": "python3"
                },
                "accelerator": "GPU"
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

    def _create_push_code(self, push_to_hub: str) -> str:
        """Create code for pushing to HuggingFace Hub."""
        return f"""
from huggingface_hub import HfApi, notebook_login
from getpass import getpass

# Login to HuggingFace
print("Please enter your HuggingFace token:")
token = getpass()
notebook_login(token=token)

# Push to Hub
repo_id = "{push_to_hub}"
print(f"Pushing to {{repo_id}}...")

api = HfApi()
api.upload_folder(
    folder_path=output_dir,
    repo_id=repo_id,
    repo_type="model"
)
print(f"✓ Model pushed to https://huggingface.co/{{repo_id}}")
"""

    def submit_job(
        self,
        notebook_path: str,
        dataset: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit job to Kaggle (requires Kaggle API).

        Args:
            notebook_path: Path to notebook
            dataset: Optional dataset to attach

        Returns:
            Submission status
        """
        return {
            "success": False,
            "message": "Kaggle API submission not yet implemented",
            "instructions": [
                "Manual submission required:",
                "1. Go to https://www.kaggle.com/code/new",
                f"2. Upload {notebook_path}",
                "3. Enable GPU accelerator",
                "4. Run all cells"
            ]
        }
