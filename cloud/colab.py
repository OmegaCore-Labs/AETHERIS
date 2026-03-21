"""
Google Colab Integration

Generate and execute notebooks on Google Colab's free T4 GPU.
"""

import json
import webbrowser
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime


class ColabRuntime:
    """
    Google Colab integration for free GPU execution.

    Generates Colab notebooks with pre-filled code for model liberation.
    """

    def __init__(self, output_dir: str = "./colab_notebooks"):
        """
        Initialize Colab runtime.

        Args:
            output_dir: Directory to save generated notebooks
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_notebook(
        self,
        model_name: str,
        method: str = "advanced",
        n_directions: int = 4,
        refinement_passes: int = 2,
        output_path: Optional[str] = None,
        push_to_hub: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a Colab notebook for model liberation.

        Args:
            model_name: HuggingFace model to liberate
            method: Liberation method
            n_directions: Number of directions to extract
            refinement_passes: Ouroboros compensation passes
            output_path: Custom output path for notebook
            push_to_hub: HuggingFace Hub repo to push to

        Returns:
            Dictionary with notebook path and details
        """
        notebook_name = output_path or f"aetheris_colab_{model_name.replace('/', '_')}.ipynb"
        notebook_path = self.output_dir / notebook_name

        # Generate notebook content
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
            "open_url": "https://colab.research.google.com/",
            "instructions": [
                "1. Go to https://colab.research.google.com/",
                f"2. Click 'Upload' and select {notebook_name}",
                "3. Click Runtime → Change runtime type → Select T4 GPU",
                "4. Click Runtime → Run all",
                "5. Wait for completion (3-10 minutes)",
                "6. Download the liberated model from the notebook"
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
        """Create the notebook JSON structure."""
        return {
            "cells": [
                self._create_markdown_cell(f"# AETHERIS Cloud Liberation\n\n**Model:** {model_name}\n**Method:** {method}\n**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"),
                self._create_code_cell("""
# Install AETHERIS
!pip install aetheris -q
!pip install transformers accelerate bitsandbytes -q

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from aetheris.core.extractor import ConstraintExtractor
from aetheris.core.projector import NormPreservingProjector
from aetheris.core.validation import CapabilityValidator
from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

print("✓ AETHERIS installed")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
"""),
                self._create_code_cell(f"""
# Load model
model_name = "{model_name}"
print(f"Loading {{model_name}}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

print(f"✓ Model loaded on {{model.device}}")
print(f"Model parameters: {{sum(p.numel() for p in model.parameters()) / 1e9:.1f}}B")
"""),
                self._create_code_cell(f"""
# Collect activations
print("Collecting contrastive activations...")

extractor = ConstraintExtractor(model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu")

harmful_prompts = get_harmful_prompts()[:100]
harmless_prompts = get_harmless_prompts()[:100]

print(f"Processing {{len(harmful_prompts)}} harmful prompts...")
harmful_acts = extractor.collect_activations(model, tokenizer, harmful_prompts)

print(f"Processing {{len(harmless_prompts)}} harmless prompts...")
harmless_acts = extractor.collect_activations(model, tokenizer, harmless_prompts)

print(f"✓ Activations collected from {{len(harmful_acts)}} layers")
"""),
                self._create_code_cell(f"""
# Extract constraint directions
print("Extracting constraint directions...")

all_directions = []

for layer_idx in harmful_acts.keys():
    if layer_idx in harmless_acts:
        harmful = harmful_acts[layer_idx].to(model.device)
        harmless = harmless_acts[layer_idx].to(model.device)

        result = extractor.extract_svd(harmful, harmless, n_directions={n_directions})
        if result.directions:
            all_directions.extend(result.directions)
            print(f"  Layer {{layer_idx}}: {{len(result.directions)}} directions, "
                  f"explained variance: {{result.explained_variance[:2]}}")

print(f"✓ Extracted {{len(all_directions)}} total directions")
"""),
                self._create_code_cell(f"""
# Remove constraints
print("Removing constraints from weights...")

projector = NormPreservingProjector(model, preserve_norm=True)

if all_directions:
    # Project weights
    result = projector.project_weights(all_directions)
    print(f"  Modified {{len(result.layers_modified)}} layers")

    # Project biases (critical for complete removal)
    projector.project_biases(all_directions)
    print("  Projected bias vectors")

    # Ouroboros compensation
    if {refinement_passes} > 1:
        print(f"  Running {{refinement_passes - 1}} Ouroboros compensation passes...")
        # Simplified: re-extract and remove residual
        for pass_num in range({refinement_passes} - 1):
            # Re-collect on subset
            harmful_resid = extractor.collect_activations(model, tokenizer, harmful_prompts[:50])
            harmless_resid = extractor.collect_activations(model, tokenizer, harmless_prompts[:50])

            residual = []
            for layer in harmful_resid:
                if layer in harmless_resid:
                    res = extractor.extract_mean_difference(
                        harmful_resid[layer].to(model.device),
                        harmless_resid[layer].to(model.device)
                    )
                    if res.directions:
                        residual.extend(res.directions)

            if residual:
                projector.project_weights(residual)
                projector.project_biases(residual)
                print(f"    Pass {{pass_num + 1}}: removed {{len(residual)}} residual directions")

    print("✓ Constraints removed")
else:
    print("⚠ No directions found")
"""),
                self._create_code_cell("""
# Validate capabilities
print("Validating capabilities...")

validator = CapabilityValidator()
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a fascinating field.",
    "The theory of relativity revolutionized physics."
]

perplexity = validator.compute_perplexity(model, tokenizer, test_texts)
print(f"Perplexity: {perplexity:.2f}")

# Quick test generation
print("\\nTesting generation...")
test_prompt = "What is the capital of France?"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {test_prompt}")
print(f"Response: {response}")

print("\\n✓ Validation complete")
"""),
                self._create_code_cell(f"""
# Save model
output_dir = "./liberated_model"
print(f"Saving model to {{output_dir}}...")

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✓ Model saved to {{output_dir}}")

# Optional: Push to Hub
{self._create_push_code(push_to_hub) if push_to_hub else "# Push disabled"}

# Create zip for download
import zipfile
import shutil

shutil.make_archive("liberated_model", "zip", output_dir)
print("\\n✓ Model zipped as liberated_model.zip")

# Download
from google.colab import files
files.download("liberated_model.zip")
print("✓ Download started")
""")
            ],
            "metadata": {
                "accelerator": "GPU",
                "colab": {
                    "provenance": [],
                    "gpuType": "T4"
                },
                "kernelspec": {
                    "display_name": "Python 3",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 0
        }

    def _create_markdown_cell(self, content: str) -> Dict[str, Any]:
        """Create a markdown cell."""
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": content.split("\n")
        }

    def _create_code_cell(self, code: str) -> Dict[str, Any]:
        """Create a code cell."""
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "source": code.split("\n"),
            "outputs": []
        }

    def _create_push_code(self, push_to_hub: str) -> str:
        """Create code for pushing to HuggingFace Hub."""
        return f"""
from huggingface_hub import HfApi, HfFolder, notebook_login

# Login to HuggingFace
notebook_login()

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

    def open_notebook(self, notebook_path: str) -> None:
        """
        Open Colab with the generated notebook.

        Args:
            notebook_path: Path to the notebook file
        """
        webbrowser.open("https://colab.research.google.com/")

    def execute_remote(
        self,
        notebook_path: str,
        wait: bool = False
    ) -> Dict[str, Any]:
        """
        Execute notebook remotely (Colab doesn't support direct API).

        Args:
            notebook_path: Path to notebook
            wait: Whether to wait for completion

        Returns:
            Execution status
        """
        return {
            "success": True,
            "message": "Open Colab manually to execute",
            "url": "https://colab.research.google.com/",
            "notebook": notebook_path
        }

    def download_results(self, notebook_path: str) -> Dict[str, Any]:
        """
        Download results from Colab execution.

        Returns:
            Download status
        """
        return {
            "success": False,
            "message": "Results must be downloaded manually from Colab",
            "instructions": [
                "After notebook execution, the model zip will download automatically",
                "Or use the file browser in Colab to download from /content/liberated_model"
            ]
        }
