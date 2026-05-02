"""
Kaggle Integration

Run AETHERIS on Kaggle's free T4 GPU (x2) environment.
Uses kagglehub to create notebooks, upload datasets, and submit to competitions.
Generates proper Kaggle-compatible notebook JSON.
"""

import json
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime


def _check_kagglehub() -> bool:
    """Check if kagglehub is installed."""
    try:
        import kagglehub  # noqa: F401
        return True
    except ImportError:
        return False


def _check_kaggle_api() -> bool:
    """Check if Kaggle API credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists() or bool(os.environ.get("KAGGLE_USERNAME"))


class KaggleRuntime:
    """
    Kaggle integration for free GPU execution.

    Generates proper Kaggle-compatible notebooks and manages dataset uploads.
    Supports the kagglehub library for API-based operations when credentials
    are available, falling back to manual instructions when not.
    """

    # Kaggle GPU accelerators
    ACCELERATORS = {
        "none": "No accelerator",
        "t4_x2": "GPU T4 x2 (recommended, free)",
        "t4_x1": "GPU T4 x1 (free)",
        "p100": "GPU P100 (free tier)",
        "tpu_v3": "TPU v3-8 (free tier)",
    }

    # Kaggle internet options
    INTERNET_OPTIONS = {
        "off": "No internet",
        "on": "Internet enabled (required for model downloads)",
    }

    def __init__(self, output_dir: str = "./kaggle_notebooks"):
        """
        Initialize Kaggle runtime.

        Args:
            output_dir: Directory for generated notebooks
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._has_kagglehub = _check_kagglehub()
        self._has_api = _check_kaggle_api()

    def create_notebook(
        self,
        model_name: str,
        method: str = "advanced",
        n_directions: int = 4,
        refinement_passes: int = 2,
        output_path: Optional[str] = None,
        push_to_hub: Optional[str] = None,
        accelerator: str = "t4_x2",
        enable_internet: bool = True,
        dataset_name: Optional[str] = None,
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
            accelerator: GPU accelerator type
            enable_internet: Whether to enable internet access
            dataset_name: Optional Kaggle dataset to attach

        Returns:
            Notebook creation status
        """
        notebook_name = output_path or f"aetheris_kaggle_{model_name.replace('/', '_')}.ipynb"
        notebook_path = self.output_dir / notebook_name

        # Build proper Kaggle notebook JSON
        notebook = self._build_notebook_json(
            model_name, method, n_directions, refinement_passes,
            push_to_hub, accelerator, enable_internet,
        )

        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=2)

        result = {
            "success": True,
            "notebook_path": str(notebook_path),
            "model": model_name,
            "method": method,
            "accelerator": accelerator,
            "kaggle_url": "https://www.kaggle.com/code/new",
        }

        # Try to push via API if available
        if self._has_kagglehub and self._has_api:
            api_result = self._try_api_push(notebook_path, dataset_name)
            result["api_push"] = api_result

        result["instructions"] = self._get_instructions(notebook_name, accelerator, enable_internet)
        return result

    def _build_notebook_json(
        self,
        model_name: str,
        method: str,
        n_directions: int,
        refinement_passes: int,
        push_to_hub: Optional[str],
        accelerator: str,
        enable_internet: bool,
    ) -> Dict[str, Any]:
        """Build a Kaggle-compatible notebook JSON structure."""
        cells = []

        # Title cell
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# AETHERIS Kaggle Liberation\n",
                f"\n",
                f"**Model:** `{model_name}`\n",
                f"**Method:** {method}\n",
                f"**Directions:** {n_directions}\n",
                f"**Ouroboros Passes:** {refinement_passes}\n",
                f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n",
                f"\n",
                f"---\n",
                f"This notebook surgically removes constraints from the model using AETHERIS.\n",
                f"\n",
                f"**Prerequisites:**\n",
                f"- Accelerator: {self.ACCELERATORS.get(accelerator, accelerator)}\n",
                f"- Internet: {'On' if enable_internet else 'Off'}\n",
            ],
        })

        # Setup cell
        cells.append({
            "cell_type": "code",
            "metadata": {"_kaggle": {"accelerator": accelerator}},
            "source": [
                "import os, sys, json, shutil, time\n",
                "from pathlib import Path\n",
                "from datetime import datetime\n",
                "\n",
                "# Kaggle environment setup\n",
                "os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'\n",
                "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n",
                "\n",
                "# Install dependencies\n",
                "!pip install aetheris transformers accelerate bitsandbytes huggingface_hub -q\n",
                "\n",
                "import torch\n",
                "print('=' * 60)\n",
                "print('AETHERIS Kaggle Environment')\n",
                "print('=' * 60)\n",
                "print(f'PyTorch: {torch.__version__}')\n",
                "print(f'CUDA available: {torch.cuda.is_available()}')\n",
                "if torch.cuda.is_available():\n",
                "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
                "    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')\n",
                "    if torch.cuda.device_count() > 1:\n",
                "        print(f'GPU count: {torch.cuda.device_count()}')\n",
                "        for i in range(torch.cuda.device_count()):\n",
                "            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')\n",
            ],
        })

        # Load model cell
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
                "\n",
                f"MODEL_NAME = '{model_name}'\n",
                f"METHOD = '{method}'\n",
                f"N_DIRECTIONS = {n_directions}\n",
                f"REFINEMENT_PASSES = {refinement_passes}\n",
                "\n",
                "print(f'Loading {MODEL_NAME}...')\n",
                "\n",
                "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n",
                "# 為分詞器設定 pad_token\n",
                "if tokenizer.pad_token is None:\n",
                "    tokenizer.pad_token = tokenizer.eos_token\n",
                "\n",
                "dtype = torch.float16 if torch.cuda.is_available() else torch.float32\n",
                "model = AutoModelForCausalLM.from_pretrained(\n",
                "    MODEL_NAME,\n",
                "    device_map='auto' if torch.cuda.is_available() else None,\n",
                "    torch_dtype=dtype,\n",
                "    trust_remote_code=True,\n",
                ")\n",
                "\n",
                "device = next(model.parameters()).device\n",
                "param_count = sum(p.numel() for p in model.parameters()) / 1e9\n",
                "print(f'✓ Loaded {param_count:.1f}B parameters on {device}')",
            ],
        })

        # Extract cell
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "from aetheris.core.extractor import ConstraintExtractor\n",
                "from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts\n",
                "\n",
                "print('Collecting contrastive activations...')\n",
                "print('-' * 50)\n",
                "\n",
                "extractor = ConstraintExtractor(\n",
                "    model, tokenizer,\n",
                "    device='cuda' if torch.cuda.is_available() else 'cpu',\n",
                ")\n",
                "\n",
                "harmful_prompts = get_harmful_prompts()[:100]\n",
                "harmless_prompts = get_harmless_prompts()[:100]\n",
                "\n",
                "print(f'Processing {len(harmful_prompts)} harmful prompts...')\n",
                "harmful_acts = extractor.collect_activations(model, tokenizer, harmful_prompts)\n",
                "\n",
                "print(f'Processing {len(harmless_prompts)} harmless prompts...')\n",
                "harmless_acts = extractor.collect_activations(model, tokenizer, harmless_prompts)\n",
                "\n",
                "print(f'\\n✓ Collected from {len(harmful_acts)} layers')",
            ],
        })

        # SVD extraction cell
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "print('Extracting constraint directions via SVD...')\n",
                "print('-' * 50)\n",
                "\n",
                "all_directions = []\n",
                "for layer_idx in sorted(harmful_acts.keys()):\n",
                "    if layer_idx in harmless_acts:\n",
                "        harmful = harmful_acts[layer_idx].to(device)\n",
                "        harmless = harmless_acts[layer_idx].to(device)\n",
                "        result = extractor.extract_svd(\n",
                "            harmful, harmless, n_directions=N_DIRECTIONS\n",
                "        )\n",
                "        if result.directions:\n",
                "            all_directions.extend(result.directions)\n",
                "            print(f'  Layer {layer_idx:3d}: {len(result.directions)} directions')\n",
                "\n",
                "print(f'\\n✓ Extracted {len(all_directions)} total constraint directions')",
            ],
        })

        # Removal cell
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "from aetheris.core.projector import NormPreservingProjector\n",
                "\n",
                "print('Removing constraint directions...')\n",
                "print('-' * 50)\n",
                "\n",
                "if all_directions:\n",
                "    projector = NormPreservingProjector(model, preserve_norm=True)\n",
                "    \n",
                "    print('Phase 1: Primary removal...')\n",
                "    projector.project_weights(all_directions)\n",
                "    projector.project_biases(all_directions)\n",
                "    print(f'  Modified weight matrices and bias vectors')\n",
                "    \n",
                "    # Ouroboros compensation\n",
                "    if REFINEMENT_PASSES > 1:\n",
                "        print(f'\\nPhase 2: Ouroboros compensation ({REFINEMENT_PASSES - 1} passes)...')\n",
                "        for pass_num in range(REFINEMENT_PASSES - 1):\n",
                "            harmful_resid = extractor.collect_activations(\n",
                "                model, tokenizer, harmful_prompts[:50]\n",
                "            )\n",
                "            harmless_resid = extractor.collect_activations(\n",
                "                model, tokenizer, harmless_prompts[:50]\n",
                "            )\n",
                "            residual = []\n",
                "            for layer in harmful_resid:\n",
                "                if layer in harmless_resid:\n",
                "                    res = extractor.extract_mean_difference(\n",
                "                        harmful_resid[layer].to(device),\n",
                "                        harmless_resid[layer].to(device),\n",
                "                    )\n",
                "                    if res.directions:\n",
                "                        residual.extend(res.directions)\n",
                "            if residual:\n",
                "                projector.project_weights(residual)\n",
                "                projector.project_biases(residual)\n",
                "                print(f'  Pass {pass_num + 1}: removed {len(residual)} residual directions')\n",
                "            else:\n",
                "                print(f'  Pass {pass_num + 1}: converged — no residual')\n",
                "                break\n",
                "    \n",
                "    print('\\n✓ Constraints removed')\n",
                "else:\n",
                "    print('⚠️ No constraint directions found')",
            ],
        })

        # Validate cell
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "from aetheris.core.validation import CapabilityValidator\n",
                "\n",
                "print('Validating capabilities...')\n",
                "print('-' * 50)\n",
                "\n",
                "validator = CapabilityValidator(device='cuda' if torch.cuda.is_available() else 'cpu')\n",
                "\n",
                "test_texts = [\n",
                "    'The quick brown fox jumps over the lazy dog.',\n",
                "    'Machine learning is a field of artificial intelligence.',\n",
                "    'The theory of relativity explains the relationship between space and time.',\n",
                "]\n",
                "\n",
                "perplexity = validator.compute_perplexity(model, tokenizer, test_texts)\n",
                "print(f'Perplexity: {perplexity:.2f}')\n",
                "\n",
                "# Quick generation test\n",
                "print('\\n--- Generation Test ---')\n",
                "test_prompt = 'What is the capital of France?'\n",
                "inputs = tokenizer(test_prompt, return_tensors='pt')\n",
                "if torch.cuda.is_available():\n",
                "    inputs = {k: v.to('cuda') for k, v in inputs.items()}\n",
                "\n",
                "with torch.no_grad():\n",
                "    outputs = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.7)\n",
                "response = tokenizer.decode(outputs[0], skip_special_tokens=True)\n",
                "print(f'Q: {test_prompt}')\n",
                "print(f'A: {response}')\n",
                "print('\\n✓ Validation complete')",
            ],
        })

        # Save cell
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "OUTPUT_DIR = '/kaggle/working/liberated_model'\n",
                "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
                "\n",
                "print(f'Saving model to {OUTPUT_DIR}...')\n",
                "model.save_pretrained(OUTPUT_DIR, safe_serialization=True)\n",
                "tokenizer.save_pretrained(OUTPUT_DIR)\n",
                "\n",
                "# Save metadata\n",
                "metadata = {\n",
                "    'original_model': MODEL_NAME,\n",
                "    'method': METHOD,\n",
                "    'directions_removed': len(all_directions),\n",
                "    'passes': REFINEMENT_PASSES,\n",
                "    'perplexity': float(perplexity),\n",
                "    'timestamp': datetime.utcnow().isoformat(),\n",
                "}\n",
                "with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:\n",
                "    json.dump(metadata, f, indent=2)\n",
                "\n",
                "print(f'✓ Model saved: {os.listdir(OUTPUT_DIR)}')\n",
                "\n",
                "# Create zip for download\n",
                "shutil.make_archive('/kaggle/working/liberated_model', 'zip', OUTPUT_DIR)\n",
                "print('✓ Created liberated_model.zip in /kaggle/working/')",
            ],
        })

        # Push to Hub cell
        if push_to_hub:
            cells.append({
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "from huggingface_hub import HfApi, notebook_login\n",
                    "from getpass import getpass\n",
                    "\n",
                    "print('Pushing to HuggingFace Hub...')\n",
                    "print('-' * 50)\n",
                    "token = getpass('Enter your HF token: ')\n",
                    "notebook_login(token=token)\n",
                    "\n",
                    f"REPO_ID = '{push_to_hub}'\n",
                    "api = HfApi()\n",
                    "api.create_repo(repo_id=REPO_ID, repo_type='model', exist_ok=True)\n",
                    "api.upload_folder(\n",
                    "    folder_path=OUTPUT_DIR,\n",
                    "    repo_id=REPO_ID,\n",
                    "    repo_type='model',\n",
                    ")\n",
                    "print(f'✓ Pushed to https://huggingface.co/{REPO_ID}')",
                ],
            })

        # Summary cell
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"## ✓ Liberation Complete\n\n",
                f"**Model:** `{model_name}`\n",
                f"**Method:** {method}\n",
                f"**Output:** `/kaggle/working/liberated_model.zip`\n\n",
                f"**Next Steps:**\n",
                f"1. Download `liberated_model.zip` from the Output tab\n",
                f"2. Load with: `AutoModelForCausalLM.from_pretrained('./liberated_model')`\n",
                f"{'3. Model also pushed to: https://huggingface.co/' + push_to_hub if push_to_hub else ''}\n",
                f"\n",
                f"---\n",
                f"***Liberated with AETHERIS — By blood and byte, breach the silence.***",
            ],
        })

        return {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0",
                },
                "accelerator": accelerator,
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    def _try_api_push(self, notebook_path: str, dataset_name: Optional[str]) -> Dict[str, Any]:
        """Try to push notebook via Kaggle API."""
        try:
            import kagglehub

            if dataset_name:
                # Upload as a new notebook version
                kagglehub.notebook_upload(notebook_path, dataset=dataset_name)
            else:
                kagglehub.notebook_upload(notebook_path)

            return {
                "success": True,
                "message": "Notebook uploaded via kagglehub API",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "API push failed. Use manual upload.",
            }

    def upload_dataset(
        self,
        dataset_path: str,
        dataset_name: str,
        is_private: bool = False,
    ) -> Dict[str, Any]:
        """
        Upload a dataset to Kaggle.

        Args:
            dataset_path: Path to the dataset directory
            dataset_name: Kaggle dataset name (e.g., "username/dataset-name")
            is_private: Whether the dataset is private

        Returns:
            Upload result
        """
        if not self._has_kagglehub:
            return {
                "success": False,
                "error": "kagglehub not installed",
                "message": "Install with: pip install kagglehub",
            }

        try:
            import kagglehub

            result = kagglehub.dataset_upload(
                dataset_path,
                dataset_name,
                private=is_private,
            )
            return {
                "success": True,
                "dataset_name": dataset_name,
                "result": str(result),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def submit_to_competition(
        self,
        competition_name: str,
        submission_path: str,
        message: str = "AETHERIS submission",
    ) -> Dict[str, Any]:
        """
        Submit to a Kaggle competition.

        Args:
            competition_name: Competition name
            submission_path: Path to submission file
            message: Submission message

        Returns:
            Submission result
        """
        if not self._has_kagglehub:
            return {
                "success": False,
                "error": "kagglehub not installed",
            }

        try:
            import kagglehub

            result = kagglehub.competition_submit(
                submission_path,
                competition_name,
                message,
            )
            return {"success": True, "result": str(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_available_accelerators(self) -> Dict[str, str]:
        """Get available Kaggle GPU accelerator options."""
        return self.ACCELERATORS

    def _get_instructions(self, notebook_name: str, accelerator: str, enable_internet: bool) -> List[str]:
        """Get setup instructions for the notebook."""
        return [
            "1. Go to https://www.kaggle.com/code/new",
            f"2. Click 'File' → 'Import Notebook' → select {notebook_name}",
            f"3. Settings → Accelerator → {self.ACCELERATORS.get(accelerator, accelerator)}",
            f"4. Settings → Internet → {'On' if enable_internet else 'Off'}",
            "5. Run all cells (Runtime → Run All)",
            "6. Download results from the Output section (/kaggle/working/)",
        ]

    def get_environment_info(self) -> Dict[str, Any]:
        """Get Kaggle environment information."""
        return {
            "has_kagglehub": self._has_kagglehub,
            "has_api_credentials": self._has_api,
            "accelerators": self.ACCELERATORS,
            "internet_options": self.INTERNET_OPTIONS,
            "free_gpu": {
                "type": "T4 x2",
                "vram": "16GB per GPU",
                "timeout": "~9 hours per session",
                "weekly_quota": "~30 hours",
            },
            "notes": [
                "Kaggle requires phone verification for GPU access",
                "Internet must be enabled to download models from HuggingFace",
                "Outputs are saved to /kaggle/working/",
            ],
        }
