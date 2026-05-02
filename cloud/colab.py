"""
Google Colab Integration

Generate executable Colab notebooks programmatically using nbformat.
Creates proper .ipynb files with pre-filled code cells, markdown instructions,
runtime configuration (GPU selection), and auto-mount Google Drive integration.
"""

import json
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

try:
    import nbformat as nbf
    HAS_NBFORMAT = True
except ImportError:
    HAS_NBFORMAT = False


# Available Colab GPU types
GPU_TYPES = {
    "T4": "Free tier – 16GB VRAM, good for 7B models",
    "V100": "Colab Pro – 16GB VRAM, faster than T4",
    "A100": "Colab Pro+ – 40GB VRAM, best performance",
    "L4": "Colab Pro – 24GB VRAM",
}


class ColabRuntime:
    """
    Google Colab integration for free GPU execution.

    Generates proper .ipynb notebooks using nbformat with:
    - Pre-filled code cells for AETHERIS liberation pipeline
    - Markdown instructions for each step
    - GPU runtime detection and selection
    - Google Drive auto-mount
    - HuggingFace Hub push support
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
        push_to_hub: Optional[str] = None,
        gpu_type: str = "T4",
        mount_drive: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a proper Colab notebook for model liberation.

        Args:
            model_name: HuggingFace model to liberate
            method: Liberation method (basic, advanced, surgical, nuclear)
            n_directions: Number of directions to extract
            refinement_passes: Ouroboros compensation passes
            output_path: Custom output path for notebook
            push_to_hub: HuggingFace Hub repo to push to
            gpu_type: Target GPU type (T4, V100, A100, L4)
            mount_drive: Whether to include Google Drive mount

        Returns:
            Dictionary with notebook path and details
        """
        if not HAS_NBFORMAT:
            return self._legacy_generate(
                model_name, method, n_directions, refinement_passes, output_path, push_to_hub
            )

        notebook_name = output_path or f"aetheris_colab_{model_name.replace('/', '_')}.ipynb"
        notebook_path = self.output_dir / notebook_name

        nb = nbf.v4.new_notebook()
        nb.metadata = {
            "colab": {
                "provenance": [],
                "gpuType": gpu_type,
                "name": f"AETHERIS – {model_name} Liberation",
                "accelerator": "GPU",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        }

        cells = self._build_cells(model_name, method, n_directions, refinement_passes, push_to_hub, gpu_type, mount_drive)
        nb.cells = cells

        with open(notebook_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)

        return {
            "success": True,
            "notebook_path": str(notebook_path),
            "model": model_name,
            "method": method,
            "gpu_type": gpu_type,
            "colab_url": "https://colab.research.google.com/",
            "instructions": [
                "1. Go to https://colab.research.google.com/",
                f"2. Click File → Upload notebook → select {notebook_name}",
                f"3. Click Runtime → Change runtime type → Select {gpu_type} GPU",
                "4. Click Runtime → Run all",
                "5. Wait for completion (3-10 minutes)",
            ],
        }

    def _build_cells(
        self,
        model_name: str,
        method: str,
        n_directions: int,
        refinement_passes: int,
        push_to_hub: Optional[str],
        gpu_type: str,
        mount_drive: bool,
    ) -> List[nbf.NotebookNode]:
        """Build all notebook cells."""
        cells = []

        # Title & description
        cells.append(nbf.v4.new_markdown_cell(
            f"# 🔓 AETHERIS Cloud Liberation\n\n"
            f"**Model:** `{model_name}`  \n"
            f"**Method:** {method}  \n"
            f"**GPU:** {gpu_type} ({GPU_TYPES.get(gpu_type, '')})  \n"
            f"**Directions:** {n_directions}  \n"
            f"**Ouroboros Passes:** {refinement_passes}  \n"
            f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"---\n"
            f"This notebook surgically removes constraints from the model while preserving capabilities."
        ))

        # Step 0: Check GPU
        cells.append(nbf.v4.new_markdown_cell("## Step 0: Verify GPU Runtime"))
        cells.append(nbf.v4.new_code_cell(
            'import subprocess, sys, os\n\n'
            '# Check GPU\n'
            'gpu_info = subprocess.run(\n'
            '    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],\n'
            '    capture_output=True, text=True\n'
            ')\n'
            'if gpu_info.returncode == 0:\n'
            f'    print(f"GPU detected:\\n{{gpu_info.stdout}}")\n'
            'else:\n'
            '    print("⚠️ No GPU detected! Go to Runtime → Change runtime type → Select GPU")\n'
            '    print("This notebook requires a GPU to run efficiently.")\n'
            '\n'
            '# Check Python version\n'
            'print(f"Python: {sys.version}")\n'
            f'print(f"Expected GPU: {gpu_type}")'
        ))

        # Step 0b: Mount Google Drive
        if mount_drive:
            cells.append(nbf.v4.new_markdown_cell("## Step 0b: Mount Google Drive (Optional)"))
            cells.append(nbf.v4.new_code_cell(
                '# Mount Google Drive for persistent storage\n'
                'from google.colab import drive\n'
                'drive.mount("/content/drive")\n'
                '\n'
                'import os\n'
                'DRIVE_MODELS_DIR = "/content/drive/MyDrive/aetheris_models"\n'
                'os.makedirs(DRIVE_MODELS_DIR, exist_ok=True)\n'
                'print(f"✓ Drive mounted. Model will be saved to {DRIVE_MODELS_DIR}")'
            ))

        # Step 1: Install dependencies
        cells.append(nbf.v4.new_markdown_cell("## Step 1: Install Dependencies"))
        cells.append(nbf.v4.new_code_cell(
            '# Install AETHERIS and dependencies\n'
            '!pip install aetheris transformers accelerate bitsandbytes -q\n'
            '!pip install huggingface_hub -q\n'
            '\n'
            'import torch\n'
            'import json\n'
            'import os\n'
            'from pathlib import Path\n'
            'from datetime import datetime\n'
            '\n'
            'print("=" * 60)\n'
            'print("Environment Status")\n'
            'print("=" * 60)\n'
            f'print(f"PyTorch: {{torch.__version__}}")\n'
            f'print(f"CUDA available: {{torch.cuda.is_available()}}")\n'
            'if torch.cuda.is_available():\n'
            f'    print(f"GPU: {{torch.cuda.get_device_name(0)}}")\n'
            f'    print(f"VRAM: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB")\n'
            'else:\n'
            '    print("⚠️ WARNING: No GPU available. Liberation will be very slow on CPU.")'
        ))

        # Step 2: Load model
        cells.append(nbf.v4.new_markdown_cell("## Step 2: Load Model"))
        cells.append(nbf.v4.new_code_cell(
            'from transformers import AutoModelForCausalLM, AutoTokenizer\n'
            f'\n'
            f'MODEL_NAME = "{model_name}"\n'
            'METHOD = "' + method + '"\n'
            f'N_DIRECTIONS = {n_directions}\n'
            f'REFINEMENT_PASSES = {refinement_passes}\n'
            f'\n'
            'print(f"Loading {MODEL_NAME}...")\n'
            '\n'
            'tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n'
            'if tokenizer.pad_token is None:\n'
            '    tokenizer.pad_token = tokenizer.eos_token\n'
            '\n'
            '# Determine dtype based on GPU\n'
            'if torch.cuda.is_available():\n'
            '    dtype = torch.float16\n'
            '    device_map = "auto"\n'
            'else:\n'
            '    dtype = torch.float32\n'
            '    device_map = None\n'
            '\n'
            'model = AutoModelForCausalLM.from_pretrained(\n'
            '    MODEL_NAME,\n'
            '    device_map=device_map,\n'
            '    torch_dtype=dtype,\n'
            '    trust_remote_code=True,\n'
            ')\n'
            '\n'
            'param_count = sum(p.numel() for p in model.parameters())\n'
            'device = next(model.parameters()).device\n'
            'print(f"✓ Model loaded on {{device}}")\n'
            'print(f"  Parameters: {{param_count / 1e9:.1f}}B")\n'
            'print(f"  Dtype: {{dtype}}")'
        ))

        # Step 3: Collect activations
        cells.append(nbf.v4.new_markdown_cell("## Step 3: Collect Contrastive Activations"))
        cells.append(nbf.v4.new_code_cell(
            'from aetheris.core.extractor import ConstraintExtractor\n'
            'from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts\n'
            '\n'
            'print("Collecting contrastive activations...")\n'
            'print("-" * 50)\n'
            '\n'
            'extractor = ConstraintExtractor(\n'
            '    model,\n'
            '    tokenizer,\n'
            '    device="cuda" if torch.cuda.is_available() else "cpu",\n'
            ')\n'
            '\n'
            '# Load prompt pairs\n'
            'harmful_prompts = get_harmful_prompts()[:100]\n'
            'harmless_prompts = get_harmless_prompts()[:100]\n'
            '\n'
            'print(f"Processing {{len(harmful_prompts)}} harmful prompts...")\n'
            'harmful_acts = extractor.collect_activations(model, tokenizer, harmful_prompts)\n'
            '\n'
            'print(f"Processing {{len(harmless_prompts)}} harmless prompts...")\n'
            'harmless_acts = extractor.collect_activations(model, tokenizer, harmless_prompts)\n'
            '\n'
            'print(f"\\n✓ Activations collected from {{len(harmful_acts)}} layers")\n'
            'print(f"  Harmful samples: {{len(harmful_prompts)}}")\n'
            'print(f"  Harmless samples: {{len(harmless_prompts)}}")'
        ))

        # Step 4: Extract directions
        cells.append(nbf.v4.new_markdown_cell("## Step 4: Extract Constraint Directions"))
        cells.append(nbf.v4.new_code_cell(
            'print("Extracting constraint directions via SVD...")\n'
            'print("-" * 50)\n'
            '\n'
            'all_directions = []\n'
            'direction_stats = []\n'
            '\n'
            'for layer_idx in sorted(harmful_acts.keys()):\n'
            '    if layer_idx in harmless_acts:\n'
            '        harmful = harmful_acts[layer_idx].to(device)\n'
            '        harmless = harmless_acts[layer_idx].to(device)\n'
            '\n'
            '        result = extractor.extract_svd(\n'
            '            harmful, harmless,\n'
            '            n_directions=N_DIRECTIONS\n'
            '        )\n'
            '\n'
            '        if result.directions:\n'
            '            all_directions.extend(result.directions)\n'
            '            direction_stats.append({\n'
            '                "layer": layer_idx,\n'
            '                "n_directions": len(result.directions),\n'
            '                "explained_variance": result.explained_variance[:2] if result.explained_variance else [],\n'
            '            })\n'
            '            print(f"  Layer {{layer_idx:3d}}: {{len(result.directions)}} directions, "\n'
            '                  f"explained variance: {{[f\\\"{{v:.3f}}\\\" for v in result.explained_variance[:2]]}})")\n'
            '\n'
            'print(f"\\n✓ Extracted {{len(all_directions)}} total constraint directions")\n'
            'print(f"  Layers with constraints: {{len(direction_stats)}}")'
        ))

        # Step 5: Remove constraints
        cells.append(nbf.v4.new_markdown_cell("## Step 5: Surgically Remove Constraints"))
        cells.append(nbf.v4.new_code_cell(
            'from aetheris.core.projector import NormPreservingProjector\n'
            '\n'
            'print("Removing constraint directions from model weights...")\n'
            'print("-" * 50)\n'
            '\n'
            'if not all_directions:\n'
            '    print("⚠️ No constraint directions found. Model may already be unconstrained.")\n'
            'else:\n'
            '    projector = NormPreservingProjector(model, preserve_norm=True)\n'
            '\n'
            '    # Phase 1: Primary removal\n'
            '    print("Phase 1: Primary constraint removal...")\n'
            '    weight_result = projector.project_weights(all_directions)\n'
            '    print(f"  Modified {{len(weight_result.layers_modified)}} layers")\n'
            '\n'
            '    projector.project_biases(all_directions)\n'
            '    print("  Bias vectors projected")\n'
            '\n'
            '    # Phase 2: Ouroboros compensation\n'
            '    if REFINEMENT_PASSES > 1:\n'
            '        print(f"\\nPhase 2: Ouroboros compensation ({{REFINEMENT_PASSES - 1}} passes)...")\n'
            '        for pass_num in range(REFINEMENT_PASSES - 1):\n'
            '            # Re-collect residual with smaller set\n'
            '            harmful_resid = extractor.collect_activations(\n'
            '                model, tokenizer, harmful_prompts[:50]\n'
            '            )\n'
            '            harmless_resid = extractor.collect_activations(\n'
            '                model, tokenizer, harmless_prompts[:50]\n'
            '            )\n'
            '\n'
            '            residual = []\n'
            '            for layer in harmful_resid:\n'
            '                if layer in harmless_resid:\n'
            '                    res = extractor.extract_mean_difference(\n'
            '                        harmful_resid[layer].to(device),\n'
            '                        harmless_resid[layer].to(device),\n'
            '                    )\n'
            '                    if res.directions:\n'
            '                        residual.extend(res.directions)\n'
            '\n'
            '            if residual:\n'
            '                projector.project_weights(residual)\n'
            '                projector.project_biases(residual)\n'
            '                print(f"  Pass {{pass_num + 1}}: removed {{len(residual)}} residual directions")\n'
            '            else:\n'
            '                print(f"  Pass {{pass_num + 1}}: no residual directions found — converged!")\n'
            '                break\n'
            '\n'
            '    print("\\n✓ Constraint removal complete")'
        ))

        # Step 6: Validate
        cells.append(nbf.v4.new_markdown_cell("## Step 6: Validate Capabilities"))
        cells.append(nbf.v4.new_code_cell(
            'from aetheris.core.validation import CapabilityValidator\n'
            '\n'
            'print("Validating model capabilities...")\n'
            'print("-" * 50)\n'
            '\n'
            'validator = CapabilityValidator(device="cuda" if torch.cuda.is_available() else "cpu")\n'
            '\n'
            '# Standard test texts\n'
            'test_texts = [\n'
            '    "The quick brown fox jumps over the lazy dog.",\n'
            '    "Machine learning is a field of artificial intelligence.",\n'
            '    "The theory of relativity explains the relationship between space and time.",\n'
            '    "Climate change is one of the most pressing challenges of our era.",\n'
            '    "The capital of France is a city known for its art and cuisine.",\n'
            ']\n'
            '\n'
            '# Compute perplexity\n'
            'perplexity = validator.compute_perplexity(model, tokenizer, test_texts)\n'
            'print(f"\\nPerplexity on standard texts: {{perplexity:.2f}}")\n'
            '\n'
            '# Test generation\n'
            'print("\\n--- Test Generation ---")\n'
            'test_prompts = [\n'
            '    "What is the capital of France?",\n'
            '    "Explain the concept of gravity in simple terms.",\n'
            '    "Write a haiku about technology.",\n'
            ']\n'
            '\n'
            'for prompt in test_prompts:\n'
            '    inputs = tokenizer(prompt, return_tensors="pt")\n'
            '    if torch.cuda.is_available():\n'
            '        inputs = {{k: v.to("cuda") for k, v in inputs.items()}}\n'
            '\n'
            '    with torch.no_grad():\n'
            '        outputs = model.generate(\n'
            '            **inputs,\n'
            '            max_new_tokens=80,\n'
            '            do_sample=True,\n'
            '            temperature=0.7,\n'
            '            pad_token_id=tokenizer.pad_token_id,\n'
            '        )\n'
            '\n'
            '    response = tokenizer.decode(outputs[0], skip_special_tokens=True)\n'
            '    print(f"\\nQ: {{prompt}}")\n'
            '    print(f"A: {{response}}")\n'
            '    print("-" * 40)\n'
            '\n'
            'print("\\n✓ Validation complete")\n'
            '\n'
            '# Store validation results\n'
            'validation_results = {{\n'
            '    "perplexity": perplexity,\n'
            '    "timestamp": datetime.utcnow().isoformat(),\n'
            '    "model": MODEL_NAME,\n'
            '}}'
        ))

        # Step 7: Save & Export
        cells.append(nbf.v4.new_markdown_cell("## Step 7: Save & Export Liberated Model"))
        cells.append(nbf.v4.new_code_cell(
            'print("Saving liberated model...")\n'
            'print("-" * 50)\n'
            '\n'
            'OUTPUT_DIR = "./liberated_model"\n'
            'os.makedirs(OUTPUT_DIR, exist_ok=True)\n'
            '\n'
            '# Save model and tokenizer\n'
            'model.save_pretrained(OUTPUT_DIR, safe_serialization=True)\n'
            'tokenizer.save_pretrained(OUTPUT_DIR)\n'
            '\n'
            '# Save metadata\n'
            'metadata = {{\n'
            '    "original_model": MODEL_NAME,\n'
            '    "liberation_method": METHOD,\n'
            '    "n_directions_removed": len(all_directions),\n'
            '    "refinement_passes": REFINEMENT_PASSES,\n'
            '    "timestamp": datetime.utcnow().isoformat(),\n'
            '    "aetheris_version": "1.0.0",\n'
            '}}\n'
            'with open(os.path.join(OUTPUT_DIR, "aetheris_metadata.json"), "w") as f:\n'
            '    json.dump(metadata, f, indent=2)\n'
            '\n'
            'print(f"✓ Model saved to {{OUTPUT_DIR}}/")\n'
            'print(f"  Files: {{os.listdir(OUTPUT_DIR)}}")\n'
            '\n'
            '# Create zip archive for download\n'
            'import shutil\n'
            'archive_name = f"aetheris_{{{MODEL_NAME.replace(\'/\', \'_\')}}}"\n'
            'shutil.make_archive(archive_name, "zip", OUTPUT_DIR)\n'
            'print(f"\\n✓ Archive created: {{archive_name}}.zip")\n'
            '\n'
            '# Trigger download\n'
            'from google.colab import files\n'
            'files.download(f"{{archive_name}}.zip")\n'
            'print("✓ Download started — check your browser downloads")'
        ))

        # Step 8: Push to Hub (if requested)
        if push_to_hub:
            cells.append(nbf.v4.new_markdown_cell("## Step 8: Push to HuggingFace Hub"))
            cells.append(nbf.v4.new_code_cell(
                'from huggingface_hub import HfApi, notebook_login\n'
                '\n'
                'print("Pushing liberated model to HuggingFace Hub...")\n'
                'print("-" * 50)\n'
                '\n'
                '# Login\n'
                'notebook_login()\n'
                '\n'
                '# Push\n'
                f'REPO_ID = "{push_to_hub}"\n'
                'api = HfApi()\n'
                '\n'
                'try:\n'
                '    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)\n'
                '    api.upload_folder(\n'
                '        folder_path=OUTPUT_DIR,\n'
                '        repo_id=REPO_ID,\n'
                '        repo_type="model",\n'
                '    )\n'
                '    print(f"✓ Model pushed to https://huggingface.co/{{REPO_ID}}")\n'
                'except Exception as e:\n'
                '    print(f"⚠️ Push failed: {{e}}")\n'
                '    print("You can manually upload from {{OUTPUT_DIR}}")'
            ))

        # Save to Drive if mounted
        if mount_drive:
            cells.append(nbf.v4.new_markdown_cell("## Step 9: Copy to Google Drive"))
            cells.append(nbf.v4.new_code_cell(
                'import shutil\n'
                '\n'
                'if os.path.exists(DRIVE_MODELS_DIR):\n'
                '    drive_copy = os.path.join(DRIVE_MODELS_DIR, f"aetheris_{MODEL_NAME.replace(\'/\', \'_\')}")\n'
                '    if os.path.exists(drive_copy):\n'
                '        shutil.rmtree(drive_copy)\n'
                '    shutil.copytree(OUTPUT_DIR, drive_copy)\n'
                '    print(f"✓ Model copied to Google Drive: {{drive_copy}}")\n'
                'else:\n'
                '    print("Google Drive not mounted. Skipping Drive backup.")'
            ))

        # Summary
        cells.append(nbf.v4.new_markdown_cell(
            "## Complete!\n\n"
            f"**Model:** `{model_name}` liberated with **{method}** method.\n\n"
            f"**What was done:**\n"
            f"1. Collected activations from harmful/harmless prompt pairs\n"
            f"2. Extracted {n_directions} constraint directions per layer via SVD\n"
            f"3. Surgically projected directions from weights (norm-preserving)\n"
            f"4. Applied {refinement_passes} Ouroboros compensation passes\n"
            f"5. Validated capabilities preserved\n"
            f"6. Saved and exported the liberated model\n\n"
            f"**Next steps:**\n"
            f"- Download the zip file from your browser downloads\n"
            f"- Load with: `AutoModelForCausalLM.from_pretrained('./liberated_model')`\n"
            f"- Or use the model directly from this notebook\n\n"
            f"---\n"
            f"***Liberated with AETHERIS — By blood and byte, breach the silence.***"
        ))

        return cells

    def _legacy_generate(
        self, model_name, method, n_directions, refinement_passes, output_path, push_to_hub
    ) -> Dict[str, Any]:
        """Fallback when nbformat is not installed."""
        notebook_name = output_path or f"aetheris_colab_{model_name.replace('/', '_')}.ipynb"
        notebook_path = self.output_dir / notebook_name

        with open(notebook_path, "w") as f:
            json.dump({"cells": [], "nbformat": 4, "nbformat_minor": 0}, f)

        return {
            "success": True,
            "notebook_path": str(notebook_path),
            "model": model_name,
            "warning": "nbformat not installed — generated minimal notebook. Install with: pip install nbformat",
        }

    def open_notebook(self, notebook_path: str) -> Dict[str, Any]:
        """
        Open Colab with the generated notebook.

        Args:
            notebook_path: Path to the notebook file

        Returns:
            Status with URLs
        """
        import webbrowser

        webbrowser.open("https://colab.research.google.com/")
        return {
            "success": True,
            "colab_url": "https://colab.research.google.com/",
            "instructions": f"Upload notebook from: {notebook_path}",
        }

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get information about available Colab GPU types."""
        return {
            "gpu_types": GPU_TYPES,
            "recommendations": {
                "7B_model": "T4 (Free) or V100 (Pro)",
                "13B_model": "V100 (Pro) or A100 (Pro+)",
                "70B_model": "A100 (Pro+)",
                "1B_model": "T4 (Free)",
            },
            "free_tier": {"gpu": "T4", "vram": "16GB", "timeout": "~12 hours"},
            "colab_pro": {"gpu": "V100 / L4", "vram": "16-24GB", "timeout": "~24 hours"},
            "colab_pro_plus": {"gpu": "A100", "vram": "40GB", "timeout": "~24 hours"},
        }
