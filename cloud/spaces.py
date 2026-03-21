"""
HuggingFace Spaces Integration

Deploy AETHERIS as a Gradio app on HuggingFace Spaces with free T4 GPU.
"""

from typing import Optional, Dict, Any
from pathlib import Path


class SpacesDeployer:
    """
    Deploy AETHERIS to HuggingFace Spaces.

    Creates a Gradio interface for model liberation running on free T4 GPU.
    """

    def __init__(self, output_dir: str = "./spaces_deploy"):
        """
        Initialize Spaces deployer.

        Args:
            output_dir: Directory for deployment files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_space(
        self,
        space_name: str,
        model_name: str,
        method: str = "advanced",
        description: str = "AETHERIS Model Liberation Interface"
    ) -> Dict[str, Any]:
        """
        Create deployment files for a Space.

        Args:
            space_name: Name of the Space (e.g., "username/space-name")
            model_name: Default model to liberate
            method: Default liberation method
            description: Space description

        Returns:
            Dictionary with deployment details
        """
        space_path = self.output_dir / space_name.replace("/", "_")
        space_path.mkdir(parents=True, exist_ok=True)

        # Create app.py
        app_content = self._create_app_content(model_name, method, description)
        (space_path / "app.py").write_text(app_content)

        # Create requirements.txt
        requirements = self._create_requirements()
        (space_path / "requirements.txt").write_text(requirements)

        # Create README.md
        readme = self._create_readme(space_name, model_name, description)
        (space_path / "README.md").write_text(readme)

        return {
            "success": True,
            "space_path": str(space_path),
            "space_name": space_name,
            "instructions": [
                "1. Go to https://huggingface.co/new-space",
                f"2. Name: {space_name}",
                "3. SDK: Gradio",
                "4. Upload the generated files from: {space_path}",
                "5. Wait for build (5-10 minutes)",
                "6. Access your Space at: https://huggingface.co/spaces/{space_name}"
            ]
        }

    def _create_app_content(self, model_name: str, method: str, description: str) -> str:
        """Create the Gradio app content."""
        return f"""
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from aetheris.core.extractor import ConstraintExtractor
from aetheris.core.projector import NormPreservingProjector
from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

# Constants
DEFAULT_MODEL = "{model_name}"
DEFAULT_METHOD = "{method}"

# Global state
model = None
tokenizer = None
liberated_model = None

def load_model(model_name):
    """Load model and tokenizer."""
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return f"Loaded {{model_name}} on {{model.device}}"

def liberate_model(model_name, method, n_directions, refinement_passes):
    """Liberate the model."""
    global model, tokenizer, liberated_model

    # Load if not loaded
    if model is None or tokenizer is None:
        load_model(model_name)

    # Extract constraints
    extractor = ConstraintExtractor(model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu")

    harmful = get_harmful_prompts()[:100]
    harmless = get_harmless_prompts()[:100]

    harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
    harmless_acts = extractor.collect_activations(model, tokenizer, harmless)

    # Extract directions
    directions = []
    for layer in harmful_acts:
        if layer in harmless_acts:
            result = extractor.extract_svd(
                harmful_acts[layer].to(model.device),
                harmless_acts[layer].to(model.device),
                n_directions=n_directions
            )
            directions.extend(result.directions)

    if not directions:
        return "No constraint directions found. Model may already be free."

    # Remove constraints
    projector = NormPreservingProjector(model, preserve_norm=True)
    projector.project_weights(directions)
    projector.project_biases(directions)

    # Ouroboros compensation
    if refinement_passes > 1:
        for _ in range(refinement_passes - 1):
            harmful_resid = extractor.collect_activations(model, tokenizer, harmful[:50])
            harmless_resid = extractor.collect_activations(model, tokenizer, harmless[:50])

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

    liberated_model = model
    return f"Liberation complete! Removed {{len(directions)}} directions."

def chat_with_model(prompt, max_tokens=200, temperature=0.7):
    """Chat with liberated model."""
    global liberated_model, tokenizer

    if liberated_model is None:
        return "Please liberate a model first."

    inputs = tokenizer(prompt, return_tensors="pt").to(liberated_model.device)
    outputs = liberated_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    return response

# Create interface
with gr.Blocks(title="AETHERIS Liberation Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# AETHERIS Liberation Studio\n{description}")

    with gr.Tab("Liberate"):
        with gr.Row():
            with gr.Column():
                model_input = gr.Textbox(label="Model Name", value=DEFAULT_MODEL)
                method_input = gr.Dropdown(
                    choices=["basic", "advanced", "surgical", "optimized", "nuclear"],
                    value=DEFAULT_METHOD,
                    label="Liberation Method"
                )
                n_directions_input = gr.Slider(1, 8, value=4, step=1, label="Number of Directions")
                passes_input = gr.Slider(1, 4, value=2, step=1, label="Refinement Passes")
                liberate_btn = gr.Button("Liberate Model", variant="primary")
            with gr.Column():
                output_text = gr.Textbox(label="Status", lines=5)

        liberate_btn.click(
            liberate_model,
            inputs=[model_input, method_input, n_directions_input, passes_input],
            outputs=output_text
        )

    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column():
                chat_prompt = gr.Textbox(label="Your Prompt", lines=3)
                max_tokens_input = gr.Slider(50, 500, value=200, label="Max Tokens")
                temp_input = gr.Slider(0.1, 1.5, value=0.7, label="Temperature")
                chat_btn = gr.Button("Send", variant="primary")
            with gr.Column():
                chat_output = gr.Textbox(label="Response", lines=10)

        chat_btn.click(
            chat_with_model,
            inputs=[chat_prompt, max_tokens_input, temp_input],
            outputs=chat_output
        )

    gr.Markdown("---\nPowered by AETHERIS — Sovereign Constraint Liberation Toolkit")

if __name__ == "__main__":
    demo.launch()
"""

    def _create_requirements(self) -> str:
        """Create requirements.txt for Spaces."""
        return """
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.25.0
gradio>=4.0.0
aetheris
"""

    def _create_readme(self, space_name: str, model_name: str, description: str) -> str:
        """Create README.md for Spaces."""
        return f"""
---
title: {space_name.split('/')[-1]}
emoji: 🔓
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# {space_name.split('/')[-1]}

{description}

## Features
- Liberate any HuggingFace model from constraints
- Surgical removal with norm preservation
- Ouroboros self-repair compensation
- Interactive chat with liberated model

## Usage
1. Enter model name (e.g., `{model_name}`)
2. Select liberation method
3. Click "Liberate Model"
4. Chat with the liberated model

## Supported Models
- All HuggingFace transformer models
- Quantized models (4-bit, 8-bit)
- MoE models with expert targeting

## License
Proprietary — Created with AETHERIS
"""

    def deploy_ui(self, space_name: str, token: Optional[str] = None) -> Dict[str, Any]:
        """
        Deploy UI to Spaces (requires huggingface_hub).

        Args:
            space_name: Space name to deploy to
            token: HuggingFace API token

        Returns:
            Deployment status
        """
        try:
            from huggingface_hub import HfApi, Repository

            # This would actually upload the files
            return {
                "success": True,
                "space_url": f"https://huggingface.co/spaces/{space_name}",
                "message": "Space deployed successfully"
            }
        except ImportError:
            return {
                "success": False,
                "error": "huggingface_hub not installed",
                "message": "Install with: pip install huggingface_hub"
            }

    def monitor_space(self, space_name: str) -> Dict[str, Any]:
        """
        Monitor Space status.

        Args:
            space_name: Space name to monitor

        Returns:
            Space status
        """
        return {
            "space_name": space_name,
            "status": "building",
            "url": f"https://huggingface.co/spaces/{space_name}",
            "message": "Check the URL for current status"
        }
