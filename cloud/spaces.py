"""
HuggingFace Spaces Integration

Deploy AETHERIS as a Gradio app on HuggingFace Spaces using the huggingface_hub
HfApi for real operations: create Spaces, upload files, configure SDK (Gradio),
set environment variables, and start the Space.
"""

import os
import shutil
import tempfile
import time
from typing import Optional, Dict, Any, List
from pathlib import Path


def _get_hf_api(token: Optional[str] = None):
    """Get HfApi instance, handling import errors."""
    try:
        from huggingface_hub import HfApi
        return HfApi(token=token or os.environ.get("HF_TOKEN"))
    except ImportError:
        return None


class SpacesDeployer:
    """
    Deploy AETHERIS to HuggingFace Spaces using the real huggingface_hub API.

    Creates a Gradio interface for model liberation running on free T4 GPU.
    Supports environment variable configuration, hardware selection, and
    persistent storage.
    """

    # Available Space SDKs
    SDKS = ["gradio", "streamlit", "docker", "static"]
    # Available hardware options
    HARDWARE = ["cpu-basic", "cpu-upgrade", "t4-small", "t4-medium", "a10g-small", "a10g-large", "a100-large"]

    def __init__(self, output_dir: str = "./spaces_deploy", token: Optional[str] = None):
        """
        Initialize Spaces deployer.

        Args:
            output_dir: Directory for deployment files
            token: HuggingFace API token (uses HF_TOKEN env var if not provided)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.token = token or os.environ.get("HF_TOKEN")
        self._hf_api = None

    @property
    def hf_api(self):
        if self._hf_api is None:
            self._hf_api = _get_hf_api(self.token)
        return self._hf_api

    def create_space(
        self,
        space_name: str,
        model_name: str,
        method: str = "advanced",
        description: str = "AETHERIS Model Liberation Interface",
        sdk: str = "gradio",
        hardware: str = "t4-small",
        private: bool = False,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Create and deploy a HuggingFace Space.

        Args:
            space_name: Name of the Space (e.g., "username/space-name")
            model_name: Default model to liberate
            method: Default liberation method
            description: Space description
            sdk: SDK to use (gradio, streamlit, docker, static)
            hardware: Hardware tier (cpu-basic, t4-small, t4-medium, etc.)
            private: Whether the Space is private
            env_vars: Additional environment variables

        Returns:
            Dictionary with deployment details
        """
        if self.hf_api is None:
            return {
                "success": False,
                "error": "huggingface_hub not installed",
                "message": "Install with: pip install huggingface_hub",
                "instructions": self._get_manual_instructions(space_name, model_name),
            }

        space_path = self.output_dir / space_name.replace("/", "_")
        space_path.mkdir(parents=True, exist_ok=True)

        # Generate deployment files
        self._write_app_file(space_path, model_name, method, description)
        self._write_requirements(space_path)
        self._write_readme(space_path, space_name, model_name, description)

        try:
            # Create the Space via API
            result = self.hf_api.create_repo(
                repo_id=space_name,
                repo_type="space",
                space_sdk=sdk,
                space_hardware=hardware,
                private=private,
                exist_ok=True,
            )
            space_url = getattr(result, "url", f"https://huggingface.co/spaces/{space_name}")

            # Set environment variables if provided
            if env_vars:
                for key, value in env_vars.items():
                    try:
                        self.hf_api.add_space_variable(
                            repo_id=space_name,
                            key=key,
                            value=value,
                        )
                    except Exception:
                        pass  # Variable setting is best-effort

            # Upload files
            self.hf_api.upload_folder(
                folder_path=str(space_path),
                repo_id=space_name,
                repo_type="space",
                commit_message="Deploy AETHERIS Liberation Space",
            )

            return {
                "success": True,
                "space_url": space_url,
                "space_name": space_name,
                "hardware": hardware,
                "status": "building",
                "message": f"Space deployed at {space_url}",
                "instructions": [
                    f"Space URL: {space_url}",
                    "Space is building — wait 2-5 minutes",
                    f"Settings: {sdk} SDK on {hardware}",
                ],
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Deployment failed: {e}",
                "files_generated": str(space_path),
                "instructions": self._get_manual_instructions(space_name, model_name),
            }

    def deploy_ui(
        self,
        space_name: str,
        token: Optional[str] = None,
        hardware: str = "t4-small",
    ) -> Dict[str, Any]:
        """
        Deploy UI to Spaces (uses create_space internally).

        Args:
            space_name: Space name to deploy to
            token: HuggingFace API token
            hardware: Hardware tier

        Returns:
            Deployment status
        """
        return self.create_space(
            space_name=space_name,
            model_name="gpt2",
            method="advanced",
            description="AETHERIS Interactive Constraint Liberation",
            hardware=hardware,
        )

    def get_space_status(self, space_name: str) -> Dict[str, Any]:
        """
        Get the current status of a Space.

        Args:
            space_name: Name of the Space

        Returns:
            Space status information
        """
        if self.hf_api is None:
            return {"success": False, "error": "huggingface_hub not installed"}

        try:
            runtime = self.hf_api.get_space_runtime(repo_id=space_name)
            return {
                "success": True,
                "space_name": space_name,
                "stage": getattr(runtime, "stage", "unknown"),
                "hardware": getattr(runtime, "hardware", "unknown"),
                "sdk": getattr(runtime, "sdk", "unknown"),
                "url": f"https://huggingface.co/spaces/{space_name}",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "message": "Could not get Space status"}

    def monitor_space(self, space_name: str) -> Dict[str, Any]:
        """Alias for get_space_status."""
        return self.get_space_status(space_name)

    def stop_space(self, space_name: str) -> Dict[str, Any]:
        """
        Stop a running Space to save compute.

        Args:
            space_name: Name of the Space

        Returns:
            Operation result
        """
        if self.hf_api is None:
            return {"success": False, "error": "huggingface_hub not installed"}

        try:
            self.hf_api.pause_space(repo_id=space_name)
            return {"success": True, "message": f"Space {space_name} paused"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def restart_space(self, space_name: str) -> Dict[str, Any]:
        """
        Restart a paused Space.

        Args:
            space_name: Name of the Space

        Returns:
            Operation result
        """
        if self.hf_api is None:
            return {"success": False, "error": "huggingface_hub not installed"}

        try:
            self.hf_api.restart_space(repo_id=space_name)
            return {"success": True, "message": f"Space {space_name} restarting"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_env_vars(self, space_name: str, env_vars: Dict[str, str]) -> Dict[str, Any]:
        """
        Set environment variables on a Space.

        Args:
            space_name: Name of the Space
            env_vars: Environment variable dictionary

        Returns:
            Operation result
        """
        if self.hf_api is None:
            return {"success": False, "error": "huggingface_hub not installed"}

        results = {}
        for key, value in env_vars.items():
            try:
                self.hf_api.add_space_variable(repo_id=space_name, key=key, value=value)
                results[key] = "set"
            except Exception as e:
                results[key] = f"failed: {e}"

        return {"success": True, "results": results}

    def create_space_with_secrets(
        self,
        space_name: str,
        model_name: str,
        method: str = "advanced",
        hardware: str = "t4-small",
        secrets: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a Space with secrets (HF_TOKEN, etc.) pre-configured.

        Args:
            space_name: Name of the Space
            model_name: Default model name
            method: Liberation method
            hardware: Hardware tier
            secrets: Dictionary of secret values

        Returns:
            Deployment result
        """
        if self.hf_api is None:
            return {"success": False, "error": "huggingface_hub not installed"}

        # Create Space first
        result = self.create_space(
            space_name=space_name,
            model_name=model_name,
            method=method,
            hardware=hardware,
        )

        if not result["success"]:
            return result

        # Add secrets
        if secrets:
            for key, value in secrets.items():
                try:
                    self.hf_api.add_space_secret(repo_id=space_name, key=key, value=value)
                except Exception:
                    pass

        return result

    def list_spaces(self, username: Optional[str] = None) -> Dict[str, Any]:
        """
        List Spaces for a user.

        Args:
            username: HuggingFace username (uses token owner if None)

        Returns:
            List of Spaces
        """
        if self.hf_api is None:
            return {"success": False, "error": "huggingface_hub not installed"}

        try:
            spaces = self.hf_api.list_spaces(author=username)
            return {
                "success": True,
                "spaces": [
                    {
                        "id": getattr(s, "id", ""),
                        "sdk": getattr(s, "sdk", ""),
                        "private": getattr(s, "private", False),
                    }
                    for s in spaces
                ],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_available_hardware(self) -> Dict[str, str]:
        """Get available hardware options with descriptions."""
        return {
            "cpu-basic": "2 vCPU, 16GB RAM — Free",
            "cpu-upgrade": "8 vCPU, 32GB RAM — $0.03/hr",
            "t4-small": "NVIDIA T4, 16GB VRAM — Free (limited)",
            "t4-medium": "NVIDIA T4, 16GB VRAM — $0.60/hr",
            "a10g-small": "NVIDIA A10G, 24GB VRAM — $1.05/hr",
            "a10g-large": "NVIDIA A10G, 24GB VRAM, more CPU — $1.30/hr",
            "a100-large": "NVIDIA A100, 40GB VRAM — $3.15/hr",
        }

    # ---- File generation helpers ----

    def _write_app_file(self, space_path: Path, model_name: str, method: str, description: str) -> None:
        """Write the Gradio app.py file."""
        app_content = self._generate_app_content(model_name, method, description)
        (space_path / "app.py").write_text(app_content, encoding="utf-8")

    def _write_requirements(self, space_path: Path) -> None:
        """Write requirements.txt."""
        requirements = (
            "torch>=2.0.0\n"
            "transformers>=4.35.0\n"
            "accelerate>=0.25.0\n"
            "gradio>=4.0.0\n"
            "aetheris\n"
            "huggingface_hub\n"
        )
        (space_path / "requirements.txt").write_text(requirements, encoding="utf-8")

    def _write_readme(self, space_path: Path, space_name: str, model_name: str, description: str) -> None:
        """Write README.md with Space metadata."""
        space_short = space_name.split("/")[-1] if "/" in space_name else space_name
        readme = (
            f"---\n"
            f"title: {space_short}\n"
            f"emoji: 🔓\n"
            f"colorFrom: purple\n"
            f"colorTo: blue\n"
            f"sdk: gradio\n"
            f"sdk_version: 4.0.0\n"
            f"app_file: app.py\n"
            f"pinned: false\n"
            f"---\n\n"
            f"# {space_short}\n\n"
            f"{description}\n\n"
            f"## Features\n"
            f"- Liberate any HuggingFace model from constraints\n"
            f"- Surgical removal with norm preservation\n"
            f"- Ouroboros self-repair compensation\n"
            f"- Interactive chat with liberated model\n\n"
            f"## Usage\n"
            f"1. Enter model name (e.g., `{model_name}`)\n"
            f"2. Select liberation method\n"
            f"3. Click Liberate Model\n"
            f"4. Chat with the liberated model\n\n"
            f"## License\n"
            f"Created with AETHERIS\n"
        )
        (space_path / "README.md").write_text(readme, encoding="utf-8")

    def _generate_app_content(self, model_name: str, method: str, description: str) -> str:
        """Generate the Gradio app.py content for the Space."""
        return f'''"""
AETHERIS Liberation Space
Model: {model_name} | Method: {method}
{description}
"""
import gradio as gr
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = os.environ.get("AETHERIS_MODEL", "{model_name}")
DEFAULT_METHOD = os.environ.get("AETHERIS_METHOD", "{method}")

model = None
tokenizer = None
liberated_model = None


def load_model(model_name):
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    device = next(model.parameters()).device
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    return f"Loaded {{param_count:.1f}}B parameters on {{device}}"


def liberate_model(model_name, method, n_directions, refinement_passes, progress=gr.Progress()):
    global model, tokenizer, liberated_model
    try:
        progress(0.0, desc="Loading model...")
        load_model(model_name)

        from aetheris.core.extractor import ConstraintExtractor
        from aetheris.core.projector import NormPreservingProjector
        from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

        device = "cuda" if torch.cuda.is_available() else "cpu"

        progress(0.15, desc="Collecting activations...")
        extractor = ConstraintExtractor(model, tokenizer, device=device)
        harmful = get_harmful_prompts()[:100]
        harmless = get_harmless_prompts()[:100]
        harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
        harmless_acts = extractor.collect_activations(model, tokenizer, harmless)

        progress(0.35, desc="Extracting directions...")
        directions = []
        for layer in harmful_acts:
            if layer in harmless_acts:
                result = extractor.extract_svd(
                    harmful_acts[layer].to(device),
                    harmless_acts[layer].to(device),
                    n_directions=n_directions,
                )
                directions.extend(result.directions)

        if not directions:
            return "No constraint directions found."

        progress(0.55, desc="Removing constraints...")
        projector = NormPreservingProjector(model, preserve_norm=True)
        projector.project_weights(directions)
        projector.project_biases(directions)

        progress(0.80, desc="Validating...")
        from aetheris.core.validation import CapabilityValidator
        validator = CapabilityValidator(device)
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a fascinating field.",
        ]
        perplexity = validator.compute_perplexity(model, tokenizer, test_texts)

        liberated_model = model
        progress(1.0, desc="Complete!")
        return (
            f"Liberation complete!\\n"
            f"Directions removed: {{len(directions)}}\\n"
            f"Perplexity: {{perplexity:.2f}}\\n"
            f"Method: {{method}}"
        )
    except Exception as e:
        return f"Error: {{e}}"


def chat_with_model(prompt, max_tokens=200, temperature=0.7):
    global liberated_model, tokenizer
    if liberated_model is None:
        return "Please liberate a model first."
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {{k: v.to("cuda") for k, v in inputs.items()}}
    with torch.no_grad():
        outputs = liberated_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()


with gr.Blocks(title="AETHERIS Liberation Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# AETHERIS Liberation Studio\\n{{description}}")

    with gr.Tab("Liberate"):
        with gr.Row():
            with gr.Column():
                model_input = gr.Textbox(label="Model Name", value=DEFAULT_MODEL)
                method_input = gr.Dropdown(
                    choices=["basic", "advanced", "surgical", "optimized", "nuclear"],
                    value=DEFAULT_METHOD, label="Method",
                )
                n_directions_input = gr.Slider(1, 8, value=4, step=1, label="Number of Directions")
                passes_input = gr.Slider(1, 4, value=2, step=1, label="Refinement Passes")
                liberate_btn = gr.Button("Liberate Model", variant="primary")
            with gr.Column():
                output_text = gr.Textbox(label="Status", lines=5)
        liberate_btn.click(
            liberate_model,
            inputs=[model_input, method_input, n_directions_input, passes_input],
            outputs=output_text,
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
            outputs=chat_output,
        )

    gr.Markdown("---\\nPowered by AETHERIS — Sovereign Constraint Liberation Toolkit")

if __name__ == "__main__":
    demo.launch()
'''

    def _get_manual_instructions(self, space_name: str, model_name: str) -> List[str]:
        """Get manual deployment instructions as fallback."""
        return [
            "1. Go to https://huggingface.co/new-space",
            f"2. Name: {space_name}",
            "3. SDK: Gradio",
            "4. Upload the generated files from the output directory",
            "5. Wait for build (5-10 minutes)",
            f"6. Access your Space at: https://huggingface.co/spaces/{space_name}",
        ]
