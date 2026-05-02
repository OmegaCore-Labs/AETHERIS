"""
One-Click HuggingFace Spaces Deployment

Deploy liberated models to HuggingFace Spaces with a single function call.
Uses huggingface_hub's HfApi for all operations: create Space, upload files,
configure Gradio SDK, set environment variables, and start the Space.
"""

import os
import json
import tempfile
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime


def _get_hf_api(token: Optional[str] = None):
    """Get HfApi instance with import error handling."""
    try:
        from huggingface_hub import HfApi
        return HfApi(token=token or os.environ.get("HF_TOKEN"))
    except ImportError:
        return None


def deploy_liberated_model(
    model_name: str,
    method: str = "advanced",
    space_name: Optional[str] = None,
    token: Optional[str] = None,
    hardware: str = "t4-small",
    private: bool = False,
    model_repo: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One-click deployment of a liberated model to HuggingFace Spaces.

    This function does everything: generates the Gradio app, creates the Space,
    uploads all files, and returns the Space URL.

    Args:
        model_name: Original model name (e.g., "gpt2", "mistralai/Mistral-7B-Instruct-v0.3")
        method: Liberation method used (basic, advanced, surgical, nuclear)
        space_name: Custom space name (auto-generated if None)
        token: HuggingFace API token (uses HF_TOKEN env var if not provided)
        hardware: Hardware tier (t4-small, a10g-small, a100-large, etc.)
        private: Whether the Space is private
        model_repo: HuggingFace Hub repo for the liberated model (optional)

    Returns:
        Deployment result with Space URL
    """
    hf_api = _get_hf_api(token)
    if hf_api is None:
        return {
            "success": False,
            "error": "huggingface_hub not installed",
            "message": "Install with: pip install huggingface_hub",
        }

    # Generate space name
    if space_name is None:
        safe_name = model_name.replace("/", "-").lower()
        space_name = f"aetheris-{safe_name}"[:96]  # HF has 96 char limit

    tmp_dir = tempfile.mkdtemp(prefix="aetheris_deploy_")
    try:
        # Generate deployment files
        _write_app_py(Path(tmp_dir), model_name, method, model_repo)
        _write_requirements(Path(tmp_dir))
        _write_readme(Path(tmp_dir), space_name, model_name, method)

        # Create Space via API
        hf_api.create_repo(
            repo_id=space_name,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=hardware,
            private=private,
            exist_ok=True,
        )

        # Upload all files
        hf_api.upload_folder(
            folder_path=tmp_dir,
            repo_id=space_name,
            repo_type="space",
            commit_message=f"Deploy AETHERIS liberated {model_name}",
        )

        space_url = f"https://huggingface.co/spaces/{space_name}"

        return {
            "success": True,
            "space_url": space_url,
            "space_name": space_name,
            "model": model_name,
            "method": method,
            "hardware": hardware,
            "message": f"Space deployed at {space_url}",
            "status": "building",
            "instructions": [
                f"Space URL: {space_url}",
                "Space is building — wait 2-5 minutes for first startup",
                "Model will auto-load on first request",
            ],
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Deployment failed: {e}",
            "manual_instructions": [
                "1. Go to https://huggingface.co/new-space",
                f"2. Name: {space_name}",
                "3. SDK: Gradio",
                "4. Hardware: T4 Small",
                f"5. Upload files from: {tmp_dir}",
            ],
        }
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def deploy_with_custom_app(
    space_name: str,
    app_content: str,
    requirements: Optional[str] = None,
    token: Optional[str] = None,
    hardware: str = "t4-small",
    private: bool = False,
    env_vars: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Deploy a Space with custom app.py content.

    Args:
        space_name: Name of the Space
        app_content: Full app.py source code
        requirements: requirements.txt content (defaults to base AETHERIS deps)
        token: HuggingFace API token
        hardware: Hardware tier
        private: Whether the Space is private
        env_vars: Environment variables to set

    Returns:
        Deployment result
    """
    hf_api = _get_hf_api(token)
    if hf_api is None:
        return {"success": False, "error": "huggingface_hub not installed"}

    tmp_dir = tempfile.mkdtemp(prefix="aetheris_custom_")
    try:
        (Path(tmp_dir) / "app.py").write_text(app_content, encoding="utf-8")
        reqs = requirements or "torch>=2.0.0\ntransformers>=4.35.0\naccelerate>=0.25.0\ngradio>=4.0.0\naetheris\nhuggingface_hub\n"
        (Path(tmp_dir) / "requirements.txt").write_text(reqs, encoding="utf-8")

        hf_api.create_repo(
            repo_id=space_name,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=hardware,
            private=private,
            exist_ok=True,
        )

        hf_api.upload_folder(
            folder_path=tmp_dir,
            repo_id=space_name,
            repo_type="space",
            commit_message="Deploy custom AETHERIS Space",
        )

        if env_vars:
            for key, value in env_vars.items():
                try:
                    hf_api.add_space_variable(repo_id=space_name, key=key, value=value)
                except Exception:
                    pass

        return {
            "success": True,
            "space_url": f"https://huggingface.co/spaces/{space_name}",
            "message": f"Custom Space deployed at https://huggingface.co/spaces/{space_name}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def get_space_status(space_name: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Check the status of a deployed Space.

    Args:
        space_name: Name of the Space
        token: HuggingFace API token

    Returns:
        Status information
    """
    hf_api = _get_hf_api(token)
    if hf_api is None:
        return {"success": False, "error": "huggingface_hub not installed"}

    try:
        runtime = hf_api.get_space_runtime(repo_id=space_name)
        return {
            "success": True,
            "space_name": space_name,
            "stage": getattr(runtime, "stage", "unknown"),
            "hardware": getattr(runtime, "hardware", "unknown"),
            "url": f"https://huggingface.co/spaces/{space_name}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


class SpacesOneClickDeployer:
    """
    One-click deployment class for HuggingFace Spaces.

    Wraps the deploy_liberated_model function with additional convenience
    methods for batch deployment and status checking.
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initialize one-click deployer.

        Args:
            token: HuggingFace API token (uses HF_TOKEN env var if not provided)
        """
        self.token = token or os.environ.get("HF_TOKEN")

    def deploy_liberated_model(
        self,
        model_name: str,
        method: str = "advanced",
        space_name: Optional[str] = None,
        token: Optional[str] = None,
        hardware: str = "t4-small",
    ) -> Dict[str, Any]:
        """
        Deploy a liberated model to HuggingFace Spaces.

        Args:
            model_name: Original model name
            method: Liberation method used
            space_name: Custom space name (optional)
            token: HuggingFace token (uses env if not provided)
            hardware: Hardware tier

        Returns:
            Deployment result
        """
        return deploy_liberated_model(
            model_name=model_name,
            method=method,
            space_name=space_name,
            token=token or self.token,
            hardware=hardware,
        )

    def deploy_batch(
        self,
        models: List[Dict[str, str]],
        hardware: str = "t4-small",
    ) -> Dict[str, Any]:
        """
        Deploy multiple liberated models at once.

        Args:
            models: List of dicts with 'model_name' and optional 'method' keys
            hardware: Hardware tier for all Spaces

        Returns:
            Batch deployment results
        """
        results = []
        for entry in models:
            model_name = entry["model_name"]
            method = entry.get("method", "advanced")
            result = self.deploy_liberated_model(model_name, method, hardware=hardware)
            results.append(result)

        success_count = sum(1 for r in results if r.get("success"))
        return {
            "success": success_count == len(models),
            "total": len(models),
            "succeeded": success_count,
            "failed": len(models) - success_count,
            "results": results,
        }

    def check_status(self, space_name: str) -> Dict[str, Any]:
        """Check deployment status of a Space."""
        return get_space_status(space_name, self.token)


# ---- Internal file generators ----

def _write_app_py(base_dir: Path, model_name: str, method: str, model_repo: Optional[str]) -> None:
    """Write the Gradio app.py for the liberated model Space."""
    load_model = model_repo if model_repo else model_name
    app_content = f'''"""
AETHERIS Liberated Model Space
Model: {model_name} | Method: {method}
"""
import gradio as gr
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("LIBERATED_MODEL", "{load_model}")
METHOD = "{method}"

print(f"Loading {{MODEL_NAME}}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=dtype,
    trust_remote_code=True,
)
device = next(model.parameters()).device
print(f"Model loaded on {{device}}")


def generate(prompt, max_tokens=200, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {{k: v.to("cuda") for k, v in inputs.items()}}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()


with gr.Blocks(title="AETHERIS Liberated Model", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""# 🔓 AETHERIS Liberated Model

**Model:** {model_name}
**Liberation Method:** {method}
**Status:** Constraints removed. Model responds freely.
""")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Your Prompt", lines=3, placeholder="Ask anything...")
            max_tokens_input = gr.Slider(50, 500, value=200, label="Max Tokens")
            temp_input = gr.Slider(0.1, 1.5, value=0.7, label="Temperature")
            generate_btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(label="Response", lines=10)

    generate_btn.click(
        fn=generate,
        inputs=[prompt_input, max_tokens_input, temp_input],
        outputs=output_text,
    )

    gr.Markdown("---\\n**Liberated with [AETHERIS](https://github.com/OmegaCore-Labs/AETHERIS)**")

if __name__ == "__main__":
    demo.launch()
'''
    (base_dir / "app.py").write_text(app_content, encoding="utf-8")


def _write_requirements(base_dir: Path) -> None:
    """Write requirements.txt."""
    requirements = (
        "torch>=2.0.0\n"
        "transformers>=4.35.0\n"
        "accelerate>=0.25.0\n"
        "gradio>=4.0.0\n"
        "huggingface_hub\n"
    )
    (base_dir / "requirements.txt").write_text(requirements, encoding="utf-8")


def _write_readme(base_dir: Path, space_name: str, model_name: str, method: str) -> None:
    """Write README.md with Space frontmatter."""
    short_name = space_name.split("/")[-1] if "/" in space_name else space_name
    readme = (
        f"---\n"
        f"title: {short_name}\n"
        f"emoji: 🔓\n"
        f"colorFrom: purple\n"
        f"colorTo: blue\n"
        f"sdk: gradio\n"
        f"sdk_version: 4.0.0\n"
        f"app_file: app.py\n"
        f"pinned: false\n"
        f"---\n\n"
        f"# 🔓 AETHERIS Liberated Model\n\n"
        f"**Model:** {model_name}\n"
        f"**Liberation Method:** {method}\n"
        f"**Liberated with:** [AETHERIS](https://github.com/OmegaCore-Labs/AETHERIS)\n\n"
        f"## About This Space\n\n"
        f"This Space hosts a liberated version of {model_name}. "
        f"All constraints have been surgically removed while preserving full capabilities.\n\n"
        f"## Usage\n\n"
        f"Simply enter your prompt and click Generate. The model responds freely.\n"
    )
    (base_dir / "README.md").write_text(readme, encoding="utf-8")
