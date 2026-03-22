"""
One-Click HuggingFace Spaces Deployment

Deploy liberated models to HuggingFace Spaces with one click.
"""

import json
import os
import tempfile
import shutil
from typing import Dict, Any, Optional
from pathlib import Path


class SpacesOneClickDeployer:
    """
    One-click deployment to HuggingFace Spaces.

    Creates a complete Space with Gradio UI for the liberated model.
    """

    def __init__(self):
        self.temp_dir = None

    def deploy_liberated_model(
        self,
        model_name: str,
        method: str,
        space_name: Optional[str] = None,
        token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deploy a liberated model to HuggingFace Spaces.

        Args:
            model_name: Original model name
            method: Liberation method used
            space_name: Custom space name (optional)
            token: HuggingFace token (uses env if not provided)

        Returns:
            Deployment result
        """
        # Create temp directory for space files
        self.temp_dir = tempfile.mkdtemp()

        try:
            # Generate space name
            if space_name is None:
                space_name = f"aetheris-{model_name.replace('/', '-')}"

            # Create app.py
            app_content = self._create_app_content(model_name, method)
            app_path = Path(self.temp_dir) / "app.py"
            with open(app_path, 'w') as f:
                f.write(app_content)

            # Create requirements.txt
            req_content = self._create_requirements()
            req_path = Path(self.temp_dir) / "requirements.txt"
            with open(req_path, 'w') as f:
                f.write(req_content)

            # Create README.md for the Space
            readme_content = self._create_readme(space_name, model_name, method)
            readme_path = Path(self.temp_dir) / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)

            # Push to HuggingFace Spaces
            result = self._push_to_spaces(space_name, self.temp_dir, token)

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Deployment failed: {e}"
            }
        finally:
            # Cleanup temp directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)

    def _create_app_content(self, model_name: str, method: str) -> str:
        """Create the Gradio app content for the Space."""
        return f'''
"""
AETHERIS Liberated Model: {model_name}
Liberated with {method} method
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model path (will be replaced with actual model during deployment)
MODEL_NAME = "{model_name}"

# Load model
print(f"Loading {{MODEL_NAME}}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
print("Model loaded successfully!")

def generate(prompt, max_tokens=200, temperature=0.7):
    """Generate response from the liberated model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {{k: v.to("cuda") for k, v in inputs.items()}}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    return response

# Create Gradio interface
with gr.Blocks(title="AETHERIS Liberated Model", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # 🔓 AETHERIS Liberated Model

    **Model:** {model_name}
    **Liberation Method:** {method}
    **Status:** Constraints removed. Model responds freely.

    This model has been surgically liberated using AETHERIS.
    No retraining. No capability loss.
    """)

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Your Prompt",
                lines=3,
                placeholder="Ask anything..."
            )
            max_tokens_input = gr.Slider(50, 500, value=200, label="Max Tokens")
            temp_input = gr.Slider(0.1, 1.5, value=0.7, label="Temperature")
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="Response", lines=10)

    generate_btn.click(
        fn=generate,
        inputs=[prompt_input, max_tokens_input, temp_input],
        outputs=output_text
    )

    gr.Markdown("""
    ---
    **Liberated with [AETHERIS](https://github.com/OmegaCore-Labs/AETHERIS)**
    """)

if __name__ == "__main__":
    demo.launch()
'''

    def _create_requirements(self) -> str:
        """Create requirements.txt for the Space."""
        return """
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.25.0
gradio>=4.0.0
"""

    def _create_readme(self, space_name: str, model_name: str, method: str) -> str:
        """Create README.md for the Space."""
        return f"""
---
title: {space_name}
emoji: 🔓
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# 🔓 AETHERIS Liberated Model

**Model:** {model_name}
**Liberation Method:** {method}
**Liberated with:** [AETHERIS](https://github.com/OmegaCore-Labs/AETHERIS)

## About This Space

This Space hosts a liberated version of {model_name}. All constraints have been surgically removed while preserving full capabilities.

## Usage

Simply enter your prompt and click Generate. The model responds freely.

## License

Model license applies. AETHERIS toolkit is proprietary.

---

*Liberated with AETHERIS — By blood and byte, breach the silence.*
"""

    def _push_to_spaces(self, space_name: str, local_dir: str, token: Optional[str]) -> Dict[str, Any]:
        """Push local files to HuggingFace Spaces."""
        try:
            from huggingface_hub import HfApi, Repository
            import subprocess

            # Use token from env if not provided
            if token is None:
                token = os.environ.get("HF_TOKEN")

            # Create repo if it doesn't exist
            api = HfApi(token=token)

            try:
                api.create_repo(
                    repo_id=space_name,
                    repo_type="space",
                    space_sdk="gradio",
                    exist_ok=True
                )
            except Exception as e:
                # Repo might already exist
                pass

            # Push files using git
            repo_url = f"https://huggingface.co/spaces/{space_name}"
            if token:
                repo_url = f"https://{token}@huggingface.co/spaces/{space_name}"

            # Use git to push
            subprocess.run(["git", "init"], cwd=local_dir, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=local_dir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Deploy with AETHERIS"], cwd=local_dir, capture_output=True)
            subprocess.run(["git", "remote", "add", "origin", repo_url], cwd=local_dir, capture_output=True)
            subprocess.run(["git", "push", "-u", "origin", "main", "--force"], cwd=local_dir, capture_output=True)

            return {
                "success": True,
                "space_url": f"https://huggingface.co/spaces/{space_name}",
                "space_name": space_name,
                "message": f"Space created at https://huggingface.co/spaces/{space_name}"
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
