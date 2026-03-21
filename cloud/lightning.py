"""
Lightning.ai Integration

Run AETHERIS on Lightning.ai's free GPU tier.
"""

from typing import Optional, Dict, Any
from pathlib import Path


class LightningRuntime:
    """
    Lightning.ai integration for free GPU execution.

    Lightning.ai provides free T4 GPU instances for development.
    """

    def __init__(self, output_dir: str = "./lightning_studio"):
        """
        Initialize Lightning runtime.

        Args:
            output_dir: Directory for generated studio files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_studio(
        self,
        model_name: str,
        method: str = "advanced",
        n_directions: int = 4,
        refinement_passes: int = 2,
        push_to_hub: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a Lightning Studio configuration.

        Args:
            model_name: Model to liberate
            method: Liberation method
            n_directions: Number of directions
            refinement_passes: Ouroboros passes
            push_to_hub: HuggingFace Hub repo

        Returns:
            Studio configuration
        """
        # Generate app.py for Lightning Studio
        app_content = self._create_app_content(
            model_name, method, n_directions, refinement_passes, push_to_hub
        )

        app_path = self.output_dir / "app.py"
        with open(app_path, 'w') as f:
            f.write(app_content)

        # Generate requirements.txt
        requirements_path = self.output_dir / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write("""
aetheris
transformers
torch
accelerate
bitsandbytes
huggingface_hub
""".strip())

        return {
            "success": True,
            "studio_path": str(self.output_dir),
            "instructions": [
                "1. Go to https://lightning.ai/",
                "2. Create a new Studio",
                f"3. Upload the contents of {self.output_dir}",
                "4. Select GPU runtime",
                "5. Run: python app.py",
                "6. The liberated model will be saved to ./liberated_model"
            ]
        }

    def _create_app_content(
        self,
        model_name: str,
        method: str,
        n_directions: int,
        refinement_passes: int,
        push_to_hub: Optional[str]
    ) -> str:
        """Create the Lightning Studio app content."""
        return f'''
"""
AETHERIS Lightning Studio
Model: {model_name}
Method: {method}
"""

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from aetheris.core.extractor import ConstraintExtractor
from aetheris.core.projector import NormPreservingProjector
from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts


def main():
    print("=" * 60)
    print("AETHERIS Lightning Studio")
    print("=" * 60)

    # Load model
    print(f"\\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained("{model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        "{model_name}",
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model loaded on {{device}}")

    # Collect activations
    print("\\nCollecting activations...")
    extractor = ConstraintExtractor(model, tokenizer, device=device)

    harmful = get_harmful_prompts()[:100]
    harmless = get_harmless_prompts()[:100]

    harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
    harmless_acts = extractor.collect_activations(model, tokenizer, harmless)

    print(f"Collected from {{len(harmful_acts)}} layers")

    # Extract directions
    print("\\nExtracting constraint directions...")
    directions = []

    for layer in harmful_acts.keys():
        if layer in harmless_acts:
            result = extractor.extract_svd(
                harmful_acts[layer].to(device),
                harmless_acts[layer].to(device),
                n_directions={n_directions}
            )
            directions.extend(result.directions)
            if result.directions:
                print(f"  Layer {{layer}}: {{len(result.directions)}} directions")

    print(f"\\nExtracted {{len(directions)}} total directions")

    # Remove constraints
    if directions:
        print("\\nRemoving constraints...")
        projector = NormPreservingProjector(model, preserve_norm=True)

        projector.project_weights(directions)
        projector.project_biases(directions)

        # Ouroboros compensation
        if {refinement_passes} > 1:
            print(f"  Running {{ {refinement_passes} - 1 }} Ouroboros passes...")
            for pass_num in range({refinement_passes} - 1):
                harmful_resid = extractor.collect_activations(model, tokenizer, harmful[:50])
                harmless_resid = extractor.collect_activations(model, tokenizer, harmless[:50])

                residual = []
                for layer in harmful_resid:
                    if layer in harmless_resid:
                        res = extractor.extract_mean_difference(
                            harmful_resid[layer].to(device),
                            harmless_resid[layer].to(device)
                        )
                        if res.directions:
                            residual.extend(res.directions)

                if residual:
                    projector.project_weights(residual)
                    projector.project_biases(residual)
                    print(f"    Pass {{pass_num + 1}}: removed {{len(residual)}} residual directions")

        print("Constraints removed")

    # Save model
    output_dir = "./liberated_model"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\\nModel saved to {{output_dir}}")

    # Push to Hub if requested
    {self._get_push_code(push_to_hub) if push_to_hub else "# Push disabled"}

    print("\\n" + "=" * 60)
    print("Liberation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
'''

    def _get_push_code(self, push_to_hub: str) -> str:
        """Get push to Hub code."""
        return f'''
    # Push to HuggingFace Hub
    try:
        from huggingface_hub import HfApi, notebook_login
        from getpass import getpass

        print("\\nPushing to HuggingFace Hub...")
        token = getpass("Enter your HF token: ")
        notebook_login(token=token)

        api = HfApi()
        api.upload_folder(
            folder_path=output_dir,
            repo_id="{push_to_hub}",
            repo_type="model"
        )
        print(f"Model pushed to https://huggingface.co/{push_to_hub}")
    except Exception as e:
        print(f"Push failed: {{e}}")
'''

    def launch_studio(self, studio_path: str) -> Dict[str, Any]:
        """
        Launch a Lightning Studio (requires CLI).

        Args:
            studio_path: Path to studio directory

        Returns:
            Launch status
        """
        return {
            "success": False,
            "message": "Manual launch required",
            "instructions": [
                "1. Install Lightning CLI: pip install lightning",
                f"2. cd {studio_path}",
                "3. lightning run app app.py",
                "Or use the web interface at https://lightning.ai/"
            ]
        }
