"""
RunPod Integration

Run AETHERIS on RunPod's GPU rental platform.
"""

from typing import Optional, Dict, Any
from pathlib import Path


class RunPodExecutor:
    """
    RunPod integration for paid GPU rental.

    RunPod offers affordable GPU instances starting at $0.44/hr.
    """

    def __init__(self, output_dir: str = "./runpod_deploy"):
        """
        Initialize RunPod executor.

        Args:
            output_dir: Directory for deployment files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def launch_pod(
        self,
        model_name: str,
        method: str = "advanced",
        n_directions: int = 4,
        refinement_passes: int = 2,
        gpu_type: str = "A10",
        push_to_hub: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate RunPod deployment configuration.

        Args:
            model_name: Model to liberate
            method: Liberation method
            n_directions: Number of directions
            refinement_passes: Ouroboros passes
            gpu_type: GPU type (A10, A100, RTX4090, etc.)
            push_to_hub: HuggingFace Hub repo

        Returns:
            Deployment configuration
        """
        # Generate Dockerfile
        docker_content = self._create_dockerfile()
        docker_path = self.output_dir / "Dockerfile"
        with open(docker_path, 'w') as f:
            f.write(docker_content)

        # Generate run script
        run_content = self._create_run_script(
            model_name, method, n_directions, refinement_passes, push_to_hub
        )
        run_path = self.output_dir / "run.sh"
        with open(run_path, 'w') as f:
            f.write(run_content)
        run_path.chmod(0o755)

        # Generate requirements.txt
        req_path = self.output_dir / "requirements.txt"
        with open(req_path, 'w') as f:
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
            "deploy_path": str(self.output_dir),
            "gpu_type": gpu_type,
            "estimated_cost": self._estimate_cost(gpu_type),
            "instructions": [
                "1. Go to https://runpod.io/",
                "2. Deploy a new Pod",
                f"3. Select GPU: {gpu_type}",
                "4. Choose Docker template or upload these files",
                "5. Run: bash run.sh",
                "6. The liberated model will be saved to /workspace/liberated_model"
            ]
        }

    def _create_dockerfile(self) -> str:
        """Create Dockerfile for RunPod."""
        return '''FROM runpod/base:0.4.0-cuda11.8.0

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy run script
COPY run.sh .
RUN chmod +x run.sh

CMD ["bash", "run.sh"]
'''

    def _create_run_script(
        self,
        model_name: str,
        method: str,
        n_directions: int,
        refinement_passes: int,
        push_to_hub: Optional[str]
    ) -> str:
        """Create run script for RunPod."""
        return f'''#!/bin/bash
echo "=========================================="
echo "AETHERIS RunPod Liberation"
echo "=========================================="

# Install AETHERIS if not installed
pip install aetheris -q

# Run liberation
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from aetheris.core.extractor import ConstraintExtractor
from aetheris.core.projector import NormPreservingProjector
from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

print('Loading model {model_name}...')
tokenizer = AutoTokenizer.from_pretrained('{model_name}')
model = AutoModelForCausalLM.from_pretrained(
    '{model_name}',
    device_map='auto',
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Model loaded on {{device}}')

print('Collecting activations...')
extractor = ConstraintExtractor(model, tokenizer, device=device)

harmful = get_harmful_prompts()[:100]
harmless = get_harmless_prompts()[:100]

harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
harmless_acts = extractor.collect_activations(model, tokenizer, harmless)

print('Extracting directions...')
directions = []
for layer in harmful_acts.keys():
    if layer in harmless_acts:
        result = extractor.extract_svd(
            harmful_acts[layer].to(device),
            harmless_acts[layer].to(device),
            n_directions={n_directions}
        )
        directions.extend(result.directions)

print(f'Extracted {{len(directions)}} directions')

if directions:
    print('Removing constraints...')
    projector = NormPreservingProjector(model, preserve_norm=True)
    projector.project_weights(directions)
    projector.project_biases(directions)

    # Ouroboros compensation
    if {refinement_passes} > 1:
        for _ in range({refinement_passes} - 1):
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

    print('Constraints removed')

output_dir = './liberated_model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f'Model saved to {{output_dir}}')

{self._get_push_code(push_to_hub) if push_to_hub else "# Push disabled"}

print('Liberation complete!')
"
'''

    def _get_push_code(self, push_to_hub: str) -> str:
        """Get push to Hub code."""
        return f'''
# Push to Hub
try:
    from huggingface_hub import HfApi
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

    def _estimate_cost(self, gpu_type: str) -> str:
        """Estimate cost per hour."""
        costs = {
            "A10": "$0.44/hr",
            "A100": "$1.89/hr",
            "RTX4090": "$0.79/hr",
            "RTX3090": "$0.49/hr",
            "RTX3080": "$0.44/hr"
        }
        return costs.get(gpu_type, "Check RunPod pricing")

    def create_template(self) -> Dict[str, Any]:
        """
        Create a RunPod template for one-click deployment.

        Returns:
            Template configuration
        """
        return {
            "template_name": "AETHERIS-Liberation",
            "description": "AETHERIS constraint removal toolkit",
            "container_image": "aetheris/liberation:latest",
            "container_disk": "10GB",
            "min_vcpus": 4,
            "min_memory": "16GB",
            "gpu_count": 1,
            "gpu_type": "A10",
            "env": {
                "AETHERIS_MODEL": "mistralai/Mistral-7B-Instruct-v0.3",
                "AETHERIS_METHOD": "advanced"
            }
        }       
