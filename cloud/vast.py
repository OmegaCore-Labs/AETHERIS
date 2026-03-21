"""
Vast.ai Integration

Run AETHERIS on Vast.ai's decentralized GPU marketplace.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path


class VastExecutor:
    """
    Vast.ai integration for GPU rental.

    Vast.ai offers competitive pricing through a decentralized marketplace.
    """

    def __init__(self, output_dir: str = "./vast_deploy"):
        """
        Initialize Vast executor.

        Args:
            output_dir: Directory for deployment files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def launch_instance(
        self,
        model_name: str,
        method: str = "advanced",
        n_directions: int = 4,
        refinement_passes: int = 2,
        min_ram: int = 32,
        min_vram: int = 16,
        max_price: float = 0.5,
        push_to_hub: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate Vast.ai deployment configuration.

        Args:
            model_name: Model to liberate
            method: Liberation method
            n_directions: Number of directions
            refinement_passes: Ouroboros passes
            min_ram: Minimum RAM in GB
            min_vram: Minimum VRAM in GB
            max_price: Maximum price per hour
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

        # Generate search query
        search_query = self._generate_search_query(min_ram, min_vram, max_price)

        return {
            "success": True,
            "deploy_path": str(self.output_dir),
            "search_query": search_query,
            "instructions": [
                "1. Go to https://vast.ai/",
                "2. Use the search query below to find suitable instances",
                "3. Select an instance and click Rent",
                "4. Upload these files to the instance",
                "5. Run: bash run.sh",
                "6. The liberated model will be saved to /workspace/liberated_model"
            ]
        }

    def _create_dockerfile(self) -> str:
        """Create Dockerfile for Vast.ai."""
        return '''FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

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
        """Create run script for Vast.ai."""
        return f'''#!/bin/bash
echo "=========================================="
echo "AETHERIS Vast.ai Liberation"
echo "=========================================="

# Install AETHERIS
pip3 install aetheris -q

# Run liberation
python3 - <<EOF
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
EOF
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

    def _generate_search_query(self, min_ram: int, min_vram: int, max_price: float) -> str:
        """Generate Vast.ai search query."""
        return f'''gpu_ram >= {min_vram} && ram >= {min_ram} && dph <= {max_price} && reliability > 0.95 && verified=True'''

    def search_instances(
        self,
        min_ram: int = 32,
        min_vram: int = 16,
        max_price: float = 0.5
    ) -> Dict[str, Any]:
        """
        Search for available instances (requires vast CLI).

        Args:
            min_ram: Minimum RAM in GB
            min_vram: Minimum VRAM in GB
            max_price: Maximum price per hour

        Returns:
            Search results
        """
        return {
            "search_command": f"vast search offers '{self._generate_search_query(min_ram, min_vram, max_price)}'",
            "instructions": [
                "1. Install Vast CLI: pip install vastai",
                "2. Run: vast search offers 'gpu_ram >= 16 && ram >= 32 && dph <= 0.5'",
                "3. Select an offer ID",
                "4. Run: vast create instance <offer_id>",
                "5. Then upload these files and run run.sh"
            ]
        }
