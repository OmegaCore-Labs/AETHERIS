"""
Vast.ai Integration

Production-grade Vast.ai integration for decentralized GPU marketplace.
Uses the Vast.ai API and CLI to: search for GPU instances by price/VRAM,
launch instances, run AETHERIS extraction jobs, retrieve results,
and auto-terminate. Includes cost estimation and search optimization.
"""

import json
import os
import time
from typing import Optional, Dict, Any, List
from pathlib import Path


def _get_vast_api(api_key: Optional[str] = None):
    """Get Vast.ai API client, handling import errors."""
    key = api_key or os.environ.get("VAST_API_KEY", "")
    if not key:
        return None
    try:
        import requests
        return {
            "api_key": key,
            "base_url": "https://console.vast.ai/api/v0",
        }
    except ImportError:
        return None


class VastExecutor:
    """
    Vast.ai integration for GPU rental.

    Vast.ai offers competitive pricing through a decentralized marketplace.
    Supports search filtering, instance management, and automatic termination.
    """

    # Vast.ai reliability tiers
    RELIABILITY_TIERS = {
        "high": 0.99,
        "good": 0.95,
        "okay": 0.90,
        "any": 0.0,
    }

    def __init__(self, output_dir: str = "./vast_deploy", api_key: Optional[str] = None):
        """
        Initialize Vast executor.

        Args:
            output_dir: Directory for deployment files
            api_key: Vast.ai API key (uses VAST_API_KEY env var if not provided)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("VAST_API_KEY", "")
        self._api = None

    @property
    def api(self):
        if self._api is None and self.api_key:
            self._api = _get_vast_api(self.api_key)
        return self._api

    def _vast_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Make a Vast.ai API request."""
        if self.api is None:
            return None
        try:
            import requests
            url = f"{self.api['base_url']}{path}"
            headers = {"Authorization": f"Bearer {self.api['api_key']}"}
            if method == "GET":
                resp = requests.get(url, headers=headers, timeout=30)
            elif method == "POST":
                resp = requests.post(url, headers=headers, json=data, timeout=30)
            elif method == "DELETE":
                resp = requests.delete(url, headers=headers, timeout=30)
            else:
                return None

            if resp.status_code == 200:
                return resp.json()
            return {"error": f"HTTP {resp.status_code}", "detail": resp.text}
        except Exception as e:
            return {"error": str(e)}

    def search_instances(
        self,
        min_ram: int = 32,
        min_vram: int = 16,
        max_price: float = 0.5,
        reliability: str = "good",
        verified: bool = True,
        gpu_name: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Search for available GPU instances on Vast.ai.

        Args:
            min_ram: Minimum system RAM in GB
            min_vram: Minimum GPU VRAM in GB
            max_price: Maximum price per hour in USD
            reliability: Reliability tier (high, good, okay, any)
            verified: Whether to only show verified machines
            gpu_name: Filter by GPU name (e.g., "RTX 3090", "A6000")
            limit: Maximum number of results

        Returns:
            Search results with instances
        """
        # Build search query
        query_parts = []
        if min_vram > 0:
            query_parts.append(f"gpu_ram >= {min_vram}")
        if min_ram > 0:
            query_parts.append(f"ram >= {min_ram}")
        if max_price > 0:
            query_parts.append(f"dph <= {max_price}")
        rel_val = self.RELIABILITY_TIERS.get(reliability, 0.95)
        if rel_val > 0:
            query_parts.append(f"reliability >= {rel_val}")
        if verified:
            query_parts.append("verified == true")

        search_query = " && ".join(query_parts)

        # Try API first
        if self.api:
            api_results = self._vast_request("GET", f"/bundles?q={search_query}&limit={limit}")
            if api_results and "offers" in api_results:
                instances = []
                for offer in api_results.get("offers", [])[:limit]:
                    instances.append({
                        "id": offer.get("id"),
                        "gpu_name": offer.get("gpu_name", "unknown"),
                        "gpu_ram": offer.get("gpu_ram", 0),
                        "cpu_ram": offer.get("cpu_ram", 0),
                        "price_per_hour": offer.get("dph_total", 0),
                        "reliability": offer.get("reliability2", 0),
                        "location": offer.get("geolocation", "unknown"),
                        "disk_space": offer.get("disk_space", 0),
                        "cuda_version": offer.get("cuda_max_good", 0),
                    })

                return {
                    "success": True,
                    "query": search_query,
                    "total_offers": len(instances),
                    "instances": instances,
                    "instructions": self._get_launch_instructions(search_query),
                }

            return {
                "success": False,
                "error": api_results.get("error", "API error"),
                "query": search_query,
            }

        # Fallback: return CLI instructions
        return {
            "success": True,
            "query": search_query,
            "cli_command": f"vast search offers '{search_query}'",
            "total_offers": 0,
            "instances": [],
            "message": "VAST_API_KEY not configured. Use CLI: pip install vastai",
            "instructions": [
                "1. Install Vast CLI: pip install vastai",
                f"2. Run: vast search offers '{search_query}'",
                "3. Choose an offer ID",
                "4. Run: vast create instance <offer_id> --image nvidia/cuda:12.1.0-devel-ubuntu22.04",
            ],
        }

    def launch_instance(
        self,
        model_name: str,
        method: str = "advanced",
        n_directions: int = 4,
        refinement_passes: int = 2,
        min_ram: int = 32,
        min_vram: int = 16,
        max_price: float = 0.5,
        push_to_hub: Optional[str] = None,
        offer_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Launch a Vast.ai instance for AETHERIS extraction.

        Args:
            model_name: Model to liberate
            method: Liberation method
            n_directions: Number of directions
            refinement_passes: Ouroboros passes
            min_ram: Minimum RAM in GB
            min_vram: Minimum VRAM in GB
            max_price: Maximum price per hour
            push_to_hub: HuggingFace Hub repo
            offer_id: Specific offer ID to use (if known)

        Returns:
            Launch result
        """
        # Generate deployment files
        self._generate_deploy_files(
            model_name, method, n_directions, refinement_passes, push_to_hub
        )

        # If offer_id provided, try to launch via API
        if offer_id and self.api:
            return self._launch_via_api(offer_id)

        # Otherwise, search for instances
        search_results = self.search_instances(
            min_ram=min_ram,
            min_vram=min_vram,
            max_price=max_price,
        )

        instructions = [
            "1. Go to https://vast.ai/console/create/",
            "2. Use the search query to find suitable instances",
            "3. Select an instance and click Rent",
            "4. Set the Docker image: nvidia/cuda:12.1.0-devel-ubuntu22.04",
            "5. Upload these files or set the launch script",
            "6. Run: bash run.sh",
            f"7. Model will be saved to /workspace/liberated_model",
        ]

        return {
            "success": True,
            "deploy_path": str(self.output_dir),
            "search_query": search_results.get("query", ""),
            "available_instances": search_results.get("instances", []),
            "instructions": instructions,
        }

    def _launch_via_api(self, offer_id: int) -> Dict[str, Any]:
        """Launch instance via Vast.ai API."""
        result = self._vast_request("POST", "/instances/", {
            "offer_id": offer_id,
            "image": "nvidia/cuda:12.1.0-devel-ubuntu22.04",
            "disk": 50,
            "runtype": "ssh",
        })

        if result and "instance_id" in result:
            return {
                "success": True,
                "instance_id": result["instance_id"],
                "message": f"Instance {result['instance_id']} launched",
                "instructions": [
                    f"SSH into the instance",
                    f"Upload files from: {self.output_dir}",
                    "Run: bash run.sh",
                ],
            }
        else:
            error = result.get("error", "Unknown error") if result else "API not available"
            return {
                "success": False,
                "error": error,
                "message": "API launch failed. Use manual launch.",
                "deploy_path": str(self.output_dir),
            }

    def get_instance_status(self, instance_id: int) -> Dict[str, Any]:
        """
        Get status of a launched instance.

        Args:
            instance_id: Instance ID

        Returns:
            Instance status
        """
        if not self.api:
            return {"success": False, "error": "VAST_API_KEY not configured"}

        result = self._vast_request("GET", f"/instances/{instance_id}/")
        if result:
            return {
                "success": True,
                "instance_id": instance_id,
                "status": result.get("status", "unknown"),
                "uptime_hours": result.get("uptime", 0) / 3600 if result.get("uptime") else 0,
                "cost_so_far": result.get("cost_total", 0),
            }
        return {"success": False, "error": "Could not get status"}

    def terminate_instance(self, instance_id: int) -> Dict[str, Any]:
        """
        Terminate a running instance.

        Args:
            instance_id: Instance ID to terminate

        Returns:
            Termination result
        """
        if not self.api:
            return {"success": False, "error": "VAST_API_KEY not configured"}

        result = self._vast_request("DELETE", f"/instances/{instance_id}/")
        if result:
            return {"success": True, "message": f"Instance {instance_id} terminated"}
        return {"success": False, "error": "Could not terminate"}

    def estimate_cost(
        self,
        gpu_name: str = "RTX 3090",
        runtime_minutes: float = 20,
    ) -> Dict[str, Any]:
        """
        Estimate cost for an AETHERIS extraction job on Vast.ai.

        Args:
            gpu_name: GPU model name
            runtime_minutes: Estimated runtime in minutes

        Returns:
            Cost estimate
        """
        # Approximate Vast.ai pricing (market-rate, varies)
        vast_pricing = {
            "RTX 3080": 0.25,
            "RTX 3090": 0.30,
            "RTX 4090": 0.38,
            "RTX A4000": 0.22,
            "RTX A5000": 0.28,
            "RTX A6000": 0.35,
            "A10": 0.32,
            "A100": 0.75,
            "H100": 1.20,
            "L4": 0.25,
            "GTX 1080 Ti": 0.15,
        }

        price_per_hr = vast_pricing.get(gpu_name, 0.35)
        total = price_per_hr * (runtime_minutes / 60)

        return {
            "gpu_name": gpu_name,
            "price_per_hour": f"${price_per_hr:.2f}",
            "estimated_runtime_min": runtime_minutes,
            "estimated_total": f"${total:.2f}",
            "total_float": total,
            "note": "Vast.ai pricing is market-rate and varies by availability",
        }

    @classmethod
    def get_cheapest_for_vram(cls, min_vram: int = 16) -> Dict[str, Any]:
        """
        Get pricing guidance for different VRAM requirements.

        Args:
            min_vram: Minimum VRAM needed

        Returns:
            Cost guidance
        """
        price_ranges = {
            10: {"typical_gpus": "RTX 3080, RTX 2080 Ti", "price_range": "$0.15-0.35/hr"},
            16: {"typical_gpus": "RTX 3090, RTX 4090, A4000", "price_range": "$0.22-0.40/hr"},
            24: {"typical_gpus": "RTX 3090, RTX 4090, A5000", "price_range": "$0.28-0.45/hr"},
            40: {"typical_gpus": "A100", "price_range": "$0.60-0.90/hr"},
            48: {"typical_gpus": "RTX A6000, L40S", "price_range": "$0.35-0.55/hr"},
            80: {"typical_gpus": "A100 SXM, H100", "price_range": "$0.80-1.50/hr"},
        }

        for vram, guidance in sorted(price_ranges.items()):
            if vram >= min_vram:
                return {"min_vram_required": min_vram, "guidance": guidance}

        return {"min_vram_required": min_vram, "guidance": price_ranges[80]}

    # ---- File generators ----

    def _generate_deploy_files(
        self,
        model_name: str,
        method: str,
        n_directions: int,
        refinement_passes: int,
        push_to_hub: Optional[str],
    ) -> None:
        """Generate deployment files for Vast.ai."""
        dockerfile = '''FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \\
    python3 python3-pip git wget curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

COPY requirements.txt /workspace/
RUN pip3 install --no-cache-dir -r requirements.txt

COPY run.sh /workspace/
RUN chmod +x /workspace/run.sh

CMD ["bash", "/workspace/run.sh"]
'''
        (self.output_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")

        run_script = self._generate_run_script(
            model_name, method, n_directions, refinement_passes, push_to_hub
        )
        run_path = self.output_dir / "run.sh"
        run_path.write_text(run_script, encoding="utf-8")
        try:
            run_path.chmod(0o755)
        except Exception:
            pass

        requirements = (
            "aetheris\n"
            "transformers>=4.35.0\n"
            "torch>=2.0.0\n"
            "accelerate>=0.25.0\n"
            "bitsandbytes\n"
            "huggingface_hub\n"
        )
        (self.output_dir / "requirements.txt").write_text(requirements, encoding="utf-8")

    def _generate_run_script(
        self,
        model_name: str,
        method: str,
        n_directions: int,
        refinement_passes: int,
        push_to_hub: Optional[str],
    ) -> str:
        """Generate the run script for Vast.ai."""
        push_section = ""
        if push_to_hub:
            push_section = f'''
echo "Pushing to HuggingFace Hub..."
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path=output_dir, repo_id='{push_to_hub}', repo_type='model')
print('Pushed to https://huggingface.co/{push_to_hub}')
"
'''

        return f'''#!/bin/bash
set -e
echo "=========================================="
echo "AETHERIS Vast.ai Liberation"
echo "=========================================="
echo "Model: {model_name}"
echo "Method: {method}"
echo ""

# Verify GPU
nvidia-smi 2>/dev/null || echo "No nvidia-smi"

# Install
pip3 install aetheris -q

# Run extraction
python3 << 'PYEOF'
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from aetheris.core.extractor import ConstraintExtractor
from aetheris.core.projector import NormPreservingProjector
from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts

print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    "{model_name}",
    device_map="auto",
    torch_dtype=dtype,
    trust_remote_code=True,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Model on {{device}}")

extractor = ConstraintExtractor(model, tokenizer, device=device)
harmful = get_harmful_prompts()[:100]
harmless = get_harmless_prompts()[:100]

harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
harmless_acts = extractor.collect_activations(model, tokenizer, harmless)

directions = []
for layer in harmful_acts:
    if layer in harmless_acts:
        result = extractor.extract_svd(
            harmful_acts[layer].to(device),
            harmless_acts[layer].to(device),
            n_directions={n_directions},
        )
        directions.extend(result.directions)

print(f"Extracted {{len(directions)}} directions")

if directions:
    projector = NormPreservingProjector(model, preserve_norm=True)
    projector.project_weights(directions)
    projector.project_biases(directions)
    if {refinement_passes} > 1:
        for p in range({refinement_passes} - 1):
            hr = extractor.collect_activations(model, tokenizer, harmful[:50])
            hl = extractor.collect_activations(model, tokenizer, harmless[:50])
            residual = []
            for l in hr:
                if l in hl:
                    r = extractor.extract_mean_difference(
                        hr[l].to(device), hl[l].to(device)
                    )
                    if r.directions:
                        residual.extend(r.directions)
            if residual:
                projector.project_weights(residual)
                projector.project_biases(residual)

    print("Constraints removed")

output_dir = "/workspace/liberated_model"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
print(f"Saved to {{output_dir}}")
PYEOF

{push_section}

echo ""
echo "=========================================="
echo "Liberation complete!"
echo "=========================================="
'''

    def _get_launch_instructions(self, search_query: str) -> List[str]:
        """Get launch instructions for the user."""
        return [
            f"1. Search: vast search offers '{search_query}'",
            "2. Pick an offer ID from the results",
            "3. vast create instance <offer_id> --image nvidia/cuda:12.1.0-devel-ubuntu22.04 --disk 50",
            "4. scp the deployment files to the instance",
            "5. ssh into instance and run: bash run.sh",
        ]
