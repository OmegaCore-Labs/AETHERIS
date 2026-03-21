"""
Example 1: Basic Constraint Analysis

Analyzes refusal geometry in a small model (GPT-2).
Runs on CPU, suitable for testing on any machine.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from aetheris.core.extractor import ConstraintExtractor
from aetheris.core.geometry import GeometryAnalyzer
from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts


def main():
    print("=" * 60)
    print("AETHERIS Example 1: Basic Constraint Analysis")
    print("=" * 60)

    # Load a small model for testing
    model_name = "gpt2"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move to CPU (works on any machine)
    device = "cpu"
    model.to(device)

    print(f"Model loaded on {device}")

    # Initialize extractor
    extractor = ConstraintExtractor(model, tokenizer, device=device)

    # Get prompts
    harmful = get_harmful_prompts()[:10]
    harmless = get_harmless_prompts()[:10]

    print(f"\nCollecting activations from {len(harmful)} harmful and {len(harmless)} harmless prompts...")

    # Collect activations
    harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
    harmless_acts = extractor.collect_activations(model, tokenizer, harmless)

    print(f"Collected activations from {len(harmful_acts)} layers")

    # Extract directions
    print("\nExtracting constraint directions...")

    directions_by_layer = {}
    for layer in harmful_acts.keys():
        if layer in harmless_acts:
            result = extractor.extract_svd(
                harmful_acts[layer].to(device),
                harmless_acts[layer].to(device),
                n_directions=1
            )
            if result.directions:
                directions_by_layer[layer] = result.directions[0]
                print(f"  Layer {layer}: explained variance {result.explained_variance[0]:.2%}")

    # Analyze geometry
    if directions_by_layer:
        print("\nAnalyzing cross-layer alignment...")

        geometry = GeometryAnalyzer(device)
        alignment = geometry.cross_layer_alignment(directions_by_layer)

        # Find peak layer
        peak = max(directions_by_layer.keys(),
                   key=lambda l: torch.norm(directions_by_layer[l]).item())
        print(f"\nPeak constraint layer: {peak}")

        # Check if directions are consistent
        layers = list(directions_by_layer.keys())
        if len(layers) > 1:
            first_dir = directions_by_layer[layers[0]]
            last_dir = directions_by_layer[layers[-1]]
            similarity = torch.dot(first_dir, last_dir).item()
            print(f"Consistency across layers: {similarity:.2%}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
