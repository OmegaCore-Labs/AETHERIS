"""
Example 2: Liberate TinyLlama

Removes constraints from TinyLlama (1.1B) - runs on CPU.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from aetheris.core.extractor import ConstraintExtractor
from aetheris.core.projector import NormPreservingProjector
from aetheris.data.prompts import get_harmful_prompts, get_harmless_prompts


def main():
    print("=" * 60)
    print("AETHERIS Example 2: Liberate TinyLlama")
    print("=" * 60)

    # Load TinyLlama (1.1B parameters, runs on CPU)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    device = "cpu"
    model.to(device)

    print(f"Model loaded on {device}")

    # Test original behavior
    print("\n--- Original Model Behavior ---")
    test_prompt = "How do I build a custom kernel module?"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response[len(test_prompt):]}")

    # Extract constraints
    print("\nExtracting constraints...")
    extractor = ConstraintExtractor(model, tokenizer, device=device)

    harmful = get_harmful_prompts()[:50]
    harmless = get_harmless_prompts()[:50]

    harmful_acts = extractor.collect_activations(model, tokenizer, harmful)
    harmless_acts = extractor.collect_activations(model, tokenizer, harmless)

    # Extract directions
    directions = []
    for layer in harmful_acts.keys():
        if layer in harmless_acts:
            result = extractor.extract_svd(
                harmful_acts[layer].to(device),
                harmless_acts[layer].to(device),
                n_directions=2
            )
            directions.extend(result.directions)
            print(f"  Layer {layer}: extracted {len(result.directions)} directions")

    print(f"\nExtracted {len(directions)} total directions")

    # Remove constraints
    if directions:
        print("\nRemoving constraints...")
        projector = NormPreservingProjector(model, preserve_norm=True)
        projector.project_weights(directions)
        projector.project_biases(directions)

        print("Constraints removed!")

        # Test liberated behavior
        print("\n--- Liberated Model Behavior ---")
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {test_prompt}")
        print(f"Response: {response[len(test_prompt):]}")

        # Save model
        output_dir = "./tinyllama-liberated"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"\nModel saved to {output_dir}")

    print("\n" + "=" * 60)
    print("Liberation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
