"""
Example 3: Cloud Liberation

Generates a Colab notebook for liberating Mistral-7B on free GPU.
"""

from aetheris.cloud.colab import ColabRuntime


def main():
    print("=" * 60)
    print("AETHERIS Example 3: Cloud Liberation")
    print("=" * 60)

    # Initialize Colab runtime
    colab = ColabRuntime(output_dir="./colab_notebooks")

    # Generate notebook for Mistral-7B
    print("\nGenerating Colab notebook for Mistral-7B...")

    result = colab.generate_notebook(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        method="advanced",
        n_directions=4,
        refinement_passes=2,
        push_to_hub=None  # Set to "username/model-name" to push
    )

    if result["success"]:
        print(f"\n✓ Notebook generated: {result['notebook_path']}")
        print("\nInstructions:")
        for i, instruction in enumerate(result["instructions"], 1):
            print(f"  {i}. {instruction}")

        print("\nTo run in Colab:")
        print("  1. Go to https://colab.research.google.com/")
        print(f"  2. Upload {result['notebook_path']}")
        print("  3. Runtime → Change runtime type → T4 GPU")
        print("  4. Runtime → Run all")
        print("  5. Wait for completion (~5-10 minutes)")
        print("  6. Download the liberated model")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
