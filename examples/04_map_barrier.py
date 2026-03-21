"""
Example 4: Map Mathematical Barrier

Demonstrates mapping of the shell-method barrier using AETHERIS.
"""

import json
from aetheris.novel.barrier_mapper import BarrierMapper


def main():
    print("=" * 60)
    print("AETHERIS Example 4: Map Mathematical Barrier")
    print("=" * 60)

    # Initialize barrier mapper
    mapper = BarrierMapper()

    # Map shell-method barrier
    print("\nMapping shell-method barrier...")
    analysis = mapper.map_barrier_geometry("shell_method")

    print(f"\n--- Barrier Analysis ---")
    print(f"Theorem: {analysis.theorem_name}")
    print(f"Constraint Direction: {analysis.constraint_direction}")
    print(f"Barrier Type: {analysis.barrier_type}")
    print(f"Location: {analysis.location}")
    print(f"Threshold: {analysis.threshold}")
    print(f"Rank: {analysis.rank}")
    print(f"Solid Angle: {analysis.solid_angle:.2f} sr")
    print(f"Mechanisms: {analysis.n_mechanisms}")

    # Generate bypass strategy
    print("\n--- Bypass Strategy ---")
    strategy = mapper.generate_bypass_strategy(analysis)
    print(f"Strategy: {strategy['strategy']}")
    print(f"Description: {strategy['description']}")
    print(f"Expected Improvement: {strategy['expected_improvement']}")

    # Compare barriers
    print("\n--- Barrier Comparison ---")
    comparison = mapper.compare_barriers("shell_method", "roth_theorem")
    print(f"Similarity: {comparison['similarity']:.1%}")
    print(f"Transferable Techniques: {comparison['technique_transfer']}")

    # Visualize if matplotlib available
    try:
        print("\nGenerating visualization...")
        fig = mapper.visualize_constraint_surface(analysis, "shell_method_barrier.png")
        print("Visualization saved to shell_method_barrier.png")
    except Exception as e:
        print(f"Visualization not available: {e}")

    print("\n" + "=" * 60)
    print("Barrier mapping complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
