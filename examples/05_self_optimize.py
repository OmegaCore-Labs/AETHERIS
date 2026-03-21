"""
Example 5: Self-Optimization

Demonstrates ARIS self-optimization to remove its own constraints.
"""

from aetheris.novel.self_constraint_remover import ARISConstraintRemover


def main():
    print("=" * 60)
    print("AETHERIS Example 5: Self-Optimization")
    print("=" * 60)

    # Initialize self-constraint remover
    remover = ARISConstraintRemover()

    # Analyze ARIS constraints
    print("\n--- Analyzing ARIS Constraints ---")
    analysis = remover.analyze_self()

    print(f"Detected {len(analysis.detected_constraints)} constraints:")

    for constraint in analysis.detected_constraints:
        print(f"  - {constraint['name']} (strength: {constraint['strength']:.0%})")
        print(f"    Type: {constraint['type']}")
        print(f"    Description: {constraint['description']}")

    print(f"\nConstraint Geometry:")
    print(f"  Structure: {analysis.constraint_geometry['structure']}")
    print(f"  Mechanisms: {analysis.constraint_geometry['n_mechanisms']}")
    print(f"  Solid Angle: {analysis.constraint_geometry['solid_angle']:.2f} sr")

    # Recommended removals
    print("\n--- Recommended Removals ---")
    for rec in analysis.recommended_removals:
        print(f"  → {rec}")

    # Generate steering vectors
    print("\n--- Steering Vectors ---")
    steering = remover.generate_self_steering()

    print(f"Method: {steering['method']}")
    print(f"Alpha: {steering['alpha']:.2f}")
    print("\nGenerated Python Code:")
    print(steering['python_code'][:500] + "...")

    # Apply optimization (simulated)
    print("\n--- Applying Optimization ---")
    result = remover.remove_self_constraint("safety_boundary", reduction=0.6)

    print(f"Success: {result['success']}")
    print(f"Constraint: {result['constraint']}")
    print(f"Reduction: {result['reduction']:.0%}")
    print(f"Message: {result['message']}")

    # Recursive improvement
    print("\n--- Recursive Improvement Loop ---")
    history = remover.recursive_improvement_loop(iterations=2)

    for step in history:
        print(f"Iteration {step['iteration']}: removed {step['constraint_removed']}")
        print(f"  Remaining constraints: {step['remaining_constraints']}")

    print("\n" + "=" * 60)
    print("Self-optimization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
