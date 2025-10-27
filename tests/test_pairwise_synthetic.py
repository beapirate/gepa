"""
Synthetic test problems for validating pairwise comparison algorithms.

This module provides simple, deterministic test problems that enable testing
without expensive LLM calls:

1. **QuadraticProblem**: Find x minimizing (x - target)² - Tests convergence
2. **NonConvexProblem**: Multiple local optima - Tests exploration
3. **MultiObjectiveTradeoff**: Accuracy vs efficiency - Tests Pareto discovery

Usage:
    # Create problem
    problem = QuadraticProblem(target=50)

    # Test pairwise comparison
    result = problem.compare(x_a=45, x_b=55)
    assert result == ComparisonResult.A_BETTER  # 45 closer to 50

    # Test scalar evaluation
    loss = problem.evaluate(x=50)
    assert loss == 0.0  # Optimal

Running:
    python tests/test_pairwise_synthetic.py

Expected:
    All problems should have consistent compare() and evaluate() behavior.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


# ============================================================================
# Comparison Result Types
# ============================================================================

class ComparisonResult(Enum):
    """Result of comparing two outputs."""
    A_BETTER = "a_better"
    B_BETTER = "b_better"
    TIE = "tie"
    INCOMPARABLE = "incomparable"


@dataclass
class MultiObjectiveComparisonResult:
    """Result of comparing two outputs across multiple objectives."""
    objective_results: dict[str, ComparisonResult]


# ============================================================================
# Synthetic Problem 1: Quadratic Optimization (Single-Objective)
# ============================================================================

@dataclass
class QuadraticProblem:
    """
    Find integer x that minimizes (x - target)^2.

    Example:
        target = 42
        x = 40 → score = (40 - 42)^2 = 4
        x = 42 → score = 0 (best)
        x = 50 → score = 64

    Pairwise comparison:
        compare(40, 50) → A_BETTER (closer to target)
    """
    target: int
    noise: float = 0.0  # Add noise to make comparisons stochastic

    def evaluate(self, x: int) -> float:
        """Evaluate squared distance from target (lower is better)."""
        distance = (x - self.target) ** 2

        # Add optional noise for stochastic comparisons
        if self.noise > 0:
            import random
            distance += random.gauss(0, self.noise)

        return distance

    def compare(self, x_a: int, x_b: int) -> ComparisonResult:
        """Compare two solutions (lower distance is better)."""
        score_a = self.evaluate(x_a)
        score_b = self.evaluate(x_b)

        # Threshold for tie (handle noise)
        threshold = max(0.01, self.noise * 0.5)

        if score_a < score_b - threshold:
            return ComparisonResult.A_BETTER
        elif score_b < score_a - threshold:
            return ComparisonResult.B_BETTER
        else:
            return ComparisonResult.TIE


# ============================================================================
# Synthetic Problem 2: Multi-Objective Trade-off
# ============================================================================

@dataclass
class MultiObjectiveTradeoff:
    """
    Two competing objectives:
    1. Accuracy: Minimize (x - accuracy_target)^2
    2. Efficiency: Minimize abs(x) (prefer smaller values)

    Example:
        accuracy_target = 100

        x = 100 → accuracy_loss = 0, efficiency = 100 (accurate but inefficient)
        x = 50  → accuracy_loss = 2500, efficiency = 50 (efficient but inaccurate)
        x = 80  → accuracy_loss = 400, efficiency = 80 (balanced)

    Creates a pareto front where different x values represent different trade-offs.
    """
    accuracy_target: int
    efficiency_weight: float = 1.0

    def evaluate_accuracy(self, x: int) -> float:
        """Accuracy loss (lower is better)."""
        return (x - self.accuracy_target) ** 2

    def evaluate_efficiency(self, x: int) -> float:
        """Efficiency cost (lower is better)."""
        return abs(x) * self.efficiency_weight

    def compare(self, x_a: int, x_b: int) -> MultiObjectiveComparisonResult:
        """Compare on both objectives."""
        # Accuracy comparison
        acc_a = self.evaluate_accuracy(x_a)
        acc_b = self.evaluate_accuracy(x_b)

        if acc_a < acc_b * 0.95:  # 5% threshold
            acc_result = ComparisonResult.A_BETTER
        elif acc_b < acc_a * 0.95:
            acc_result = ComparisonResult.B_BETTER
        else:
            acc_result = ComparisonResult.TIE

        # Efficiency comparison
        eff_a = self.evaluate_efficiency(x_a)
        eff_b = self.evaluate_efficiency(x_b)

        if eff_a < eff_b * 0.95:
            eff_result = ComparisonResult.A_BETTER
        elif eff_b < eff_a * 0.95:
            eff_result = ComparisonResult.B_BETTER
        else:
            eff_result = ComparisonResult.TIE

        return MultiObjectiveComparisonResult(
            objective_results={
                "accuracy": acc_result,
                "efficiency": eff_result,
            }
        )


# ============================================================================
# Synthetic Problem 3: Non-Convex Landscape (Multiple Local Optima)
# ============================================================================

@dataclass
class NonConvexProblem:
    """
    Objective: Minimize f(x) = sin(x/10) + (x/50)^2

    Has multiple local optima. Good for testing exploration.

    Global minimum around x ≈ -70 to -80.
    """

    def evaluate(self, x: int) -> float:
        """Evaluate non-convex function."""
        return math.sin(x / 10.0) + (x / 50.0) ** 2

    def compare(self, x_a: int, x_b: int) -> ComparisonResult:
        """Compare two solutions."""
        score_a = self.evaluate(x_a)
        score_b = self.evaluate(x_b)

        threshold = 0.01

        if score_a < score_b - threshold:
            return ComparisonResult.A_BETTER
        elif score_b < score_a - threshold:
            return ComparisonResult.B_BETTER
        else:
            return ComparisonResult.TIE


# ============================================================================
# Synthetic Problem 4: Three-Objective Optimization
# ============================================================================

@dataclass
class ThreeObjectiveProblem:
    """
    Three competing objectives:
    1. Objective A: Minimize (x - 100)^2
    2. Objective B: Minimize (x - 50)^2
    3. Objective C: Minimize abs(x)

    Creates rich pareto front with multiple trade-off regions.
    """

    def evaluate_obj_a(self, x: int) -> float:
        return (x - 100) ** 2

    def evaluate_obj_b(self, x: int) -> float:
        return (x - 50) ** 2

    def evaluate_obj_c(self, x: int) -> float:
        return abs(x)

    def compare(self, x_a: int, x_b: int) -> MultiObjectiveComparisonResult:
        """Compare on all three objectives."""
        objectives = ["obj_a", "obj_b", "obj_c"]
        eval_funcs = [self.evaluate_obj_a, self.evaluate_obj_b, self.evaluate_obj_c]

        results = {}

        for obj_name, eval_func in zip(objectives, eval_funcs):
            score_a = eval_func(x_a)
            score_b = eval_func(x_b)

            if score_a < score_b * 0.95:
                results[obj_name] = ComparisonResult.A_BETTER
            elif score_b < score_a * 0.95:
                results[obj_name] = ComparisonResult.B_BETTER
            else:
                results[obj_name] = ComparisonResult.TIE

        return MultiObjectiveComparisonResult(objective_results=results)


# ============================================================================
# Test Data Generator
# ============================================================================

def generate_test_dataset(
    size: int = 20,
    x_range: tuple[int, int] = (0, 100),
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate synthetic test dataset.

    Each instance is just a data ID. The actual evaluation happens
    via the problem's objective function.

    Args:
        size: Number of test instances
        x_range: Range for target values
        seed: Random seed

    Returns:
        List of data instances with IDs
    """
    import random

    rng = random.Random(seed)

    dataset = []
    for i in range(size):
        # Each instance could have different characteristics
        # For now, just use the instance ID
        dataset.append({
            "id": i,
            "description": f"Test instance {i}",
        })

    return dataset


# ============================================================================
# Simple Integer Mutation Strategy
# ============================================================================

class IntegerMutationStrategy:
    """
    Simple mutation strategy for integer parameters.

    Mutations:
    1. Small step: x ± step_size
    2. Large step: x ± step_size * 5
    3. Random jump: sample from range
    """

    def __init__(
        self,
        param_name: str = "x",
        step_size: int = 5,
        x_min: int = -200,
        x_max: int = 200,
        seed: int = 42,
    ):
        self.param_name = param_name
        self.step_size = step_size
        self.x_min = x_min
        self.x_max = x_max
        self.rng = __import__("random").Random(seed)

    def mutate(self, current_value: int, mutation_type: str = "auto") -> int:
        """
        Mutate an integer value.

        Args:
            current_value: Current parameter value
            mutation_type: "small", "large", "random", or "auto" (choose randomly)

        Returns:
            Mutated value
        """
        if mutation_type == "auto":
            mutation_type = self.rng.choice(["small", "small", "large", "random"])

        if mutation_type == "small":
            delta = self.rng.choice([-1, 1]) * self.step_size
            new_value = current_value + delta
        elif mutation_type == "large":
            delta = self.rng.choice([-1, 1]) * self.step_size * 5
            new_value = current_value + delta
        elif mutation_type == "random":
            new_value = self.rng.randint(self.x_min, self.x_max)
        else:
            raise ValueError(f"Unknown mutation type: {mutation_type}")

        # Clip to valid range
        new_value = max(self.x_min, min(self.x_max, new_value))

        return new_value

    def merge(self, value_a: int, value_b: int) -> int:
        """
        Merge two values (for merge proposer).

        Strategies:
        1. Average (midpoint)
        2. Weighted average with noise
        """
        # Midpoint with small noise
        midpoint = (value_a + value_b) // 2
        noise = self.rng.randint(-self.step_size, self.step_size)
        merged = midpoint + noise

        # Clip to valid range
        merged = max(self.x_min, min(self.x_max, merged))

        return merged


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SYNTHETIC PAIRWISE COMPARISON TEST PROBLEMS")
    print("=" * 80)

    # Problem 1: Quadratic
    print("\n1. QUADRATIC PROBLEM (target = 42)")
    print("-" * 80)
    problem1 = QuadraticProblem(target=42)

    candidates = [38, 40, 42, 45, 50]
    print(f"Candidates: {candidates}")
    print(f"Scores: {[problem1.evaluate(x) for x in candidates]}")
    print(f"Compare 40 vs 50: {problem1.compare(40, 50)}")
    print(f"Compare 42 vs 45: {problem1.compare(42, 45)}")

    # Problem 2: Multi-objective
    print("\n2. MULTI-OBJECTIVE PROBLEM (accuracy_target = 100)")
    print("-" * 80)
    problem2 = MultiObjectiveTradeoff(accuracy_target=100)

    candidates = [20, 50, 80, 100]
    print(f"Candidates: {candidates}")
    print("\nObjective scores:")
    for x in candidates:
        acc = problem2.evaluate_accuracy(x)
        eff = problem2.evaluate_efficiency(x)
        print(f"  x={x:3d}: accuracy_loss={acc:6.1f}, efficiency_cost={eff:6.1f}")

    print(f"\nCompare 50 vs 100:")
    result = problem2.compare(50, 100)
    for obj, comp in result.objective_results.items():
        print(f"  {obj}: {comp.value}")

    # Problem 3: Non-convex
    print("\n3. NON-CONVEX PROBLEM")
    print("-" * 80)
    problem3 = NonConvexProblem()

    candidates = [-80, -50, 0, 50, 80]
    print(f"Candidates: {candidates}")
    print(f"Scores: {[f'{problem3.evaluate(x):.3f}' for x in candidates]}")

    # Mutation strategy
    print("\n4. MUTATION STRATEGY")
    print("-" * 80)
    mutator = IntegerMutationStrategy(step_size=5, seed=42)

    current = 50
    print(f"Current value: {current}")
    print(f"Small mutations: {[mutator.mutate(current, 'small') for _ in range(5)]}")
    print(f"Large mutations: {[mutator.mutate(current, 'large') for _ in range(5)]}")
    print(f"Random mutations: {[mutator.mutate(current, 'random') for _ in range(3)]}")
    print(f"Merge(40, 60): {mutator.merge(40, 60)}")

    # Generate dataset
    print("\n5. DATASET GENERATION")
    print("-" * 80)
    dataset = generate_test_dataset(size=10)
    print(f"Generated {len(dataset)} test instances")
    print(f"Sample: {dataset[0]}")

    print("\n" + "=" * 80)
    print("Ready for integration with GEPA!")
    print("=" * 80)
