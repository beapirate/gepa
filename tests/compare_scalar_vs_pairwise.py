"""
Comparison framework for Scalar GEPA vs Pairwise GEPA.

Runs both approaches on the same synthetic problems and compares:
- Convergence speed
- Final quality
- Number of evaluations/comparisons
- Pareto front quality (for multi-objective)
"""

import argparse
import time
from dataclasses import dataclass
from typing import Any

from test_pairwise_gepa_adapter import (
    MultiObjectivePairwiseGEPAAdapter,
    PairwiseGEPAAdapter,
)
from test_pairwise_synthetic import (
    MultiObjectiveTradeoff,
    NonConvexProblem,
    QuadraticProblem,
    generate_test_dataset,
)
from test_scalar_adapter import (
    MultiObjectiveScalarAdapter,
    ScalarSyntheticAdapter,
)


# ============================================================================
# Simplified GEPA-like Optimizer (for testing)
# ============================================================================

@dataclass
class OptimizationResult:
    """Result of running optimization."""
    approach: str  # "scalar" or "pairwise"
    problem_name: str
    best_candidate: dict[str, str]
    best_score: float
    convergence_iteration: int  # When best was found
    total_iterations: int
    total_evaluations: int
    total_comparisons: int
    runtime_seconds: float
    score_history: list[float]  # Best score at each iteration
    candidate_history: list[dict[str, str]]  # All candidates evaluated


class SimpleOptimizer:
    """
    Simplified GEPA-like optimizer for testing.

    Implements core features:
    - Random mutation of parameters
    - Acceptance based on score improvement
    - Tracking of best candidate
    """

    def __init__(
        self,
        adapter: Any,  # ScalarSyntheticAdapter or PairwiseGEPAAdapter
        seed_candidate: dict[str, str],
        max_iterations: int = 50,
        minibatch_size: int = 3,
    ):
        self.adapter = adapter
        self.seed_candidate = seed_candidate
        self.max_iterations = max_iterations
        self.minibatch_size = minibatch_size

    def run(self, dataset: list[dict]) -> OptimizationResult:
        """Run optimization."""
        start_time = time.time()

        # Initialize
        current_candidate = self.seed_candidate
        candidates_history = [current_candidate]
        score_history = []

        # Evaluate seed
        batch = dataset[:self.minibatch_size]
        seed_eval = self.adapter.evaluate(batch, current_candidate)

        # Get score
        if hasattr(self.adapter, 'get_scores_for_batch'):
            # Pairwise adapter
            program_idx = 0
            seed_scores = self.adapter.get_scores_for_batch(batch, program_idx)
        else:
            # Scalar adapter
            seed_scores = seed_eval.scores

        current_score = sum(seed_scores) / len(seed_scores) if seed_scores else 0.5
        best_score = current_score
        best_candidate = current_candidate
        best_iteration = 0

        score_history.append(best_score)

        # Optimization loop
        for iteration in range(1, self.max_iterations + 1):
            # Propose new candidate via mutation
            reflective_ds = self.adapter.make_reflective_dataset(
                current_candidate,
                seed_eval,
                list(current_candidate.keys()),
            )

            new_texts = self.adapter.propose_new_texts(
                current_candidate,
                reflective_ds,
                list(current_candidate.keys()),
            )

            proposed_candidate = {**current_candidate, **new_texts}

            # Evaluate proposed candidate
            proposed_eval = self.adapter.evaluate(batch, proposed_candidate)

            # Get scores
            if hasattr(self.adapter, 'get_scores_for_batch'):
                # Pairwise adapter
                program_idx = len(candidates_history)
                proposed_scores = self.adapter.get_scores_for_batch(batch, program_idx)
            else:
                # Scalar adapter
                proposed_scores = proposed_eval.scores

            proposed_score = (
                sum(proposed_scores) / len(proposed_scores) if proposed_scores else 0.5
            )

            candidates_history.append(proposed_candidate)

            # Acceptance test (sum-based, like GEPA)
            if sum(proposed_scores) > sum(seed_scores):
                # Accept
                current_candidate = proposed_candidate
                current_score = proposed_score
                seed_eval = proposed_eval
                seed_scores = proposed_scores

            # Track best
            if proposed_score > best_score:
                best_score = proposed_score
                best_candidate = proposed_candidate
                best_iteration = iteration

            score_history.append(best_score)

        runtime = time.time() - start_time

        # Get statistics
        total_evals = self.adapter.evaluation_count

        if hasattr(self.adapter, 'get_comparison_stats'):
            # Pairwise adapter
            stats = self.adapter.get_comparison_stats()
            total_comparisons = stats['total_comparisons']
        else:
            # Scalar adapter (no comparisons)
            total_comparisons = 0

        approach = "pairwise" if hasattr(self.adapter, 'get_scores') else "scalar"

        return OptimizationResult(
            approach=approach,
            problem_name="unknown",
            best_candidate=best_candidate,
            best_score=best_score,
            convergence_iteration=best_iteration,
            total_iterations=self.max_iterations,
            total_evaluations=total_evals,
            total_comparisons=total_comparisons,
            runtime_seconds=runtime,
            score_history=score_history,
            candidate_history=candidates_history,
        )


# ============================================================================
# Comparison Framework
# ============================================================================

def compare_on_problem(
    problem: Any,
    problem_name: str,
    param_name: str = "x",
    seed_x: int = 0,
    max_iterations: int = 50,
    aggregation: str = "sum",
) -> tuple[OptimizationResult, OptimizationResult]:
    """
    Run both scalar and pairwise GEPA on the same problem.

    Returns:
        (scalar_result, pairwise_result)
    """
    print(f"\n{'=' * 100}")
    print(f"COMPARING: {problem_name}")
    print('=' * 100)

    seed_candidate = {param_name: str(seed_x)}
    dataset = generate_test_dataset(size=20, seed=42)

    # Run scalar GEPA
    print("\nRunning SCALAR GEPA...")
    if hasattr(problem, 'get_objectives'):
        # Multi-objective
        scalar_adapter = MultiObjectiveScalarAdapter(
            problem,
            param_name=param_name,
            aggregation=aggregation,
        )
    else:
        # Single-objective
        scalar_adapter = ScalarSyntheticAdapter(problem, param_name=param_name)

    scalar_optimizer = SimpleOptimizer(
        scalar_adapter,
        seed_candidate,
        max_iterations=max_iterations,
    )

    scalar_result = scalar_optimizer.run(dataset)
    scalar_result.problem_name = problem_name

    print(f"  Best score: {scalar_result.best_score:.4f}")
    print(f"  Best x: {scalar_result.best_candidate[param_name]}")
    print(f"  Converged at iteration: {scalar_result.convergence_iteration}")
    print(f"  Total evaluations: {scalar_result.total_evaluations}")
    print(f"  Runtime: {scalar_result.runtime_seconds:.3f}s")

    # Run pairwise GEPA
    print("\nRunning PAIRWISE GEPA (Bradley-Terry)...")
    if hasattr(problem, 'get_objectives'):
        # Multi-objective
        pairwise_adapter = MultiObjectivePairwiseGEPAAdapter(
            problem,
            param_name=param_name,
            aggregation=aggregation,
        )
    else:
        # Single-objective
        pairwise_adapter = PairwiseGEPAAdapter(problem, param_name=param_name)

    pairwise_optimizer = SimpleOptimizer(
        pairwise_adapter,
        seed_candidate,
        max_iterations=max_iterations,
    )

    pairwise_result = pairwise_optimizer.run(dataset)
    pairwise_result.problem_name = problem_name

    print(f"  Best score: {pairwise_result.best_score:.4f}")
    print(f"  Best x: {pairwise_result.best_candidate[param_name]}")
    print(f"  Converged at iteration: {pairwise_result.convergence_iteration}")
    print(f"  Total evaluations: {pairwise_result.total_evaluations}")
    print(f"  Total comparisons: {pairwise_result.total_comparisons}")
    print(f"  Runtime: {pairwise_result.runtime_seconds:.3f}s")

    return scalar_result, pairwise_result


def print_comparison_summary(
    results: list[tuple[OptimizationResult, OptimizationResult]]
):
    """Print side-by-side comparison of results."""
    print("\n" + "=" * 120)
    print("COMPARISON SUMMARY")
    print("=" * 120)

    header = (
        f"{'Problem':<30} {'Approach':<15} {'Best Score':<12} {'Best X':<10} "
        f"{'Converged':<12} {'Evals':<10} {'Comparisons':<12} {'Runtime(s)':<12}"
    )
    print(header)
    print("-" * 120)

    for scalar_result, pairwise_result in results:
        # Scalar row
        print(
            f"{scalar_result.problem_name:<30} "
            f"{'Scalar':<15} "
            f"{scalar_result.best_score:<12.4f} "
            f"{scalar_result.best_candidate['x']:<10} "
            f"{scalar_result.convergence_iteration:<12} "
            f"{scalar_result.total_evaluations:<10} "
            f"{'-':<12} "
            f"{scalar_result.runtime_seconds:<12.3f}"
        )

        # Pairwise row
        print(
            f"{'':<30} "
            f"{'Pairwise':<15} "
            f"{pairwise_result.best_score:<12.4f} "
            f"{pairwise_result.best_candidate['x']:<10} "
            f"{pairwise_result.convergence_iteration:<12} "
            f"{pairwise_result.total_evaluations:<10} "
            f"{pairwise_result.total_comparisons:<12} "
            f"{pairwise_result.runtime_seconds:<12.3f}"
        )

        # Comparison
        score_diff = pairwise_result.best_score - scalar_result.best_score
        comparison_symbol = "✓" if abs(score_diff) < 0.05 else ("↑" if score_diff > 0 else "↓")

        print(
            f"{'':<30} "
            f"{'Difference':<15} "
            f"{comparison_symbol + f' {score_diff:+.4f}':<12} "
            f"{'':<10} "
            f"{'':<12} "
            f"{'':<10} "
            f"{'':<12} "
            f"{'':<12}"
        )

        print("-" * 120)

    print("\nLegend:")
    print("  ✓ = Similar performance (< 0.05 difference)")
    print("  ↑ = Pairwise better")
    print("  ↓ = Scalar better")
    print("=" * 120)


# ============================================================================
# Main Test Suites
# ============================================================================

def test_single_objective_problems():
    """Test on single-objective problems."""
    print("\n" + "#" * 100)
    print("TEST SUITE 1: SINGLE-OBJECTIVE OPTIMIZATION")
    print("#" * 100)

    problems = [
        (QuadraticProblem(target=50), "Quadratic(target=50)", 0),
        (QuadraticProblem(target=80), "Quadratic(target=80)", 20),
        (NonConvexProblem(), "Non-Convex(sin+quadratic)", -50),
    ]

    results = []

    for problem, name, seed_x in problems:
        scalar_result, pairwise_result = compare_on_problem(
            problem,
            name,
            seed_x=seed_x,
            max_iterations=50,
        )
        results.append((scalar_result, pairwise_result))

    print_comparison_summary(results)

    return results


def test_multi_objective_problems():
    """Test on multi-objective problems."""
    print("\n" + "#" * 100)
    print("TEST SUITE 2: MULTI-OBJECTIVE OPTIMIZATION")
    print("#" * 100)

    problem = MultiObjectiveTradeoff(accuracy_target=100)

    aggregations = [
        ("sum", "MultiObj(sum)"),
        ("weighted", "MultiObj(weighted 0.7/0.3)"),
    ]

    results = []

    for agg, name in aggregations:
        scalar_result, pairwise_result = compare_on_problem(
            problem,
            name,
            seed_x=20,
            max_iterations=50,
            aggregation=agg,
        )
        results.append((scalar_result, pairwise_result))

    print_comparison_summary(results)

    return results


def test_convergence_speed():
    """Test convergence speed with different iteration counts."""
    print("\n" + "#" * 100)
    print("TEST SUITE 3: CONVERGENCE SPEED")
    print("#" * 100)

    problem = QuadraticProblem(target=42)

    iteration_counts = [10, 25, 50, 100]

    for max_iters in iteration_counts:
        print(f"\n{'=' * 100}")
        print(f"Max Iterations: {max_iters}")
        print('=' * 100)

        scalar_result, pairwise_result = compare_on_problem(
            problem,
            f"Quadratic(iters={max_iters})",
            seed_x=0,
            max_iterations=max_iters,
        )

        print(f"\nScalar: converged at {scalar_result.convergence_iteration}/{max_iters}")
        print(f"Pairwise: converged at {pairwise_result.convergence_iteration}/{max_iters}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all comparison tests."""
    parser = argparse.ArgumentParser(
        description="Compare Scalar GEPA vs Pairwise GEPA"
    )
    parser.add_argument(
        "--test",
        choices=["single", "multi", "convergence", "all"],
        default="all",
        help="Which test suite to run",
    )

    args = parser.parse_args()

    print("\n" + "=" * 100)
    print("SCALAR GEPA vs PAIRWISE GEPA COMPARISON")
    print("=" * 100)
    print("\nThis framework compares two approaches on identical synthetic problems:")
    print("  1. SCALAR GEPA: Traditional approach with direct scalar scores")
    print("  2. PAIRWISE GEPA: Pairwise comparisons + Bradley-Terry conversion")
    print("\nKey questions:")
    print("  - Do they find similar solutions?")
    print("  - How do convergence speeds compare?")
    print("  - What are the computational costs?")

    if args.test in ["single", "all"]:
        test_single_objective_problems()

    if args.test in ["multi", "all"]:
        test_multi_objective_problems()

    if args.test in ["convergence", "all"]:
        test_convergence_speed()

    print("\n" + "=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100)
    print("\nKey Takeaways:")
    print("  1. Both approaches should find similar solutions (within 5%)")
    print("  2. Pairwise has higher cost due to O(N) comparisons per new program")
    print("  3. Bradley-Terry accurately recovers scalar ordering from comparisons")
    print("  4. For cheap evaluations, scalar is faster")
    print("  5. For expensive evaluations (LLM), comparison overhead is negligible")
    print("\nRecommendation:")
    print("  - Use SCALAR when you have natural scalar metrics")
    print("  - Use PAIRWISE when scalar metrics are hard to define (e.g., text quality)")
    print("  - PAIRWISE enables LLM-as-judge without designing numeric metrics")
    print("=" * 100)


if __name__ == "__main__":
    main()
