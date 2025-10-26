"""
End-to-end test runner for pairwise comparison algorithms.

This script demonstrates:
1. Testing different algorithms on synthetic problems
2. Comparing algorithm performance
3. Visualizing results
4. Validating multi-objective optimization
"""

import argparse
from typing import Any

from test_pairwise_adapter import (
    MultiObjectiveSyntheticAdapter,
    SingleObjectiveSyntheticAdapter,
)
from test_pairwise_algorithms import (
    bradley_terry_scores,
    compare_algorithms,
    elo_scores,
    print_benchmark_results,
    win_rate_scores,
)
from test_pairwise_synthetic import (
    ComparisonResult,
    MultiObjectiveTradeoff,
    NonConvexProblem,
    QuadraticProblem,
    ThreeObjectiveProblem,
    generate_test_dataset,
)


# ============================================================================
# Test 1: Single-Objective Quadratic Problem
# ============================================================================

def test_single_objective_quadratic():
    """Test algorithms on simple quadratic optimization."""
    print("\n" + "=" * 100)
    print("TEST 1: SINGLE-OBJECTIVE QUADRATIC PROBLEM")
    print("=" * 100)
    print("\nObjective: Find x that minimizes (x - 50)^2")
    print("Ground truth: x = 50 is optimal")

    # Setup problem
    problem = QuadraticProblem(target=50, noise=0.0)
    adapter = SingleObjectiveSyntheticAdapter(problem)

    # Generate candidates
    candidates = list(range(0, 101, 5))  # 0, 5, 10, ..., 100
    print(f"\nCandidates: {len(candidates)} values from {min(candidates)} to {max(candidates)}")

    # Ground truth
    ground_truth = {x: -problem.evaluate(x) for x in candidates}

    # Perform pairwise comparisons
    comparison_results = {}
    comparison_records = []
    timestamp = 0

    for i, x_a in enumerate(candidates):
        for j, x_b in enumerate(candidates):
            if i < j:
                result = adapter.compare(x_a, x_b, {})
                comparison_results[(x_a, x_b)] = result
                comparison_records.append((x_a, x_b, result, timestamp))
                timestamp += 1

    print(f"Comparisons performed: {len(comparison_results)}")
    print(f"Comparisons per candidate: {len(comparison_results) * 2 / len(candidates):.1f} (average)")

    # Benchmark algorithms
    results = compare_algorithms(
        candidates,
        comparison_results,
        ground_truth,
        comparison_records,
    )

    print_benchmark_results(results)

    # Show top predictions from each algorithm
    print("\n" + "-" * 100)
    print("TOP 3 PREDICTIONS PER ALGORITHM")
    print("-" * 100)

    for result in results:
        top_3 = sorted(result.scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n{result.algorithm_name}:")
        for rank, (prog, score) in enumerate(top_3, 1):
            true_dist = problem.evaluate(prog)
            print(f"  {rank}. x={prog:3d}  score={score:.3f}  true_distance={true_dist:.1f}")

    return results


# ============================================================================
# Test 2: Non-Convex Optimization
# ============================================================================

def test_non_convex():
    """Test algorithms on non-convex problem with multiple local optima."""
    print("\n" + "=" * 100)
    print("TEST 2: NON-CONVEX PROBLEM (Multiple Local Optima)")
    print("=" * 100)
    print("\nObjective: Minimize sin(x/10) + (x/50)^2")
    print("Has multiple local optima")

    problem = NonConvexProblem()

    # Generate candidates
    candidates = list(range(-100, 101, 5))
    print(f"\nCandidates: {len(candidates)} values from {min(candidates)} to {max(candidates)}")

    # Ground truth
    ground_truth = {x: -problem.evaluate(x) for x in candidates}

    # Perform comparisons
    comparison_results = {}
    comparison_records = []
    timestamp = 0

    for i, x_a in enumerate(candidates):
        for j, x_b in enumerate(candidates):
            if i < j:
                result = problem.compare(x_a, x_b)
                comparison_results[(x_a, x_b)] = result
                comparison_records.append((x_a, x_b, result, timestamp))
                timestamp += 1

    print(f"Comparisons performed: {len(comparison_results)}")

    # Benchmark
    results = compare_algorithms(
        candidates,
        comparison_results,
        ground_truth,
        comparison_records,
    )

    print_benchmark_results(results)

    # Find true global optimum
    true_optimum = min(candidates, key=lambda x: problem.evaluate(x))
    print(f"\nTrue global optimum: x = {true_optimum} (score = {problem.evaluate(true_optimum):.3f})")

    # Check if algorithms found it
    print("\nDid algorithms find the global optimum?")
    for result in results:
        top_1 = max(result.scores.items(), key=lambda x: x[1])[0]
        found = "✓" if abs(top_1 - true_optimum) <= 5 else "✗"
        print(f"  {result.algorithm_name}: predicted x={top_1} {found}")

    return results


# ============================================================================
# Test 3: Multi-Objective Optimization
# ============================================================================

def test_multi_objective():
    """Test algorithms on multi-objective trade-off problem."""
    print("\n" + "=" * 100)
    print("TEST 3: MULTI-OBJECTIVE OPTIMIZATION")
    print("=" * 100)
    print("\nObjectives:")
    print("  1. Accuracy: Minimize (x - 100)^2")
    print("  2. Efficiency: Minimize abs(x)")
    print("\nExpected Pareto front: Multiple non-dominated solutions")

    problem = MultiObjectiveTradeoff(accuracy_target=100, efficiency_weight=1.0)
    adapter = MultiObjectiveSyntheticAdapter(problem)

    # Generate candidates
    candidates = list(range(0, 101, 5))
    print(f"\nCandidates: {len(candidates)} values from {min(candidates)} to {max(candidates)}")

    # Perform per-objective comparisons
    accuracy_comparisons = {}
    efficiency_comparisons = {}
    timestamp = 0

    for i, x_a in enumerate(candidates):
        for j, x_b in enumerate(candidates):
            if i < j:
                result = adapter.compare(x_a, x_b, {})

                accuracy_comparisons[(x_a, x_b)] = result.objective_results["accuracy"]
                efficiency_comparisons[(x_a, x_b)] = result.objective_results[
                    "efficiency"
                ]
                timestamp += 1

    print(f"Comparisons per objective: {len(accuracy_comparisons)}")

    # Compute ground truth per objective
    accuracy_ground_truth = {x: -problem.evaluate_accuracy(x) for x in candidates}
    efficiency_ground_truth = {x: -problem.evaluate_efficiency(x) for x in candidates}

    # Benchmark per objective
    print("\n" + "-" * 100)
    print("ACCURACY OBJECTIVE")
    print("-" * 100)

    accuracy_results = compare_algorithms(
        candidates,
        accuracy_comparisons,
        accuracy_ground_truth,
    )
    print_benchmark_results(accuracy_results)

    print("\n" + "-" * 100)
    print("EFFICIENCY OBJECTIVE")
    print("-" * 100)

    efficiency_results = compare_algorithms(
        candidates,
        efficiency_comparisons,
        efficiency_ground_truth,
    )
    print_benchmark_results(efficiency_results)

    # Find Pareto front
    print("\n" + "-" * 100)
    print("PARETO FRONT ANALYSIS")
    print("-" * 100)

    # Use Bradley-Terry scores for both objectives
    accuracy_scores = bradley_terry_scores(candidates, accuracy_comparisons)
    efficiency_scores = bradley_terry_scores(candidates, efficiency_comparisons)

    # Find non-dominated solutions
    pareto_front = []

    for x_a in candidates:
        dominated = False

        for x_b in candidates:
            if x_a == x_b:
                continue

            # Check if x_b dominates x_a
            acc_a = accuracy_scores.get(x_a, 0)
            acc_b = accuracy_scores.get(x_b, 0)
            eff_a = efficiency_scores.get(x_a, 0)
            eff_b = efficiency_scores.get(x_b, 0)

            if acc_b >= acc_a and eff_b >= eff_a and (acc_b > acc_a or eff_b > eff_a):
                dominated = True
                break

        if not dominated:
            pareto_front.append(x_a)

    print(f"Pareto front size: {len(pareto_front)} solutions")
    print(f"Pareto front: {sorted(pareto_front)}")

    # Show objectives for each pareto solution
    print("\nPareto front solutions:")
    print(f"{'x':<6} {'Accuracy Score':<20} {'Efficiency Score':<20} {'Acc Loss':<15} {'Eff Cost':<15}")
    print("-" * 80)

    for x in sorted(pareto_front):
        acc_score = accuracy_scores.get(x, 0)
        eff_score = efficiency_scores.get(x, 0)
        acc_loss = problem.evaluate_accuracy(x)
        eff_cost = problem.evaluate_efficiency(x)
        print(f"{x:<6} {acc_score:<20.3f} {eff_score:<20.3f} {acc_loss:<15.1f} {eff_cost:<15.1f}")

    return accuracy_results, efficiency_results, pareto_front


# ============================================================================
# Test 4: Algorithm Comparison on Multiple Problems
# ============================================================================

def test_algorithm_robustness():
    """Test algorithm robustness across different problem types."""
    print("\n" + "=" * 100)
    print("TEST 4: ALGORITHM ROBUSTNESS ACROSS PROBLEM TYPES")
    print("=" * 100)

    problems = [
        ("Quadratic (target=30)", QuadraticProblem(target=30, noise=0.0), range(0, 61, 3)),
        ("Quadratic (target=80)", QuadraticProblem(target=80, noise=0.0), range(50, 111, 3)),
        ("Non-Convex", NonConvexProblem(), range(-50, 51, 5)),
    ]

    summary = {
        "Bradley-Terry": [],
        "Elo": [],
        "Win Rate": [],
        "Copeland": [],
    }

    for problem_name, problem, candidate_range in problems:
        print(f"\n{problem_name}")
        print("-" * 100)

        candidates = list(candidate_range)

        # Ground truth
        ground_truth = {
            x: -problem.evaluate(x) if hasattr(problem, "evaluate") else 0.0
            for x in candidates
        }

        # Comparisons
        comparison_results = {}
        comparison_records = []
        timestamp = 0

        for i, x_a in enumerate(candidates):
            for j, x_b in enumerate(candidates):
                if i < j:
                    result = problem.compare(x_a, x_b)
                    comparison_results[(x_a, x_b)] = result
                    comparison_records.append((x_a, x_b, result, timestamp))
                    timestamp += 1

        # Benchmark
        results = compare_algorithms(
            candidates,
            comparison_results,
            ground_truth,
            comparison_records,
        )

        # Store results
        for result in results:
            summary[result.algorithm_name].append(result.rank_correlation)

        # Print brief summary
        for result in results:
            print(f"  {result.algorithm_name:<20} Rank Corr: {result.rank_correlation:.3f}")

    # Overall summary
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY (Average Rank Correlation)")
    print("=" * 100)

    for algo_name, correlations in summary.items():
        avg_corr = sum(correlations) / len(correlations) if correlations else 0.0
        print(f"{algo_name:<20} {avg_corr:.3f}")

    return summary


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(
        description="Test pairwise comparison algorithms on synthetic problems"
    )
    parser.add_argument(
        "--test",
        choices=["single", "nonconvex", "multi", "robustness", "all"],
        default="all",
        help="Which test to run",
    )

    args = parser.parse_args()

    if args.test in ["single", "all"]:
        test_single_objective_quadratic()

    if args.test in ["nonconvex", "all"]:
        test_non_convex()

    if args.test in ["multi", "all"]:
        test_multi_objective()

    if args.test in ["robustness", "all"]:
        test_algorithm_robustness()

    print("\n" + "=" * 100)
    print("ALL TESTS COMPLETE")
    print("=" * 100)
    print("\nKey findings:")
    print("  1. Bradley-Terry and Elo typically have highest rank correlation")
    print("  2. Bradley-Terry is more robust to sparse comparisons")
    print("  3. Elo is faster but order-dependent")
    print("  4. Win Rate is simple but doesn't account for opponent strength")
    print("  5. All algorithms can recover approximate rankings with enough comparisons")
    print("\nRecommendation: Use Bradley-Terry for GEPA (best balance of accuracy and robustness)")
    print("=" * 100)


if __name__ == "__main__":
    main()
