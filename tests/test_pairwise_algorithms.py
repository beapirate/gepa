"""
Implementation and benchmarking of pairwise-to-scalar conversion algorithms.

Implements:
1. Bradley-Terry Model
2. Elo Rating System
3. Win Rate (simple baseline)
4. PageRank-style
5. Copeland Score

Provides utilities to compare algorithms on synthetic problems.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from test_pairwise_synthetic import ComparisonResult


# ============================================================================
# Algorithm 1: Bradley-Terry Model
# ============================================================================

def bradley_terry_scores(
    programs: list[int],
    comparison_results: dict[tuple[int, int], ComparisonResult],
    iterations: int = 100,
    tolerance: float = 1e-6,
) -> dict[int, float]:
    """
    Compute Bradley-Terry scores from pairwise comparisons.

    Uses MM (Majorization-Minimization) algorithm for maximum likelihood estimation.

    Model: P(i beats j) = score_i / (score_i + score_j)

    Args:
        programs: List of program indices
        comparison_results: Dict mapping (prog_a, prog_b) -> ComparisonResult
        iterations: Maximum number of iterations
        tolerance: Convergence tolerance

    Returns:
        Dict mapping program index -> score
    """
    # Initialize scores uniformly
    scores = {prog_idx: 1.0 for prog_idx in programs}

    # Build win/loss records
    wins = defaultdict(float)
    comparisons = defaultdict(float)

    for (prog_a, prog_b), result in comparison_results.items():
        if result == ComparisonResult.A_BETTER:
            wins[prog_a] += 1.0
            comparisons[prog_a] += 1.0
            comparisons[prog_b] += 1.0
        elif result == ComparisonResult.B_BETTER:
            wins[prog_b] += 1.0
            comparisons[prog_a] += 1.0
            comparisons[prog_b] += 1.0
        elif result == ComparisonResult.TIE:
            wins[prog_a] += 0.5
            wins[prog_b] += 0.5
            comparisons[prog_a] += 1.0
            comparisons[prog_b] += 1.0

    # MM algorithm
    for iteration in range(iterations):
        new_scores = {}
        max_change = 0.0

        for prog_idx in programs:
            if comparisons[prog_idx] == 0:
                new_scores[prog_idx] = 1.0
                continue

            # Compute denominator sum
            denom_sum = 0.0
            for other_idx in programs:
                if other_idx == prog_idx:
                    continue

                # Count comparisons between prog_idx and other_idx
                count_ab = 0.0
                count_ba = 0.0

                if (prog_idx, other_idx) in comparison_results:
                    result = comparison_results[(prog_idx, other_idx)]
                    if result != ComparisonResult.INCOMPARABLE:
                        count_ab = 1.0

                if (other_idx, prog_idx) in comparison_results:
                    result = comparison_results[(other_idx, prog_idx)]
                    if result != ComparisonResult.INCOMPARABLE:
                        count_ba = 1.0

                if count_ab > 0 or count_ba > 0:
                    denom_sum += (count_ab + count_ba) / (
                        scores[prog_idx] + scores[other_idx]
                    )

            # Update score
            if denom_sum > 0:
                new_scores[prog_idx] = wins[prog_idx] / denom_sum
            else:
                new_scores[prog_idx] = 1.0

            # Track convergence
            max_change = max(max_change, abs(new_scores[prog_idx] - scores[prog_idx]))

        scores = new_scores

        # Check convergence
        if max_change < tolerance:
            break

    # Normalize scores to [0, 1] for interpretability
    if scores:
        min_score = min(scores.values())
        max_score = max(scores.values())
        if max_score > min_score:
            scores = {
                prog: (score - min_score) / (max_score - min_score)
                for prog, score in scores.items()
            }

    return scores


# ============================================================================
# Algorithm 2: Elo Rating System
# ============================================================================

def elo_scores(
    programs: list[int],
    comparison_records: list[tuple[int, int, ComparisonResult, int]],
    k_factor: float = 32.0,
    initial_rating: float = 1500.0,
) -> dict[int, float]:
    """
    Compute Elo ratings from pairwise comparisons.

    Processes comparisons in temporal order (by timestamp).

    Args:
        programs: List of program indices
        comparison_records: List of (prog_a, prog_b, result, timestamp)
        k_factor: Learning rate (higher = faster adaptation)
        initial_rating: Starting rating for all programs

    Returns:
        Dict mapping program index -> Elo rating
    """
    # Initialize ratings
    ratings = {prog_idx: initial_rating for prog_idx in programs}

    # Sort by timestamp
    sorted_records = sorted(comparison_records, key=lambda r: r[3])

    for prog_a, prog_b, result, _timestamp in sorted_records:
        if result == ComparisonResult.INCOMPARABLE:
            continue

        # Expected scores
        rating_a = ratings.get(prog_a, initial_rating)
        rating_b = ratings.get(prog_b, initial_rating)

        expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1.0 - expected_a

        # Actual scores
        if result == ComparisonResult.A_BETTER:
            actual_a, actual_b = 1.0, 0.0
        elif result == ComparisonResult.B_BETTER:
            actual_a, actual_b = 0.0, 1.0
        elif result == ComparisonResult.TIE:
            actual_a, actual_b = 0.5, 0.5
        else:
            continue

        # Update ratings
        ratings[prog_a] = rating_a + k_factor * (actual_a - expected_a)
        ratings[prog_b] = rating_b + k_factor * (actual_b - expected_b)

    # Normalize to [0, 1]
    if ratings:
        min_rating = min(ratings.values())
        max_rating = max(ratings.values())
        if max_rating > min_rating:
            ratings = {
                prog: (rating - min_rating) / (max_rating - min_rating)
                for prog, rating in ratings.items()
            }

    return ratings


# ============================================================================
# Algorithm 3: Win Rate (Baseline)
# ============================================================================

def win_rate_scores(
    programs: list[int],
    comparison_results: dict[tuple[int, int], ComparisonResult],
) -> dict[int, float]:
    """
    Compute simple win rate for each program.

    Score = wins / (wins + losses)

    Args:
        programs: List of program indices
        comparison_results: Dict mapping (prog_a, prog_b) -> ComparisonResult

    Returns:
        Dict mapping program index -> win rate
    """
    wins = defaultdict(float)
    total = defaultdict(float)

    for (prog_a, prog_b), result in comparison_results.items():
        if result == ComparisonResult.A_BETTER:
            wins[prog_a] += 1.0
            total[prog_a] += 1.0
            total[prog_b] += 1.0
        elif result == ComparisonResult.B_BETTER:
            wins[prog_b] += 1.0
            total[prog_a] += 1.0
            total[prog_b] += 1.0
        elif result == ComparisonResult.TIE:
            wins[prog_a] += 0.5
            wins[prog_b] += 0.5
            total[prog_a] += 1.0
            total[prog_b] += 1.0

    scores = {}
    for prog_idx in programs:
        if total[prog_idx] > 0:
            scores[prog_idx] = wins[prog_idx] / total[prog_idx]
        else:
            scores[prog_idx] = 0.5  # No comparisons yet

    return scores


# ============================================================================
# Algorithm 4: Copeland Score
# ============================================================================

def copeland_scores(
    programs: list[int],
    comparison_results: dict[tuple[int, int], ComparisonResult],
) -> dict[int, float]:
    """
    Compute Copeland scores: net wins against other programs.

    Score = (# programs beaten) - (# programs that beat this)

    Args:
        programs: List of program indices
        comparison_results: Dict mapping (prog_a, prog_b) -> ComparisonResult

    Returns:
        Dict mapping program index -> Copeland score
    """
    # Track which programs beat which
    beats = defaultdict(set)

    for (prog_a, prog_b), result in comparison_results.items():
        if result == ComparisonResult.A_BETTER:
            beats[prog_a].add(prog_b)
        elif result == ComparisonResult.B_BETTER:
            beats[prog_b].add(prog_a)

    # Compute Copeland score
    scores = {}
    for prog_idx in programs:
        beaten_by_me = len(beats[prog_idx])
        beats_me = sum(1 for other in programs if prog_idx in beats[other])
        scores[prog_idx] = beaten_by_me - beats_me

    # Normalize to [0, 1]
    if scores:
        min_score = min(scores.values())
        max_score = max(scores.values())
        if max_score > min_score:
            scores = {
                prog: (score - min_score) / (max_score - min_score)
                for prog, score in scores.items()
            }

    return scores


# ============================================================================
# Benchmarking Utilities
# ============================================================================

@dataclass
class AlgorithmBenchmarkResult:
    """Result of benchmarking an algorithm."""
    algorithm_name: str
    scores: dict[int, float]
    rank_correlation: float  # Correlation with ground truth ranking
    score_correlation: float  # Correlation with ground truth scores
    top_k_accuracy: dict[int, float]  # Accuracy of top-k predictions
    runtime_ms: float


def benchmark_algorithm(
    algorithm_name: str,
    algorithm_fn: Callable,
    programs: list[int],
    comparison_results: dict[tuple[int, int], ComparisonResult],
    ground_truth_scores: dict[int, float] | None = None,
    comparison_records: list | None = None,
) -> AlgorithmBenchmarkResult:
    """
    Benchmark a single algorithm.

    Args:
        algorithm_name: Name of the algorithm
        algorithm_fn: Function that computes scores
        programs: List of program indices
        comparison_results: Pairwise comparison results
        ground_truth_scores: True scores (if available) for validation
        comparison_records: Temporal records (for Elo)

    Returns:
        AlgorithmBenchmarkResult
    """
    import time

    # Run algorithm
    start_time = time.time()

    if algorithm_name == "Elo" and comparison_records:
        scores = algorithm_fn(programs, comparison_records)
    else:
        scores = algorithm_fn(programs, comparison_results)

    runtime_ms = (time.time() - start_time) * 1000

    # Compute metrics if ground truth available
    rank_correlation = 0.0
    score_correlation = 0.0
    top_k_accuracy = {}

    if ground_truth_scores:
        # Rank correlation (Spearman)
        rank_correlation = compute_rank_correlation(scores, ground_truth_scores)

        # Score correlation (Pearson)
        score_correlation = compute_score_correlation(scores, ground_truth_scores)

        # Top-k accuracy
        for k in [1, 3, 5]:
            top_k_accuracy[k] = compute_top_k_accuracy(
                scores, ground_truth_scores, k
            )

    return AlgorithmBenchmarkResult(
        algorithm_name=algorithm_name,
        scores=scores,
        rank_correlation=rank_correlation,
        score_correlation=score_correlation,
        top_k_accuracy=top_k_accuracy,
        runtime_ms=runtime_ms,
    )


def compute_rank_correlation(
    predicted_scores: dict[int, float],
    ground_truth_scores: dict[int, float],
) -> float:
    """Compute Spearman rank correlation."""
    common_programs = set(predicted_scores.keys()) & set(ground_truth_scores.keys())

    if len(common_programs) < 2:
        return 0.0

    # Get rankings
    pred_ranking = {
        prog: rank
        for rank, prog in enumerate(
            sorted(common_programs, key=lambda p: predicted_scores[p], reverse=True)
        )
    }

    truth_ranking = {
        prog: rank
        for rank, prog in enumerate(
            sorted(common_programs, key=lambda p: ground_truth_scores[p], reverse=True)
        )
    }

    # Compute Spearman correlation
    n = len(common_programs)
    d_squared_sum = sum(
        (pred_ranking[prog] - truth_ranking[prog]) ** 2 for prog in common_programs
    )

    correlation = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))

    return correlation


def compute_score_correlation(
    predicted_scores: dict[int, float],
    ground_truth_scores: dict[int, float],
) -> float:
    """Compute Pearson correlation coefficient."""
    common_programs = set(predicted_scores.keys()) & set(ground_truth_scores.keys())

    if len(common_programs) < 2:
        return 0.0

    pred_values = [predicted_scores[p] for p in common_programs]
    truth_values = [ground_truth_scores[p] for p in common_programs]

    # Compute Pearson correlation
    n = len(pred_values)
    mean_pred = sum(pred_values) / n
    mean_truth = sum(truth_values) / n

    numerator = sum(
        (p - mean_pred) * (t - mean_truth) for p, t in zip(pred_values, truth_values)
    )

    denom_pred = math.sqrt(sum((p - mean_pred) ** 2 for p in pred_values))
    denom_truth = math.sqrt(sum((t - mean_truth) ** 2 for t in truth_values))

    if denom_pred == 0 or denom_truth == 0:
        return 0.0

    correlation = numerator / (denom_pred * denom_truth)

    return correlation


def compute_top_k_accuracy(
    predicted_scores: dict[int, float],
    ground_truth_scores: dict[int, float],
    k: int,
) -> float:
    """
    Compute top-k accuracy: fraction of true top-k programs in predicted top-k.
    """
    common_programs = set(predicted_scores.keys()) & set(ground_truth_scores.keys())

    if len(common_programs) < k:
        k = len(common_programs)

    if k == 0:
        return 0.0

    # Get top-k programs
    pred_top_k = set(
        sorted(common_programs, key=lambda p: predicted_scores[p], reverse=True)[:k]
    )

    truth_top_k = set(
        sorted(common_programs, key=lambda p: ground_truth_scores[p], reverse=True)[:k]
    )

    # Compute overlap
    overlap = len(pred_top_k & truth_top_k)
    accuracy = overlap / k

    return accuracy


def compare_algorithms(
    programs: list[int],
    comparison_results: dict[tuple[int, int], ComparisonResult],
    ground_truth_scores: dict[int, float] | None = None,
    comparison_records: list | None = None,
) -> list[AlgorithmBenchmarkResult]:
    """
    Compare all algorithms on the same comparison data.

    Args:
        programs: List of program indices
        comparison_results: Pairwise comparison results
        ground_truth_scores: True scores for validation
        comparison_records: Temporal records for Elo

    Returns:
        List of benchmark results, one per algorithm
    """
    algorithms = [
        ("Bradley-Terry", bradley_terry_scores),
        ("Elo", elo_scores),
        ("Win Rate", win_rate_scores),
        ("Copeland", copeland_scores),
    ]

    results = []

    for name, fn in algorithms:
        result = benchmark_algorithm(
            name,
            fn,
            programs,
            comparison_results,
            ground_truth_scores,
            comparison_records,
        )
        results.append(result)

    return results


def print_benchmark_results(results: list[AlgorithmBenchmarkResult]):
    """Pretty-print benchmark results."""
    print("\n" + "=" * 100)
    print("ALGORITHM BENCHMARK RESULTS")
    print("=" * 100)

    print(
        f"\n{'Algorithm':<20} {'Runtime (ms)':<15} {'Rank Corr':<12} {'Score Corr':<12} "
        f"{'Top-1 Acc':<12} {'Top-3 Acc':<12} {'Top-5 Acc':<12}"
    )
    print("-" * 100)

    for result in results:
        top_1 = result.top_k_accuracy.get(1, 0.0)
        top_3 = result.top_k_accuracy.get(3, 0.0)
        top_5 = result.top_k_accuracy.get(5, 0.0)

        print(
            f"{result.algorithm_name:<20} {result.runtime_ms:<15.2f} "
            f"{result.rank_correlation:<12.3f} {result.score_correlation:<12.3f} "
            f"{top_1:<12.3f} {top_3:<12.3f} {top_5:<12.3f}"
        )

    print("=" * 100)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from test_pairwise_synthetic import QuadraticProblem

    print("=" * 80)
    print("ALGORITHM COMPARISON ON SYNTHETIC PROBLEM")
    print("=" * 80)

    # Create a simple problem
    problem = QuadraticProblem(target=50, noise=0.0)

    # Generate candidate programs (different x values)
    candidates = list(range(20, 80, 5))  # 20, 25, 30, ..., 75
    print(f"\nCandidates: {candidates}")

    # Compute ground truth scores
    ground_truth = {x: -problem.evaluate(x) for x in candidates}  # Negate for higher=better
    print(f"Ground truth (best candidates): {sorted(ground_truth.items(), key=lambda x: x[1], reverse=True)[:5]}")

    # Perform pairwise comparisons
    comparison_results = {}
    comparison_records = []
    timestamp = 0

    for i, x_a in enumerate(candidates):
        for j, x_b in enumerate(candidates):
            if i < j:  # Only compare each pair once
                result = problem.compare(x_a, x_b)
                comparison_results[(x_a, x_b)] = result
                comparison_records.append((x_a, x_b, result, timestamp))
                timestamp += 1

    print(f"\nPerformed {len(comparison_results)} pairwise comparisons")

    # Benchmark all algorithms
    results = compare_algorithms(
        candidates,
        comparison_results,
        ground_truth,
        comparison_records,
    )

    # Print results
    print_benchmark_results(results)

    # Show detailed scores for best algorithm
    best_algo = max(results, key=lambda r: r.rank_correlation)
    print(f"\n\nBEST ALGORITHM: {best_algo.algorithm_name}")
    print("-" * 80)
    print("Top 5 programs by predicted score:")
    sorted_scores = sorted(best_algo.scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (prog, score) in enumerate(sorted_scores[:5], 1):
        true_score = ground_truth[prog]
        print(f"  {rank}. x={prog:3d}  predicted_score={score:.3f}  true_score={true_score:.1f}")

    print("\n" + "=" * 80)
