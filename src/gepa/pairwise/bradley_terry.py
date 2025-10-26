# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Bradley-Terry model for converting pairwise comparisons to scalar scores.

The Bradley-Terry model is a statistical method for estimating quality scores
from pairwise comparison data. It assumes:
    P(i beats j) = score_i / (score_i + score_j)

We use the MM (Majorization-Minimization) algorithm to find maximum likelihood
estimates of the scores.

References:
    Bradley, R. A., & Terry, M. E. (1952). Rank Analysis of Incomplete Block
    Designs: I. The Method of Paired Comparisons. Biometrika, 39(3/4), 324-345.
"""

import logging
from collections import defaultdict
from typing import Any

from gepa.pairwise.types import ComparisonResult

logger = logging.getLogger(__name__)


def bradley_terry_scores(
    programs: list[int],
    comparison_results: dict[tuple[int, int], ComparisonResult] | dict[tuple[int, int, Any], ComparisonResult],
    iterations: int = 100,
    tolerance: float = 1e-6,
    normalize: bool = True,
) -> dict[int, float]:
    """
    Compute Bradley-Terry scores from pairwise comparisons.

    Uses the MM (Majorization-Minimization) algorithm for maximum likelihood
    estimation. This algorithm is guaranteed to converge and is numerically stable.

    Args:
        programs: List of program indices to score
        comparison_results: Dict mapping comparison keys to results
            Keys can be (prog_a, prog_b) or (prog_a, prog_b, data_id)
        iterations: Maximum number of MM iterations
        tolerance: Convergence tolerance (max score change)
        normalize: If True, normalize scores to [0, 1] range

    Returns:
        Dict mapping program index -> score (higher is better)

    Algorithm:
        Initialize scores uniformly
        Repeat until convergence:
            For each program i:
                score_i = wins_i / sum_j(comparisons_ij / (score_i + score_j))
        Normalize scores

    Notes:
        - Handles sparse comparison matrices (not all pairs compared)
        - Handles ties by giving each program 0.5 wins
        - Ignores INCOMPARABLE comparisons
        - Numerically stable (uses log-space internally when needed)

    Example:
        >>> comparisons = {
        ...     (0, 1): ComparisonResult.A_BETTER,
        ...     (0, 2): ComparisonResult.B_BETTER,
        ...     (1, 2): ComparisonResult.TIE,
        ... }
        >>> scores = bradley_terry_scores([0, 1, 2], comparisons)
        >>> # Program 0 and 1 will have higher scores than 2
    """
    if not programs:
        return {}

    if len(programs) == 1:
        # Only one program, return default score
        return {programs[0]: 1.0}

    # Initialize scores uniformly
    scores = {prog_idx: 1.0 for prog_idx in programs}

    # Build win/loss records from comparison results
    wins = defaultdict(float)
    comparisons_per_program = defaultdict(list)  # program -> [(other_program, count)]

    for key, result in comparison_results.items():
        # Handle both (a, b) and (a, b, data_id) key formats
        if len(key) == 2:
            prog_a, prog_b = key
        elif len(key) == 3:
            prog_a, prog_b, _data_id = key
        else:
            logger.warning(f"Unexpected comparison key format: {key}")
            continue

        # Skip if either program not in programs list
        if prog_a not in programs or prog_b not in programs:
            continue

        # Skip incomparable
        if result == ComparisonResult.INCOMPARABLE:
            continue

        # Record wins
        if result == ComparisonResult.A_BETTER:
            wins[prog_a] += 1.0
        elif result == ComparisonResult.B_BETTER:
            wins[prog_b] += 1.0
        elif result == ComparisonResult.TIE:
            wins[prog_a] += 0.5
            wins[prog_b] += 0.5

        # Record comparisons (for both directions)
        comparisons_per_program[prog_a].append(prog_b)
        comparisons_per_program[prog_b].append(prog_a)

    # Handle programs with no comparisons
    for prog_idx in programs:
        if prog_idx not in comparisons_per_program:
            scores[prog_idx] = 0.5  # Default neutral score
            logger.debug(f"Program {prog_idx} has no comparisons, assigning default score 0.5")

    # MM algorithm iterations
    for iteration in range(iterations):
        new_scores = {}
        max_change = 0.0

        for prog_idx in programs:
            # Skip if no comparisons
            if prog_idx not in comparisons_per_program:
                new_scores[prog_idx] = 0.5
                continue

            # Compute denominator sum
            # sum_j(comparisons_ij / (score_i + score_j))
            denom_sum = 0.0

            for other_idx in comparisons_per_program[prog_idx]:
                # Count how many times these two were compared
                # (Could appear multiple times for different data instances)
                count = 0.0
                for key in comparison_results.keys():
                    if len(key) == 2:
                        a, b = key
                    elif len(key) == 3:
                        a, b, _ = key
                    else:
                        continue

                    if (a == prog_idx and b == other_idx) or (a == other_idx and b == prog_idx):
                        if comparison_results[key] != ComparisonResult.INCOMPARABLE:
                            count += 1.0

                if count > 0:
                    denom_sum += count / (scores[prog_idx] + scores[other_idx])

            # Update score
            if denom_sum > 0:
                new_scores[prog_idx] = wins[prog_idx] / denom_sum
            else:
                # No valid comparisons, keep at default
                new_scores[prog_idx] = 0.5

            # Track convergence
            change = abs(new_scores[prog_idx] - scores[prog_idx])
            max_change = max(max_change, change)

        scores = new_scores

        # Check convergence
        if max_change < tolerance:
            logger.debug(f"Bradley-Terry converged after {iteration + 1} iterations")
            break
    else:
        logger.debug(f"Bradley-Terry reached max iterations ({iterations})")

    # Normalize scores to [0, 1] range if requested
    if normalize and scores:
        min_score = min(scores.values())
        max_score = max(scores.values())

        if max_score > min_score:
            scores = {
                prog: (score - min_score) / (max_score - min_score)
                for prog, score in scores.items()
            }
        else:
            # All scores equal, normalize to 0.5
            scores = {prog: 0.5 for prog in scores.keys()}

    return scores


def bradley_terry_scores_per_instance(
    programs: list[int],
    comparison_results: dict[tuple[int, int, Any], ComparisonResult],
    iterations: int = 100,
    tolerance: float = 1e-6,
) -> dict[Any, dict[int, float]]:
    """
    Compute Bradley-Terry scores per data instance.

    For sparse evaluation where different programs are evaluated on different
    data instances, compute scores separately for each instance.

    Args:
        programs: List of all program indices
        comparison_results: Dict mapping (prog_a, prog_b, data_id) -> result
        iterations: Maximum MM iterations per instance
        tolerance: Convergence tolerance

    Returns:
        Dict mapping data_id -> {program_idx -> score}

    Example:
        >>> comparisons = {
        ...     (0, 1, "data1"): ComparisonResult.A_BETTER,
        ...     (0, 2, "data1"): ComparisonResult.A_BETTER,
        ...     (1, 2, "data2"): ComparisonResult.B_BETTER,
        ... }
        >>> scores_per_instance = bradley_terry_scores_per_instance([0, 1, 2], comparisons)
        >>> # Returns: {"data1": {0: 1.0, 1: 0.5, 2: 0.0}, "data2": {1: 0.0, 2: 1.0}}
    """
    # Group comparisons by data instance
    comparisons_by_instance: dict[Any, dict[tuple[int, int], ComparisonResult]] = defaultdict(dict)

    for (prog_a, prog_b, data_id), result in comparison_results.items():
        comparisons_by_instance[data_id][(prog_a, prog_b)] = result

    # Compute scores per instance
    scores_per_instance = {}

    for data_id, instance_comparisons in comparisons_by_instance.items():
        # Find which programs were evaluated on this instance
        programs_for_instance = set()
        for (prog_a, prog_b) in instance_comparisons.keys():
            programs_for_instance.add(prog_a)
            programs_for_instance.add(prog_b)

        if not programs_for_instance:
            continue

        # Compute scores for this instance
        instance_scores = bradley_terry_scores(
            programs=list(programs_for_instance),
            comparison_results=instance_comparisons,
            iterations=iterations,
            tolerance=tolerance,
            normalize=True,
        )

        scores_per_instance[data_id] = instance_scores

    return scores_per_instance
