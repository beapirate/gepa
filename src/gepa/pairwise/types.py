# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Core types for pairwise comparison in GEPA.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar

# Type variables
RolloutOutput = TypeVar("RolloutOutput")
DataInst = TypeVar("DataInst")
DataId = TypeVar("DataId")
ObjectiveId = str


class ComparisonResult(Enum):
    """Result of comparing two outputs."""

    A_BETTER = "a_better"
    """Output A is strictly better than output B."""

    B_BETTER = "b_better"
    """Output B is strictly better than output A."""

    TIE = "tie"
    """Outputs A and B are equally good."""

    INCOMPARABLE = "incomparable"
    """Cannot determine which is better (e.g., different quality dimensions)."""


@dataclass
class MultiObjectiveComparisonResult:
    """
    Result of comparing two outputs across multiple objectives.

    For multi-objective optimization, we compare outputs on each objective
    separately and store per-objective results.

    Example:
        result = MultiObjectiveComparisonResult(
            objective_results={
                "accuracy": ComparisonResult.A_BETTER,
                "latency": ComparisonResult.B_BETTER,
                "readability": ComparisonResult.TIE,
            }
        )
    """
    objective_results: dict[ObjectiveId, ComparisonResult]

    def dominates_other(self, other: "MultiObjectiveComparisonResult") -> bool:
        """
        Check if this result represents A dominating B in Pareto sense.

        A dominates B if:
        - A >= B on all objectives
        - A > B on at least one objective

        Returns:
            True if A dominates B
        """
        at_least_one_better = False

        for obj_id in self.objective_results.keys():
            self_result = self.objective_results[obj_id]

            if self_result == ComparisonResult.B_BETTER:
                # A is worse on this objective, doesn't dominate
                return False
            elif self_result == ComparisonResult.A_BETTER:
                at_least_one_better = True

        return at_least_one_better


@dataclass
class PairwiseComparisonRecord(Generic[DataId]):
    """
    Record of a single pairwise comparison.

    Stores metadata about when and how the comparison was performed.
    Used for debugging, analysis, and temporal ordering (e.g., for Elo).
    """
    program_a: int  # Program index
    program_b: int  # Program index
    data_id: DataId  # Which data instance was compared
    result: ComparisonResult | MultiObjectiveComparisonResult
    timestamp: int  # Iteration when comparison was made
    metadata: dict[str, Any] | None = None  # Optional metadata (e.g., confidence scores)


class PairwiseComparator(Protocol[RolloutOutput, DataInst]):
    """
    Protocol for comparing two outputs pairwise.

    Implementers provide a comparison function that determines which of two
    outputs is better for a given input. This could use:
    - LLM-as-judge ("Which answer is better?")
    - Human-in-the-loop feedback
    - Rule-based comparison (e.g., correctness + efficiency)
    - Hybrid approaches

    The comparison should be:
    - Deterministic (same inputs → same output) OR
    - Stochastic but unbiased (random noise averages out)

    Example implementations:
        - LLMJudgeComparator: Uses LLM to judge quality
        - RuleBasedComparator: Uses programmatic rules
        - HumanComparator: Collects human preferences
    """

    def compare(
        self,
        output_a: RolloutOutput,
        output_b: RolloutOutput,
        data_instance: DataInst,
    ) -> ComparisonResult | MultiObjectiveComparisonResult:
        """
        Compare two outputs for the same data instance.

        Args:
            output_a: First output to compare
            output_b: Second output to compare
            data_instance: The input data both outputs were generated from

        Returns:
            ComparisonResult indicating which output is better, or
            MultiObjectiveComparisonResult for multi-objective comparison

        Notes:
            - Should be consistent: comparing (A, B) and (B, A) should give
              opposite results (unless TIE)
            - Can return INCOMPARABLE if no meaningful comparison is possible
            - For best results, should be transitive: if A > B and B > C,
              then A > C (though Bradley-Terry handles some intransitivity)
        """
        ...


class MultiObjectiveComparator(Protocol[RolloutOutput, DataInst]):
    """
    Protocol for multi-objective pairwise comparison.

    Like PairwiseComparator but explicitly returns per-objective comparisons.
    """

    def compare(
        self,
        output_a: RolloutOutput,
        output_b: RolloutOutput,
        data_instance: DataInst,
    ) -> MultiObjectiveComparisonResult:
        """Compare outputs across all objectives."""
        ...

    def get_objectives(self) -> list[ObjectiveId]:
        """Return list of objective IDs this comparator evaluates."""
        ...
