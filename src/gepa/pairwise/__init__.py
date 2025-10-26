"""
Pairwise comparison support for GEPA.

This module provides infrastructure for using pairwise comparisons instead of
scalar metrics in GEPA. The hybrid approach stores pairwise comparisons and
converts them to scalar scores on-demand using Bradley-Terry model, allowing
reuse of existing GEPA code.

Key components:
- ComparisonResult: Result of comparing two outputs
- PairwiseComparator: Protocol for comparing outputs
- bradley_terry_scores: Convert comparisons to scalar scores
- HybridGEPAState: State that works with pairwise comparisons

Example usage:
    from gepa.pairwise import PairwiseComparator, ComparisonResult

    class MyComparator(PairwiseComparator):
        def compare(self, output_a, output_b, data_instance):
            # Your comparison logic (e.g., LLM-as-judge)
            return ComparisonResult.A_BETTER

    # Use with GEPA via pairwise adapter
"""

from gepa.pairwise.types import (
    ComparisonResult,
    MultiObjectiveComparisonResult,
    PairwiseComparisonRecord,
    PairwiseComparator,
)
from gepa.pairwise.bradley_terry import bradley_terry_scores

__all__ = [
    "ComparisonResult",
    "MultiObjectiveComparisonResult",
    "PairwiseComparisonRecord",
    "PairwiseComparator",
    "bradley_terry_scores",
]
