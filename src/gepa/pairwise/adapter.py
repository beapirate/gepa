# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Pairwise comparison adapter protocol for GEPA.

This module defines the adapter interface for using pairwise comparisons
instead of scalar metrics in GEPA. Adapters implementing this protocol can
work with HybridGEPAState to enable pairwise comparison workflows.
"""

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from gepa.core.adapter import DataInst, RolloutOutput, Trajectory
from gepa.pairwise.types import ComparisonResult, MultiObjectiveComparisonResult, PairwiseComparator

__all__ = [
    "PairwiseEvaluationBatch",
    "PairwiseGEPAAdapter",
]


@dataclass
class PairwiseEvaluationBatch(Generic[Trajectory, RolloutOutput]):
    """
    Container for pairwise evaluation results.

    Unlike traditional EvaluationBatch, this doesn't include scalar scores.
    Scores are computed later via pairwise comparisons.

    Attributes:
        outputs: Raw per-example outputs (will be compared pairwise)
        trajectories: Optional per-example traces for reflection
        reference_comparisons: Optional pre-computed comparisons against reference
    """
    outputs: list[RolloutOutput]
    trajectories: list[Trajectory] | None = None
    reference_comparisons: list[ComparisonResult | MultiObjectiveComparisonResult] | None = None


class PairwiseGEPAAdapter(Protocol[DataInst, Trajectory, RolloutOutput]):
    """
    GEPA adapter protocol for pairwise comparisons.

    This is the main integration point between your domain and GEPA when using
    pairwise comparisons. It extends the standard GEPAAdapter interface with
    pairwise comparison capabilities.

    Key differences from standard GEPAAdapter:
    1. evaluate() returns PairwiseEvaluationBatch (outputs only, no scores)
    2. comparator() method provides pairwise comparison function
    3. make_reflective_dataset() can use comparison results for feedback

    Typical implementation flow:
        1. evaluate(batch, candidate) → outputs
        2. Engine compares outputs pairwise using comparator()
        3. Bradley-Terry converts comparisons → scalar scores
        4. Standard GEPA logic continues

    Example implementation:
        class MyPairwiseAdapter(PairwiseGEPAAdapter):
            def __init__(self, llm_client, judge_client):
                self.llm = llm_client
                self.judge = judge_client

            def evaluate(self, batch, candidate, capture_traces=False):
                outputs = [self.llm.complete(candidate["instruction"] + ex["question"])
                          for ex in batch]
                return PairwiseEvaluationBatch(outputs=outputs)

            def comparator(self):
                return LLMJudgeComparator(self.judge)

            def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
                # Build dataset using outputs and trajectories
                ...

    Type parameters:
        DataInst: User-defined type of input data
        Trajectory: User-defined type of execution trace
        RolloutOutput: User-defined type of program output
    """

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> PairwiseEvaluationBatch[Trajectory, RolloutOutput]:
        """
        Evaluate candidate on batch and return outputs (no scores).

        This is similar to standard GEPAAdapter.evaluate() but returns only
        outputs. Scores will be computed via pairwise comparisons.

        Args:
            batch: List of data instances to evaluate
            candidate: Program candidate (component_name -> component_text)
            capture_traces: If True, populate trajectories for reflection

        Returns:
            PairwiseEvaluationBatch with:
            - outputs: Per-example outputs
            - trajectories: Per-example traces (if capture_traces=True)
            - reference_comparisons: Optional pre-computed comparisons

        Notes:
            - Do not compute scalar scores here
            - Outputs should be comparable via comparator()
            - Trajectories used for make_reflective_dataset()
            - Handle errors gracefully (return placeholder outputs)
        """
        ...

    def comparator(self) -> PairwiseComparator[RolloutOutput, DataInst]:
        """
        Return pairwise comparator for this adapter.

        The comparator will be called by GEPA engine to compare outputs.
        It should be consistent across calls (deterministic or unbiased random).

        Returns:
            PairwiseComparator that can compare two outputs

        Notes:
            - Comparator may be called many times (O(N²) for N programs)
            - Consider caching comparison results if expensive
            - Ensure consistency: compare(A, B) opposite of compare(B, A)

        Example:
            def comparator(self):
                return LLMJudgeComparator(
                    llm_client=self.judge_llm,
                    prompt_template="Which answer is better?",
                )
        """
        ...

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: PairwiseEvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Build reflective dataset for instruction refinement.

        This is the same as standard GEPAAdapter but works with
        PairwiseEvaluationBatch instead of EvaluationBatch.

        Args:
            candidate: Current candidate being evaluated
            eval_batch: Evaluation results (outputs + trajectories)
            components_to_update: Which components to update

        Returns:
            Reflective dataset: component_name -> list of examples

        Notes:
            - Use trajectories to extract feedback
            - Can use reference_comparisons if available
            - Format should be consumable by propose_new_texts()
        """
        ...

    # Optional: custom proposal function (same as standard GEPAAdapter)
    propose_new_texts: Any | None = None  # ProposalFn
