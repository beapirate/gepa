# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Hybrid GEPA state for pairwise comparisons.

This module provides a state implementation that stores pairwise comparisons
and computes scalar scores on-demand using Bradley-Terry, allowing it to work
with existing GEPA engine code.
"""

import logging
from collections import defaultdict
from typing import Any, Generic

from gepa.core.adapter import RolloutOutput
from gepa.core.data_loader import DataId
from gepa.core.state import GEPAState, ProgramIdx
from gepa.pairwise.bradley_terry import bradley_terry_scores_per_instance
from gepa.pairwise.types import ComparisonResult, PairwiseComparisonRecord

logger = logging.getLogger(__name__)


class HybridGEPAState(GEPAState[RolloutOutput, DataId]):
    """
    GEPA state that supports pairwise comparisons with Bradley-Terry conversion.

    This hybrid approach stores pairwise comparison results and converts them
    to scalar scores on-demand, allowing full compatibility with existing GEPA
    engine code while enabling pairwise comparison workflows.

    Key features:
    - Stores pairwise comparisons and outputs
    - Computes scores lazily via Bradley-Terry
    - Caches scores until new comparisons added
    - Compatible with existing GEPA engine

    Architecture:
        1. Outputs stored per program: prog_candidate_val_outputs
        2. Comparisons cached: comparison_cache
        3. Scores computed on access: prog_candidate_val_subscores (property)
        4. Cache invalidated on new comparisons

    Example usage:
        state = HybridGEPAState.create_from_seed(
            seed_candidate={"instruction": "Answer the question."},
            base_valset_outputs=base_outputs,
            comparator=my_comparator,
        )

        # Access scores triggers Bradley-Terry conversion
        scores = state.prog_candidate_val_subscores

        # Add new program with comparisons
        state.add_program_with_comparisons(
            new_program,
            new_outputs,
            comparator,
            valset_data_instances,
        )
    """

    def __init__(
        self,
        seed_candidate: dict[str, str],
        base_valset_eval_output: tuple[dict[DataId, RolloutOutput], dict[DataId, float]],
        track_best_outputs: bool = False,
        bradley_terry_iterations: int = 100,
        bradley_terry_tolerance: float = 1e-6,
    ):
        """
        Initialize hybrid state.

        Args:
            seed_candidate: Initial candidate program
            base_valset_eval_output: (outputs, scores) for seed program
            track_best_outputs: Whether to track best outputs per instance
            bradley_terry_iterations: Max iterations for Bradley-Terry
            bradley_terry_tolerance: Convergence tolerance for Bradley-Terry
        """
        # Initialize parent with base scores
        super().__init__(seed_candidate, base_valset_eval_output, track_best_outputs)

        base_outputs, base_scores = base_valset_eval_output

        # Pairwise comparison data
        self.prog_candidate_val_outputs: list[dict[DataId, RolloutOutput]] = [base_outputs]
        self.comparison_cache: dict[tuple[ProgramIdx, ProgramIdx, DataId], ComparisonResult] = {}
        self.comparison_records: list[PairwiseComparisonRecord] = []

        # Score computation settings
        self.bradley_terry_iterations = bradley_terry_iterations
        self.bradley_terry_tolerance = bradley_terry_tolerance

        # Cache management
        self._cached_scores: list[dict[DataId, float]] | None = None
        self._cache_valid = False

        # Override parent's scores with cached property access
        self._override_prog_candidate_val_subscores()

    def _override_prog_candidate_val_subscores(self):
        """
        Override prog_candidate_val_subscores to use cached/computed scores.

        This allows existing GEPA code to access scores transparently.
        """
        # Store reference to parent's scores (for initial seed)
        self._parent_scores = self.prog_candidate_val_subscores

    @property
    def prog_candidate_val_subscores(self) -> list[dict[DataId, float]]:
        """
        Get scores for all programs, computing via Bradley-Terry if needed.

        This property is accessed by existing GEPA code. It returns cached
        scores if valid, otherwise computes them from pairwise comparisons.

        Returns:
            List where element i is dict mapping data_id -> score for program i
        """
        # Use cached scores if valid
        if self._cache_valid and self._cached_scores is not None:
            return self._cached_scores

        # Recompute scores from pairwise comparisons
        self._recompute_scores()
        self._cache_valid = True

        return self._cached_scores

    @prog_candidate_val_subscores.setter
    def prog_candidate_val_subscores(self, value: list[dict[DataId, float]]):
        """
        Allow setting scores directly (for compatibility with parent class).

        When scores are set directly, we assume they're authoritative and
        invalidate any cached computed scores.
        """
        self._cached_scores = value
        self._cache_valid = True

    def _recompute_scores(self):
        """
        Recompute all scores from pairwise comparisons using Bradley-Terry.

        For each validation instance:
        1. Find all programs evaluated on that instance
        2. Extract comparisons for those programs
        3. Run Bradley-Terry to get scores
        4. Store in cache
        """
        logger.debug(f"Recomputing scores for {len(self.program_candidates)} programs using Bradley-Terry")

        # Initialize score cache
        self._cached_scores = [{} for _ in range(len(self.program_candidates))]

        # Get all validation instances
        all_val_ids = set()
        for outputs in self.prog_candidate_val_outputs:
            all_val_ids.update(outputs.keys())

        if not all_val_ids:
            logger.warning("No validation instances found, cannot compute scores")
            return

        # Compute scores per validation instance
        scores_per_instance = bradley_terry_scores_per_instance(
            programs=list(range(len(self.program_candidates))),
            comparison_results=self.comparison_cache,
            iterations=self.bradley_terry_iterations,
            tolerance=self.bradley_terry_tolerance,
        )

        # Reorganize from {data_id -> {prog_idx -> score}} to {prog_idx -> {data_id -> score}}
        for data_id, prog_scores in scores_per_instance.items():
            for prog_idx, score in prog_scores.items():
                if prog_idx < len(self._cached_scores):
                    self._cached_scores[prog_idx][data_id] = score

        # Fill in missing scores for programs not evaluated on all instances
        for prog_idx in range(len(self.program_candidates)):
            evaluated_instances = set(self.prog_candidate_val_outputs[prog_idx].keys())
            for val_id in evaluated_instances:
                if val_id not in self._cached_scores[prog_idx]:
                    # Program was evaluated but no comparisons yet - assign neutral score
                    self._cached_scores[prog_idx][val_id] = 0.5

        logger.debug(f"Recomputed scores for {len(all_val_ids)} validation instances")

    def add_program_with_comparisons(
        self,
        parent_program_idx: list[ProgramIdx],
        new_program: dict[str, str],
        valset_outputs: dict[DataId, RolloutOutput],
        comparator: Any,  # PairwiseComparator
        valset_data_instances: dict[DataId, Any],  # DataInst
        run_dir: str | None,
        num_metric_calls_by_discovery: int,
    ) -> ProgramIdx:
        """
        Add a new program and perform pairwise comparisons with existing programs.

        This method:
        1. Stores new program and outputs
        2. Compares with all existing programs on common instances
        3. Records comparisons
        4. Invalidates score cache
        5. Updates pareto fronts (will use recomputed scores)

        Args:
            parent_program_idx: Parent program indices
            new_program: New program candidate
            valset_outputs: Outputs for new program on validation set
            comparator: Pairwise comparator to use
            valset_data_instances: Validation data instances
            run_dir: Directory to save outputs
            num_metric_calls_by_discovery: Number of metric calls so far

        Returns:
            Index of newly added program
        """
        new_program_idx = len(self.program_candidates)

        # Store program and outputs
        self.program_candidates.append(new_program)
        self.prog_candidate_val_outputs.append(valset_outputs)
        self.parent_program_for_candidate.append(list(parent_program_idx))
        self.num_metric_calls_by_discovery.append(num_metric_calls_by_discovery)

        # Handle named predictor tracking
        max_predictor_id = max(
            [self.named_predictor_id_to_update_next_for_program_candidate[p] for p in parent_program_idx],
            default=0,
        )
        self.named_predictor_id_to_update_next_for_program_candidate.append(max_predictor_id)

        # Perform pairwise comparisons with all existing programs
        for existing_idx in range(new_program_idx):
            existing_outputs = self.prog_candidate_val_outputs[existing_idx]

            # Find common validation instances
            common_ids = set(valset_outputs.keys()) & set(existing_outputs.keys())

            for data_id in common_ids:
                new_output = valset_outputs[data_id]
                existing_output = existing_outputs[data_id]
                data_instance = valset_data_instances.get(data_id)

                if data_instance is None:
                    logger.warning(f"Data instance {data_id} not found, skipping comparison")
                    continue

                # Perform comparison
                try:
                    result = comparator.compare(new_output, existing_output, data_instance)

                    # Store comparison (new vs existing)
                    self.comparison_cache[(new_program_idx, existing_idx, data_id)] = result

                    # Record for logging/analysis
                    self.comparison_records.append(
                        PairwiseComparisonRecord(
                            program_a=new_program_idx,
                            program_b=existing_idx,
                            data_id=data_id,
                            result=result,
                            timestamp=self.i + 1,
                        )
                    )

                except Exception as e:
                    logger.error(
                        f"Comparison failed for programs {new_program_idx} vs {existing_idx} "
                        f"on instance {data_id}: {e}"
                    )
                    # Continue with other comparisons

        # Invalidate score cache - scores will be recomputed on next access
        self._cache_valid = False

        # Note: Pareto front updates happen in parent class via update_state_with_new_program
        # Those will use the recomputed scores from Bradley-Terry

        return new_program_idx

    def get_comparison_stats(self) -> dict[str, Any]:
        """
        Get statistics about pairwise comparisons.

        Returns:
            Dict with comparison statistics
        """
        total_comparisons = len(self.comparison_cache)

        # Count by result type
        result_counts = defaultdict(int)
        for result in self.comparison_cache.values():
            result_counts[result.value if hasattr(result, 'value') else str(result)] += 1

        # Comparisons per program
        comparisons_per_program = defaultdict(int)
        for (prog_a, prog_b, _), _ in self.comparison_cache.items():
            comparisons_per_program[prog_a] += 1
            comparisons_per_program[prog_b] += 1

        avg_comparisons_per_program = (
            sum(comparisons_per_program.values()) / len(self.program_candidates)
            if self.program_candidates
            else 0
        )

        return {
            "total_programs": len(self.program_candidates),
            "total_comparisons": total_comparisons,
            "result_counts": dict(result_counts),
            "avg_comparisons_per_program": avg_comparisons_per_program,
            "cache_valid": self._cache_valid,
            "bradley_terry_iterations": self.bradley_terry_iterations,
        }

    @staticmethod
    def create_from_seed(
        seed_candidate: dict[str, str],
        base_valset_outputs: dict[DataId, RolloutOutput],
        comparator: Any | None = None,
        track_best_outputs: bool = False,
        bradley_terry_iterations: int = 100,
    ) -> "HybridGEPAState":
        """
        Create hybrid state from seed candidate without initial scores.

        For pairwise-only mode, we don't have initial scores. This method
        creates the state with default scores that will be overwritten once
        comparisons are made.

        Args:
            seed_candidate: Initial candidate program
            base_valset_outputs: Outputs for seed on validation set
            comparator: Optional comparator (not used for seed)
            track_best_outputs: Whether to track best outputs
            bradley_terry_iterations: Max iterations for Bradley-Terry

        Returns:
            Initialized HybridGEPAState
        """
        # Create default scores (will be recomputed after comparisons)
        default_scores = {data_id: 0.5 for data_id in base_valset_outputs.keys()}

        return HybridGEPAState(
            seed_candidate=seed_candidate,
            base_valset_eval_output=(base_valset_outputs, default_scores),
            track_best_outputs=track_best_outputs,
            bradley_terry_iterations=bradley_terry_iterations,
        )
