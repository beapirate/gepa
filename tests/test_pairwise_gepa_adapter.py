"""
Pairwise GEPA adapter with Bradley-Terry score conversion.

This adapter stores pairwise comparisons and converts them to scalar scores
on-demand, allowing it to work with standard GEPA while preserving pairwise
comparison data.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from test_pairwise_adapter import EvaluationBatch
from test_pairwise_algorithms import bradley_terry_scores
from test_pairwise_synthetic import (
    ComparisonResult,
    IntegerMutationStrategy,
    MultiObjectiveComparisonResult,
    MultiObjectiveTradeoff,
    NonConvexProblem,
    QuadraticProblem,
    ThreeObjectiveProblem,
)


# ============================================================================
# Pairwise Adapter with Score Conversion (Hybrid Approach)
# ============================================================================

class PairwiseGEPAAdapter:
    """
    GEPA adapter that performs pairwise comparisons and converts to scalar scores.

    This is the "hybrid" approach: store pairwise comparisons but compute
    scalar scores on-demand using Bradley-Terry. This allows reusing all
    standard GEPA code while working with pairwise comparison data.

    Architecture:
    1. evaluate() returns outputs (no scores initially)
    2. Pairwise comparisons performed and cached
    3. Scores computed on-demand via get_scores()
    4. Bradley-Terry conversion happens lazily
    """

    def __init__(
        self,
        problem: QuadraticProblem | NonConvexProblem,
        param_name: str = "x",
    ):
        self.problem = problem
        self.param_name = param_name

        # Track all outputs and comparisons
        self.program_outputs: list[int] = []  # Output for each program evaluated
        self.comparison_cache: dict[tuple[int, int], ComparisonResult] = {}
        self.comparison_count = 0
        self.evaluation_count = 0

        # Cached scores (invalidated when new comparisons added)
        self._cached_scores: dict[int, float] | None = None
        self._cache_valid = False

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """
        Evaluate candidate and store outputs.

        Returns outputs but no scores initially. Scores will be computed
        via pairwise comparisons later.
        """
        x_value = int(candidate[self.param_name])
        outputs = [x_value] * len(batch)

        # Store this program's output
        program_idx = len(self.program_outputs)
        self.program_outputs.append(x_value)

        self.evaluation_count += len(batch)

        # Compare with all previous programs
        for prev_idx in range(program_idx):
            prev_x = self.program_outputs[prev_idx]
            result = self.problem.compare(x_value, prev_x)
            self.comparison_cache[(program_idx, prev_idx)] = result
            self.comparison_count += 1

        # Invalidate score cache
        self._cache_valid = False

        trajectories = None
        if capture_traces:
            trajectories = [
                {
                    "candidate": candidate,
                    "x_value": x_value,
                    "program_idx": program_idx,
                    "instance_id": instance["id"],
                }
                for instance in batch
            ]

        # Return empty scores - they'll be computed on-demand
        return EvaluationBatch(
            outputs=outputs,
            scores=[],  # Empty - use get_scores() to compute
            trajectories=trajectories,
        )

    def get_scores(self, program_indices: list[int] | None = None) -> dict[int, float]:
        """
        Compute scalar scores from pairwise comparisons using Bradley-Terry.

        Args:
            program_indices: Which programs to score (None = all)

        Returns:
            Dict mapping program index -> scalar score
        """
        if program_indices is None:
            program_indices = list(range(len(self.program_outputs)))

        # Use cached scores if valid
        if self._cache_valid and self._cached_scores is not None:
            return {
                idx: self._cached_scores.get(idx, 0.5)
                for idx in program_indices
            }

        # Compute scores using Bradley-Terry
        scores = bradley_terry_scores(
            programs=list(range(len(self.program_outputs))),
            comparison_results=self.comparison_cache,
        )

        # Cache scores
        self._cached_scores = scores
        self._cache_valid = True

        return {idx: scores.get(idx, 0.5) for idx in program_indices}

    def get_scores_for_batch(
        self, batch: list[dict], program_idx: int
    ) -> list[float]:
        """
        Get scores for a specific program on a batch.

        For synthetic problems, all instances have same score.
        """
        scores = self.get_scores([program_idx])
        score = scores.get(program_idx, 0.5)
        return [score] * len(batch)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build reflective dataset using comparison-based feedback."""
        x_value = int(candidate[self.param_name])
        program_idx = len(self.program_outputs) - 1  # Most recent

        # Get score for this program
        scores = self.get_scores([program_idx])
        score = scores.get(program_idx, 0.5)

        dataset = []

        for i, (output, traj) in enumerate(
            zip(eval_batch.outputs, eval_batch.trajectories or [])
        ):
            feedback = f"Current value: {x_value}, Estimated score: {score:.3f}"

            # Add comparison feedback
            if self.comparison_count > 1:
                # Find how this compares to other programs
                wins = sum(
                    1
                    for (a, b), result in self.comparison_cache.items()
                    if a == program_idx and result == ComparisonResult.A_BETTER
                )
                losses = sum(
                    1
                    for (a, b), result in self.comparison_cache.items()
                    if a == program_idx and result == ComparisonResult.B_BETTER
                )

                feedback += f"\nWins: {wins}, Losses: {losses}"

            dataset.append({
                "Inputs": {"instance": traj.get("instance_id") if traj else i},
                "Generated Output": output,
                "Feedback": feedback,
                "Score": score,
            })

        return {self.param_name: dataset}

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Propose new parameter value."""
        mutator = IntegerMutationStrategy(
            param_name=self.param_name,
            seed=hash(str(candidate)) % (2**31),
        )

        current_value = int(candidate[self.param_name])
        new_value = mutator.mutate(current_value)

        return {self.param_name: str(new_value)}

    def get_comparison_stats(self) -> dict:
        """Get statistics about pairwise comparisons."""
        return {
            "total_programs": len(self.program_outputs),
            "total_comparisons": self.comparison_count,
            "comparisons_per_program": (
                self.comparison_count / len(self.program_outputs)
                if self.program_outputs
                else 0
            ),
            "cache_valid": self._cache_valid,
        }


# ============================================================================
# Multi-Objective Pairwise Adapter
# ============================================================================

class MultiObjectivePairwiseGEPAAdapter:
    """
    Multi-objective pairwise adapter with per-objective Bradley-Terry conversion.

    Stores per-objective comparisons and computes per-objective scores.
    For compatibility with standard GEPA, aggregates to single score.
    """

    def __init__(
        self,
        problem: MultiObjectiveTradeoff | ThreeObjectiveProblem,
        param_name: str = "x",
        aggregation: str = "sum",
        weights: dict[str, float] | None = None,
    ):
        self.problem = problem
        self.param_name = param_name
        self.aggregation = aggregation

        # Determine objectives
        if isinstance(problem, MultiObjectiveTradeoff):
            self.objectives = ["accuracy", "efficiency"]
        elif isinstance(problem, ThreeObjectiveProblem):
            self.objectives = ["obj_a", "obj_b", "obj_c"]
        else:
            self.objectives = []

        self.weights = weights or {obj: 1.0 for obj in self.objectives}

        # Track outputs and per-objective comparisons
        self.program_outputs: list[int] = []
        self.comparison_cache_per_objective: dict[
            str, dict[tuple[int, int], ComparisonResult]
        ] = {obj: {} for obj in self.objectives}
        self.comparison_count = 0
        self.evaluation_count = 0

        # Cached per-objective scores
        self._cached_obj_scores: dict[str, dict[int, float]] | None = None
        self._cache_valid = False

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Evaluate and perform per-objective comparisons."""
        x_value = int(candidate[self.param_name])
        outputs = [x_value] * len(batch)

        # Store output
        program_idx = len(self.program_outputs)
        self.program_outputs.append(x_value)

        self.evaluation_count += len(batch)

        # Compare with all previous programs on each objective
        for prev_idx in range(program_idx):
            prev_x = self.program_outputs[prev_idx]
            multi_result = self.problem.compare(x_value, prev_x)

            for obj_id, result in multi_result.objective_results.items():
                self.comparison_cache_per_objective[obj_id][
                    (program_idx, prev_idx)
                ] = result
                self.comparison_count += 1

        # Invalidate cache
        self._cache_valid = False

        trajectories = None
        if capture_traces:
            trajectories = [
                {
                    "candidate": candidate,
                    "x_value": x_value,
                    "program_idx": program_idx,
                    "instance_id": instance["id"],
                }
                for instance in batch
            ]

        return EvaluationBatch(
            outputs=outputs,
            scores=[],
            trajectories=trajectories,
        )

    def get_objective_scores(
        self, program_indices: list[int] | None = None
    ) -> dict[str, dict[int, float]]:
        """
        Compute per-objective scores using Bradley-Terry.

        Returns:
            Dict mapping objective_id -> {program_idx -> score}
        """
        if program_indices is None:
            program_indices = list(range(len(self.program_outputs)))

        # Use cache if valid
        if self._cache_valid and self._cached_obj_scores is not None:
            return {
                obj_id: {
                    idx: self._cached_obj_scores[obj_id].get(idx, 0.5)
                    for idx in program_indices
                }
                for obj_id in self.objectives
            }

        # Compute per-objective scores
        obj_scores = {}

        for obj_id in self.objectives:
            scores = bradley_terry_scores(
                programs=list(range(len(self.program_outputs))),
                comparison_results=self.comparison_cache_per_objective[obj_id],
            )
            obj_scores[obj_id] = scores

        # Cache
        self._cached_obj_scores = obj_scores
        self._cache_valid = True

        return {
            obj_id: {idx: obj_scores[obj_id].get(idx, 0.5) for idx in program_indices}
            for obj_id in self.objectives
        }

    def get_scores(self, program_indices: list[int] | None = None) -> dict[int, float]:
        """
        Get aggregated scalar scores for compatibility with standard GEPA.
        """
        obj_scores = self.get_objective_scores(program_indices)

        # Aggregate per-objective scores
        aggregated = {}

        for prog_idx in program_indices or range(len(self.program_outputs)):
            obj_vals = {
                obj_id: obj_scores[obj_id].get(prog_idx, 0.5)
                for obj_id in self.objectives
            }

            if self.aggregation == "sum":
                agg_score = sum(
                    self.weights[obj] * val for obj, val in obj_vals.items()
                )
            elif self.aggregation == "weighted":
                total_weight = sum(self.weights.values())
                agg_score = (
                    sum(self.weights[obj] * val for obj, val in obj_vals.items())
                    / total_weight
                )
            elif self.aggregation == "product":
                agg_score = 1.0
                for obj, val in obj_vals.items():
                    agg_score *= val ** self.weights[obj]
            else:
                agg_score = sum(obj_vals.values()) / len(obj_vals)

            aggregated[prog_idx] = agg_score

        return aggregated

    def get_scores_for_batch(
        self, batch: list[dict], program_idx: int
    ) -> list[float]:
        """Get aggregated scores for a program on a batch."""
        scores = self.get_scores([program_idx])
        score = scores.get(program_idx, 0.5)
        return [score] * len(batch)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build reflective dataset with per-objective feedback."""
        x_value = int(candidate[self.param_name])
        program_idx = len(self.program_outputs) - 1

        # Get per-objective scores
        obj_scores = self.get_objective_scores([program_idx])
        aggregated = self.get_scores([program_idx])

        dataset = []

        for i, (output, traj) in enumerate(
            zip(eval_batch.outputs, eval_batch.trajectories or [])
        ):
            feedback_parts = [f"Current value: {x_value}"]
            feedback_parts.append(
                f"Aggregated score: {aggregated.get(program_idx, 0.5):.3f}"
            )
            feedback_parts.append("\nPer-objective scores:")

            for obj_id in self.objectives:
                obj_score = obj_scores[obj_id].get(program_idx, 0.5)
                feedback_parts.append(f"  {obj_id}: {obj_score:.3f}")

            dataset.append({
                "Inputs": {"instance": traj.get("instance_id") if traj else i},
                "Generated Output": output,
                "Feedback": "\n".join(feedback_parts),
            })

        return {self.param_name: dataset}

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Propose new parameter value."""
        mutator = IntegerMutationStrategy(
            param_name=self.param_name,
            seed=hash(str(candidate)) % (2**31),
        )

        current_value = int(candidate[self.param_name])
        new_value = mutator.mutate(current_value)

        return {self.param_name: str(new_value)}

    def get_objectives(self) -> list[str]:
        """Return list of objective IDs."""
        return self.objectives

    def get_comparison_stats(self) -> dict:
        """Get statistics about comparisons."""
        return {
            "total_programs": len(self.program_outputs),
            "total_comparisons_per_objective": {
                obj: len(cache)
                for obj, cache in self.comparison_cache_per_objective.items()
            },
            "total_comparisons": self.comparison_count,
            "cache_valid": self._cache_valid,
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PAIRWISE GEPA ADAPTER WITH BRADLEY-TERRY CONVERSION")
    print("=" * 80)

    # Test single-objective
    print("\n1. SINGLE-OBJECTIVE PAIRWISE ADAPTER")
    print("-" * 80)

    problem = QuadraticProblem(target=42)
    adapter = PairwiseGEPAAdapter(problem)

    batch = [{"id": 0}, {"id": 1}]

    # Evaluate several candidates
    candidates_x = [20, 40, 42, 50, 70]

    for x in candidates_x:
        candidate = {"x": str(x)}
        result = adapter.evaluate(batch, candidate, capture_traces=True)
        print(f"Evaluated x={x}")

    # Get scores via Bradley-Terry
    print("\nComputed scores (Bradley-Terry):")
    scores = adapter.get_scores()
    for idx, x in enumerate(candidates_x):
        score = scores.get(idx, 0.5)
        true_dist = (x - 42) ** 2
        print(f"  x={x:3d}  score={score:.4f}  true_distance={true_dist:4.0f}")

    stats = adapter.get_comparison_stats()
    print(f"\nComparison stats: {stats}")

    # Test multi-objective
    print("\n2. MULTI-OBJECTIVE PAIRWISE ADAPTER")
    print("-" * 80)

    problem2 = MultiObjectiveTradeoff(accuracy_target=100)
    adapter2 = MultiObjectivePairwiseGEPAAdapter(
        problem2,
        aggregation="weighted",
        weights={"accuracy": 0.7, "efficiency": 0.3},
    )

    for x in [20, 50, 80, 100]:
        candidate = {"x": str(x)}
        adapter2.evaluate(batch, candidate)
        print(f"Evaluated x={x}")

    # Get per-objective scores
    print("\nPer-objective scores (Bradley-Terry):")
    obj_scores = adapter2.get_objective_scores()

    for idx, x in enumerate([20, 50, 80, 100]):
        print(f"  x={x:3d}  ", end="")
        for obj in adapter2.get_objectives():
            score = obj_scores[obj].get(idx, 0.5)
            print(f"{obj}={score:.3f}  ", end="")
        agg = adapter2.get_scores([idx])[idx]
        print(f"aggregated={agg:.3f}")

    stats2 = adapter2.get_comparison_stats()
    print(f"\nComparison stats: {stats2}")

    print("\n" + "=" * 80)
    print("Pairwise adapters ready for GEPA comparison!")
    print("=" * 80)
