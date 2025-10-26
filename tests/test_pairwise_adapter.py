"""
Test adapter for synthetic pairwise comparison problems.

Integrates synthetic problems with GEPA's adapter interface.
"""

from dataclasses import dataclass
from typing import Any

from test_pairwise_synthetic import (
    ComparisonResult,
    IntegerMutationStrategy,
    MultiObjectiveComparisonResult,
    MultiObjectiveTradeoff,
    QuadraticProblem,
    ThreeObjectiveProblem,
)


# ============================================================================
# Simplified GEPA Adapter Interface (for testing)
# ============================================================================

@dataclass
class EvaluationBatch:
    """Container for evaluation results."""
    outputs: list[int]  # For synthetic problems, output is just the integer value
    scores: list[float]  # Optional: can be empty for pure pairwise
    trajectories: list[dict] | None = None


# ============================================================================
# Single-Objective Synthetic Adapter
# ============================================================================

class SingleObjectiveSyntheticAdapter:
    """
    Adapter for single-objective synthetic problems.

    Candidate format: {"x": str(integer_value)}
    Output: integer value
    Comparison: via problem.compare()
    """

    def __init__(self, problem: QuadraticProblem, param_name: str = "x"):
        self.problem = problem
        self.param_name = param_name
        self.comparison_count = 0

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """
        Evaluate candidate on batch.

        Args:
            batch: List of data instances (unused for synthetic problems)
            candidate: {"x": "42"} - parameter as string
            capture_traces: Whether to capture execution traces

        Returns:
            EvaluationBatch with outputs (integer values)
        """
        # Parse parameter value
        x_value = int(candidate[self.param_name])

        # For synthetic problems, output is the same for all instances
        # (We could vary by instance if needed)
        outputs = [x_value] * len(batch)

        # Optional: compute scores (for comparison with scalar GEPA)
        scores = [self.problem.evaluate(x_value)] * len(batch)

        trajectories = None
        if capture_traces:
            trajectories = [
                {
                    "candidate": candidate,
                    "x_value": x_value,
                    "instance_id": instance["id"],
                }
                for instance in batch
            ]

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def compare(
        self,
        output_a: int,
        output_b: int,
        data_instance: dict,
    ) -> ComparisonResult:
        """
        Compare two outputs using the problem's comparison function.

        Args:
            output_a: First output (integer value)
            output_b: Second output (integer value)
            data_instance: Data instance (unused for synthetic problems)

        Returns:
            ComparisonResult
        """
        self.comparison_count += 1
        return self.problem.compare(output_a, output_b)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
        reference_outputs: list[int] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Build reflective dataset.

        For synthetic problems, we can provide simple feedback based on comparisons.
        """
        x_value = int(candidate[self.param_name])

        dataset = []

        for i, (output, traj) in enumerate(
            zip(eval_batch.outputs, eval_batch.trajectories or [])
        ):
            feedback = f"Current value: {x_value}"

            # If we have reference outputs, compare
            if reference_outputs and reference_outputs[i] is not None:
                ref_value = reference_outputs[i]
                comp_result = self.compare(output, ref_value, {})

                if comp_result == ComparisonResult.B_BETTER:
                    feedback += f"\nReference value {ref_value} is better."
                elif comp_result == ComparisonResult.A_BETTER:
                    feedback += f"\nCurrent value is better than reference {ref_value}."
                else:
                    feedback += f"\nCurrent value is similar to reference {ref_value}."

            dataset.append({
                "Inputs": {"instance": traj.get("instance_id") if traj else i},
                "Generated Output": output,
                "Feedback": feedback,
            })

        return {self.param_name: dataset}

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        Propose new parameter value using mutation strategy.

        For synthetic problems, we use simple integer mutation.
        """
        mutator = IntegerMutationStrategy(
            param_name=self.param_name,
            seed=hash(str(candidate)) % (2**31),  # Deterministic but varied
        )

        current_value = int(candidate[self.param_name])
        new_value = mutator.mutate(current_value)

        return {self.param_name: str(new_value)}


# ============================================================================
# Multi-Objective Synthetic Adapter
# ============================================================================

class MultiObjectiveSyntheticAdapter:
    """
    Adapter for multi-objective synthetic problems.

    Supports problems with multiple competing objectives.
    """

    def __init__(
        self,
        problem: MultiObjectiveTradeoff | ThreeObjectiveProblem,
        param_name: str = "x",
    ):
        self.problem = problem
        self.param_name = param_name
        self.comparison_count = 0

        # Determine objectives based on problem type
        if isinstance(problem, MultiObjectiveTradeoff):
            self.objectives = ["accuracy", "efficiency"]
        elif isinstance(problem, ThreeObjectiveProblem):
            self.objectives = ["obj_a", "obj_b", "obj_c"]
        else:
            self.objectives = []

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Evaluate candidate on batch."""
        x_value = int(candidate[self.param_name])
        outputs = [x_value] * len(batch)

        # For multi-objective, scores could be vectors
        # For now, return empty (will use pairwise comparison)
        scores = []

        trajectories = None
        if capture_traces:
            trajectories = [
                {
                    "candidate": candidate,
                    "x_value": x_value,
                    "instance_id": instance["id"],
                }
                for instance in batch
            ]

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def compare(
        self,
        output_a: int,
        output_b: int,
        data_instance: dict,
    ) -> MultiObjectiveComparisonResult:
        """Compare two outputs across all objectives."""
        self.comparison_count += 1
        return self.problem.compare(output_a, output_b)

    def get_objectives(self) -> list[str]:
        """Return list of objective IDs."""
        return self.objectives

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
        reference_outputs: list[int] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Build reflective dataset for multi-objective case."""
        x_value = int(candidate[self.param_name])

        dataset = []

        for i, (output, traj) in enumerate(
            zip(eval_batch.outputs, eval_batch.trajectories or [])
        ):
            feedback_parts = [f"Current value: {x_value}"]

            # If we have reference outputs, compare per objective
            if reference_outputs and reference_outputs[i] is not None:
                ref_value = reference_outputs[i]
                comp_result = self.compare(output, ref_value, {})

                feedback_parts.append("\nPer-objective comparison with reference:")
                for obj_id, result in comp_result.objective_results.items():
                    feedback_parts.append(f"  {obj_id}: {result.value}")

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


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SYNTHETIC ADAPTER TESTING")
    print("=" * 80)

    # Test single-objective adapter
    print("\n1. SINGLE-OBJECTIVE ADAPTER")
    print("-" * 80)

    problem = QuadraticProblem(target=42)
    adapter = SingleObjectiveSyntheticAdapter(problem)

    # Create test batch
    batch = [{"id": 0}, {"id": 1}, {"id": 2}]

    # Evaluate a candidate
    candidate = {"x": "40"}
    result = adapter.evaluate(batch, candidate, capture_traces=True)

    print(f"Candidate: {candidate}")
    print(f"Outputs: {result.outputs}")
    print(f"Scores: {result.scores}")
    print(f"Trajectories: {result.trajectories[0]}")

    # Compare two candidates
    comp_result = adapter.compare(40, 50, {})
    print(f"\nCompare 40 vs 50: {comp_result}")

    # Build reflective dataset
    ref_outputs = [42, 42, 42]  # Reference is optimal
    reflective_ds = adapter.make_reflective_dataset(
        candidate, result, ["x"], ref_outputs
    )
    print(f"\nReflective dataset sample:")
    print(f"  {reflective_ds['x'][0]}")

    # Propose new candidate
    new_texts = adapter.propose_new_texts(candidate, reflective_ds, ["x"])
    print(f"\nProposed new candidate: {new_texts}")

    # Test multi-objective adapter
    print("\n2. MULTI-OBJECTIVE ADAPTER")
    print("-" * 80)

    problem2 = MultiObjectiveTradeoff(accuracy_target=100)
    adapter2 = MultiObjectiveSyntheticAdapter(problem2)

    print(f"Objectives: {adapter2.get_objectives()}")

    candidate2 = {"x": "50"}
    result2 = adapter2.evaluate(batch, candidate2)

    print(f"Candidate: {candidate2}")
    print(f"Outputs: {result2.outputs}")

    # Compare with different candidate
    comp_result2 = adapter2.compare(50, 100, {})
    print(f"\nCompare 50 vs 100:")
    for obj, res in comp_result2.objective_results.items():
        print(f"  {obj}: {res.value}")

    print("\n" + "=" * 80)
    print("Adapters ready for use with GEPA!")
    print("=" * 80)
