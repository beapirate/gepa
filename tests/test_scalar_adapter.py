"""
Scalar GEPA adapters for synthetic problems.

These adapters return scalar scores directly (traditional GEPA approach).
Used for comparison with pairwise GEPA.
"""

from dataclasses import dataclass
from typing import Any

from test_pairwise_synthetic import (
    IntegerMutationStrategy,
    MultiObjectiveTradeoff,
    NonConvexProblem,
    QuadraticProblem,
    ThreeObjectiveProblem,
)


# ============================================================================
# Simplified GEPA Adapter Interface (for testing)
# ============================================================================

@dataclass
class ScalarEvaluationBatch:
    """Container for scalar evaluation results."""
    outputs: list[int]  # Output values
    scores: list[float]  # Scalar scores (higher is better)
    trajectories: list[dict] | None = None


# ============================================================================
# Single-Objective Scalar Adapter
# ============================================================================

class ScalarSyntheticAdapter:
    """
    Traditional GEPA adapter that returns scalar scores directly.

    For comparison with pairwise approach.
    """

    def __init__(
        self,
        problem: QuadraticProblem | NonConvexProblem,
        param_name: str = "x",
    ):
        self.problem = problem
        self.param_name = param_name
        self.evaluation_count = 0

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> ScalarEvaluationBatch:
        """
        Evaluate candidate on batch and return scalar scores.

        Args:
            batch: List of data instances
            candidate: {"x": "42"} - parameter as string
            capture_traces: Whether to capture execution traces

        Returns:
            ScalarEvaluationBatch with scalar scores
        """
        # Parse parameter value
        x_value = int(candidate[self.param_name])

        # For synthetic problems, output is the same for all instances
        outputs = [x_value] * len(batch)

        # Compute scalar scores (negate because problem returns loss, we want higher=better)
        raw_score = self.problem.evaluate(x_value)

        # Convert to "higher is better" and normalize to reasonable range
        # For quadratic: distance can be 0-10000, we want scores in [0, 1] range
        # Simple transformation: score = 1 / (1 + distance)
        scalar_score = 1.0 / (1.0 + raw_score)

        scores = [scalar_score] * len(batch)

        self.evaluation_count += len(batch)

        trajectories = None
        if capture_traces:
            trajectories = [
                {
                    "candidate": candidate,
                    "x_value": x_value,
                    "instance_id": instance["id"],
                    "raw_loss": raw_score,
                    "scalar_score": scalar_score,
                }
                for instance in batch
            ]

        return ScalarEvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: ScalarEvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build reflective dataset using scalar scores."""
        x_value = int(candidate[self.param_name])

        dataset = []

        for i, (output, score, traj) in enumerate(
            zip(
                eval_batch.outputs,
                eval_batch.scores,
                eval_batch.trajectories or [{}] * len(eval_batch.outputs),
            )
        ):
            feedback = f"Current value: {x_value}, Score: {score:.3f}"

            if score < 0.5:
                feedback += " (Poor - needs improvement)"
            elif score < 0.8:
                feedback += " (Moderate - can be better)"
            else:
                feedback += " (Good - close to optimal)"

            dataset.append({
                "Inputs": {"instance": traj.get("instance_id", i)},
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
        """Propose new parameter value using mutation strategy."""
        mutator = IntegerMutationStrategy(
            param_name=self.param_name,
            seed=hash(str(candidate)) % (2**31),
        )

        current_value = int(candidate[self.param_name])
        new_value = mutator.mutate(current_value)

        return {self.param_name: str(new_value)}


# ============================================================================
# Multi-Objective Scalar Adapter
# ============================================================================

class MultiObjectiveScalarAdapter:
    """
    Scalar adapter for multi-objective problems.

    Returns score vectors (one score per objective).
    Compatible with standard GEPA if we aggregate to single score.
    """

    def __init__(
        self,
        problem: MultiObjectiveTradeoff | ThreeObjectiveProblem,
        param_name: str = "x",
        aggregation: str = "sum",  # "sum", "product", "min", or "weighted"
        weights: dict[str, float] | None = None,
    ):
        self.problem = problem
        self.param_name = param_name
        self.aggregation = aggregation
        self.evaluation_count = 0

        # Determine objectives
        if isinstance(problem, MultiObjectiveTradeoff):
            self.objectives = ["accuracy", "efficiency"]
            self.eval_funcs = {
                "accuracy": problem.evaluate_accuracy,
                "efficiency": problem.evaluate_efficiency,
            }
        elif isinstance(problem, ThreeObjectiveProblem):
            self.objectives = ["obj_a", "obj_b", "obj_c"]
            self.eval_funcs = {
                "obj_a": problem.evaluate_obj_a,
                "obj_b": problem.evaluate_obj_b,
                "obj_c": problem.evaluate_obj_c,
            }
        else:
            self.objectives = []
            self.eval_funcs = {}

        # Default weights (equal)
        self.weights = weights or {obj: 1.0 for obj in self.objectives}

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> ScalarEvaluationBatch:
        """
        Evaluate candidate and return aggregated scalar scores.

        For multi-objective, we aggregate to a single score for traditional GEPA.
        """
        x_value = int(candidate[self.param_name])
        outputs = [x_value] * len(batch)

        # Compute per-objective scores
        obj_losses = {
            obj_id: eval_func(x_value)
            for obj_id, eval_func in self.eval_funcs.items()
        }

        # Convert losses to scores (higher is better)
        obj_scores = {
            obj_id: 1.0 / (1.0 + loss)
            for obj_id, loss in obj_losses.items()
        }

        # Aggregate to single score
        if self.aggregation == "sum":
            aggregated_score = sum(
                self.weights[obj] * score for obj, score in obj_scores.items()
            )
        elif self.aggregation == "product":
            aggregated_score = 1.0
            for obj, score in obj_scores.items():
                aggregated_score *= (score ** self.weights[obj])
        elif self.aggregation == "min":
            aggregated_score = min(obj_scores.values())
        elif self.aggregation == "weighted":
            total_weight = sum(self.weights.values())
            aggregated_score = sum(
                self.weights[obj] * score for obj, score in obj_scores.items()
            ) / total_weight
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        scores = [aggregated_score] * len(batch)
        self.evaluation_count += len(batch)

        trajectories = None
        if capture_traces:
            trajectories = [
                {
                    "candidate": candidate,
                    "x_value": x_value,
                    "instance_id": instance["id"],
                    "obj_losses": obj_losses,
                    "obj_scores": obj_scores,
                    "aggregated_score": aggregated_score,
                }
                for instance in batch
            ]

        return ScalarEvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: ScalarEvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build reflective dataset."""
        x_value = int(candidate[self.param_name])

        dataset = []

        for i, (output, score, traj) in enumerate(
            zip(
                eval_batch.outputs,
                eval_batch.scores,
                eval_batch.trajectories or [{}] * len(eval_batch.outputs),
            )
        ):
            feedback_parts = [f"Current value: {x_value}"]
            feedback_parts.append(f"Aggregated score: {score:.3f}")

            if "obj_scores" in traj:
                feedback_parts.append("\nPer-objective scores:")
                for obj_id, obj_score in traj["obj_scores"].items():
                    feedback_parts.append(f"  {obj_id}: {obj_score:.3f}")

            dataset.append({
                "Inputs": {"instance": traj.get("instance_id", i)},
                "Generated Output": output,
                "Feedback": "\n".join(feedback_parts),
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

    def get_objectives(self) -> list[str]:
        """Return list of objective IDs."""
        return self.objectives


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SCALAR ADAPTER TESTING")
    print("=" * 80)

    # Test single-objective adapter
    print("\n1. SINGLE-OBJECTIVE SCALAR ADAPTER")
    print("-" * 80)

    problem = QuadraticProblem(target=42)
    adapter = ScalarSyntheticAdapter(problem)

    # Create test batch
    batch = [{"id": 0}, {"id": 1}, {"id": 2}]

    # Evaluate candidates
    for x in [30, 40, 42, 45, 60]:
        candidate = {"x": str(x)}
        result = adapter.evaluate(batch, candidate)
        print(f"x={x:3d}  score={result.scores[0]:.4f}  "
              f"(distance={(x-42)**2:4.0f})")

    print("\n2. MULTI-OBJECTIVE SCALAR ADAPTER")
    print("-" * 80)

    problem2 = MultiObjectiveTradeoff(accuracy_target=100)

    # Test different aggregation strategies
    for agg in ["sum", "product", "weighted"]:
        print(f"\nAggregation: {agg}")
        adapter2 = MultiObjectiveScalarAdapter(
            problem2,
            aggregation=agg,
            weights={"accuracy": 0.7, "efficiency": 0.3},
        )

        for x in [20, 50, 80, 100]:
            candidate = {"x": str(x)}
            result = adapter2.evaluate(batch, candidate, capture_traces=True)
            traj = result.trajectories[0]
            print(f"  x={x:3d}  aggregated={result.scores[0]:.4f}  "
                  f"acc={traj['obj_scores']['accuracy']:.3f}  "
                  f"eff={traj['obj_scores']['efficiency']:.3f}")

    print("\n" + "=" * 80)
    print("Scalar adapters ready for GEPA comparison!")
    print("=" * 80)
