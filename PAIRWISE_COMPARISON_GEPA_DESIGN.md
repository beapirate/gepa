# GEPA with Pairwise Comparison Metrics: Design Document

## Executive Summary

This document describes how to implement a version of GEPA that uses **pairwise comparison metrics** instead of scalar floating-point metrics. In pairwise comparison mode, we evaluate programs by comparing their outputs directly (e.g., "Which output is better: A or B?") rather than assigning numerical scores.

**Key Constraint**: We cannot map pairwise comparisons to scalar metrics. All decisions must be made directly from comparison results.

## 1. Current GEPA Architecture and Scalar Dependencies

### 1.1 Core Data Flow

Current GEPA uses scalar scores throughout:

```python
# adapter.py - Evaluation returns scalar scores
EvaluationBatch:
    outputs: list[RolloutOutput]
    scores: list[float]  # ← SCALAR SCORES
    trajectories: list[Trajectory] | None

# state.py - State stores scalar scores
GEPAState:
    prog_candidate_val_subscores: list[dict[DataId, float]]  # ← SCALARS
    pareto_front_valset: dict[DataId, float]  # ← SCALARS
    program_at_pareto_front_valset: dict[DataId, set[ProgramIdx]]
```

### 1.2 Critical Scalar Operations

| Operation | Location | Purpose | Scalar Dependency |
|-----------|----------|---------|-------------------|
| **Minibatch Acceptance** | `engine.py:261` | Accept if `sum(new_scores) > sum(old_scores)` | **SUM** aggregation |
| **Validation Scoring** | `state.py:157` | `sum(scores.values()) / len(scores)` | **MEAN** aggregation |
| **Pareto Front Updates** | `state.py:190` | `if score > prev_score` | **>** comparison |
| **Pareto Ties** | `state.py:200` | `elif score == prev_score` | **==** comparison |
| **Best Program Selection** | `eval_policy.py:48` | `max(avg_scores)` | **MAX** aggregation |
| **Merge Parent Sampling** | `merge.py:81` | `weights=[scores[i] for i in ...]` | **Weighted sampling** |
| **Merge Subsample Selection** | `merge.py:232` | `scores1[idx] > scores2[idx]` | **>** comparison |
| **Perfect Score Check** | `reflective_mutation.py:107` | `all(s >= perfect_score for s in scores)` | **≥** comparison |
| **Feedback Classification** | `default_adapter.py:114` | `if score > 0.0: "correct"` | **>** threshold |
| **Stop Conditions** | `stop_condition.py:76` | `max(scores) > threshold` | **> threshold** |

## 2. Pairwise Comparison GEPA: Core Design

### 2.1 New Evaluation Interface

Replace scalar scores with pairwise comparison results:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeVar, Generic

class ComparisonResult(Enum):
    """Result of comparing two outputs."""
    A_BETTER = "a_better"      # A is strictly better than B
    B_BETTER = "b_better"      # B is strictly better than A
    TIE = "tie"                # A and B are equally good
    INCOMPARABLE = "incomparable"  # Cannot determine (e.g., different dimensions)

@dataclass
class PairwiseEvaluationBatch(Generic[Trajectory, RolloutOutput]):
    """
    Container for pairwise evaluation results.

    Instead of scalar scores, we store outputs that can be compared pairwise.
    Actual comparisons are performed on-demand by the comparison function.
    """
    outputs: list[RolloutOutput]
    trajectories: list[Trajectory] | None = None

    # Optional: pre-computed comparison against a reference
    reference_comparisons: list[ComparisonResult] | None = None


class PairwiseComparator(Protocol[RolloutOutput]):
    """
    Protocol for comparing two outputs.

    Implementers must provide a function that compares two outputs and returns
    which one is better. This could use:
    - LLM-based evaluation ("Which answer is better?")
    - Human-in-the-loop feedback
    - Rule-based comparison (e.g., code correctness + efficiency)
    """

    def compare(
        self,
        output_a: RolloutOutput,
        output_b: RolloutOutput,
        data_instance: DataInst,
    ) -> ComparisonResult:
        """
        Compare two outputs for the same data instance.

        Args:
            output_a: First output to compare
            output_b: Second output to compare
            data_instance: The input data both outputs were generated from

        Returns:
            ComparisonResult indicating which output is better
        """
        ...


class PairwiseGEPAAdapter(Protocol[DataInst, Trajectory, RolloutOutput]):
    """
    Adapter for GEPA with pairwise comparisons.

    Key differences from standard GEPAAdapter:
    1. evaluate() returns outputs but no scalar scores
    2. New comparator() method provides pairwise comparison function
    3. make_reflective_dataset() must work without scalar scores
    """

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> PairwiseEvaluationBatch[Trajectory, RolloutOutput]:
        """
        Evaluate candidate on batch, returning outputs but no scalar scores.

        Returns:
            PairwiseEvaluationBatch with outputs and optional trajectories
        """
        ...

    def comparator(self) -> PairwiseComparator[RolloutOutput]:
        """
        Return a comparator that can compare two outputs.

        The comparator will be called many times during optimization,
        so consider caching or batching comparison requests if using LLMs.
        """
        ...

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: PairwiseEvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
        reference_outputs: list[RolloutOutput] | None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Build reflective dataset for instruction refinement.

        Args:
            candidate: Current candidate being evaluated
            eval_batch: Evaluation results (outputs + trajectories)
            components_to_update: Which components to update
            reference_outputs: Optional reference outputs for comparison
                (e.g., outputs from current best program)

        Returns:
            Reflective dataset suitable for LLM-based instruction proposal
        """
        ...
```

### 2.2 New State Management

Replace scalar score storage with pairwise comparison records:

```python
from collections import defaultdict

@dataclass
class PairwiseComparisonRecord:
    """Record of a single pairwise comparison."""
    program_a: ProgramIdx
    program_b: ProgramIdx
    data_id: DataId
    result: ComparisonResult
    timestamp: int  # Iteration when comparison was made


class PairwiseGEPAState(Generic[RolloutOutput, DataId]):
    """
    State for pairwise comparison GEPA.

    Key differences from standard GEPAState:
    1. No scalar scores stored
    2. Comparison records tracked instead
    3. Pareto front based on dominance relationships, not scalar comparisons
    4. Outputs stored for comparison purposes
    """

    # Program candidates and lineage (unchanged)
    program_candidates: list[dict[str, str]]
    parent_program_for_candidate: list[list[ProgramIdx | None]]

    # Comparison records: stores all pairwise comparisons performed
    comparison_records: list[PairwiseComparisonRecord]

    # Cached comparison results for efficiency
    # (program_a, program_b, data_id) -> ComparisonResult
    comparison_cache: dict[tuple[ProgramIdx, ProgramIdx, DataId], ComparisonResult]

    # Validation outputs: need to store to enable future comparisons
    # program_idx -> {data_id -> output}
    prog_candidate_val_outputs: list[dict[DataId, RolloutOutput]]

    # Pareto front: programs that are not dominated by any other program
    # data_id -> set of program indices on pareto front for that instance
    program_at_pareto_front_valset: dict[DataId, set[ProgramIdx]]

    # Win/loss records for bookkeeping (NOT used as primary metric)
    # program_idx -> {data_id -> (wins, losses, ties)}
    program_win_loss_records: list[dict[DataId, tuple[int, int, int]]]

    # Evaluation tracking (unchanged)
    i: int
    num_full_ds_evals: int
    total_num_evals: int
    num_metric_calls_by_discovery: list[int]

    def compare_programs(
        self,
        program_a: ProgramIdx,
        program_b: ProgramIdx,
        data_id: DataId,
        comparator: PairwiseComparator,
        data_instance: DataInst,
    ) -> ComparisonResult:
        """
        Compare two programs on a specific data instance.

        Uses cached result if available, otherwise performs comparison.
        """
        # Check cache (both orderings)
        cache_key = (program_a, program_b, data_id)
        if cache_key in self.comparison_cache:
            return self.comparison_cache[cache_key]

        reverse_key = (program_b, program_a, data_id)
        if reverse_key in self.comparison_cache:
            result = self.comparison_cache[reverse_key]
            # Reverse the result
            if result == ComparisonResult.A_BETTER:
                return ComparisonResult.B_BETTER
            elif result == ComparisonResult.B_BETTER:
                return ComparisonResult.A_BETTER
            else:
                return result

        # Perform comparison
        output_a = self.prog_candidate_val_outputs[program_a][data_id]
        output_b = self.prog_candidate_val_outputs[program_b][data_id]

        result = comparator.compare(output_a, output_b, data_instance)

        # Cache result
        self.comparison_cache[cache_key] = result

        # Record comparison
        self.comparison_records.append(
            PairwiseComparisonRecord(
                program_a=program_a,
                program_b=program_b,
                data_id=data_id,
                result=result,
                timestamp=self.i,
            )
        )

        return result

    def update_pareto_front(
        self,
        new_program_idx: ProgramIdx,
        data_ids: list[DataId],
        comparator: PairwiseComparator,
        data_instances: dict[DataId, DataInst],
    ) -> None:
        """
        Update pareto fronts after adding a new program.

        For each validation instance:
        1. Compare new program against all programs on current pareto front
        2. If new program dominates any existing program, remove those
        3. If new program is not dominated, add to pareto front
        """
        for data_id in data_ids:
            if data_id not in self.program_at_pareto_front_valset:
                # First program evaluated on this instance
                self.program_at_pareto_front_valset[data_id] = {new_program_idx}
                continue

            current_front = self.program_at_pareto_front_valset[data_id].copy()
            data_instance = data_instances[data_id]

            dominated_by_new = set()
            new_dominated_by_existing = False
            new_ties_with_existing = False

            for existing_idx in current_front:
                result = self.compare_programs(
                    new_program_idx,
                    existing_idx,
                    data_id,
                    comparator,
                    data_instance,
                )

                if result == ComparisonResult.A_BETTER:
                    # New program dominates existing
                    dominated_by_new.add(existing_idx)
                elif result == ComparisonResult.B_BETTER:
                    # Existing dominates new
                    new_dominated_by_existing = True
                elif result == ComparisonResult.TIE:
                    # Tie: both should be on pareto front
                    new_ties_with_existing = True

            # Update pareto front
            if not new_dominated_by_existing:
                # New program is not dominated, add to front
                self.program_at_pareto_front_valset[data_id].add(new_program_idx)

                # Remove dominated programs
                for dominated_idx in dominated_by_new:
                    self.program_at_pareto_front_valset[data_id].discard(dominated_idx)
```

### 2.3 Modified Engine: Minibatch Acceptance

Replace sum-based acceptance with tournament-based acceptance:

```python
class PairwiseGEPAEngine(Generic[DataId, DataInst, Trajectory, RolloutOutput]):
    """
    GEPA engine for pairwise comparison metrics.
    """

    def _accept_minibatch_proposal(
        self,
        old_outputs: list[RolloutOutput],
        new_outputs: list[RolloutOutput],
        batch: list[DataInst],
        comparator: PairwiseComparator,
    ) -> bool:
        """
        Decide whether to accept a proposed candidate based on minibatch comparison.

        Strategy: Use pairwise tournament on minibatch.
        Accept if new program wins majority of comparisons.

        Args:
            old_outputs: Outputs from current program on minibatch
            new_outputs: Outputs from proposed program on minibatch
            batch: Minibatch data instances
            comparator: Pairwise comparison function

        Returns:
            True if proposal should be accepted
        """
        assert len(old_outputs) == len(new_outputs) == len(batch)

        wins = 0
        losses = 0
        ties = 0

        for old_out, new_out, data_inst in zip(old_outputs, new_outputs, batch):
            result = comparator.compare(new_out, old_out, data_inst)

            if result == ComparisonResult.A_BETTER:
                wins += 1
            elif result == ComparisonResult.B_BETTER:
                losses += 1
            elif result == ComparisonResult.TIE:
                ties += 1
            # INCOMPARABLE treated as neutral

        # Accept if new program wins more than it loses
        # (Ties don't count toward acceptance)
        return wins > losses

    def _accept_merge_proposal(
        self,
        parent1_outputs: list[RolloutOutput],
        parent2_outputs: list[RolloutOutput],
        merged_outputs: list[RolloutOutput],
        batch: list[DataInst],
        comparator: PairwiseComparator,
    ) -> bool:
        """
        Accept merge if merged program beats both parents.

        Args:
            parent1_outputs: Outputs from first parent
            parent2_outputs: Outputs from second parent
            merged_outputs: Outputs from merged program
            batch: Evaluation batch
            comparator: Pairwise comparison function

        Returns:
            True if merge should be accepted
        """
        # Compare merged against parent 1
        wins_vs_p1 = 0
        losses_vs_p1 = 0

        for merged_out, p1_out, data_inst in zip(merged_outputs, parent1_outputs, batch):
            result = comparator.compare(merged_out, p1_out, data_inst)
            if result == ComparisonResult.A_BETTER:
                wins_vs_p1 += 1
            elif result == ComparisonResult.B_BETTER:
                losses_vs_p1 += 1

        # Compare merged against parent 2
        wins_vs_p2 = 0
        losses_vs_p2 = 0

        for merged_out, p2_out, data_inst in zip(merged_outputs, parent2_outputs, batch):
            result = comparator.compare(merged_out, p2_out, data_inst)
            if result == ComparisonResult.A_BETTER:
                wins_vs_p2 += 1
            elif result == ComparisonResult.B_BETTER:
                losses_vs_p2 += 1

        # Accept if merged doesn't lose to either parent
        # (At least ties or wins against both)
        return (wins_vs_p1 >= losses_vs_p1) and (wins_vs_p2 >= losses_vs_p2)
```

### 2.4 Modified Proposers

#### Reflective Mutation Proposer

```python
class PairwiseReflectiveMutationProposer:
    """
    Reflective mutation for pairwise comparison GEPA.

    Key changes:
    1. No perfect score check (no scalar scores)
    2. Comparison against reference outputs for feedback
    3. Accept based on pairwise tournament
    """

    def propose(
        self,
        state: PairwiseGEPAState,
        adapter: PairwiseGEPAAdapter,
        valset: DataLoader,
    ) -> CandidateProposal | None:
        """Propose new candidate via reflective mutation."""

        # 1. Select candidate to mutate (uses tournament selection)
        candidate_idx = self.candidate_selector.select_candidate_idx(state)
        current_program = state.program_candidates[candidate_idx]

        # 2. Sample minibatch
        minibatch_ids = self.batch_sampler.sample(valset)
        minibatch = valset.fetch(minibatch_ids)

        # 3. Evaluate current program on minibatch
        eval_curr = adapter.evaluate(minibatch, current_program, capture_traces=True)

        # 4. Build reflective dataset
        # Compare against reference (e.g., best program outputs)
        reference_program_idx = self._get_reference_program(state)
        reference_outputs = [
            state.prog_candidate_val_outputs[reference_program_idx].get(mid)
            for mid in minibatch_ids
        ]

        reflective_dataset = adapter.make_reflective_dataset(
            candidate=current_program,
            eval_batch=eval_curr,
            components_to_update=self.components_to_update,
            reference_outputs=reference_outputs,
        )

        # 5. Propose new texts
        new_texts = adapter.propose_new_texts(
            current_program,
            reflective_dataset,
            self.components_to_update,
        )

        # 6. Create new program
        new_program = {**current_program, **new_texts}

        # 7. Evaluate new program on minibatch
        eval_new = adapter.evaluate(minibatch, new_program, capture_traces=False)

        return CandidateProposal(
            candidate=new_program,
            parent_program_ids=[candidate_idx],
            subsample_indices=minibatch_ids,
            subsample_outputs_before=eval_curr.outputs,
            subsample_outputs_after=eval_new.outputs,
            tag="reflective_mutation",
            metadata={},
        )

    def _get_reference_program(self, state: PairwiseGEPAState) -> ProgramIdx:
        """
        Get reference program for comparison feedback.

        Could use:
        - Program with most pareto front memberships
        - Most recent pareto program
        - Tournament winner
        """
        # Simple: use program that appears most on pareto fronts
        program_counts = defaultdict(int)
        for front in state.program_at_pareto_front_valset.values():
            for prog_idx in front:
                program_counts[prog_idx] += 1

        if not program_counts:
            return 0  # Fallback to seed

        return max(program_counts.items(), key=lambda x: x[1])[0]
```

#### Merge Proposer

```python
class PairwiseMergeProposer:
    """
    Merge proposer for pairwise comparison GEPA.

    Key changes:
    1. Parent selection uses pareto front membership counts (no scalar weights)
    2. Subsample selection based on pairwise comparisons
    3. Acceptance uses pairwise tournament
    """

    def _select_merge_parents(
        self,
        state: PairwiseGEPAState,
    ) -> tuple[ProgramIdx, ProgramIdx, ProgramIdx] | None:
        """
        Select two programs to merge and their common ancestor.

        Uses pareto front membership as proxy for quality.
        """
        # Find all programs on any pareto front
        pareto_programs = set()
        for front in state.program_at_pareto_front_valset.values():
            pareto_programs.update(front)

        if len(pareto_programs) < 2:
            return None

        # Find pairs with common ancestors
        pareto_list = list(pareto_programs)
        for i in range(len(pareto_list)):
            for j in range(i + 1, len(pareto_list)):
                prog_i = pareto_list[i]
                prog_j = pareto_list[j]

                # Find common ancestor
                ancestors_i = self._get_all_ancestors(prog_i, state)
                ancestors_j = self._get_all_ancestors(prog_j, state)
                common = ancestors_i & ancestors_j

                if common:
                    # Use most recent common ancestor
                    ancestor = max(common)
                    return prog_i, prog_j, ancestor

        return None

    def _select_merge_subsample(
        self,
        parent1_idx: ProgramIdx,
        parent2_idx: ProgramIdx,
        state: PairwiseGEPAState,
        comparator: PairwiseComparator,
        valset: DataLoader,
    ) -> list[DataId]:
        """
        Select evaluation subsample for merged program.

        Partition validation instances by comparison results:
        - P1: instances where parent1 beats parent2
        - P2: instances where parent2 beats parent1
        - P3: instances where parents tie or are incomparable

        Sample from each partition.
        """
        # Find common evaluated instances
        outputs1 = state.prog_candidate_val_outputs[parent1_idx]
        outputs2 = state.prog_candidate_val_outputs[parent2_idx]
        common_ids = set(outputs1.keys()) & set(outputs2.keys())

        p1_wins = []
        p2_wins = []
        ties = []

        for data_id in common_ids:
            result = state.compare_programs(
                parent1_idx,
                parent2_idx,
                data_id,
                comparator,
                valset.fetch([data_id])[0],
            )

            if result == ComparisonResult.A_BETTER:
                p1_wins.append(data_id)
            elif result == ComparisonResult.B_BETTER:
                p2_wins.append(data_id)
            else:  # TIE or INCOMPARABLE
                ties.append(data_id)

        # Sample from each partition
        rng = self.rng
        sample_per_partition = 5  # Configurable

        subsample = []
        subsample.extend(rng.sample(p1_wins, min(sample_per_partition, len(p1_wins))))
        subsample.extend(rng.sample(p2_wins, min(sample_per_partition, len(p2_wins))))
        subsample.extend(rng.sample(ties, min(sample_per_partition, len(ties))))

        return subsample
```

### 2.5 Candidate Selection Strategies

```python
class TournamentCandidateSelector:
    """
    Select candidate using tournament selection based on pairwise comparisons.

    Instead of selecting by maximum scalar score, run tournaments among
    candidate programs.
    """

    def __init__(self, tournament_size: int = 3, seed: int = 0):
        self.tournament_size = tournament_size
        self.rng = random.Random(seed)

    def select_candidate_idx(
        self,
        state: PairwiseGEPAState,
        valset: DataLoader,
        comparator: PairwiseComparator,
    ) -> ProgramIdx:
        """
        Select candidate via tournament selection.

        Args:
            state: Current GEPA state
            valset: Validation dataset
            comparator: Pairwise comparison function

        Returns:
            Index of selected candidate program
        """
        # Find all programs on pareto fronts
        pareto_programs = set()
        for front in state.program_at_pareto_front_valset.values():
            pareto_programs.update(front)

        if not pareto_programs:
            return 0  # Fallback to seed

        # Run tournament
        tournament_participants = self.rng.sample(
            list(pareto_programs),
            min(self.tournament_size, len(pareto_programs)),
        )

        if len(tournament_participants) == 1:
            return tournament_participants[0]

        # Compare all pairs, track wins
        wins = defaultdict(int)

        for i in range(len(tournament_participants)):
            for j in range(i + 1, len(tournament_participants)):
                prog_i = tournament_participants[i]
                prog_j = tournament_participants[j]

                # Compare on a random validation instance
                common_ids = (
                    set(state.prog_candidate_val_outputs[prog_i].keys()) &
                    set(state.prog_candidate_val_outputs[prog_j].keys())
                )

                if not common_ids:
                    continue

                sample_id = self.rng.choice(list(common_ids))
                data_instance = valset.fetch([sample_id])[0]

                result = state.compare_programs(
                    prog_i, prog_j, sample_id, comparator, data_instance
                )

                if result == ComparisonResult.A_BETTER:
                    wins[prog_i] += 1
                elif result == ComparisonResult.B_BETTER:
                    wins[prog_j] += 1

        # Return tournament winner
        if not wins:
            return tournament_participants[0]

        return max(wins.items(), key=lambda x: x[1])[0]


class ParetoFrontCandidateSelector:
    """
    Select candidate uniformly from programs on pareto fronts.

    Simpler than tournament: just pick randomly from pareto programs.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def select_candidate_idx(
        self,
        state: PairwiseGEPAState,
    ) -> ProgramIdx:
        """Select uniformly from pareto front programs."""
        pareto_programs = set()
        for front in state.program_at_pareto_front_valset.values():
            pareto_programs.update(front)

        if not pareto_programs:
            return 0

        return self.rng.choice(list(pareto_programs))
```

## 3. Features That CANNOT Be Ported

### 3.1 Scalar-Dependent Features

The following features **cannot be implemented** with pure pairwise comparisons:

| Feature | Current Implementation | Why Incompatible | Possible Workaround |
|---------|----------------------|------------------|---------------------|
| **Weighted Parent Sampling** | `weights=[scores[i] for i in ...]` | Requires scalar weights | Use pareto front membership counts or uniform sampling |
| **Perfect Score Threshold** | `if all(s >= perfect_score for s in scores)` | Requires scalar threshold | Remove feature; rely on stop conditions |
| **Threshold-Based Stopping** | `if max(scores) > threshold: stop` | Requires scalar comparison | Use iteration budget or manual stopping |
| **Score-Based Logging** | Log scalar metrics to W&B/MLflow | Requires scalar values | Log win/loss ratios (informational only) |
| **Validation Set Averaging** | `sum(scores) / len(scores)` | Requires scalar aggregation | Use pareto front size as proxy |
| **Exact Scalar Serialization** | Save `list[float]` in results | No scalar scores | Save comparison records |

### 3.2 Stop Conditions

Current stop conditions that won't work:

```python
# stop_condition.py:76 - Max score threshold
class MaxScoreStopper:
    def __call__(self, state):
        current_score = max(state.program_full_scores_val_set)
        return current_score >= self.target_score  # ← NO SCALAR SCORES

# stop_condition.py:99 - Improvement tracking
class NoImprovementStopper:
    def __call__(self, state):
        current_score = max(state.program_full_scores_val_set)  # ← NO SCALARS
        if current_score > self.best_score:  # ← NO SCALAR COMPARISON
            self.best_score = current_score
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
        return self.iterations_without_improvement >= self.patience
```

**Replacement**: Only support iteration budget and manual stopping:

```python
class IterationBudgetStopper:
    """Stop after N iterations."""
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def __call__(self, state: PairwiseGEPAState) -> bool:
        return state.i >= self.max_iterations


class ComparisonBudgetStopper:
    """Stop after N pairwise comparisons."""
    def __init__(self, max_comparisons: int):
        self.max_comparisons = max_comparisons

    def __call__(self, state: PairwiseGEPAState) -> bool:
        return len(state.comparison_records) >= self.max_comparisons
```

### 3.3 Logging and Metrics

Current logging that won't work:

```python
# logging/utils.py:44 - Pareto score logging
pareto_scores = [state.program_full_scores_val_set[i] for i in pareto_programs]
avg_pareto_score = sum(pareto_scores) / len(pareto_scores)  # ← NO SCALARS
```

**Replacement**: Log pareto front statistics:

```python
def log_pairwise_metrics(
    state: PairwiseGEPAState,
    new_program_idx: ProgramIdx,
):
    """Log metrics for pairwise comparison GEPA."""

    # Count pareto front memberships
    pareto_memberships = defaultdict(int)
    for front in state.program_at_pareto_front_valset.values():
        for prog_idx in front:
            pareto_memberships[prog_idx] += 1

    new_prog_memberships = pareto_memberships[new_program_idx]
    total_val_instances = len(state.program_at_pareto_front_valset)

    # Log statistics
    metrics = {
        "pareto_front_memberships": new_prog_memberships,
        "pareto_front_coverage": new_prog_memberships / total_val_instances,
        "total_comparisons": len(state.comparison_records),
        "iteration": state.i,
    }

    return metrics
```

### 3.4 Evaluation Policies

Current validation evaluation policies assume scalar aggregation:

```python
# strategies/eval_policy.py
class FullEvaluationPolicy:
    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Return program with highest average score."""
        best_idx, best_score = -1, float("-inf")
        for idx, scores in enumerate(state.prog_candidate_val_subscores):
            avg = sum(scores.values()) / len(scores)  # ← SCALAR AGGREGATION
            if avg > best_score:
                best_idx, best_score = idx, avg
        return best_idx
```

**Replacement**: Use pareto-based policies:

```python
class ParetoEvaluationPolicy:
    """Evaluation policy for pairwise comparison GEPA."""

    def get_best_program(self, state: PairwiseGEPAState) -> ProgramIdx:
        """
        Return 'best' program based on pareto front memberships.

        Since we can't aggregate scores, use program that appears most
        on pareto fronts as a proxy for quality.
        """
        pareto_counts = defaultdict(int)
        for front in state.program_at_pareto_front_valset.values():
            for prog_idx in front:
                pareto_counts[prog_idx] += 1

        if not pareto_counts:
            return 0

        return max(pareto_counts.items(), key=lambda x: x[1])[0]
```

## 4. Implementation Roadmap

### Phase 1: Core Infrastructure

1. **Define pairwise comparison interfaces**
   - `ComparisonResult` enum
   - `PairwiseComparator` protocol
   - `PairwiseEvaluationBatch` dataclass
   - `PairwiseGEPAAdapter` protocol

2. **Implement pairwise state management**
   - `PairwiseGEPAState` class
   - Comparison caching
   - Pareto front updates based on dominance

3. **Implement pairwise engine core**
   - `PairwiseGEPAEngine` class
   - Tournament-based acceptance testing
   - Comparison-based full evaluation

### Phase 2: Proposers and Strategies

4. **Adapt reflective mutation proposer**
   - Remove perfect score checks
   - Use comparison-based feedback
   - Tournament acceptance

5. **Adapt merge proposer**
   - Pareto-based parent selection
   - Comparison-based subsample selection
   - Tournament acceptance

6. **Implement candidate selection strategies**
   - `TournamentCandidateSelector`
   - `ParetoFrontCandidateSelector`

### Phase 3: Logging and Utilities

7. **Implement pairwise logging**
   - Pareto front statistics
   - Comparison counts
   - Win/loss records (informational)

8. **Implement pairwise stop conditions**
   - `IterationBudgetStopper`
   - `ComparisonBudgetStopper`

9. **Implement result serialization**
   - Save comparison records
   - Save pareto fronts
   - Export best programs

### Phase 4: Adapters and Examples

10. **Create example pairwise adapters**
    - LLM-as-judge adapter
    - Rule-based comparison adapter
    - Human-in-the-loop adapter

11. **Create example applications**
    - Pairwise prompt optimization
    - Code generation with quality comparisons
    - RAG system optimization

## 5. Trade-offs and Considerations

### 5.1 Advantages of Pairwise Comparison

1. **More Natural for Some Domains**
   - Text generation quality (LLM-as-judge)
   - Code quality (correctness + style)
   - Multi-objective problems (accuracy + latency)

2. **Avoids Metric Design**
   - Don't need to design scalar metric
   - Don't need to tune metric weights
   - More robust to outliers

3. **Richer Information**
   - Captures ordinal relationships
   - Preserves incomparability
   - Better for multi-objective scenarios

### 5.2 Disadvantages of Pairwise Comparison

1. **Higher Comparison Cost**
   - O(N²) comparisons for N programs
   - LLM-as-judge can be expensive
   - Need comparison caching

2. **No Absolute Progress Tracking**
   - Can't track "score improvement"
   - Can only track relative rankings
   - Harder to debug

3. **Limited Stopping Criteria**
   - Can't use score thresholds
   - Must rely on iteration budgets
   - Less intuitive stopping

4. **Reduced Feature Set**
   - No weighted sampling
   - No perfect score checks
   - No scalar logging

5. **Complexity in Logging**
   - Can't log scalar metrics to MLflow/W&B
   - Win/loss ratios are informational only
   - Harder to compare runs

### 5.3 When to Use Pairwise GEPA

**Use pairwise comparison GEPA when:**
- Scalar metrics are hard to define (e.g., text quality)
- You have access to reliable comparison function (LLM-as-judge, human)
- Multi-objective optimization with incomparable objectives
- Ordinal relationships more important than cardinal values

**Use scalar GEPA when:**
- Scalar metrics are natural and reliable
- Comparison cost is high (need many comparisons)
- You need absolute progress tracking
- You need threshold-based stopping conditions

## 6. Example Usage

```python
# Example: LLM-as-judge pairwise comparison adapter

class LLMJudgeComparator:
    """Use an LLM to compare two outputs."""

    def __init__(self, llm_client, judge_prompt_template: str):
        self.llm_client = llm_client
        self.judge_prompt_template = judge_prompt_template

    def compare(
        self,
        output_a: str,
        output_b: str,
        data_instance: dict,
    ) -> ComparisonResult:
        """Compare two outputs using LLM."""

        prompt = self.judge_prompt_template.format(
            question=data_instance["question"],
            answer_a=output_a,
            answer_b=output_b,
        )

        response = self.llm_client.complete(prompt)

        # Parse response
        if "A is better" in response or "Answer A" in response:
            return ComparisonResult.A_BETTER
        elif "B is better" in response or "Answer B" in response:
            return ComparisonResult.B_BETTER
        elif "tie" in response.lower() or "equal" in response.lower():
            return ComparisonResult.TIE
        else:
            return ComparisonResult.INCOMPARABLE


class QAPairwiseAdapter:
    """Pairwise comparison adapter for Q&A tasks."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.comparator_instance = LLMJudgeComparator(
            llm_client,
            judge_prompt_template="...",
        )

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> PairwiseEvaluationBatch:
        """Evaluate candidate on batch."""

        outputs = []
        trajectories = [] if capture_traces else None

        for example in batch:
            # Execute program with instruction from candidate
            instruction = candidate["instruction"]
            output = self.llm_client.complete(
                instruction + "\n\n" + example["question"]
            )
            outputs.append(output)

            if capture_traces:
                trajectories.append({
                    "instruction": instruction,
                    "question": example["question"],
                    "output": output,
                })

        return PairwiseEvaluationBatch(
            outputs=outputs,
            trajectories=trajectories,
        )

    def comparator(self) -> PairwiseComparator:
        """Return comparison function."""
        return self.comparator_instance

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: PairwiseEvaluationBatch,
        components_to_update: list[str],
        reference_outputs: list[str] | None,
    ) -> dict[str, list[dict]]:
        """Build reflective dataset using comparisons."""

        dataset = []

        for i, traj in enumerate(eval_batch.trajectories):
            # Compare against reference if available
            feedback = ""
            if reference_outputs and reference_outputs[i]:
                comparison = self.comparator_instance.compare(
                    traj["output"],
                    reference_outputs[i],
                    {"question": traj["question"]},
                )

                if comparison == ComparisonResult.B_BETTER:
                    feedback = f"Your output is worse than the reference: {reference_outputs[i]}"
                elif comparison == ComparisonResult.A_BETTER:
                    feedback = "Your output is better than the reference"
                else:
                    feedback = "Your output is similar to the reference"

            dataset.append({
                "Inputs": {"question": traj["question"]},
                "Generated Output": traj["output"],
                "Feedback": feedback,
            })

        return {"instruction": dataset}


# Usage
adapter = QAPairwiseAdapter(llm_client)

result = pairwise_gepa_optimize(
    adapter=adapter,
    seed_candidate={"instruction": "Answer the question accurately."},
    trainset=trainset,
    valset=valset,
    max_iterations=50,
    seed=42,
)
```

## 7. Summary

### What Can Be Ported

- ✅ Core optimization loop
- ✅ Reflective mutation proposer (with modifications)
- ✅ Merge proposer (with modifications)
- ✅ Pareto front management (using dominance relationships)
- ✅ Candidate selection (using tournaments)
- ✅ State management (storing outputs + comparisons)
- ✅ Minibatch acceptance (using tournaments)
- ✅ Result serialization (storing comparison records)

### What Cannot Be Ported

- ❌ Weighted random sampling by score
- ❌ Perfect score threshold checks
- ❌ Score-based stopping conditions
- ❌ Scalar metric logging (W&B/MLflow)
- ❌ Validation set averaging
- ❌ Score improvement tracking
- ❌ Threshold-based early stopping

### Key Implementation Changes

1. **Evaluation**: Return outputs only, no scalar scores
2. **State**: Store outputs + comparison records, not scalar scores
3. **Acceptance**: Tournament-based, not sum-based
4. **Selection**: Tournament or uniform from pareto front
5. **Pareto**: Dominance-based, not scalar comparison
6. **Merge**: Pareto membership counts, not weighted sampling
7. **Stopping**: Iteration/comparison budget only
8. **Logging**: Pareto statistics, not scalar metrics

This design preserves the core GEPA algorithm while adapting it to work purely with pairwise comparisons, without mapping to scalar metrics.
