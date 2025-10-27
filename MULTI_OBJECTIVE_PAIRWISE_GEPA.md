# Multi-Objective GEPA with Pairwise Comparisons

> **⚠️ Status: Future Work** - This is a design document for future implementation.
>
> Core single-objective pairwise comparison GEPA is implemented and ready to use.
> See `README_PAIRWISE.md` for current implementation.

## Executive Summary

Bradley-Terry can be extended to multi-objective optimization by running it **separately for each objective**. This would give:

1. ✅ **Score vectors** (one score per objective) instead of single scalars
2. ✅ **True Pareto fronts** based on multi-objective dominance
3. ✅ **~85% code reuse** from original GEPA (vs 98% for single-objective)
4. ✅ **No information loss** from collapsing to single dimension

**Key insight**: Run Bradley-Terry independently per objective → get score vectors → use standard Pareto dominance.

---

## 1. Multi-Objective Bradley-Terry Architecture

### 1.1 Core Idea

Instead of one comparison question:
```
"Is program A better than program B?"
```

Ask multiple comparison questions (one per objective):
```
"Is program A more accurate than program B?"
"Is program A faster than program B?"
"Is program A more readable than program B?"
```

Run Bradley-Terry **separately** for each objective to get a **score vector** per program.

### 1.2 Data Structures

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

ObjectiveId = str  # e.g., "accuracy", "latency", "readability"

@dataclass
class MultiObjectiveComparisonResult:
    """Result of comparing two outputs across multiple objectives."""

    # Per-objective comparison results
    # e.g., {"accuracy": A_BETTER, "latency": B_BETTER, "readability": TIE}
    objective_results: dict[ObjectiveId, ComparisonResult]

    def dominates(self, other: "MultiObjectiveComparisonResult") -> bool:
        """
        Check if this result dominates another (Pareto dominance).

        A dominates B if:
        - A >= B on all objectives
        - A > B on at least one objective
        """
        at_least_one_better = False

        for obj_id in self.objective_results.keys():
            self_result = self.objective_results[obj_id]
            other_result = other.objective_results.get(obj_id, ComparisonResult.TIE)

            if self_result == ComparisonResult.B_BETTER:  # Other is better
                return False  # Not dominated if worse on any objective
            elif self_result == ComparisonResult.A_BETTER:
                at_least_one_better = True

        return at_least_one_better


class MultiObjectiveComparator(Protocol[RolloutOutput]):
    """
    Protocol for multi-objective pairwise comparison.

    Returns comparison results for each objective independently.
    """

    def compare(
        self,
        output_a: RolloutOutput,
        output_b: RolloutOutput,
        data_instance: DataInst,
    ) -> MultiObjectiveComparisonResult:
        """
        Compare two outputs across all objectives.

        Returns:
            MultiObjectiveComparisonResult with per-objective comparisons
        """
        ...

    def get_objectives(self) -> list[ObjectiveId]:
        """Return list of objective IDs this comparator evaluates."""
        ...


@dataclass
class MultiObjectivePairwiseRecord:
    """Record of a multi-objective pairwise comparison."""
    program_a: ProgramIdx
    program_b: ProgramIdx
    data_id: DataId
    objective_results: dict[ObjectiveId, ComparisonResult]
    timestamp: int
```

### 1.3 Multi-Objective State

```python
class MultiObjectiveGEPAState(Generic[RolloutOutput, DataId]):
    """
    GEPA state for multi-objective optimization with pairwise comparisons.

    Key differences from single-objective:
    1. Stores per-objective comparison records
    2. Computes per-objective scores using Bradley-Terry
    3. Maintains Pareto fronts based on multi-objective dominance
    """

    # Program candidates (unchanged)
    program_candidates: list[dict[str, str]]
    parent_program_for_candidate: list[list[ProgramIdx | None]]

    # Multi-objective comparison data
    comparison_records: list[MultiObjectivePairwiseRecord]
    prog_candidate_val_outputs: list[dict[DataId, RolloutOutput]]

    # Per-objective comparison caches
    # objective_id -> (program_a, program_b, data_id) -> ComparisonResult
    comparison_cache_per_objective: dict[
        ObjectiveId,
        dict[tuple[ProgramIdx, ProgramIdx, DataId], ComparisonResult]
    ]

    # Objectives being optimized
    objectives: list[ObjectiveId]

    # Cached score vectors (computed on-demand)
    # program_idx -> objective_id -> data_id -> score
    _cached_score_vectors: dict[ProgramIdx, dict[ObjectiveId, dict[DataId, float]]] | None = None
    _cache_invalidated: bool = True

    # Pareto fronts (per validation instance)
    # data_id -> set of program indices on pareto front
    pareto_front_valset: dict[DataId, set[ProgramIdx]]

    # Tracking
    i: int
    num_full_ds_evals: int
    total_num_evals: int

    def __init__(self, objectives: list[ObjectiveId], ...):
        self.objectives = objectives
        self.comparison_cache_per_objective = {obj_id: {} for obj_id in objectives}
        # ... rest of initialization

    @property
    def prog_candidate_val_subscores(self) -> list[dict[DataId, dict[ObjectiveId, float]]]:
        """
        Get score vectors for each program.

        Returns:
            List where element i is a dict mapping:
            data_id -> {objective_id -> score}

        This allows reusing GEPA code with minimal modifications.
        Instead of dict[DataId, float], we return dict[DataId, dict[ObjectiveId, float]].
        """
        if self._cache_invalidated or self._cached_score_vectors is None:
            self._recompute_score_vectors()
            self._cache_invalidated = False

        # Restructure for compatibility
        result = []
        for prog_idx in range(len(self.program_candidates)):
            prog_scores = {}
            if prog_idx in self._cached_score_vectors:
                for data_id, obj_scores in self._cached_score_vectors[prog_idx].items():
                    prog_scores[data_id] = obj_scores
            result.append(prog_scores)

        return result

    def _recompute_score_vectors(self) -> None:
        """
        Compute score vectors from pairwise comparisons.

        For each objective:
        1. Extract comparisons for that objective
        2. Run Bradley-Terry to get scalar scores
        3. Store in score vector
        """
        from collections import defaultdict

        self._cached_score_vectors = defaultdict(lambda: defaultdict(dict))

        # Get all validation instances
        all_val_ids = set()
        for outputs in self.prog_candidate_val_outputs:
            all_val_ids.update(outputs.keys())

        # For each objective, compute scores independently
        for obj_id in self.objectives:
            # For each validation instance
            for val_id in all_val_ids:
                # Find programs evaluated on this instance
                programs_for_val_id = [
                    prog_idx for prog_idx, outputs in enumerate(self.prog_candidate_val_outputs)
                    if val_id in outputs
                ]

                if len(programs_for_val_id) <= 1:
                    # Not enough programs to compare
                    if len(programs_for_val_id) == 1:
                        prog_idx = programs_for_val_id[0]
                        self._cached_score_vectors[prog_idx][val_id][obj_id] = 1.0
                    continue

                # Extract comparison matrix for this objective and validation instance
                comparison_matrix = {
                    (prog_a, prog_b, val_id): result
                    for (prog_a, prog_b, vid), result in
                        self.comparison_cache_per_objective[obj_id].items()
                    if vid == val_id and prog_a in programs_for_val_id and prog_b in programs_for_val_id
                }

                # Run Bradley-Terry for this objective
                obj_scores = bradley_terry_scores(programs_for_val_id, comparison_matrix)

                # Store scores
                for prog_idx, score in obj_scores.items():
                    self._cached_score_vectors[prog_idx][val_id][obj_id] = score

    def update_pareto_front(
        self,
        new_program_idx: ProgramIdx,
        data_ids: list[DataId],
    ) -> None:
        """
        Update Pareto fronts using multi-objective dominance.

        A program dominates another if:
        - It's >= on all objectives
        - It's > on at least one objective
        """
        # Ensure scores are computed
        if self._cache_invalidated:
            self._recompute_score_vectors()

        new_prog_scores = self._cached_score_vectors.get(new_program_idx, {})

        for data_id in data_ids:
            if data_id not in new_prog_scores:
                continue

            if data_id not in self.pareto_front_valset:
                # First program on this instance
                self.pareto_front_valset[data_id] = {new_program_idx}
                continue

            current_front = self.pareto_front_valset[data_id].copy()

            dominated_by_new = set()
            new_is_dominated = False

            for existing_idx in current_front:
                existing_scores = self._cached_score_vectors.get(existing_idx, {}).get(data_id, {})
                new_scores = new_prog_scores[data_id]

                # Check dominance
                new_dominates = self._dominates(new_scores, existing_scores)
                existing_dominates = self._dominates(existing_scores, new_scores)

                if new_dominates:
                    dominated_by_new.add(existing_idx)
                elif existing_dominates:
                    new_is_dominated = True

            # Update front
            if not new_is_dominated:
                self.pareto_front_valset[data_id].add(new_program_idx)
                for dominated_idx in dominated_by_new:
                    self.pareto_front_valset[data_id].discard(dominated_idx)

    def _dominates(
        self,
        scores_a: dict[ObjectiveId, float],
        scores_b: dict[ObjectiveId, float],
    ) -> bool:
        """
        Check if scores_a dominates scores_b (Pareto dominance).

        Returns True if a >= b on all objectives and a > b on at least one.
        """
        if not scores_a or not scores_b:
            return False

        at_least_one_better = False

        for obj_id in self.objectives:
            score_a = scores_a.get(obj_id, float("-inf"))
            score_b = scores_b.get(obj_id, float("-inf"))

            if score_a < score_b:
                return False  # a is worse on this objective
            elif score_a > score_b:
                at_least_one_better = True

        return at_least_one_better
```

---

## 2. How This Differs from Single-Objective

### 2.1 Comparison Questions

**Single-Objective:**
```python
comparator.compare(output_a, output_b, data)
# Returns: A_BETTER | B_BETTER | TIE
```

**Multi-Objective:**
```python
comparator.compare(output_a, output_b, data)
# Returns: {
#     "accuracy": A_BETTER,
#     "latency": B_BETTER,
#     "readability": TIE
# }
```

### 2.2 Score Storage

**Single-Objective:**
```python
prog_candidate_val_subscores: list[dict[DataId, float]]
# e.g., program 0: {data_1: 0.8, data_2: 0.9}
```

**Multi-Objective:**
```python
prog_candidate_val_subscores: list[dict[DataId, dict[ObjectiveId, float]]]
# e.g., program 0: {
#     data_1: {"accuracy": 0.8, "latency": 0.3, "readability": 0.7},
#     data_2: {"accuracy": 0.9, "latency": 0.4, "readability": 0.8}
# }
```

### 2.3 Pareto Fronts

**Single-Objective:**
```python
# Only one "best" program per validation instance (or ties)
pareto_front_valset: dict[DataId, set[ProgramIdx]]
# e.g., {data_1: {program_5}, data_2: {program_3, program_5}}
```

**Multi-Objective:**
```python
# Multiple non-dominated programs per validation instance
pareto_front_valset: dict[DataId, set[ProgramIdx]]
# e.g., {data_1: {program_2, program_5, program_7}, ...}
# program_2 might be best on accuracy
# program_5 might be best on latency
# program_7 might be balanced
```

**Key difference**: Multi-objective Pareto fronts are typically **larger** because programs can be incomparable (A better on accuracy, B better on latency).

---

## 3. Code Reuse Analysis

### 3.1 What Changes Are Needed

| Component | Original Type | Multi-Objective Type | Changes Needed |
|-----------|---------------|---------------------|----------------|
| **Score storage** | `dict[DataId, float]` | `dict[DataId, dict[ObjectiveId, float]]` | Medium - add objective layer |
| **Pareto updates** | `if score > prev_score` | `if dominates(scores, prev_scores)` | Medium - multi-objective dominance |
| **Acceptance test** | `sum(scores)` | `sum(scores[obj])` per objective | Medium - check dominance |
| **Aggregation** | `sum(scores) / len(scores)` | Per-objective aggregation | Small - loop over objectives |
| **Best program** | `argmax(scores)` | Return pareto front | Small - return set instead of single |
| **Weighted sampling** | `weights=[scores[i]]` | `weights=[hypervolume(i)]` | Large - need scalarization |
| **Logging** | Scalar metrics | Vector metrics | Small - log per objective |

### 3.2 Code Reuse Estimate

| Component | Reuse % | Reason |
|-----------|---------|--------|
| **GEPAEngine.run()** | **90%** | Main loop unchanged; acceptance logic needs multi-objective check |
| **GEPAState** | **70%** | Structure same; dominance replaces scalar comparison |
| **Reflective Mutation** | **85%** | Works same; acceptance uses dominance |
| **Merge Proposer** | **60%** | Selection needs scalarization (hypervolume or random) |
| **Pareto Updates** | **100%** | Actually more natural with real Pareto dominance! |
| **Evaluation Policies** | **80%** | Return pareto set instead of best single |
| **Stop Conditions** | **50%** | Thresholds per objective or hypervolume-based |
| **Logging** | **90%** | Loop over objectives |

**Overall Code Reuse: ~85%** (vs 98% for single-objective hybrid)

### 3.3 New Challenges

**1. Minibatch Acceptance** - Need multi-objective comparison:
```python
def accept_proposal(old_scores_vec, new_scores_vec) -> bool:
    """
    Accept if new dominates old or is non-dominated.

    Old single-objective: sum(new) > sum(old)
    New multi-objective: new dominates old OR (new is non-dominated AND improves any objective)
    """
    # Check if new dominates old
    if dominates(new_scores_vec, old_scores_vec):
        return True

    # Accept if new improves on any objective without regressing on others
    improved_any = False
    regressed_any = False

    for obj_id in objectives:
        new_score = sum(new_scores_vec[example][obj_id] for example in batch)
        old_score = sum(old_scores_vec[example][obj_id] for example in batch)

        if new_score > old_score:
            improved_any = True
        elif new_score < old_score:
            regressed_any = True

    return improved_any and not regressed_any
```

**2. Weighted Sampling in Merge** - Need scalarization:
```python
def select_merge_candidates(state: MultiObjectiveGEPAState):
    """
    Select merge candidates in multi-objective setting.

    Options:
    1. Random from pareto front (simple)
    2. Weighted by hypervolume contribution (complex but principled)
    3. Weighted by pareto front membership count (proxy for quality)
    """
    # Option 1: Random from pareto front (SIMPLEST)
    pareto_programs = set()
    for front in state.pareto_front_valset.values():
        pareto_programs.update(front)

    return random.sample(list(pareto_programs), 2)

    # Option 2: Weighted by pareto front membership (RECOMMENDED)
    membership_counts = defaultdict(int)
    for front in state.pareto_front_valset.values():
        for prog_idx in front:
            membership_counts[prog_idx] += 1

    weights = [membership_counts[i] for i in range(len(state.program_candidates))]
    return random.choices(range(len(state.program_candidates)), weights=weights, k=2)

    # Option 3: Weighted by hypervolume contribution (MOST PRINCIPLED)
    # Compute hypervolume contribution of each program
    # (expensive but theoretically sound)
    hv_contributions = compute_hypervolume_contributions(state)
    weights = list(hv_contributions.values())
    return random.choices(list(hv_contributions.keys()), weights=weights, k=2)
```

**3. Stop Conditions** - Multiple approaches:
```python
# Option 1: Threshold per objective
class MultiObjectiveThresholdStopper:
    def __init__(self, thresholds: dict[ObjectiveId, float]):
        self.thresholds = thresholds

    def __call__(self, state: MultiObjectiveGEPAState) -> bool:
        # Stop if any program achieves all thresholds
        for prog_idx in range(len(state.program_candidates)):
            scores = state._cached_score_vectors.get(prog_idx, {})
            avg_scores = self._average_scores(scores)

            if all(avg_scores.get(obj_id, 0) >= threshold
                   for obj_id, threshold in self.thresholds.items()):
                return True
        return False

# Option 2: Hypervolume-based
class HypervolumeStopper:
    def __init__(self, target_hypervolume: float, reference_point: dict[ObjectiveId, float]):
        self.target_hypervolume = target_hypervolume
        self.reference_point = reference_point

    def __call__(self, state: MultiObjectiveGEPAState) -> bool:
        # Compute hypervolume of current pareto front
        hv = compute_hypervolume(state.pareto_front_valset, state, self.reference_point)
        return hv >= self.target_hypervolume

# Option 3: Iteration budget (simplest)
class IterationBudgetStopper:
    # Same as before - works for both single and multi-objective
    ...
```

---

## 4. Advantages of Multi-Objective Approach

### 4.1 vs Single-Objective Conversion

| Aspect | Single-Objective Bradley-Terry | Multi-Objective Bradley-Terry |
|--------|-------------------------------|------------------------------|
| **Information preservation** | ❌ Loses multi-objective structure | ✅ Preserves all objectives |
| **Pareto fronts** | ⚠️ Artificial (based on single score) | ✅ True Pareto dominance |
| **Trade-off exploration** | ❌ Collapses to single dimension | ✅ Discovers trade-off frontier |
| **Code reuse** | ✅ 98% | ⚠️ 85% |
| **Implementation complexity** | ✅ Simple (500 LOC) | ⚠️ Moderate (800 LOC) |
| **Interpretability** | ⚠️ Single number (may hide trade-offs) | ✅ Explicit per-objective scores |

### 4.2 vs Pure Pairwise (No Conversion)

| Aspect | Pure Pairwise | Multi-Objective Bradley-Terry |
|--------|---------------|------------------------------|
| **Information preservation** | ✅ Full comparison records | ✅ Score vectors (lossy but principled) |
| **Code reuse** | ❌ ~10% | ✅ ~85% |
| **Weighted sampling** | ❌ Not supported | ✅ Supported (via hypervolume or membership) |
| **Stop conditions** | ❌ No thresholds | ✅ Per-objective thresholds |
| **Logging** | ❌ Win/loss ratios only | ✅ Scalar metrics per objective |
| **Implementation complexity** | ❌ High (2000 LOC) | ✅ Moderate (800 LOC) |

---

## 5. Example Implementation

### 5.1 Multi-Objective LLM-as-Judge

```python
class MultiObjectiveLLMJudge:
    """
    LLM-as-judge that compares outputs across multiple objectives.
    """

    def __init__(self, judge_client, objectives: list[str]):
        self.judge_client = judge_client
        self.objectives = objectives  # e.g., ["accuracy", "clarity", "conciseness"]

    def compare(
        self,
        output_a: str,
        output_b: str,
        data_instance: dict,
    ) -> MultiObjectiveComparisonResult:
        """Compare outputs across all objectives."""

        objective_results = {}

        for objective in self.objectives:
            prompt = f"""
            Compare these two answers on {objective}:

            Question: {data_instance['question']}

            Answer A: {output_a}
            Answer B: {output_b}

            Which answer has better {objective}? Respond with "A", "B", or "Tie".
            """

            response = self.judge_client.complete(prompt)

            if "A" in response and "B" not in response:
                objective_results[objective] = ComparisonResult.A_BETTER
            elif "B" in response and "A" not in response:
                objective_results[objective] = ComparisonResult.B_BETTER
            else:
                objective_results[objective] = ComparisonResult.TIE

        return MultiObjectiveComparisonResult(objective_results=objective_results)

    def get_objectives(self) -> list[ObjectiveId]:
        return self.objectives


# Usage
adapter = MultiObjectivePairwiseAdapter(
    llm_client=llm_client,
    comparator=MultiObjectiveLLMJudge(
        judge_client=judge_client,
        objectives=["accuracy", "clarity", "conciseness"]
    )
)

result = multi_objective_gepa_optimize(
    adapter=adapter,
    seed_candidate={"instruction": "Answer the question."},
    trainset=trainset,
    valset=valset,
    max_iterations=50,
    score_algorithm="bradley_terry",  # Per-objective
)

# Result contains pareto front programs
# e.g., program 5 might be most accurate
#       program 12 might be most concise
#       program 8 might be balanced
```

### 5.2 Result Structure

```python
@dataclass
class MultiObjectiveGEPAResult:
    """Result of multi-objective GEPA optimization."""

    # All discovered programs
    candidates: list[dict[str, str]]

    # Score vectors per program
    # program_idx -> data_id -> objective_id -> score
    val_score_vectors: list[dict[DataId, dict[ObjectiveId, float]]]

    # Pareto front per validation instance
    # data_id -> set of program indices
    pareto_fronts: dict[DataId, set[ProgramIdx]]

    # Global pareto front (averaged across validation set)
    global_pareto_front: set[ProgramIdx]

    # Per-objective best programs (for reference)
    best_per_objective: dict[ObjectiveId, ProgramIdx]

    def get_pareto_programs(self) -> list[dict[str, str]]:
        """Get all programs on global pareto front."""
        return [self.candidates[i] for i in self.global_pareto_front]

    def get_best_for_objective(self, objective_id: ObjectiveId) -> dict[str, str]:
        """Get program that's best on a specific objective."""
        return self.candidates[self.best_per_objective[objective_id]]

    def plot_pareto_front(self, obj_x: ObjectiveId, obj_y: ObjectiveId):
        """Plot 2D pareto front for two objectives."""
        import matplotlib.pyplot as plt

        # Extract scores for pareto programs
        x_scores = []
        y_scores = []

        for prog_idx in self.global_pareto_front:
            # Average across validation set
            scores = self.val_score_vectors[prog_idx]
            avg_x = sum(s[obj_x] for s in scores.values()) / len(scores)
            avg_y = sum(s[obj_y] for s in scores.values()) / len(scores)
            x_scores.append(avg_x)
            y_scores.append(avg_y)

        plt.scatter(x_scores, y_scores, label="Pareto Front")
        plt.xlabel(obj_x)
        plt.ylabel(obj_y)
        plt.title("Pareto Front")
        plt.legend()
        plt.show()
```

---

## 6. Pareto Front Semantics

### 6.1 What is a Pareto Front?

A **Pareto front** is the set of solutions where:
- No solution dominates any other solution in the set
- All solutions outside the set are dominated by at least one solution in the set

**Example** (accuracy vs latency):
```
Program A: accuracy=0.9, latency=100ms  ← On pareto front
Program B: accuracy=0.8, latency=50ms   ← On pareto front (faster, less accurate)
Program C: accuracy=0.85, latency=120ms ← NOT on front (dominated by A)
Program D: accuracy=0.95, latency=80ms  ← On pareto front (best of both worlds)
```

Pareto front: {A, B, D}

### 6.2 Single vs Multiple Pareto Fronts

**You have ONE pareto front**, but you can think of it in different ways:

1. **Global pareto front**: Programs that are non-dominated when averaged across all validation instances

2. **Per-instance pareto fronts**: For each validation instance, the set of programs that are non-dominated on that specific instance

3. **Pareto layers** (sometimes called "fronts"):
   - **First layer**: True pareto front (non-dominated)
   - **Second layer**: Dominated only by first layer
   - **Third layer**: Dominated by first or second layer
   - etc.

In GEPA, we typically care about:
- **Per-instance pareto fronts** (stored in `pareto_front_valset`)
- **Global pareto front** (computed by aggregating across instances)

### 6.3 How Pareto Fronts Grow During Optimization

```
Iteration 0: {seed_program}
    accuracy=0.7, latency=200ms

Iteration 10: {prog_0, prog_5, prog_8}
    prog_0: accuracy=0.7, latency=200ms
    prog_5: accuracy=0.8, latency=150ms  ← Discovered fast+accurate variant
    prog_8: accuracy=0.85, latency=180ms ← Discovered very accurate variant

Iteration 20: {prog_5, prog_8, prog_15, prog_17}
    prog_0 removed (dominated by prog_5)
    prog_5: accuracy=0.8, latency=150ms
    prog_8: accuracy=0.85, latency=180ms
    prog_15: accuracy=0.88, latency=160ms ← New discovery
    prog_17: accuracy=0.75, latency=80ms  ← Very fast variant

Final: True pareto front with diverse trade-offs
```

**Key insight**: Pareto front **grows** during optimization as you discover programs in different regions of objective space.

---

## 7. Implementation Roadmap

### Phase 1: Multi-Objective State (2-3 days)
1. Implement `MultiObjectiveComparisonResult`
2. Implement `MultiObjectiveGEPAState` with score vector storage
3. Implement per-objective Bradley-Terry conversion
4. Implement multi-objective dominance checking

### Phase 2: Multi-Objective Engine (2-3 days)
5. Modify minibatch acceptance for multi-objective
6. Implement multi-objective pareto front updates
7. Modify evaluation to perform per-objective comparisons

### Phase 3: Proposers and Strategies (2-3 days)
8. Adapt reflective mutation for multi-objective acceptance
9. Adapt merge proposer with hypervolume or membership weighting
10. Implement multi-objective candidate selectors

### Phase 4: Utilities and Logging (1-2 days)
11. Implement multi-objective stop conditions
12. Implement per-objective logging
13. Implement pareto front visualization

**Total: ~1.5-2 weeks** (vs 1 week for single-objective, 1 month for pure pairwise)

---

## 8. Recommendation

### 8.1 When to Use Each Approach

| Use Case | Recommended Approach | Reason |
|----------|---------------------|--------|
| **Single quality metric** | Single-objective Bradley-Terry | Simplest (98% reuse) |
| **Multiple objectives with trade-offs** | **Multi-objective Bradley-Terry** | Preserves objectives, discovers pareto front |
| **Incomparable dimensions** | Multi-objective Bradley-Terry | Natural representation |
| **Need scalar score** | Single-objective Bradley-Terry | Forces total ordering |
| **Research/exploration** | Multi-objective Bradley-Terry | More informative |

### 8.2 Implementation Priority

**Recommended implementation order:**

1. **Start with single-objective Bradley-Terry** (1 week)
   - Validates the hybrid approach
   - Provides immediate value
   - Simpler to debug

2. **Extend to multi-objective** (1 additional week)
   - Add objective dimension to storage
   - Implement multi-objective dominance
   - Add per-objective Bradley-Terry
   - Most code is reusable!

3. **Optional: Add pure pairwise mode** (3-4 additional weeks)
   - For cases where conversion isn't appropriate
   - Much more complex

### 8.3 Final Recommendation

**Use multi-objective Bradley-Terry** if you have multiple objectives. It provides:
- ✅ True pareto fronts with meaningful trade-offs
- ✅ No information loss from collapsing to single dimension
- ✅ 85% code reuse (vs 10% for pure pairwise)
- ✅ All GEPA features (weighted sampling, thresholds, logging)
- ✅ Statistically principled (Bradley-Terry per objective)
- ✅ Moderate implementation effort (~2 weeks vs 1 month)

The only downside vs single-objective is slightly more implementation complexity, but this is **far outweighed** by the benefits of preserving multi-objective structure.

---

## 9. Example: Real-World Use Case

**Optimizing a code generation system with three objectives:**

1. **Correctness**: Does the code pass tests?
2. **Efficiency**: How fast does the code run?
3. **Readability**: How clean/maintainable is the code?

```python
# Multi-objective comparator
class CodeGenerationComparator:
    def compare(self, output_a: str, output_b: str, data_instance: dict):
        results = {}

        # Correctness: Run tests
        tests_a = run_tests(output_a, data_instance["tests"])
        tests_b = run_tests(output_b, data_instance["tests"])

        if tests_a["passed"] > tests_b["passed"]:
            results["correctness"] = ComparisonResult.A_BETTER
        elif tests_b["passed"] > tests_a["passed"]:
            results["correctness"] = ComparisonResult.B_BETTER
        else:
            results["correctness"] = ComparisonResult.TIE

        # Efficiency: Measure runtime
        runtime_a = tests_a["avg_runtime"]
        runtime_b = tests_b["avg_runtime"]

        if runtime_a < runtime_b * 0.9:  # 10% faster
            results["efficiency"] = ComparisonResult.A_BETTER
        elif runtime_b < runtime_a * 0.9:
            results["efficiency"] = ComparisonResult.B_BETTER
        else:
            results["efficiency"] = ComparisonResult.TIE

        # Readability: LLM-as-judge
        prompt = f"Which code is more readable? A or B?\n\nCode A:\n{output_a}\n\nCode B:\n{output_b}"
        response = judge_llm.complete(prompt)

        if "A" in response:
            results["readability"] = ComparisonResult.A_BETTER
        elif "B" in response:
            results["readability"] = ComparisonResult.B_BETTER
        else:
            results["readability"] = ComparisonResult.TIE

        return MultiObjectiveComparisonResult(objective_results=results)


# Run optimization
result = multi_objective_gepa_optimize(
    adapter=CodeGenerationAdapter(comparator=CodeGenerationComparator()),
    seed_candidate={"instruction": "Write Python code to solve the problem."},
    valset=coding_problems,
    max_iterations=100,
)

# Get pareto front
pareto_programs = result.get_pareto_programs()

# User can choose based on preference:
# - result.get_best_for_objective("correctness") → Most correct
# - result.get_best_for_objective("efficiency") → Fastest
# - result.get_best_for_objective("readability") → Most readable
# - Or any pareto-optimal trade-off
```

This discovers programs like:
- **Program A**: 100% correct, slow, very readable (for production)
- **Program B**: 95% correct, very fast, less readable (for performance-critical)
- **Program C**: 98% correct, medium speed, medium readability (balanced)

All three are on the pareto front and useful for different scenarios!
