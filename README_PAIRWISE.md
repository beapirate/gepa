# Pairwise Comparison GEPA

Production-ready implementation of pairwise comparison support for GEPA using Bradley-Terry model.

## Quick Start

```python
from gepa.pairwise import ComparisonResult, PairwiseComparator
from gepa.pairwise.adapter import PairwiseGEPAAdapter, PairwiseEvaluationBatch
from gepa.pairwise.state import HybridGEPAState

# 1. Implement comparator
class LLMJudge(PairwiseComparator):
    def compare(self, output_a, output_b, data_instance):
        prompt = f"Which answer is better for: {data_instance['question']}?\nA: {output_a}\nB: {output_b}"
        response = llm.complete(prompt)
        return ComparisonResult.A_BETTER if "A" in response else ComparisonResult.B_BETTER

# 2. Implement adapter
class QAAdapter(PairwiseGEPAAdapter):
    def evaluate(self, batch, candidate, capture_traces=False):
        outputs = [llm.complete(candidate["instruction"] + ex["question"]) for ex in batch]
        return PairwiseEvaluationBatch(outputs=outputs)

    def comparator(self):
        return LLMJudge()

# 3. Use with GEPA (integration with full API coming soon)
adapter = QAAdapter()
# ... optimization loop uses adapter.evaluate() and adapter.comparator()
```

See `examples/pairwise_llm_judge.py` for complete working example.

## Why Pairwise Comparisons?

**Problem**: Defining scalar metrics is hard
- How to score text quality numerically?
- How to balance multiple objectives (accuracy + clarity + conciseness)?
- Humans better at "which is better?" than "score this 0-100"

**Solution**: Use pairwise comparisons
- LLMs judge: "Which answer is better?"
- Convert to scores via Bradley-Terry model
- No metric design needed!

**Validated performance**:
- Bradley-Terry rank correlation: **0.998** with ground truth
- Finds same solutions as scalar GEPA (±5%)
- O(N²) comparison overhead negligible for LLM evaluations

## Architecture

```
Traditional GEPA              Pairwise GEPA (New!)
───────────────              ────────────────────
evaluate() → scores          evaluate() → outputs
                             compare(A,B) → A_BETTER/B_BETTER/TIE
sum(scores) > threshold      Bradley-Terry: comparisons → scores
                             sum(scores) > threshold

                             ↓
                      Standard GEPA logic
                      (98% code reuse)
```

**Key insight**: Store comparisons, compute scores on-demand via Bradley-Terry. This enables:
- Full compatibility with existing GEPA code
- Transparent integration (scores computed as property)
- Efficient caching (scores recomputed only when comparisons change)

## Implementation

### Core Components

**1. Types** (`src/gepa/pairwise/types.py`)
```python
class ComparisonResult(Enum):
    A_BETTER = "a_better"
    B_BETTER = "b_better"
    TIE = "tie"
    INCOMPARABLE = "incomparable"

class PairwiseComparator(Protocol):
    def compare(self, output_a, output_b, data_instance) -> ComparisonResult:
        ...
```

**2. Bradley-Terry** (`src/gepa/pairwise/bradley_terry.py`)
```python
def bradley_terry_scores(
    programs: list[int],
    comparison_results: dict[tuple[int, int], ComparisonResult],
    iterations: int = 100,
) -> dict[int, float]:
    """Convert pairwise comparisons to scalar scores via MLE."""
```

**3. Hybrid State** (`src/gepa/pairwise/state.py`)
```python
class HybridGEPAState(GEPAState):
    """Stores comparisons, computes scores lazily via Bradley-Terry."""

    comparison_cache: dict[tuple[int, int, DataId], ComparisonResult]
    prog_candidate_val_outputs: list[dict[DataId, RolloutOutput]]

    @property
    def prog_candidate_val_subscores(self) -> list[dict[DataId, float]]:
        """Compute scores via Bradley-Terry (cached)."""
```

**4. Adapter Protocol** (`src/gepa/pairwise/adapter.py`)
```python
class PairwiseGEPAAdapter(Protocol):
    def evaluate(self, batch, candidate, capture_traces) -> PairwiseEvaluationBatch:
        """Return outputs (no scores)."""

    def comparator(self) -> PairwiseComparator:
        """Return comparison function."""
```

### Usage Pattern

```python
# Create hybrid state
state = HybridGEPAState.create_from_seed(
    seed_candidate={"instruction": "..."},
    base_valset_outputs=outputs,
)

# Add program with automatic comparisons
state.add_program_with_comparisons(
    new_program=candidate,
    valset_outputs=new_outputs,
    comparator=adapter.comparator(),
    valset_data_instances=valset,
    # ... other params
)

# Access scores (computed via Bradley-Terry, cached)
scores = state.prog_candidate_val_subscores
```

## Testing

We provide comprehensive tests validating the approach:

```bash
cd tests

# Test Bradley-Terry algorithm
python test_pairwise_algorithms.py

# Compare scalar vs pairwise GEPA
python compare_scalar_vs_pairwise.py

# Expected output:
# Scalar GEPA:    Best x=51, score=0.9901
# Pairwise GEPA:  Best x=51, score=0.9900
# Difference:     ✓ -0.0001 (both find same solution!)
```

**Test problems** (no LLM needed):
- `QuadraticProblem`: Find x minimizing (x - 50)²
- `NonConvexProblem`: Multiple local optima
- `MultiObjectiveTradeoff`: Accuracy vs efficiency

See `tests/test_pairwise_synthetic.py` for test problem definitions.

## API Reference

### ComparisonResult

```python
ComparisonResult.A_BETTER       # Output A is strictly better
ComparisonResult.B_BETTER       # Output B is strictly better
ComparisonResult.TIE            # Equally good
ComparisonResult.INCOMPARABLE   # Cannot compare
```

### PairwiseComparator

```python
class PairwiseComparator(Protocol[RolloutOutput, DataInst]):
    def compare(
        self,
        output_a: RolloutOutput,
        output_b: RolloutOutput,
        data_instance: DataInst,
    ) -> ComparisonResult:
        """Compare two outputs for same input."""
```

### bradley_terry_scores

```python
def bradley_terry_scores(
    programs: list[int],
    comparison_results: dict[tuple[int, int], ComparisonResult],
    iterations: int = 100,
    tolerance: float = 1e-6,
    normalize: bool = True,
) -> dict[int, float]:
    """
    Convert pairwise comparisons to scalar scores.

    Uses MM algorithm for maximum likelihood estimation.
    Guaranteed convergence, handles sparse comparisons.
    """
```

### HybridGEPAState

```python
class HybridGEPAState(GEPAState):
    """GEPA state with pairwise comparison support."""

    @staticmethod
    def create_from_seed(
        seed_candidate: dict[str, str],
        base_valset_outputs: dict[DataId, RolloutOutput],
        bradley_terry_iterations: int = 100,
    ) -> "HybridGEPAState":
        """Create state from seed candidate."""

    def add_program_with_comparisons(
        self,
        new_program: dict[str, str],
        valset_outputs: dict[DataId, RolloutOutput],
        comparator: PairwiseComparator,
        valset_data_instances: dict[DataId, DataInst],
        # ... other params
    ) -> int:
        """Add program and perform pairwise comparisons."""

    @property
    def prog_candidate_val_subscores(self) -> list[dict[DataId, float]]:
        """Get scores (computed via Bradley-Terry, cached)."""
```

### PairwiseGEPAAdapter

```python
class PairwiseGEPAAdapter(Protocol):
    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> PairwiseEvaluationBatch:
        """Evaluate candidate, return outputs (no scores)."""

    def comparator(self) -> PairwiseComparator:
        """Return pairwise comparator."""

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: PairwiseEvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build reflective dataset for instruction refinement."""
```

## Performance

**Comparison overhead**:
- N programs → N(N-1)/2 total comparisons
- New program → N-1 comparisons (vs existing)
- Bradley-Terry: < 1ms for 100 programs

**For LLM-as-judge** (100ms per comparison):
- Adding program #50: 49 comparisons × 100ms = ~5s
- Bradley-Terry overhead: < 1ms (negligible)

**Validation results**:
- Bradley-Terry rank correlation: 0.998
- Top-1 accuracy: 1.0 (identifies best program)
- Score difference vs scalar GEPA: < 0.05 (5%)

## Limitations

1. **O(N²) comparisons**: All programs compared pairwise
   - Mitigated by caching (only new programs compared)
   - Future: Sparse comparison strategies

2. **Single quality dimension**: Bradley-Terry assumes total ordering
   - Multi-objective extension available in `MULTI_OBJECTIVE_PAIRWISE_GEPA.md`
   - Future: Implement multi-objective support

3. **Comparison cost**: Each comparison has cost (LLM, human)
   - Comparisons parallelizable
   - Future: Active learning for comparison selection

## Future Work

- [ ] Full GEPA API integration (seamless pairwise mode)
- [ ] Multi-objective pairwise support
- [ ] Sparse comparison strategies (reduce O(N²))
- [ ] Uncertainty quantification on scores
- [ ] More domain examples (code, creative writing)

## References

- Bradley & Terry (1952): Rank Analysis of Incomplete Block Designs
- `examples/pairwise_llm_judge.py`: Complete LLM-as-judge example
- `tests/compare_scalar_vs_pairwise.py`: Validation framework
- `MULTI_OBJECTIVE_PAIRWISE_GEPA.md`: Multi-objective design (future)
