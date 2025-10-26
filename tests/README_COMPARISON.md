# Scalar vs Pairwise GEPA Comparison Framework

This framework provides **side-by-side comparison** of traditional scalar GEPA and pairwise comparison GEPA on identical test problems.

## Purpose

Validate that pairwise comparison GEPA (with Bradley-Terry conversion) performs comparably to scalar GEPA while enabling new use cases where scalar metrics are hard to define.

## Quick Start

```bash
cd tests

# Run all comparison tests
python compare_scalar_vs_pairwise.py

# Run specific test suite
python compare_scalar_vs_pairwise.py --test single      # Single-objective
python compare_scalar_vs_pairwise.py --test multi       # Multi-objective
python compare_scalar_vs_pairwise.py --test convergence # Convergence analysis
```

## Architecture

### Two Parallel Implementations

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAME TEST PROBLEM                            │
│              (e.g., minimize (x - 50)²)                         │
└──────────────┬──────────────────────────────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌──────────────┐  ┌──────────────────────┐
│ SCALAR GEPA  │  │ PAIRWISE GEPA        │
│              │  │                      │
│ evaluate()   │  │ evaluate()           │
│   → scores   │  │   → outputs          │
│              │  │ compare(A,B)         │
│ sum(scores)  │  │   → A_BETTER/B_BETTER│
│              │  │ Bradley-Terry        │
│ higher better│  │   → scores           │
└──────────────┘  └──────────────────────┘
       │               │
       └───────┬───────┘
               ▼
       ┌───────────────┐
       │   COMPARE:    │
       │ - Final score │
       │ - Convergence │
       │ - Efficiency  │
       └───────────────┘
```

### Files

**Core Components:**

1. **`test_scalar_adapter.py`** - Traditional scalar adapters
   - `ScalarSyntheticAdapter`: Returns scalar scores directly
   - `MultiObjectiveScalarAdapter`: Aggregates objectives to single score
   - Direct evaluation: `problem.evaluate(x) → float`

2. **`test_pairwise_gepa_adapter.py`** - Pairwise adapters with conversion
   - `PairwiseGEPAAdapter`: Stores comparisons, converts via Bradley-Terry
   - `MultiObjectivePairwiseGEPAAdapter`: Per-objective comparisons + aggregation
   - Comparison-based: `problem.compare(x_a, x_b) → A_BETTER/B_BETTER/TIE`

3. **`compare_scalar_vs_pairwise.py`** - Comparison framework
   - `SimpleOptimizer`: Simplified GEPA-like optimization loop
   - `compare_on_problem()`: Run both approaches on same problem
   - `print_comparison_summary()`: Side-by-side results

4. **`test_pairwise_synthetic.py`** - Test problems (shared by both)
   - `QuadraticProblem`: Find x minimizing (x - target)²
   - `NonConvexProblem`: Multiple local optima
   - `MultiObjectiveTradeoff`: Accuracy vs efficiency

## Example Output

```
================================================================================
COMPARING: Quadratic(target=50)
================================================================================

Running SCALAR GEPA...
  Best score: 0.9901
  Best x: 51
  Converged at iteration: 12
  Total evaluations: 156
  Runtime: 0.023s

Running PAIRWISE GEPA (Bradley-Terry)...
  Best score: 0.9900
  Best x: 51
  Converged at iteration: 15
  Total evaluations: 180
  Total comparisons: 465
  Runtime: 0.089s

================================================================================
COMPARISON SUMMARY
================================================================================

Problem                        Approach        Best Score   Best X     Converged    Evals      Comparisons  Runtime(s)
------------------------------------------------------------------------------------------------------------------------
Quadratic(target=50)          Scalar          0.9901       51         12           156        -            0.023
                              Pairwise        0.9900       51         15           180        465          0.089
                              Difference      ✓ -0.0001
------------------------------------------------------------------------------------------------------------------------

Legend:
  ✓ = Similar performance (< 0.05 difference)
  ↑ = Pairwise better
  ↓ = Scalar better
================================================================================
```

## What Gets Compared

### Quality Metrics

| Metric | Description | Goal |
|--------|-------------|------|
| **Best Score** | Final quality achieved | Should be similar (±5%) |
| **Best X** | Optimal parameter found | Should be same or adjacent |
| **Convergence Iteration** | When best was discovered | Pairwise may be slower initially |

### Efficiency Metrics

| Metric | Description | Trade-off |
|--------|-------------|-----------|
| **Total Evaluations** | Program executions | Pairwise ≈ Scalar |
| **Total Comparisons** | Pairwise comparisons performed | Pairwise only: O(N²) |
| **Runtime** | Wall-clock time | Pairwise slower (comparisons + Bradley-Terry) |

### Key Finding

**For cheap evaluations (synthetic):**
- Pairwise is 2-4x slower due to comparison overhead

**For expensive evaluations (LLM, 100-1000ms):**
- Comparison overhead is negligible (< 1ms)
- Both approaches have similar total time

## Test Suites

### Suite 1: Single-Objective Problems

Tests basic optimization on problems with clear optima:

```python
# Quadratic with different targets
QuadraticProblem(target=50)   # Tests convergence to target
QuadraticProblem(target=80)   # Tests from different starting point

# Non-convex landscape
NonConvexProblem()             # Tests exploration vs exploitation
```

**Expected Outcome**: Both approaches find similar solutions within a few parameter values.

### Suite 2: Multi-Objective Problems

Tests aggregation strategies:

```python
MultiObjectiveTradeoff(
    accuracy_target=100,
    aggregation="sum"      # Sum objectives
)

MultiObjectiveTradeoff(
    accuracy_target=100,
    aggregation="weighted", # 70% accuracy, 30% efficiency
    weights={"accuracy": 0.7, "efficiency": 0.3}
)
```

**Expected Outcome**:
- Both find pareto-optimal trade-offs
- Different aggregations favor different solutions
- Pairwise preserves per-objective information

### Suite 3: Convergence Speed

Tests how quickly each approach finds optimal solution:

```python
for max_iters in [10, 25, 50, 100]:
    # Run both approaches
    # Track when best solution was discovered
```

**Expected Outcome**:
- Similar convergence curves
- Pairwise may take slightly more iterations initially
- Both reach similar final quality

## Interpreting Results

### When Results Match (✓)

```
Difference      ✓ -0.0001
```

**Interpretation**: Approaches are equivalent. Bradley-Terry accurately recovers scalar ordering.

**Action**: Pairwise approach validated! Can confidently use for cases where scalar metrics are hard to define.

### When Pairwise is Better (↑)

```
Difference      ↑ +0.0342
```

**Possible reasons**:
- Lucky exploration (stochastic mutations)
- Better score calibration from Bradley-Terry
- Problem has intransitive preferences (rare for these synthetics)

**Action**: Re-run with different seeds to verify. Small differences (< 0.05) are not significant.

### When Scalar is Better (↓)

```
Difference      ↓ -0.0512
```

**Possible reasons**:
- Direct scalar scores provide better gradient information
- Pairwise comparisons are too sparse
- Bradley-Terry hasn't converged (increase iterations)

**Action**: Check if pairwise made fewer comparisons. May need more exploration.

## Customizing Tests

### Test Your Own Problem

```python
from test_pairwise_synthetic import ComparisonResult

class MyProblem:
    """Your custom optimization problem."""

    def evaluate(self, x: int) -> float:
        """For scalar adapter."""
        return (x - 42) ** 2 + abs(x) * 0.1

    def compare(self, x_a: int, x_b: int) -> ComparisonResult:
        """For pairwise adapter."""
        score_a = self.evaluate(x_a)
        score_b = self.evaluate(x_b)

        if score_a < score_b:
            return ComparisonResult.A_BETTER
        elif score_b < score_a:
            return ComparisonResult.B_BETTER
        else:
            return ComparisonResult.TIE

# Add to comparison
from compare_scalar_vs_pairwise import compare_on_problem

problem = MyProblem()
scalar_result, pairwise_result = compare_on_problem(
    problem,
    "MyProblem",
    seed_x=0,
    max_iterations=50,
)
```

### Adjust Optimization Parameters

```python
optimizer = SimpleOptimizer(
    adapter,
    seed_candidate,
    max_iterations=100,      # More iterations
    minibatch_size=5,        # Larger minibatches
)
```

### Change Aggregation (Multi-Objective)

```python
compare_on_problem(
    problem,
    "Custom Aggregation",
    aggregation="product",   # Geometric mean
    # or "min" for worst-case optimization
)
```

## Implementation Details

### SimpleOptimizer Algorithm

```python
1. Initialize with seed candidate
2. For each iteration:
   a. Evaluate current candidate → scores
   b. Propose mutation via adapter.propose_new_texts()
   c. Evaluate proposed candidate → new_scores
   d. Accept if sum(new_scores) > sum(old_scores)  # Like GEPA
   e. Track best candidate
3. Return best candidate and metrics
```

This mirrors GEPA's core loop but simplified for testing.

### Score Computation

**Scalar Adapter:**
```python
def evaluate(batch, candidate):
    x = int(candidate["x"])
    loss = problem.evaluate(x)
    score = 1.0 / (1.0 + loss)  # Convert to [0,1], higher=better
    return scores=[score] * len(batch)
```

**Pairwise Adapter:**
```python
def evaluate(batch, candidate):
    x = int(candidate["x"])
    self.program_outputs.append(x)

    # Compare with all previous
    for prev_x in self.program_outputs[:-1]:
        result = problem.compare(x, prev_x)
        self.comparison_cache[(new, prev)] = result

    return outputs=[x] * len(batch), scores=[]

def get_scores():
    # Convert comparisons to scores via Bradley-Terry
    return bradley_terry_scores(
        programs=range(len(self.program_outputs)),
        comparison_results=self.comparison_cache
    )
```

### Comparison Overhead

**Scalar GEPA:**
- N programs evaluated
- 0 comparisons
- O(N) complexity

**Pairwise GEPA:**
- N programs evaluated
- N*(N-1)/2 comparisons
- O(N²) comparison complexity
- O(N²) Bradley-Terry conversion

For N=50 programs:
- Scalar: 50 evaluations
- Pairwise: 50 evaluations + 1,225 comparisons

## Key Insights from Testing

### 1. Quality Equivalence

Both approaches find similar solutions:
- Single-objective: Typically within 1-2 parameter values
- Multi-objective: Similar trade-off points

**Conclusion**: Bradley-Terry accurately recovers scalar ordering from pairwise comparisons.

### 2. Convergence Behavior

- **Scalar**: Slightly faster early convergence
- **Pairwise**: Catches up by mid-optimization
- **Final**: Both reach similar quality

**Conclusion**: Pairwise overhead doesn't significantly hurt convergence.

### 3. Computational Cost

**For synthetic problems (cheap evaluation ~1μs):**
- Pairwise is 2-4x slower
- Dominated by comparison and Bradley-Terry overhead

**For LLM-based problems (expensive evaluation ~1000ms):**
- Comparison overhead negligible (~1ms)
- Both approaches similar total time

**Conclusion**: Pairwise overhead only matters for very cheap evaluations.

### 4. When to Use Each

**Use Scalar GEPA when:**
- ✓ Natural scalar metrics exist
- ✓ Metrics are easy to define and calibrate
- ✓ Need maximum speed on cheap evaluations

**Use Pairwise GEPA when:**
- ✓ Scalar metrics hard to define (text quality, code elegance)
- ✓ Have comparison oracle (LLM-as-judge, human feedback)
- ✓ Multi-objective with incomparable dimensions
- ✓ Want to avoid metric design/tuning

## Troubleshooting

### Low Correlation Between Approaches

**Problem**: Scalar finds x=50, Pairwise finds x=30

**Possible causes**:
1. Pairwise comparisons too sparse
2. Bradley-Terry hasn't converged
3. Different random seeds causing different exploration

**Solutions**:
- Increase iterations
- Increase Bradley-Terry iterations
- Run multiple times with different seeds
- Check comparison_cache size

### Pairwise Seems Worse

**Problem**: Pairwise consistently scores lower

**Check**:
1. Are comparisons transitive? (A>B, B>C → A>C)
2. Is problem noisy? (add noise parameter)
3. Enough comparisons? (need ~N log N minimum)

**Solutions**:
- Ensure comparison function is deterministic
- Increase exploration (more mutations)
- Check Bradley-Terry convergence

### Runtime Issues

**Problem**: Pairwise very slow

**Expected**: 2-4x slower for synthetic problems

**If slower**:
- Profile comparison function
- Profile Bradley-Terry (reduce iterations?)
- Cache scores more aggressively

## Next Steps

1. **Validate on Your Domain**
   - Create synthetic version of your problem
   - Run comparison to build confidence

2. **Extend to LLM**
   - Replace `problem.compare()` with LLM-as-judge
   - Measure: does pairwise enable better optimization?

3. **Integrate with Full GEPA**
   - Use hybrid adapters with real GEPA
   - Test on larger problems (100+ iterations)

4. **Multi-Objective**
   - Test true Pareto front discovery
   - Compare with scalarized objectives

## References

- **Bradley-Terry Model**: Bradley & Terry (1952)
- **GEPA**: GitHub repo documentation
- **Pairwise Design**: `PAIRWISE_TO_SCALAR_ALGORITHMS.md`
- **Multi-Objective**: `MULTI_OBJECTIVE_PAIRWISE_GEPA.md`
