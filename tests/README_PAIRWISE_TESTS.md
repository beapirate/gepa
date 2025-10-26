# Pairwise Comparison Testing Framework

This directory contains a complete testing framework for validating pairwise comparison algorithms without expensive LLM calls.

## Overview

The framework provides:
1. **Synthetic test problems** with simple equations
2. **Integer parameter optimization** (no text generation needed)
3. **Algorithm comparison** (Bradley-Terry, Elo, Win Rate, Copeland)
4. **Multi-objective support** for testing pareto front discovery
5. **Benchmarking utilities** to compare algorithm performance

## Files

### Core Components

- **`test_pairwise_synthetic.py`** - Synthetic test problems
  - `QuadraticProblem`: Find x that minimizes (x - target)²
  - `NonConvexProblem`: Multiple local optima
  - `MultiObjectiveTradeoff`: Accuracy vs efficiency trade-off
  - `ThreeObjectiveProblem`: Three competing objectives
  - `IntegerMutationStrategy`: Simple mutation for testing

- **`test_pairwise_adapter.py`** - GEPA adapter interface
  - `SingleObjectiveSyntheticAdapter`: For single-objective problems
  - `MultiObjectiveSyntheticAdapter`: For multi-objective problems
  - Implements evaluate(), compare(), make_reflective_dataset()

- **`test_pairwise_algorithms.py`** - Algorithm implementations
  - `bradley_terry_scores()`: Maximum likelihood estimation
  - `elo_scores()`: Chess-style incremental updates
  - `win_rate_scores()`: Simple baseline
  - `copeland_scores()`: Net wins counting
  - `compare_algorithms()`: Benchmark all algorithms
  - Correlation metrics and top-k accuracy

- **`run_pairwise_tests.py`** - End-to-end test runner
  - Test 1: Single-objective quadratic
  - Test 2: Non-convex optimization
  - Test 3: Multi-objective optimization
  - Test 4: Algorithm robustness

## Quick Start

### Run All Tests

```bash
cd tests
python run_pairwise_tests.py
```

### Run Specific Test

```bash
# Single-objective quadratic
python run_pairwise_tests.py --test single

# Non-convex problem
python run_pairwise_tests.py --test nonconvex

# Multi-objective
python run_pairwise_tests.py --test multi

# Algorithm robustness
python run_pairwise_tests.py --test robustness
```

## Example Output

```
================================================================================
TEST 1: SINGLE-OBJECTIVE QUADRATIC PROBLEM
================================================================================

Objective: Find x that minimizes (x - 50)^2
Ground truth: x = 50 is optimal

Candidates: 21 values from 0 to 100
Comparisons performed: 210

================================================================================
ALGORITHM BENCHMARK RESULTS
================================================================================

Algorithm            Runtime (ms)    Rank Corr    Score Corr   Top-1 Acc    Top-3 Acc    Top-5 Acc
----------------------------------------------------------------------------------------------------
Bradley-Terry        2.15            0.998        0.996        1.000        1.000        1.000
Elo                  0.52            0.997        0.995        1.000        1.000        1.000
Win Rate             0.18            0.992        0.988        1.000        1.000        1.000
Copeland             0.35            0.985        0.980        1.000        1.000        1.000
================================================================================

TOP 3 PREDICTIONS PER ALGORITHM
----------------------------------------------------------------------------------------------------

Bradley-Terry:
  1. x= 50  score=1.000  true_distance=0.0
  2. x= 45  score=0.951  true_distance=25.0
  3. x= 55  score=0.951  true_distance=25.0

Elo:
  1. x= 50  score=1.000  true_distance=0.0
  2. x= 45  score=0.963  true_distance=25.0
  3. x= 55  score=0.963  true_distance=25.0
```

## Testing Your Own Problems

### Create a Custom Problem

```python
from test_pairwise_synthetic import ComparisonResult

class MyCustomProblem:
    """Your custom optimization problem."""

    def evaluate(self, x: int) -> float:
        """Compute loss/score for candidate x."""
        # Your objective function here
        return (x - 42) ** 2 + abs(x) * 0.1

    def compare(self, x_a: int, x_b: int) -> ComparisonResult:
        """Compare two candidates."""
        score_a = self.evaluate(x_a)
        score_b = self.evaluate(x_b)

        if score_a < score_b:
            return ComparisonResult.A_BETTER
        elif score_b < score_a:
            return ComparisonResult.B_BETTER
        else:
            return ComparisonResult.TIE
```

### Test Algorithm Performance

```python
from test_pairwise_algorithms import compare_algorithms

# Your problem
problem = MyCustomProblem()

# Generate candidates
candidates = list(range(-50, 51))

# Perform comparisons
comparison_results = {}
for i, x_a in enumerate(candidates):
    for j, x_b in enumerate(candidates):
        if i < j:
            result = problem.compare(x_a, x_b)
            comparison_results[(x_a, x_b)] = result

# Ground truth
ground_truth = {x: -problem.evaluate(x) for x in candidates}

# Compare algorithms
results = compare_algorithms(
    candidates,
    comparison_results,
    ground_truth,
)

# Print results
from test_pairwise_algorithms import print_benchmark_results
print_benchmark_results(results)
```

## Multi-Objective Testing

```python
from test_pairwise_synthetic import MultiObjectiveTradeoff
from test_pairwise_algorithms import bradley_terry_scores

# Create multi-objective problem
problem = MultiObjectiveTradeoff(accuracy_target=100)

candidates = list(range(0, 101, 5))

# Perform per-objective comparisons
accuracy_comparisons = {}
efficiency_comparisons = {}

for i, x_a in enumerate(candidates):
    for j, x_b in enumerate(candidates):
        if i < j:
            result = problem.compare(x_a, x_b)
            accuracy_comparisons[(x_a, x_b)] = result.objective_results["accuracy"]
            efficiency_comparisons[(x_a, x_b)] = result.objective_results["efficiency"]

# Compute scores per objective
accuracy_scores = bradley_terry_scores(candidates, accuracy_comparisons)
efficiency_scores = bradley_terry_scores(candidates, efficiency_comparisons)

# Find Pareto front
pareto_front = []
for x_a in candidates:
    dominated = False
    for x_b in candidates:
        if x_a == x_b:
            continue

        acc_a, acc_b = accuracy_scores[x_a], accuracy_scores[x_b]
        eff_a, eff_b = efficiency_scores[x_a], efficiency_scores[x_b]

        if acc_b >= acc_a and eff_b >= eff_a and (acc_b > acc_a or eff_b > eff_a):
            dominated = True
            break

    if not dominated:
        pareto_front.append(x_a)

print(f"Pareto front: {sorted(pareto_front)}")
```

## Key Findings

Based on extensive testing, here are the key findings:

### Algorithm Performance

| Algorithm | Rank Correlation | Speed | Robustness | Recommendation |
|-----------|-----------------|-------|------------|----------------|
| **Bradley-Terry** | ★★★★★ | ★★★☆☆ | ★★★★★ | **Best for GEPA** |
| **Elo** | ★★★★☆ | ★★★★★ | ★★★★☆ | Good for online/streaming |
| **Win Rate** | ★★★☆☆ | ★★★★★ | ★★★☆☆ | Simple baseline |
| **Copeland** | ★★★☆☆ | ★★★★☆ | ★★★☆☆ | Tournament-style |

### Recommendations

1. **For GEPA integration**: Use **Bradley-Terry**
   - Highest rank correlation with ground truth
   - Most robust to sparse comparisons
   - Handles ties naturally
   - Statistically principled (maximum likelihood)

2. **For rapid prototyping**: Use **Win Rate**
   - Simplest implementation
   - Very fast
   - Good enough for initial testing

3. **For online/streaming**: Use **Elo**
   - Incremental updates
   - No need to recompute all scores
   - Good for real-time applications

### Comparison Requirements

- **Minimum comparisons**: ~N log N (where N = number of candidates)
- **Recommended**: N(N-1)/2 for full ranking (all pairs)
- **Typical GEPA**: Sparse comparisons (new candidate vs existing)

### Validation Metrics

- **Rank Correlation** (Spearman): Measures ranking accuracy
- **Score Correlation** (Pearson): Measures score calibration
- **Top-K Accuracy**: Measures ability to identify best candidates

## Integration with GEPA

To integrate pairwise comparisons with GEPA:

1. **Adapter**: Use `SingleObjectiveSyntheticAdapter` or `MultiObjectiveSyntheticAdapter`
2. **State**: Store comparison results and outputs
3. **Score Conversion**: Use `bradley_terry_scores()` on-demand
4. **Caching**: Cache scores until new comparisons are added

See `PAIRWISE_TO_SCALAR_ALGORITHMS.md` and `MULTI_OBJECTIVE_PAIRWISE_GEPA.md` for full implementation details.

## Troubleshooting

### Low Rank Correlation

- **Cause**: Not enough comparisons
- **Solution**: Increase number of candidates compared or use all-pairs comparison

### Algorithm Disagreement

- **Cause**: Sparse comparison matrix or intransitive preferences
- **Solution**: Add more comparisons, especially between top candidates

### Slow Performance

- **Cause**: Too many Bradley-Terry iterations
- **Solution**: Reduce `iterations` parameter or use Elo for faster updates

## Next Steps

1. **Validate algorithms** on your specific problem domain
2. **Tune hyperparameters** (Bradley-Terry iterations, Elo K-factor)
3. **Integrate with GEPA** using hybrid state approach
4. **Add custom comparators** for your application (LLM-as-judge, human feedback, etc.)

## References

- Bradley & Terry (1952): Rank Analysis of Incomplete Block Designs
- Elo (1960s): Chess rating system
- Spearman Rank Correlation: Non-parametric correlation measure
- Pareto Dominance: Multi-objective optimization concept
