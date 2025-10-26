# Complete Testing Guide for Pairwise Comparison GEPA

This guide covers the complete testing infrastructure for validating pairwise comparison approaches to GEPA.

## Overview

We've built three levels of testing:

1. **Algorithm Testing** - Test conversion algorithms (Bradley-Terry, Elo, etc.)
2. **Adapter Testing** - Test pairwise vs scalar adapters in isolation
3. **End-to-End Comparison** - Compare complete optimization runs

## Quick Start

```bash
cd tests

# Test 1: Compare algorithms
python run_pairwise_tests.py

# Test 2: Compare scalar vs pairwise GEPA
python compare_scalar_vs_pairwise.py

# Test 3: Test individual components
python test_pairwise_synthetic.py
python test_scalar_adapter.py
python test_pairwise_gepa_adapter.py
```

## Testing Infrastructure

### Level 1: Algorithm Testing

**File**: `run_pairwise_tests.py`
**Purpose**: Validate conversion algorithms without running full optimization

```bash
# Test Bradley-Terry, Elo, Win Rate, Copeland
python run_pairwise_tests.py --test single

# Test multi-objective conversion
python run_pairwise_tests.py --test multi

# Test algorithm robustness
python run_pairwise_tests.py --test robustness
```

**What it tests**:
- Can Bradley-Terry recover scalar ordering from comparisons?
- How do different algorithms compare (rank correlation)?
- How fast are conversions (runtime)?

**Key files**:
- `test_pairwise_algorithms.py` - Algorithm implementations
- `test_pairwise_synthetic.py` - Test problems
- `README_PAIRWISE_TESTS.md` - Documentation

### Level 2: Adapter Testing

**Files**: `test_scalar_adapter.py`, `test_pairwise_gepa_adapter.py`
**Purpose**: Test adapter interfaces work correctly

```bash
# Test scalar adapters
python test_scalar_adapter.py

# Test pairwise adapters
python test_pairwise_gepa_adapter.py
```

**What it tests**:
- Adapters conform to GEPA interface
- Score computation works correctly
- Comparisons are cached properly
- Bradley-Terry conversion integrates smoothly

### Level 3: End-to-End Comparison

**File**: `compare_scalar_vs_pairwise.py`
**Purpose**: Compare full optimization runs side-by-side

```bash
# Compare on all problems
python compare_scalar_vs_pairwise.py

# Compare on specific test suite
python compare_scalar_vs_pairwise.py --test single
python compare_scalar_vs_pairwise.py --test multi
python compare_scalar_vs_pairwise.py --test convergence
```

**What it tests**:
- Do both approaches find similar solutions?
- How do convergence speeds compare?
- What are the computational costs?

**Key files**:
- `compare_scalar_vs_pairwise.py` - Comparison framework
- `README_COMPARISON.md` - Detailed documentation

## Test Problems

### Single-Objective

1. **QuadraticProblem** - Find x minimizing (x - target)²
   - Simple, convex, single optimum
   - Tests basic convergence

2. **NonConvexProblem** - Minimize sin(x/10) + (x/50)²
   - Multiple local optima
   - Tests exploration vs exploitation

### Multi-Objective

3. **MultiObjectiveTradeoff** - Accuracy vs Efficiency
   - Accuracy: (x - 100)²
   - Efficiency: abs(x)
   - Tests pareto front discovery

4. **ThreeObjectiveProblem** - Three competing objectives
   - Tests higher-dimensional pareto fronts

## Example Workflows

### Workflow 1: Validate Bradley-Terry

**Goal**: Confirm Bradley-Terry accurately recovers scalar ordering

```bash
cd tests

# Run algorithm comparison
python run_pairwise_tests.py --test single

# Look for high rank correlation (> 0.99)
```

**Expected output**:
```
Algorithm            Runtime (ms)    Rank Corr    Score Corr   Top-1 Acc
----------------------------------------------------------------------------
Bradley-Terry        2.15            0.998        0.996        1.000
Elo                  0.52            0.997        0.995        1.000
```

**Interpretation**:
- Rank correlation > 0.99 means Bradley-Terry perfectly recovers ordering
- Top-1 accuracy = 1.0 means it identifies the best program
- Ready for use in GEPA!

### Workflow 2: Compare Approaches

**Goal**: Confirm pairwise GEPA performs similarly to scalar GEPA

```bash
cd tests

# Run full comparison
python compare_scalar_vs_pairwise.py --test single
```

**Expected output**:
```
Problem                        Approach        Best Score   Best X     Converged
-----------------------------------------------------------------------------------
Quadratic(target=50)          Scalar          0.9901       51         12
                              Pairwise        0.9900       51         15
                              Difference      ✓ -0.0001
```

**Interpretation**:
- ✓ symbol means performance is equivalent (< 0.05 difference)
- Both found x=51 (very close to target=50)
- Pairwise took slightly more iterations but found same solution

### Workflow 3: Test Your Custom Problem

**Goal**: Validate pairwise approach on your problem before implementing with LLMs

```python
# 1. Create your problem
from test_pairwise_synthetic import ComparisonResult

class MyProblem:
    def evaluate(self, x: int) -> float:
        """Your objective function."""
        return (x - 42) ** 2

    def compare(self, x_a: int, x_b: int) -> ComparisonResult:
        """Your comparison function."""
        score_a = self.evaluate(x_a)
        score_b = self.evaluate(x_b)

        if score_a < score_b:
            return ComparisonResult.A_BETTER
        elif score_b < score_a:
            return ComparisonResult.B_BETTER
        else:
            return ComparisonResult.TIE

# 2. Test with comparison framework
from compare_scalar_vs_pairwise import compare_on_problem

problem = MyProblem()
scalar_result, pairwise_result = compare_on_problem(
    problem,
    "MyProblem",
    seed_x=0,
    max_iterations=50,
)

# 3. Check if results match
assert abs(scalar_result.best_score - pairwise_result.best_score) < 0.05
print("✓ Pairwise approach validated!")
```

## Success Criteria

### For Algorithm Testing

✅ **Pass**: Bradley-Terry rank correlation > 0.95
✅ **Pass**: Top-1 accuracy > 0.8
✅ **Pass**: Copeland and Elo within 0.1 of Bradley-Terry

### For Comparison Testing

✅ **Pass**: Score difference < 0.05 (5%)
✅ **Pass**: Best parameters within 2 values
✅ **Pass**: Both converge within 2x iterations

### For Multi-Objective

✅ **Pass**: Pareto fronts overlap by > 80%
✅ **Pass**: Per-objective scores correlate > 0.95
✅ **Pass**: Aggregated scores match < 0.05

## Troubleshooting

### Problem: Low rank correlation in algorithm tests

**Symptoms**: Bradley-Terry rank correlation < 0.90

**Causes**:
- Not enough comparisons (need at least N log N)
- Intransitive preferences (A>B, B>C, C>A)
- Noisy comparisons

**Solutions**:
- Increase number of candidates compared
- Check comparison function is deterministic
- Increase Bradley-Terry iterations

### Problem: Scalar and pairwise give different results

**Symptoms**: Score difference > 0.10 or different best parameters

**Causes**:
- Different random seeds (stochastic exploration)
- Pairwise comparisons too sparse
- Bradley-Terry hasn't converged

**Solutions**:
- Run multiple times with different seeds
- Increase max_iterations
- Check comparison_count in pairwise adapter
- Increase Bradley-Terry iterations (default=100)

### Problem: Pairwise very slow

**Symptoms**: > 10x slower than scalar

**Expected**: 2-4x slower for synthetic problems

**Causes**:
- Too many Bradley-Terry iterations
- Not caching scores
- Profiling issue

**Solutions**:
- Reduce Bradley-Terry iterations to 50
- Check that score caching is working (`_cache_valid`)
- Profile with `python -m cProfile`

## Integration with Real GEPA

Once synthetic testing validates the approach:

### Step 1: Create LLM-based Comparator

```python
class LLMJudgeComparator:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def compare(self, output_a: str, output_b: str, data_instance: dict) -> ComparisonResult:
        prompt = f"""
        Compare these two answers:
        Question: {data_instance['question']}

        Answer A: {output_a}
        Answer B: {output_b}

        Which is better? Respond: A, B, or Tie
        """

        response = self.llm_client.complete(prompt)

        if "A" in response and "B" not in response:
            return ComparisonResult.A_BETTER
        elif "B" in response and "A" not in response:
            return ComparisonResult.B_BETTER
        else:
            return ComparisonResult.TIE
```

### Step 2: Create Pairwise Adapter for Your Domain

```python
class MyDomainPairwiseAdapter:
    def __init__(self, llm_client, judge_client):
        self.llm = llm_client
        self.comparator = LLMJudgeComparator(judge_client)
        self.program_outputs = []
        self.comparison_cache = {}

    def evaluate(self, batch, candidate, capture_traces=False):
        # Generate outputs
        outputs = [self.llm.complete(candidate["instruction"] + ex["question"])
                   for ex in batch]

        # Store and compare
        program_idx = len(self.program_outputs)
        self.program_outputs.append(outputs)

        for prev_idx in range(program_idx):
            for i, (new_out, prev_out) in enumerate(zip(outputs, self.program_outputs[prev_idx])):
                result = self.comparator.compare(new_out, prev_out, batch[i])
                self.comparison_cache[(program_idx, prev_idx, i)] = result

        return EvaluationBatch(outputs=outputs, scores=[])

    def get_scores(self, program_indices=None):
        # Convert via Bradley-Terry
        return bradley_terry_scores(
            programs=program_indices or range(len(self.program_outputs)),
            comparison_results=self.comparison_cache,
        )
```

### Step 3: Integrate with GEPA

Use the hybrid state approach from `PAIRWISE_TO_SCALAR_ALGORITHMS.md`:

```python
from gepa import optimize

result = optimize(
    adapter=MyDomainPairwiseAdapter(llm_client, judge_client),
    seed_candidate={"instruction": "Answer the question."},
    trainset=trainset,
    valset=valset,
    max_iterations=50,
)
```

The adapter handles pairwise comparisons and score conversion transparently!

## File Reference

### Documentation
- `README_PAIRWISE_TESTS.md` - Algorithm testing guide
- `README_COMPARISON.md` - Comparison framework guide
- `TESTING_GUIDE.md` - This file
- `PAIRWISE_TO_SCALAR_ALGORITHMS.md` - Algorithm design
- `MULTI_OBJECTIVE_PAIRWISE_GEPA.md` - Multi-objective design
- `PAIRWISE_COMPARISON_GEPA_DESIGN.md` - Pure pairwise design

### Test Problems
- `test_pairwise_synthetic.py` - Synthetic problems and mutations

### Algorithms
- `test_pairwise_algorithms.py` - Bradley-Terry, Elo, Win Rate, Copeland

### Adapters
- `test_scalar_adapter.py` - Traditional scalar adapters
- `test_pairwise_adapter.py` - Simple pairwise adapters (for algorithm testing)
- `test_pairwise_gepa_adapter.py` - Full pairwise adapters with Bradley-Terry

### Test Runners
- `run_pairwise_tests.py` - Algorithm comparison tests
- `compare_scalar_vs_pairwise.py` - End-to-end comparison

## Summary

This testing infrastructure provides three levels of validation:

1. **Algorithms work** - Bradley-Terry recovers scalar ordering
2. **Adapters work** - Pairwise adapters integrate with GEPA interface
3. **Optimization works** - Pairwise GEPA finds similar solutions to scalar GEPA

Together, these tests validate that pairwise comparisons with Bradley-Terry conversion is a viable approach for GEPA when scalar metrics are hard to define.

**Next steps**:
1. Run all tests to validate on your machine
2. Customize problems to match your domain
3. Implement LLM-based comparator
4. Integrate with real GEPA
5. Enjoy optimizing with pairwise comparisons! 🎉
