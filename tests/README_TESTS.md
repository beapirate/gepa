# Pairwise Comparison Tests

Tests for pairwise comparison implementation.

## Test Files

### test_pairwise_synthetic.py
Synthetic test problems that don't require LLMs:
- `QuadraticProblem`: Find x minimizing (x - 50)²
- `NonConvexProblem`: Multiple local optima
- `MultiObjectiveTradeoff`: Accuracy vs efficiency trade-off

### test_pairwise_algorithms.py
Tests for Bradley-Terry algorithm:
- Validates score conversion from pairwise comparisons
- Tests rank correlation with ground truth
- Benchmarks against other algorithms (Elo, Win Rate)

## Running Tests

```bash
# Test Bradley-Terry algorithm
python -m pytest tests/test_pairwise_algorithms.py -v

# Or run directly
python tests/test_pairwise_algorithms.py
```

## Expected Results

Bradley-Terry should achieve:
- Rank correlation > 0.95 with ground truth
- Top-1 accuracy > 0.8
- Runtime < 10ms for 50 programs

## Adding New Tests

To test pairwise comparisons on your domain:

```python
from tests.test_pairwise_synthetic import ComparisonResult

class MyProblem:
    def evaluate(self, x: int) -> float:
        # Your objective function
        return (x - 42) ** 2

    def compare(self, x_a: int, x_b: int) -> ComparisonResult:
        score_a = self.evaluate(x_a)
        score_b = self.evaluate(x_b)
        return ComparisonResult.A_BETTER if score_a < score_b else ComparisonResult.B_BETTER

# Test with Bradley-Terry
from tests.test_pairwise_algorithms import bradley_terry_scores

# ... create comparisons ...
scores = bradley_terry_scores(programs, comparisons)
```
