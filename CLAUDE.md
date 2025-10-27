# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Note**: This project uses [bd (beads)](https://github.com/steveyegge/beads) for issue tracking. Use `bd` commands instead of markdown TODOs. See the "Issue Tracking with bd" section below for workflow details.

## Overview

GEPA (Genetic-Pareto) is a framework for optimizing text components (AI prompts, code, instructions) using LLM-based reflection and evolutionary search. The codebase is organized around a flexible adapter pattern that allows GEPA to plug into arbitrary systems and optimize different types of text snippets.

## Development Commands

### Environment Setup

**Using uv (recommended):**
```bash
# Install dependencies
uv sync --extra dev --python 3.11

# All commands require 'uv run' prefix
uv run pytest tests/
uv run python script.py
```

**Using conda + pip:**
```bash
conda create -n gepa-dev python=3.11
conda activate gepa-dev
pip install -e ".[dev]"
pytest tests/
```

### Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_state.py

# Run pairwise comparison tests
uv run pytest tests/test_pairwise_algorithms.py
uv run pytest tests/test_pairwise_synthetic.py

# Run specific adapter tests
uv run pytest tests/test_rag_adapter/
```

### Code Quality

```bash
# Install pre-commit hooks (once)
uv run pre-commit install

# Run linting/formatting manually
uv run pre-commit run                    # Check staged files
uv run pre-commit run --all-files        # Check all files
uv run pre-commit run --files path/to/file.py

# Pre-commit hooks run automatically on commit
git add .
git commit -m "message"  # Hooks run here
```

**Linting Rules:**
- Uses `ruff` for both linting and formatting
- Line length: 120 characters
- Target: Python 3.10+
- Follows Google Python Style Guide
- Key ignores: complexity (C901), line length via formatter (E501)

## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Auto-syncs to JSONL for version control
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**
```bash
bd ready --json
```

**Create new issues:**
```bash
bd create "Issue title" -t bug|feature|task -p 0-4 --json
bd create "Issue title" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**
```bash
bd update bd-42 --status in_progress --json
bd update bd-42 --priority 1 --json
```

**Complete work:**
```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task**: `bd update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`

### Auto-Sync

bd automatically syncs with git:
- Exports to `.beads/issues.jsonl` after changes (5s debounce)
- Imports from JSONL when newer (e.g., after `git pull`)
- No manual export/import needed!

### MCP Server (Recommended)

If using Claude or MCP-compatible clients, the beads MCP server is available with these functions:
- `mcp__plugin_beads_beads__ready` - Find ready tasks
- `mcp__plugin_beads_beads__create` - Create new issues
- `mcp__plugin_beads_beads__update` - Update issue status/priority
- `mcp__plugin_beads_beads__close` - Close completed issues
- `mcp__plugin_beads_beads__list` - List issues with filters
- `mcp__plugin_beads_beads__show` - Show issue details

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use with CLI
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

## Architecture

### Core Concepts

**Adapter Pattern** (`src/gepa/core/adapter.py`):
- **GEPAAdapter**: Single integration point between your system and GEPA
- **DataInst**: User-defined type for input data
- **Trajectory**: User-defined type capturing execution traces
- **RolloutOutput**: User-defined type for program outputs
- **EvaluationBatch**: Container for outputs, scores, and optional trajectories

**Key Responsibilities:**
1. `evaluate()`: Execute candidate program on batch, return scores + outputs
2. `make_reflective_dataset()`: Extract textual feedback from trajectories
3. `propose_new_texts()`: Optional custom proposal logic (defaults to reflection LM)

**State Management** (`src/gepa/core/state.py`):
- **GEPAState**: Tracks candidate programs, Pareto fronts, evaluation history
- Maintains parent-child relationships between program mutations
- Tracks per-data-instance scores for Pareto optimization

**Engine** (`src/gepa/core/engine.py`):
- **GEPAEngine**: Orchestrates the optimization loop
- Coordinates reflective mutation and merge proposers
- Handles evaluation policies (full eval vs incremental)
- Manages stopping conditions and progress tracking

### Directory Structure

```
src/gepa/
├── core/              # Core abstractions
│   ├── adapter.py     # GEPAAdapter protocol
│   ├── engine.py      # Optimization loop orchestration
│   ├── state.py       # State management
│   ├── data_loader.py # Data loading utilities
│   └── result.py      # Result container
├── adapters/          # Concrete adapter implementations
│   ├── default_adapter/          # Single-turn LLM system prompt optimization
│   ├── dspy_adapter/             # DSPy program optimization
│   ├── dspy_full_program_adapter/# Full DSPy program evolution
│   ├── generic_rag_adapter/      # Vector store-agnostic RAG
│   ├── terminal_bench_adapter/   # Terminal agent optimization
│   └── anymaths_adapter/         # Math problem solving
├── proposer/          # Candidate proposal strategies
│   ├── reflective_mutation/      # LLM-guided reflection
│   └── merge.py                  # Merge two Pareto programs
├── strategies/        # Pluggable strategies
│   ├── candidate_selector.py     # Select from Pareto front
│   ├── component_selector.py     # Choose components to update
│   ├── batch_sampler.py          # Training example selection
│   └── eval_policy.py            # Validation evaluation strategy
├── pairwise/          # Pairwise comparison support
│   ├── types.py                  # ComparisonResult enum
│   ├── bradley_terry.py          # Bradley-Terry scoring
│   ├── state.py                  # HybridGEPAState
│   └── adapter.py                # PairwiseGEPAAdapter protocol
├── logging/           # Experiment tracking
├── utils/             # Utilities and helpers
├── examples/          # Example adapters and use cases
└── api.py             # Main optimize() function

tests/
├── test_state.py                 # State management tests
├── test_pairwise_algorithms.py   # Bradley-Terry benchmarks
├── test_pairwise_synthetic.py    # Synthetic optimization problems
├── test_rag_adapter/             # RAG adapter tests
└── proposer/                     # Proposer tests
```

### Key Design Patterns

**1. Score-Based Optimization:**
- Higher scores = better performance
- Minibatch acceptance: `sum(new_scores) > sum(old_scores)`
- Validation tracking: `mean(scores)` for Pareto fronts
- Per-example scoring enables fine-grained optimization

**2. Pareto Optimization:**
- Maintains front of non-dominated programs on validation set
- Programs can excel on different subsets of data
- Enables multi-objective optimization naturally

**3. Reflective Mutation:**
- Uses execution traces to provide feedback to reflection LM
- Reflection LM proposes improved component text
- Component selection strategies (round-robin, all-at-once)
- Batch sampling strategies control training example selection

**4. Merge Strategy:**
- Combines successful programs from Pareto front
- Discovers synergies between different approaches
- Configurable via `use_merge` and `max_merge_invocations`

### Pairwise Comparison Extension

**Current Implementation** (see `README_PAIRWISE.md` for details):
- **Hybrid approach**: Store comparisons, compute scores via Bradley-Terry
- **98% code reuse** with standard GEPA
- **Components**:
  - `ComparisonResult`: A_BETTER, B_BETTER, TIE, INCOMPARABLE
  - `PairwiseComparator`: Protocol for comparing outputs
  - `HybridGEPAState`: Extends GEPAState with comparison caching
  - `bradley_terry_scores()`: Convert comparisons to scalar scores

**Testing without LLMs:**
- `test_pairwise_synthetic.py`: QuadraticProblem, NonConvexProblem, etc.
- `test_pairwise_algorithms.py`: Bradley-Terry vs Elo vs Win Rate benchmarks
- Validates pairwise ≈ scalar on synthetic problems (0.9901 vs 0.9900 scores)

**Future Multi-Objective Extension:**
- Per-objective comparisons: `{accuracy: A_BETTER, latency: B_BETTER}`
- Per-objective Bradley-Terry scoring
- True Pareto dominance checking across objectives
- ~85% code reuse estimated

## Common Development Tasks

### Adding a New Adapter

1. Implement `GEPAAdapter` protocol in `src/gepa/adapters/your_adapter/`
2. Define your `DataInst`, `Trajectory`, `RolloutOutput` types
3. Implement `evaluate()`: run program, return scores + outputs
4. Implement `make_reflective_dataset()`: extract feedback from trajectories
5. Optional: Implement `propose_new_texts` for custom proposal logic
6. Add tests in `tests/test_your_adapter/`
7. Add example usage in `src/gepa/examples/your_adapter/`

**Reference implementations:**
- Simple: `src/gepa/adapters/default_adapter/`
- Complex: `src/gepa/adapters/generic_rag_adapter/`
- External environment: `src/gepa/adapters/terminal_bench_adapter/`

### Working with Pairwise Comparisons

1. Implement `PairwiseComparator` protocol
2. Implement `PairwiseGEPAAdapter` (returns outputs, not scores)
3. Create `HybridGEPAState` from seed candidate
4. Add programs with `add_program_with_comparisons()`
5. Access scores via `state.prog_candidate_val_subscores` (computed lazily)

**Example:** `examples/pairwise_llm_judge.py`

### Running Examples

```bash
# Default adapter example (simple prompt optimization)
uv run python -c "import gepa; gepa.examples.aime.init_dataset()"

# Pairwise comparison example
uv run python examples/pairwise_llm_judge.py

# Terminal-bench integration
pip install terminal-bench
uv run python src/gepa/examples/terminal-bench/train_terminus.py --model_name=gpt-5-mini

# RAG optimization examples
# See src/gepa/examples/rag_adapter/RAG_GUIDE.md
```

## Integration Points

### Main API (`src/gepa/api.py`)

```python
gepa.optimize(
    seed_candidate: dict[str, str],           # Initial program
    trainset: list | DataLoader,              # Training data
    valset: list | DataLoader | None,         # Validation data
    adapter: GEPAAdapter | None,              # Custom adapter
    task_lm: str | Callable | None,           # Task model (if no adapter)
    reflection_lm: str | LanguageModel,       # Reflection model
    max_metric_calls: int | None,             # Budget
    stop_callbacks: StopperProtocol | None,   # Custom stopping
    # ... many other config options
) -> GEPAResult
```

### Adapter Protocol

```python
class GEPAAdapter(Protocol):
    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Run program on batch, return outputs + scores."""

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        """Extract feedback for reflection."""

    propose_new_texts: ProposalFn | None = None  # Optional
```

### Error Handling

- **Never raise for individual example failures** in `evaluate()`
- Return valid `EvaluationBatch` with failure scores (e.g., 0.0)
- Populate trajectories with error messages for reflection
- Only raise for unrecoverable systemic failures
- Engine catches exceptions and logs errors if `raise_on_exception=False`

## Important Conventions

**Scoring:**
- Higher is better
- Minibatch acceptance uses `sum(scores)`
- Validation tracking uses `mean(scores)`
- Calibrate your metrics consistently

**Component Naming:**
- Components are arbitrary strings: "system_prompt", "query_rewriter", etc.
- Candidate is `dict[str, str]` mapping component names to text
- Component selection strategies determine update order

**State Persistence:**
- If `run_dir` provided, state saved/loaded automatically
- Create `gepa.stop` file in `run_dir` to gracefully stop optimization
- Use `use_cloudpickle=True` for dynamically generated DSPy signatures

**Reproducibility:**
- Set `seed` parameter for reproducible runs
- Batch samplers use seeded RNG
- Candidate selection uses seeded RNG

## DSPy Integration

GEPA is directly integrated into DSPy via `dspy.GEPA` API:
- Easiest path for prompt optimization
- Tutorials: https://dspy.ai/tutorials/gepa_ai_program/
- Adapters: `src/gepa/adapters/dspy_adapter/` and `dspy_full_program_adapter/`

## Testing Philosophy

- **Unit tests** for core algorithms (Bradley-Terry, state management)
- **Synthetic tests** for optimization without LLMs (pairwise comparison tests)
- **Integration tests** for adapters with minimal LLM calls
- **Example scripts** serve as integration tests and documentation

## Performance Considerations

**Pairwise Comparison Overhead:**
- O(N²) comparisons for N programs
- For expensive evaluations (LLM, 100ms+), overhead negligible
- Bradley-Terry computation: <1ms for 100 programs
- Comparisons are parallelizable

**Evaluation Policies:**
- `FullEvaluationPolicy`: Evaluate all validation data every iteration
- Incremental policies: Trade accuracy for speed

**Batch Sampling:**
- `EpochShuffledBatchSampler`: Shuffle each epoch, sample minibatches
- Custom samplers: Implement `BatchSampler` protocol

## Key Files to Read

When understanding GEPA's architecture:
1. `src/gepa/core/adapter.py` - Core protocol definitions
2. `src/gepa/api.py` - Main entry point and configuration
3. `src/gepa/core/engine.py` - Optimization loop
4. `src/gepa/adapters/default_adapter/` - Simplest adapter example
5. `README_PAIRWISE.md` - Pairwise comparison design and implementation

## Experimentation and Logging

**Experiment Tracking:**
- Weights & Biases: Set `use_wandb=True`, `wandb_api_key=...`
- MLflow: Set `use_mlflow=True`, `mlflow_tracking_uri=...`
- Both can be used simultaneously
- Custom logging: Implement `LoggerProtocol`

**Progress Monitoring:**
- Set `display_progress_bar=True` for tqdm progress bar
- Set `track_best_outputs=True` to store best outputs per validation instance
- `run_dir` enables state persistence and resumption

## Citation and Paper

Paper: "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"
- arXiv: https://arxiv.org/abs/2507.19457
- Reproduction artifact: https://github.com/gepa-ai/gepa-artifact
