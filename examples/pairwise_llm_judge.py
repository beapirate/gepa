"""
Example: Using LLM-as-Judge for pairwise comparison in GEPA.

This example shows how to optimize prompts using pairwise comparisons
instead of scalar metrics. An LLM judges which of two answers is better.

Key components:
1. LLMJudgeComparator: Compares two outputs using an LLM
2. QAPairwiseAdapter: Adapter for question-answering with pairwise comparisons
3. Integration with GEPA using HybridGEPAState

Benefits over scalar metrics:
- No need to design numeric scoring rubric
- More natural for subjective quality assessment
- Handles multi-dimensional quality (accuracy, clarity, conciseness)
- LLMs often better at comparison than absolute scoring
"""

from dataclasses import dataclass
from typing import Any

from gepa.pairwise import ComparisonResult, PairwiseComparator
from gepa.pairwise.adapter import PairwiseEvaluationBatch, PairwiseGEPAAdapter


# ============================================================================
# LLM-as-Judge Comparator
# ============================================================================

class LLMJudgeComparator(PairwiseComparator):
    """
    Compares two outputs using an LLM as judge.

    The judge LLM is asked which of two answers is better, returning
    a comparison result without assigning numeric scores.
    """

    def __init__(
        self,
        judge_client,  # LLM client for judging
        prompt_template: str | None = None,
        system_message: str | None = None,
    ):
        """
        Initialize LLM judge comparator.

        Args:
            judge_client: LLM client with .complete() or .chat() method
            prompt_template: Template for comparison prompt
            system_message: Optional system message for judge
        """
        self.judge_client = judge_client
        self.prompt_template = prompt_template or self._default_prompt_template()
        self.system_message = system_message or self._default_system_message()

    def _default_system_message(self) -> str:
        return (
            "You are an expert judge evaluating the quality of answers to questions. "
            "Your task is to compare two answers and determine which is better. "
            "Consider accuracy, clarity, completeness, and helpfulness."
        )

    def _default_prompt_template(self) -> str:
        return """
Compare these two answers to the question:

Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Which answer is better? Respond with exactly one of:
- "A" if Answer A is better
- "B" if Answer B is better
- "Tie" if they are equally good

Your response:"""

    def compare(
        self,
        output_a: str,
        output_b: str,
        data_instance: dict,
    ) -> ComparisonResult:
        """
        Compare two outputs using LLM judge.

        Args:
            output_a: First answer
            output_b: Second answer
            data_instance: Question data with "question" field

        Returns:
            ComparisonResult indicating which is better
        """
        # Format prompt
        prompt = self.prompt_template.format(
            question=data_instance.get("question", ""),
            answer_a=output_a,
            answer_b=output_b,
        )

        # Get judge response
        try:
            if hasattr(self.judge_client, "chat"):
                # Chat-based API
                response = self.judge_client.chat(
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ]
                )
            else:
                # Completion-based API
                full_prompt = f"{self.system_message}\n\n{prompt}"
                response = self.judge_client.complete(full_prompt)

            # Parse response
            response_lower = response.lower().strip()

            if "a" in response_lower and "b" not in response_lower:
                return ComparisonResult.A_BETTER
            elif "b" in response_lower and "a" not in response_lower:
                return ComparisonResult.B_BETTER
            elif "tie" in response_lower:
                return ComparisonResult.TIE
            else:
                # Unclear response, default to tie
                return ComparisonResult.TIE

        except Exception as e:
            print(f"Warning: Judge comparison failed: {e}")
            return ComparisonResult.TIE


# ============================================================================
# Q&A Pairwise Adapter
# ============================================================================

class QAPairwiseAdapter(PairwiseGEPAAdapter):
    """
    Adapter for question-answering using pairwise comparisons.

    Uses LLM to generate answers and LLM-as-judge to compare them.
    No scalar metrics needed!
    """

    def __init__(
        self,
        llm_client,  # LLM for generating answers
        judge_client,  # LLM for judging answers
        judge_prompt_template: str | None = None,
    ):
        """
        Initialize Q&A pairwise adapter.

        Args:
            llm_client: LLM for generating answers
            judge_client: LLM for judging quality
            judge_prompt_template: Optional custom judge prompt
        """
        self.llm_client = llm_client
        self.judge_comparator = LLMJudgeComparator(
            judge_client,
            prompt_template=judge_prompt_template,
        )

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> PairwiseEvaluationBatch:
        """
        Generate answers for batch of questions.

        Args:
            batch: List of {"question": str} dicts
            candidate: {"instruction": str} program
            capture_traces: Whether to capture execution traces

        Returns:
            PairwiseEvaluationBatch with answers
        """
        instruction = candidate.get("instruction", "")

        outputs = []
        trajectories = [] if capture_traces else None

        for example in batch:
            question = example.get("question", "")

            # Generate answer
            prompt = f"{instruction}\n\nQuestion: {question}\nAnswer:"

            try:
                if hasattr(self.llm_client, "chat"):
                    answer = self.llm_client.chat(
                        messages=[{"role": "user", "content": prompt}]
                    )
                else:
                    answer = self.llm_client.complete(prompt)

                outputs.append(answer)

                if capture_traces:
                    trajectories.append({
                        "instruction": instruction,
                        "question": question,
                        "answer": answer,
                    })

            except Exception as e:
                # Handle generation failures
                error_msg = f"[Error: {str(e)}]"
                outputs.append(error_msg)

                if capture_traces:
                    trajectories.append({
                        "instruction": instruction,
                        "question": question,
                        "answer": error_msg,
                        "error": str(e),
                    })

        return PairwiseEvaluationBatch(
            outputs=outputs,
            trajectories=trajectories,
        )

    def comparator(self) -> PairwiseComparator:
        """Return LLM judge comparator."""
        return self.judge_comparator

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: PairwiseEvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Build reflective dataset for instruction refinement.

        Since we don't have scalar scores, we focus on the outputs and
        trajectories to build feedback.
        """
        dataset = []

        for i, (output, traj) in enumerate(
            zip(eval_batch.outputs, eval_batch.trajectories or [])
        ):
            # Simple feedback based on output characteristics
            feedback = self._generate_feedback(output, traj)

            dataset.append({
                "Inputs": {"question": traj.get("question", "")},
                "Generated Output": output,
                "Feedback": feedback,
            })

        return {"instruction": dataset}

    def _generate_feedback(self, output: str, trajectory: dict) -> str:
        """Generate feedback for an output (can be enhanced with comparisons)."""
        feedback_parts = []

        # Basic checks
        if "[Error" in output:
            feedback_parts.append("Generation failed with error")
        elif len(output) < 10:
            feedback_parts.append("Answer is very short, may lack detail")
        elif len(output) > 500:
            feedback_parts.append("Answer is very long, consider being more concise")
        else:
            feedback_parts.append("Answer generated successfully")

        return ". ".join(feedback_parts) if feedback_parts else "No specific feedback"


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """
    Example of using pairwise comparison GEPA with LLM-as-judge.

    Note: This requires actual LLM clients. For testing, use mock clients
    or the synthetic adapters from tests/.
    """
    # Pseudocode - replace with actual LLM clients
    class MockLLM:
        def complete(self, prompt):
            return "Mock answer to the question"

        def chat(self, messages):
            return "Mock answer to the question"

    llm_client = MockLLM()
    judge_client = MockLLM()

    # Create adapter
    adapter = QAPairwiseAdapter(
        llm_client=llm_client,
        judge_client=judge_client,
    )

    # Create dataset
    trainset = [
        {"question": "What is Python?"},
        {"question": "How do you write a for loop?"},
        {"question": "What is machine learning?"},
    ]

    valset = [
        {"question": "What is a function in Python?"},
        {"question": "Explain object-oriented programming."},
    ]

    # Run GEPA with pairwise comparisons
    # (This requires integration with GEPA engine - see documentation)

    print("Pairwise comparison GEPA with LLM-as-judge example")
    print("Replace MockLLM with actual LLM clients to run")
    print(f"Adapter configured with {len(trainset)} training examples")
    print(f"Adapter configured with {len(valset)} validation examples")


if __name__ == "__main__":
    example_usage()
