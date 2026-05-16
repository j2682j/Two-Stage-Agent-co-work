from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptContract:
    benchmark: str = "generic"
    task_context: Any | None = None

    def stage1_output_contract(self, expected_weight_count: int, has_formers: bool) -> str:
        weights = "[]" if not has_formers else f"[w1, w2, ..., w{expected_weight_count}]"
        former_rule = (
            "- WEIGHTS must be [] because there are no previous agents."
            if not has_formers
            else (
                f"- WEIGHTS must contain exactly {expected_weight_count} integers.\n"
                "- Each weight corresponds to the previous agents in the same order shown above.\n"
                "- Each weight must be an integer between 1 and 5."
            )
        )
        return f"""
Task:
1. Solve the question carefully.
2. Check whether your first answer could be wrong.
3. Keep the reasoning short and focused on the key checks.
4. Make sure the final answer uses exactly the unit requested in the question.
5. If relevant reflection rules apply, use them as compact error checks rather than as answer lookup.

Return plain text only in exactly this format:
REASONING=<brief key steps and self-checks only>
FINAL_ANSWER=<your final answer>
WEIGHTS={weights}

Rules:
- REASONING must be short and only include the essential checks.
- FINAL_ANSWER must contain only your final answer.
{former_rule}
- The WEIGHTS line must be the final line of your reply.
- Do not include markdown fences or any extra text outside the required format.
        """.strip()

    def stage2_output_contract(self) -> str:
        return """
Task:
1. Solve the question again using the tool evidence if it helps.
2. Use the most relevant tool evidence instead of repeating everything.
3. Before giving the final answer, verify that the answer unit matches the unit requested in the question.
4. If needed, convert the result before giving the final answer.

Return JSON only with:
{
  "reasoning": "your reasoning",
  "final_answer": "your final answer"
}
        """.strip()

    def repair_contract(self, expected_weight_count: int) -> str:
        return f"""
Your previous reply did not follow the required output format.

Return plain text only.
Do not include markdown fences.
Do not include any extra text outside the required format.

Required format:
REASONING=<brief key steps and correction checks only>
FINAL_ANSWER=<your final answer as a string>
WEIGHTS=[w1, w2, ..., w{expected_weight_count}]

Requirements:
- REASONING must be a short string with only the essential correction/checking steps.
- FINAL_ANSWER must be a string, even if the answer is numeric.
- WEIGHTS must contain exactly {expected_weight_count} integers.
- If there are no previous agents, WEIGHTS must be [].
- If previous agents made mistakes, correct them briefly in REASONING before giving FINAL_ANSWER.
- The WEIGHTS line must be the final line of your reply.
        """.strip()

    def critic_system_prompt(self) -> str:
        return (
            "You are a careful critic reviewing a solver's answer. "
            "Return plain text only using the required key=value format. "
            "Do not use JSON."
        )

    def critic_output_contract(self) -> str:
        return """
Rules:
- Do not use JSON.
- Keep CRITIQUE short.
- REVISED_ANSWER may be empty if you agree.
- Do not copy old answers from memory unless they are independently supported by the current question.
        """.strip()

    def solver_revision_system_prompt(self) -> str:
        return (
            "You are the final solver. Revise your answer using the critics' feedback. "
            "Return plain text only using the required key=value format. "
            "Do not use JSON."
        )

    def solver_revision_output_instruction(self) -> str:
        return "Return plain text only in exactly this format:"

    def solver_revision_output_contract(self) -> str:
        return """
Rules:
- Do not use JSON.
- Keep REASONING short.
- FINAL_ANSWER must contain only the final answer.
- FINAL_ANSWER should be the last non-empty line if possible.
- Do not copy a past answer from memory unless it is independently justified for the current question.
        """.strip()
