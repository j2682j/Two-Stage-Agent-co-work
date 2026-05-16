from __future__ import annotations

import json

from .base import PromptContract


_BFCL_LIST_EXAMPLE = '[{"name":"function_name","arguments":{"param":"value"}}]'
_BFCL_LIST_EXAMPLE_AS_JSON_STRING = json.dumps(_BFCL_LIST_EXAMPLE)


class BFCLPromptContract(PromptContract):
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
1. Extract the required BFCL function call or calls from the current question.
2. Use only functions and parameter names available in the BFCL function schema.
3. Keep reasoning short; do not put explanations inside FINAL_ANSWER.

Return plain text only in exactly this format:
REASONING=<brief function-call selection checks only>
FINAL_ANSWER=<compact BFCL JSON list>
WEIGHTS={weights}

BFCL output rules:
- FINAL_ANSWER must be one compact JSON list only, for example {_BFCL_LIST_EXAMPLE}.
- Use double quotes, not single quotes.
- Use exact function names from Available functions.
- Use exact parameter names from the function schema.
- Do not wrap calls in {{"function_call": ...}}, {{"final_answer": ...}}, or any other object.
- If no function should be called, FINAL_ANSWER=[].
{former_rule}
- The WEIGHTS line must be the final line of your reply.
- Do not include markdown fences or any extra text outside the required format.
        """.strip()

    def stage2_output_contract(self) -> str:
        return f"""
Task:
1. Produce the final BFCL function-call answer using the current question, Stage-1 answer, and tool evidence only if it helps.
2. Preserve exact function names and parameter names from the BFCL function schema.
3. Do not answer in prose.

Return JSON only with this exact outer schema:
{{
  "reasoning": "brief function-call checks",
  "final_answer": {_BFCL_LIST_EXAMPLE_AS_JSON_STRING}
}}

BFCL rules:
- final_answer must be a string containing one compact BFCL JSON list.
- Use double quotes inside the BFCL list.
- Do not wrap calls in function_call, tool_calls, final_answer, or any other object inside final_answer.
- If no function should be called, final_answer must be "[]".
        """.strip()

    def repair_contract(self, expected_weight_count: int) -> str:
        return f"""
Your previous reply did not follow the required BFCL Stage-1 format.

Return plain text only.
Do not include markdown fences.
Do not include any extra text outside the required format.

Required format:
REASONING=<brief function-call correction checks only>
FINAL_ANSWER=<compact BFCL JSON list>
WEIGHTS=[w1, w2, ..., w{expected_weight_count}]

BFCL requirements:
- FINAL_ANSWER must be one compact JSON list only, for example {_BFCL_LIST_EXAMPLE}.
- Use double quotes, exact function names, and exact parameter names.
- Do not wrap calls in {{"function_call": ...}}, {{"final_answer": ...}}, or any other object.
- If no function should be called, FINAL_ANSWER=[].
- WEIGHTS must contain exactly {expected_weight_count} integers; if there are no previous agents, WEIGHTS must be [].
- The WEIGHTS line must be the final line of your reply.
        """.strip()

    def critic_system_prompt(self) -> str:
        return (
            "You are a careful critic reviewing a BFCL function-calling answer. "
            "Return plain text only using the required key=value format. "
            "When REVISED_ANSWER is not empty, it must be a single-line BFCL JSON list."
        )

    def critic_output_contract(self) -> str:
        return f"""
BFCL rules:
- REVISED_ANSWER may be empty if you agree.
- If you revise, REVISED_ANSWER must be one compact JSON list only, for example {_BFCL_LIST_EXAMPLE}.
- Do not wrap calls in {{"function_call": ...}}, {{"final_answer": ...}}, or any other object.
- Use double quotes and the exact function name from Available functions.
- Keep CRITIQUE short.
- Do not copy old answers from memory unless they are independently supported by the current question.
        """.strip()

    def solver_revision_system_prompt(self) -> str:
        return (
            "You are the final solver for a BFCL function-calling answer. "
            "Return plain text only using the required key=value format. "
            "FINAL_ANSWER must be a single-line BFCL JSON list."
        )

    def solver_revision_output_instruction(self) -> str:
        return "Return plain text only in exactly this format; FINAL_ANSWER must be compact JSON:"

    def solver_revision_output_contract(self) -> str:
        return f"""
BFCL rules:
- FINAL_ANSWER must be one compact JSON list only, for example {_BFCL_LIST_EXAMPLE}.
- Do not wrap calls in {{"function_call": ...}}, {{"final_answer": ...}}, or any other object.
- Use double quotes and the exact function name from Available functions.
- If no function should be called, FINAL_ANSWER=[].
- Keep REASONING short.
- FINAL_ANSWER should be the last non-empty line if possible.
- Do not copy a past answer from memory unless it is independently justified for the current question.
        """.strip()
