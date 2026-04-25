from __future__ import annotations

import json
from typing import Any

from .builder import PromptBuilder, PromptPacket


class _CriticPromptBuilder(PromptBuilder):
    def gather(self, **kwargs) -> list[PromptPacket]:
        return [
            PromptPacket(self._normalize_text(kwargs.get("question", "")), "question", priority=10.0),
            PromptPacket(self._normalize_text(kwargs.get("stage1_result", "")), "stage1_result", priority=8.0),
            PromptPacket(self._normalize_text(kwargs.get("solver_answer", "")), "solver_answer", priority=9.0),
            PromptPacket(self._normalize_text(kwargs.get("critic_answer", "")), "critic_answer", priority=9.0),
            PromptPacket(self._normalize_text(kwargs.get("memory_context", "")), "memory_context", priority=7.0),
        ]

    def select(self, packets: list[PromptPacket], **kwargs) -> list[PromptPacket]:
        return packets

    def structure(self, packets: list[PromptPacket], **kwargs) -> dict[str, Any]:
        structured = {
            "question": "",
            "stage1_result": "",
            "solver_answer": "",
            "critic_answer": "",
            "memory_context": "",
        }
        for packet in packets:
            structured[packet.packet_type] = packet.content
        return structured

    def compress(self, structured: dict[str, Any], **kwargs) -> dict[str, Any]:
        return structured

    def render(self, compressed: dict[str, Any], **kwargs):
        system_prompt = (
            "You are a careful critic reviewing a solver's answer. "
            "Return plain text only using the required key=value format. "
            "Do not use JSON."
        )
        user_prompt = f"""
Question:
{compressed["question"]}

Stage-1 answer:
{compressed["stage1_result"]}

Relevant memory lessons and cases:
{compressed["memory_context"] or "No relevant memory."}

Solver answer:
{compressed["solver_answer"]}

Your own answer:
{compressed["critic_answer"]}

Instructions:
1. Compare the solver answer against your own answer.
2. If the solver answer is already acceptable, set AGREE=true.
3. If not, provide a concise critique and a better revised answer.
4. Use memory as lessons or error checks, not as direct answer lookup.
5. If a relevant lesson applies, treat it as a constraint against repeating the same mistake.
6. If the solver answer conflicts with a relevant lesson, set AGREE=false unless current evidence clearly overrides that lesson.
7. If memory suggests a likely mistake pattern, explicitly account for it in your critique.
8. Return plain text only in exactly this format:

AGREE=<true or false>
CRITIQUE=<brief critique>
REVISED_ANSWER=<better answer or empty>

Rules:
- Do not use JSON.
- Keep CRITIQUE short.
- REVISED_ANSWER may be empty if you agree.
 - Do not copy old answers from memory unless they are independently supported by the current question.
        """.strip()
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


class _SolverRevisionPromptBuilder(PromptBuilder):
    def gather(self, **kwargs) -> list[PromptPacket]:
        critiques = kwargs.get("critiques", []) or []
        return [
            PromptPacket(self._normalize_text(kwargs.get("question", "")), "question", priority=10.0),
            PromptPacket(self._normalize_text(kwargs.get("stage1_result", "")), "stage1_result", priority=8.0),
            PromptPacket(self._normalize_text(kwargs.get("solver_answer", "")), "solver_answer", priority=9.0),
            PromptPacket(self._normalize_text(kwargs.get("memory_context", "")), "memory_context", priority=7.0),
            PromptPacket(json.dumps(critiques, ensure_ascii=False, indent=2), "critiques", priority=8.0),
        ]

    def select(self, packets: list[PromptPacket], **kwargs) -> list[PromptPacket]:
        return packets

    def structure(self, packets: list[PromptPacket], **kwargs) -> dict[str, Any]:
        structured = {
            "question": "",
            "stage1_result": "",
            "solver_answer": "",
            "memory_context": "",
            "critiques": "",
        }
        for packet in packets:
            structured[packet.packet_type] = packet.content
        return structured

    def compress(self, structured: dict[str, Any], **kwargs) -> dict[str, Any]:
        structured["critiques"] = self._compress_multiline_text(
            structured["critiques"],
            max_lines=20,
            max_chars=1800,
        )
        return structured

    def render(self, compressed: dict[str, Any], **kwargs):
        system_prompt = (
            "You are the final solver. Revise your answer using the critics' feedback. "
            "Return plain text only using the required key=value format. "
            "Do not use JSON."
        )
        user_prompt = f"""
Question:
{compressed["question"]}

Stage-1 answer:
{compressed["stage1_result"]}

Relevant memory lessons and cases:
{compressed["memory_context"] or "No relevant memory."}

Current solver answer:
{compressed["solver_answer"]}

Critiques:
{compressed["critiques"]}

Instructions:
1. Revise the current solver answer using only useful critiques.
2. Ignore critiques that are weak or not actually improvements.
3. Use memory as lessons or error-avoidance rules, not as direct answer lookup.
4. If memory reveals a relevant mistake pattern, make sure your revision addresses that risk explicitly.
5. If a relevant lesson warns against the current answer pattern, revise away from that pattern unless the current evidence clearly supports it.
6. Return plain text only in exactly this format:

REASONING=<brief revision reasoning only>
FINAL_ANSWER=<your final answer>

Rules:
- Do not use JSON.
- Keep REASONING short.
- FINAL_ANSWER must contain only the final answer.
- FINAL_ANSWER should be the last non-empty line if possible.
 - Do not copy a past answer from memory unless it is independently justified for the current question.
        """.strip()
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


class DecisionPromptBuilder:
    """Decision prompt 門面，對外提供 critic 與 solver revision 兩種 prompt。"""

    def __init__(self):
        self.critic_prompt_builder = _CriticPromptBuilder()
        self.solver_revision_prompt_builder = _SolverRevisionPromptBuilder()

    def build_critic_messages(
        self,
        question: str,
        stage1_result: str | None,
        solver_answer: str,
        critic_answer: str,
        memory_context: str = "",
    ) -> list[dict[str, str]]:
        return self.critic_prompt_builder.build(
            question=question,
            stage1_result=stage1_result or "",
            solver_answer=solver_answer,
            critic_answer=critic_answer,
            memory_context=memory_context,
        )

    def build_solver_revision_messages(
        self,
        question: str,
        stage1_result: str | None,
        solver_answer: str,
        critiques: list[dict[str, Any]],
        memory_context: str = "",
    ) -> list[dict[str, str]]:
        return self.solver_revision_prompt_builder.build(
            question=question,
            stage1_result=stage1_result or "",
            solver_answer=solver_answer,
            critiques=critiques,
            memory_context=memory_context,
        )
