from __future__ import annotations

from typing import Any

from builder import DecisionTraceBuilder
from parser import DecisionParser
from network.slm_agent import SLM_4b_Agent
from prompt.decision_prompt_builder import DecisionPromptBuilder
from utils.network_utils import answer_equivalence

from .base_decision_maker import BaseDecisionMaker


class VerticalSolverFirstDecisionMaker(BaseDecisionMaker):
    name = "vertical_solver_first"

    def __init__(
        self,
        fallback_model_name: str = "gpt-oss:20b",
        max_inner_turns: int = 1,
    ):
        super().__init__(max_inner_turns=max_inner_turns)
        self.fallback_model_name = fallback_model_name
        self.decision_parser = DecisionParser()
        self.message_builder = DecisionPromptBuilder()
        self.trace_builder = DecisionTraceBuilder()

    def decide(
        self,
        question: str,
        stage1_result: str | None,
        top_k_outputs: list[dict[str, Any]],
        top_k_indices: list[int],
        importance_scores: list[float] | None = None,
        memory_context: str = "",
    ) -> dict[str, Any]:
        successful = self._successful_outputs(top_k_outputs)
        if not successful:
            return self._build_result(
                mode=self.name,
                success=False,
                error="No successful top-k outputs provided.",
            )

        judged_outputs, judge_step, judge_prompt_tokens, judge_completion_tokens = self._judge_stage2_outputs(
            question=question,
            stage1_result=stage1_result,
            successful_outputs=successful,
            memory_context=memory_context,
        )
        solver_output = self._select_solver_output(judged_outputs, importance_scores)
        if solver_output is None:
            return self._build_result(
                mode=self.name,
                success=False,
                error="Failed to select a solver candidate.",
            )

        current_answer = str(solver_output.get("answer", "")).strip()
        current_reply = solver_output.get("reply")
        solver_agent_idx = solver_output.get("agent_idx")
        solver_model_name = self._resolve_model_name(solver_output)

        critiques: list[dict[str, Any]] = []
        intermediate_steps: list[dict[str, Any]] = [judge_step]
        prompt_tokens = judge_prompt_tokens
        completion_tokens = judge_completion_tokens

        critics = [
            item for item in judged_outputs
            if item.get("agent_idx") != solver_agent_idx
        ]
        critics_by_idx = {
            item.get("agent_idx"): item
            for item in critics
            if item.get("agent_idx") is not None
        }

        if not critics:
            return self._build_result(
                mode=self.name,
                success=True,
                final_answer=current_answer,
                final_reply=current_reply,
                selected_agent_idx=solver_agent_idx,
                selected_indices=[idx for idx in top_k_indices if idx is not None],
                critiques=[],
                intermediate_steps=[],
                prompt_tokens=0,
                completion_tokens=0,
                error=None,
            )

        for round_idx in range(self.max_inner_turns):
            round_critiques, round_prompt_tokens, round_completion_tokens = self._collect_critiques(
                question=question,
                stage1_result=stage1_result,
                solver_answer=current_answer,
                critics=critics,
                memory_context=memory_context,
            )
            prompt_tokens += round_prompt_tokens
            completion_tokens += round_completion_tokens
            critiques.extend(round_critiques)
            intermediate_steps.append(
                self.trace_builder.build_critic_round_step(
                    round_idx=round_idx,
                    solver_agent_idx=solver_agent_idx,
                    critiques=round_critiques,
                )
            )

            actionable_critiques = [item for item in round_critiques if not item.get("agree")]
            actionable_critiques = self._filter_actionable_critiques(
                solver_output=solver_output,
                critiques=actionable_critiques,
                critics_by_idx=critics_by_idx,
            )
            if not actionable_critiques:
                break

            revised_reply, revised_answer, revision_prompt_tokens, revision_completion_tokens = self._revise_with_solver(
                question=question,
                stage1_result=stage1_result,
                solver_answer=current_answer,
                critiques=actionable_critiques,
                solver_model_name=solver_model_name,
                memory_context=memory_context,
            )
            prompt_tokens += revision_prompt_tokens
            completion_tokens += revision_completion_tokens
            intermediate_steps.append(
                self.trace_builder.build_solver_revision_step(
                    round_idx=round_idx,
                    solver_agent_idx=solver_agent_idx,
                    revised_reply=revised_reply,
                    revised_answer=revised_answer,
                )
            )

            if revised_answer and revised_answer.strip():
                current_reply = revised_reply
                current_answer = revised_answer.strip()

        return self._build_result(
            mode=self.name,
            success=True,
            final_answer=current_answer,
            final_reply=current_reply,
            selected_agent_idx=solver_agent_idx,
            selected_indices=[idx for idx in top_k_indices if idx is not None],
            critiques=critiques,
            intermediate_steps=intermediate_steps,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=None,
        )

    def _select_solver_output(
        self,
        successful_outputs: list[dict[str, Any]],
        importance_scores: list[float] | None,
    ) -> dict[str, Any] | None:
        if not successful_outputs:
            return None

        def importance_score(item: dict[str, Any]) -> float:
            agent_idx = item.get("agent_idx")
            if isinstance(agent_idx, int) and 0 <= agent_idx < len(importance_scores):
                return importance_scores[agent_idx]
            return float("-inf")

        def score(item: dict[str, Any]) -> tuple[float, float, float]:
            judge_score = float(item.get("stage2_judge_score", float("-inf")))
            acceptable_bonus = 1.0 if item.get("stage2_judge_is_acceptable") else 0.0
            importance = importance_score(item) if importance_scores else 0.0
            return acceptable_bonus, judge_score, importance

        return sorted(successful_outputs, key=score, reverse=True)[0]

    def _judge_stage2_outputs(
        self,
        question: str,
        stage1_result: str | None,
        successful_outputs: list[dict[str, Any]],
        memory_context: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, Any], int, int]:
        judged_outputs: list[dict[str, Any]] = []
        prompt_tokens = 0
        completion_tokens = 0

        for output in successful_outputs:
            judged = dict(output)
            evaluation, used_prompt_tokens, used_completion_tokens = self._evaluate_stage2_candidate(
                question=question,
                stage1_result=stage1_result,
                candidate_answer=str(output.get("answer", "")).strip(),
                candidate_reply=str(output.get("reply") or "").strip(),
                memory_context=memory_context,
            )
            prompt_tokens += used_prompt_tokens
            completion_tokens += used_completion_tokens
            judged["stage2_judge_is_acceptable"] = evaluation.get("is_acceptable", False)
            judged["stage2_judge_score"] = evaluation.get("score", 0.0)
            judged["stage2_judge_revised_answer"] = evaluation.get("revised_answer", "")
            judged["stage2_judge_reasoning"] = evaluation.get("judge_reasoning", "")
            judged_outputs.append(judged)

        judge_step = {
            "step": "stage2_judge_rerank",
            "candidates": [
                {
                    "agent_idx": item.get("agent_idx"),
                    "answer": item.get("answer"),
                    "judge_score": item.get("stage2_judge_score", 0.0),
                    "is_acceptable": item.get("stage2_judge_is_acceptable", False),
                    "revised_answer": item.get("stage2_judge_revised_answer", ""),
                }
                for item in judged_outputs
            ],
        }
        return judged_outputs, judge_step, prompt_tokens, completion_tokens

    def _filter_actionable_critiques(
        self,
        solver_output: dict[str, Any],
        critiques: list[dict[str, Any]],
        critics_by_idx: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not critiques:
            return []

        solver_score = float(solver_output.get("stage2_judge_score", 0.0) or 0.0)
        solver_ok = bool(solver_output.get("stage2_judge_is_acceptable", False))
        strong_solver = solver_ok and solver_score >= 8.0
        if not strong_solver:
            return critiques

        trusted_critiques: list[dict[str, Any]] = []
        for critique in critiques:
            critic_idx = critique.get("critic_agent_idx")
            critic_meta = critics_by_idx.get(critic_idx, {})
            critic_ok = bool(critic_meta.get("stage2_judge_is_acceptable", False))
            critic_score = float(critic_meta.get("stage2_judge_score", 0.0) or 0.0)

            # Protect a high-confidence solver from being revised by a single
            # low-confidence or judge-rejected critic.
            if critic_ok and critic_score >= solver_score:
                trusted_critiques.append(critique)

        return trusted_critiques

    def _evaluate_stage2_candidate(
        self,
        question: str,
        stage1_result: str | None,
        candidate_answer: str,
        candidate_reply: str,
        memory_context: str = "",
    ) -> tuple[dict[str, Any], int, int]:
        if not candidate_answer:
            return {
                "is_acceptable": False,
                "score": 0.0,
                "revised_answer": "",
                "judge_reasoning": "Empty candidate answer.",
            }, 0, 0

        system_prompt = (
            "You are a strict JSON-only judge for final candidate answers. "
            "Assess whether the candidate answer is likely correct for the question. "
            "Be conservative and prioritize answer correctness over writing style."
        )
        user_prompt = f"""
Question:
{question}

Stage-1 result:
{stage1_result or ""}

Relevant memory:
{memory_context or "No relevant memory."}

Candidate final answer:
{candidate_answer}

Candidate reply:
{candidate_reply}

Return JSON only with this exact schema:
{{
  "is_acceptable": true,
  "score": 0,
  "revised_answer": "string",
  "judge_reasoning": "string"
}}

Scoring guide:
- 0 to 3: very likely wrong
- 4 to 6: partially plausible but still doubtful
- 7 to 8: likely correct
- 9 to 10: strongly supported and very likely correct

Rules:
- revised_answer may be empty.
- If the answer is likely wrong but clearly close to a better answer, put the better answer in revised_answer.
- If a relevant memory lesson identifies the candidate as a likely repeated mistake, lower the score unless current evidence clearly overrules that lesson.
- Do not invent long explanations.
        """.strip()

        try:
            judge_agent = SLM_4b_Agent(model_name=self.fallback_model_name)
            raw, prompt_tokens, completion_tokens = self._invoke_with_usage(
                judge_agent,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            parsed = self.decision_parser.parse_json(raw)
            if isinstance(parsed, dict):
                return {
                    "is_acceptable": bool(parsed.get("is_acceptable", False)),
                    "score": self._coerce_judge_score(parsed.get("score")),
                    "revised_answer": str(parsed.get("revised_answer", "")).strip(),
                    "judge_reasoning": str(parsed.get("judge_reasoning", "")).strip(),
                }, prompt_tokens, completion_tokens
        except Exception:
            pass

        fallback_score = 8.0 if candidate_answer == str(stage1_result or "").strip() else 5.0
        return {
            "is_acceptable": bool(candidate_answer),
            "score": fallback_score,
            "revised_answer": "",
            "judge_reasoning": "Fallback stage2 judge heuristic was used.",
        }, 0, 0

    def _coerce_judge_score(self, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(10.0, score))

    def _resolve_model_name(self, output: dict[str, Any]) -> str:
        model_name = str(output.get("model_name", "")).strip()
        if model_name:
            return model_name
        return self.fallback_model_name

    def _collect_critiques(
        self,
        question: str,
        stage1_result: str | None,
        solver_answer: str,
        critics: list[dict[str, Any]],
        memory_context: str = "",
    ) -> tuple[list[dict[str, Any]], int, int]:
        critiques: list[dict[str, Any]] = []
        prompt_tokens = 0
        completion_tokens = 0

        for critic in critics:
            critic_answer = str(critic.get("answer", "")).strip()
            critic_idx = critic.get("agent_idx")
            critic_model_name = self._resolve_model_name(critic)
            if not critic_answer:
                continue

            if answer_equivalence(solver_answer, critic_answer):
                critiques.append(
                    {
                        "critic_agent_idx": critic_idx,
                        "agree": True,
                        "critique": "",
                        "revised_answer": critic_answer,
                    }
                )
                continue

            messages = self.message_builder.build_critic_messages(
                question=question,
                stage1_result=stage1_result,
                solver_answer=solver_answer,
                critic_answer=critic_answer,
                memory_context=memory_context,
            )

            try:
                critic_agent = SLM_4b_Agent(model_name=critic_model_name)
                raw, used_prompt_tokens, used_completion_tokens = self._invoke_with_usage(
                    critic_agent,
                    messages,
                )
                prompt_tokens += used_prompt_tokens
                completion_tokens += used_completion_tokens
                critiques.append(
                    self.decision_parser.parse_critique(
                        raw_reply=raw,
                        critic_agent_idx=critic_idx,
                        fallback_answer=critic_answer,
                    )
                )
            except Exception as e:
                critiques.append(
                    self.trace_builder.build_critic_fallback(
                        critic_agent_idx=critic_idx,
                        critique=f"Critique generation failed: {e}",
                        revised_answer=critic_answer,
                    )
                )

        return critiques, prompt_tokens, completion_tokens

    def _revise_with_solver(
        self,
        question: str,
        stage1_result: str | None,
        solver_answer: str,
        critiques: list[dict[str, Any]],
        solver_model_name: str,
        memory_context: str = "",
    ) -> tuple[str | None, str, int, int]:
        solver_agent = SLM_4b_Agent(model_name=solver_model_name)
        messages = self.message_builder.build_solver_revision_messages(
            question=question,
            stage1_result=stage1_result,
            solver_answer=solver_answer,
            critiques=critiques,
            memory_context=memory_context,
        )
        raw, prompt_tokens, completion_tokens = self._invoke_with_usage(
            solver_agent,
            messages,
        )
        try:
            parsed = self.decision_parser.parse_solver_revision(raw)
            return raw, parsed["final_answer"], prompt_tokens, completion_tokens
        except Exception:
            # 保守 fallback：revision 無法解析時，不讓整個 decision maker 崩掉，
            # 直接保留目前 solver answer。
            return raw, solver_answer, prompt_tokens, completion_tokens

    def _invoke_with_usage(
        self,
        agent: SLM_4b_Agent,
        messages: list[dict[str, str]],
    ) -> tuple[str, int, int]:
        if hasattr(agent, "invoke_with_usage"):
            return agent.invoke_with_usage(messages)
        return agent.invoke(messages), 0, 0
