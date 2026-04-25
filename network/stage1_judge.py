from __future__ import annotations

import re
from typing import Any

from parser import try_parse_json

from .slm_agent import SLM_4b_Agent


class Stage1Judge:
    def __init__(self, judge_model_name: str = "gpt-oss:20b"):
        self.judge_model_name = judge_model_name

    def evaluate_stage1_candidate(
        self,
        question: str,
        reasoning: str,
        final_answer: str,
    ) -> dict[str, Any]:
        question_text = self._normalize_text(question)
        reasoning_text = self._normalize_text(reasoning)
        answer_text = self._normalize_text(final_answer)

        if not reasoning_text:
            return {
                "is_acceptable": False,
                "score": 0.0,
                "approved_answer": "",
                "suggested_fix": "No stage-1 reasoning was produced.",
                "revised_answer": "",
                "judge_reasoning": "The candidate did not produce usable stage-1 reasoning.",
                "raw_response": None,
                "used_fallback": True,
            }

        prompt = f"""
            You are reviewing a stage-1 reasoning trace in a multi-agent reasoning system.

            Your job:
            1. Check whether the candidate's reasoning steps are logically correct and appropriate for the question.
            2. Check whether the candidate's final answer is correct.
            3. Check whether the unit of the final answer matches the unit requested in the question.
            4. If both the reasoning and the final answer are correct, give the highest score.
            5. If the reasoning is good enough, approve its reasoning direction.
            6. If the reasoning or answer is not good enough, provide a concise correction and a better revised direction.
            7. Return valid JSON only.

            Important output rule:
            - approved_answer and revised_answer may be empty in stage 1, but if the candidate already has a correct final answer, keep it.
            - Prefer candidates whose reasoning is correct, whose final answer is correct, and whose final answer uses the unit requested in the question.
            - If the candidate has a correct reasoning path but the final answer uses the wrong unit, do not give the highest score.
            - If the candidate's numeric value looks plausible but the unit does not match the question, treat it as incorrect for scoring purposes.
            - If the question asks for a converted unit (for example thousand-hours instead of hours), an answer left in the pre-conversion unit must receive a clearly lower score than the correctly converted answer.
            - If the candidate has the correct final answer but weak or incomplete reasoning, do not give the highest score.
            - Only give the highest score when the reasoning is correct, the answer is correct, and the answer unit matches the question.

            Scoring guide:
            - 0 to 3: reasoning or answer is mostly wrong, or the final answer uses the wrong unit in a major way
            - 4 to 6: partially useful but has important flaws in reasoning, answer correctness, or unit matching
            - 7 to 8: mostly correct reasoning and answer, but still has minor issues or incomplete justification
            - 9 to 10: reasoning is correct, final answer is correct, and the final answer uses the correct unit requested by the question

            Specific judging rule:
            - When comparing a fully correct converted answer against an unconverted raw-unit answer, always score the converted answer higher.
            - Do not mark a raw-unit answer as acceptable if the question explicitly asks for a different final unit.

            Return this schema exactly:
            {{
            "is_acceptable": true,
            "score": 0,
            "approved_answer": "string",
            "suggested_fix": "string",
            "revised_answer": "string",
            "judge_reasoning": "string"
            }}

            Question:
            {question_text}

            Candidate reasoning:
            {reasoning_text}
        """.strip()

        if answer_text:
            prompt += f"\n\nCandidate current answer (may be empty in stage 1):\n{answer_text}"

        try:
            judge_agent = SLM_4b_Agent(model_name=self.judge_model_name)
            raw_response = judge_agent.invoke(
                [
                    {"role": "system", "content": "You are a strict JSON-only stage-1 reasoning judge."},
                    {"role": "user", "content": prompt},
                ]
            )
            parsed = try_parse_json(raw_response)
            if isinstance(parsed, dict):
                score = self._coerce_score(parsed.get("score"))
                is_acceptable = bool(parsed.get("is_acceptable"))
                approved_answer = self._normalize_text(
                    parsed.get("approved_answer") or (answer_text if is_acceptable else "")
                )
                suggested_fix = self._normalize_text(parsed.get("suggested_fix"))
                revised_answer = self._normalize_text(parsed.get("revised_answer"))
                judge_reasoning = self._normalize_text(parsed.get("judge_reasoning"))

                return {
                    "is_acceptable": is_acceptable,
                    "score": score,
                    "approved_answer": approved_answer,
                    "suggested_fix": suggested_fix,
                    "revised_answer": revised_answer,
                    "judge_reasoning": judge_reasoning,
                    "raw_response": raw_response,
                    "used_fallback": False,
                }
        except Exception as e:
            return {
                "is_acceptable": False,
                "score": self._heuristic_stage1_score(question_text, reasoning_text, answer_text),
                "approved_answer": "",
                "suggested_fix": f"Judge model unavailable: {type(e).__name__}: {e}",
                "revised_answer": "",
                "judge_reasoning": "",
                "raw_response": None,
                "used_fallback": True,
            }

        return {
            "is_acceptable": False,
            "score": self._heuristic_stage1_score(question_text, reasoning_text, answer_text),
            "approved_answer": "",
            "suggested_fix": "Judge response could not be parsed into the required schema.",
            "revised_answer": "",
            "judge_reasoning": "",
            "raw_response": None,
            "used_fallback": True,
        }

    def stage1_importance_score(self, question: str, reasoning: str, final_answer: str) -> float:
        evaluation = self.evaluate_stage1_candidate(question, reasoning, final_answer)
        return self.adjust_stage1_importance(evaluation)

    def adjust_stage1_importance(self, evaluation: dict[str, Any]) -> float:
        raw_score = self._coerce_score(evaluation.get("score", 0.0))
        is_acceptable = bool(evaluation.get("is_acceptable", False))
        approved_answer = self._normalize_text(evaluation.get("approved_answer"))
        revised_answer = self._normalize_text(evaluation.get("revised_answer"))

        adjusted = raw_score
        if not is_acceptable:
            if raw_score <= 2.0:
                adjusted *= 0.2
            elif raw_score <= 4.0:
                adjusted *= 0.45
            elif raw_score <= 6.0:
                adjusted *= 0.75
            else:
                adjusted *= 0.9
        else:
            if raw_score >= 8.0:
                adjusted *= 1.05
            elif raw_score >= 6.0:
                adjusted *= 1.0
            else:
                adjusted *= 0.95

        if approved_answer:
            adjusted += 0.25
        elif revised_answer and not is_acceptable:
            adjusted += 0.15

        return max(adjusted, 0.0)

    def _heuristic_stage1_score(self, question_text: str, reasoning_text: str, answer_text: str) -> float:
        if not reasoning_text:
            return 0.0

        score = 0.0
        score += 1.0
        if reasoning_text:
            score += 1.0

        question_tokens = self._extract_keywords(question_text)
        reasoning_tokens = self._extract_keywords(reasoning_text)
        if question_tokens and reasoning_tokens:
            overlap = len(question_tokens & reasoning_tokens)
            coverage = overlap / max(1, len(question_tokens))
            score += min(2.0, coverage * 2.0)

        marker_count = 0
        markers = ["first", "then", "next", "finally", "therefore", "so", "="]
        lower_reasoning = reasoning_text.lower()
        for marker in markers:
            if marker in lower_reasoning:
                marker_count += 1
        score += min(1.0, marker_count * 0.25)

        question_has_number = bool(re.search(r"\d", question_text))
        answer_has_number = bool(re.search(r"\d", answer_text))
        reasoning_has_number = bool(re.search(r"\d", reasoning_text))
        if question_has_number and answer_has_number:
            score += 1.0
        if question_has_number and reasoning_has_number:
            score += 0.5

        if answer_text and answer_text in reasoning_text:
            score += 0.5

        if len(answer_text) <= 1 and not answer_has_number:
            score -= 0.5
        if reasoning_text and len(reasoning_text.split()) < 8:
            score -= 0.25

        return max(score, 0.0)

    def _coerce_score(self, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(10.0, score))

    def _normalize_text(self, text: Any) -> str:
        if text is None:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip()

    def _extract_keywords(self, text: str) -> set[str]:
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on",
            "at", "for", "and", "or", "that", "this", "it", "as", "with", "by",
            "from", "your", "their", "into", "then", "than", "will", "would",
            "should", "could", "how", "what", "when", "where", "why", "use",
            "using", "answer", "final", "question",
        }
        tokens = re.findall(r"[A-Za-z0-9_./:-]+", text.lower())
        return {token for token in tokens if len(token) > 2 and token not in stopwords}
