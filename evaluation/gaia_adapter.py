from __future__ import annotations

from typing import Any

from memory.lesson_rule import (
    SemanticLesson,
    build_applicability,
    build_correction_checklist,
    build_semantic_lesson,
    build_tags,
    classify_error_type,
    classify_failure_mode,
    normalize_text,
)

from .benchmark_adapter import BaseBenchmarkAdapter


class GAIAAdapter(BaseBenchmarkAdapter):
    """Benchmark adapter for running AgentNetwork on GAIA."""

    def __init__(
        self,
        agent: Any,
        use_two_stage: bool = True,
        include_reasoning: bool = False,
        name: str | None = None,
    ):
        super().__init__(agent=agent, name=name or "AgentNetwork")
        self.use_two_stage = use_two_stage
        self.include_reasoning = include_reasoning

    def normalize_question(self, question: str) -> str:
        if not question:
            return ""

        text = question.strip()
        file_note = "\n\nNote: This question may require reference to the file:"
        if file_note in text:
            text = text.split(file_note, 1)[0].strip()

        return text

    def run(self, prompt: str) -> str:
        normalized_prompt = self.normalize_question(prompt)

        if self.use_two_stage:
            result = self.agent.forward_two_stage(normalized_prompt)
            final_answer = result.get("final_result", "")
            reasoning = result.get("stage1_result", "")
        else:
            final_answer, *_ = self.agent.forward(normalized_prompt)
            reasoning = ""

        if self.include_reasoning and reasoning:
            return f"REASONING: {reasoning}\nFINAL ANSWER: {final_answer}"

        return f"FINAL ANSWER: {final_answer}"

    def record_evaluation_feedback(
        self,
        *,
        benchmark: str,
        sample: dict[str, Any],
        sample_result: dict[str, Any],
    ) -> None:
        if benchmark.upper() != "GAIA":
            return

        memory_tool = getattr(self.agent, "memory_tool", None)
        if memory_tool is None:
            return

        question = self.normalize_question(sample.get("question", ""))
        if not question:
            return

        predicted = str(
            sample_result.get("predicted", sample_result.get("predicted_answer", "")) or ""
        ).strip() or "(empty)"
        expected = str(
            sample_result.get("expected", sample_result.get("expected_answer", "")) or ""
        ).strip() or "(unknown)"
        task_id = str(sample.get("task_id", "") or "").strip() or "unknown"
        exact_match = bool(sample_result.get("exact_match", False))
        partial_match = bool(sample_result.get("partial_match", False))
        score = sample_result.get("score", 0.0)

        reflection = self._build_gaia_reflection_record(
            question=question,
            predicted=predicted,
            expected=expected,
            exact_match=exact_match,
            partial_match=partial_match,
        )
        base_metadata = {
            "record_type": "gaia_feedback",
            "benchmark": "GAIA",
            "task_id": task_id,
            "question_excerpt": normalize_text(question)[:220],
            "predicted": predicted,
            "expected": expected,
            "exact_match": exact_match,
            "partial_match": partial_match,
            "score": score,
            "error_type": reflection["error_type"],
            "failure_mode": reflection["failure_mode"],
            "failure_stage": reflection["failure_stage"],
            "severity": reflection["severity"],
        }

        try:
            if exact_match:
                success_note = self._build_gaia_success_working_note(
                    task_id=task_id,
                    question=question,
                    predicted=predicted,
                    score=score,
                    reflection=reflection,
                )
                memory_tool.run(
                    {
                        "action": "add",
                        "content": success_note,
                        "memory_type": "working",
                        "importance": 0.55,
                        "metadata": {
                            **base_metadata,
                            "record_type": "gaia_success_reminder",
                            "lesson": reflection["lesson"],
                            "correction_checklist": reflection["correction_checklist"],
                        },
                    }
                )
            else:
                failure_case_note = self._build_gaia_failure_case_note(
                    task_id=task_id,
                    question=question,
                    predicted=predicted,
                    expected=expected,
                    exact_match=exact_match,
                    partial_match=partial_match,
                    score=score,
                    reflection=reflection,
                )
                lesson_note, semantic_lesson = self._build_gaia_lesson_note(
                    task_id=task_id,
                    question=question,
                    predicted=predicted,
                    expected=expected,
                    reflection=reflection,
                )

                memory_tool.run(
                    {
                        "action": "add",
                        "content": failure_case_note,
                        "memory_type": "episodic",
                        "importance": 0.9,
                        "metadata": {
                            **base_metadata,
                            "record_type": "gaia_failure_case",
                            "failure_summary": reflection["failure_summary"],
                            "correction_checklist": reflection["correction_checklist"],
                        },
                    }
                )
                memory_tool.run(
                    {
                        "action": "add",
                        "content": lesson_note,
                        "memory_type": "semantic",
                        "importance": 0.97,
                        "metadata": {
                            **base_metadata,
                            "record_type": "gaia_semantic_lesson",
                            "semantic_lesson": semantic_lesson.to_dict(),
                            "correction_checklist": reflection["correction_checklist"],
                        },
                    }
                )
        except Exception as exc:
            print(f"[WARN] GAIA feedback memory write failed: {exc}")

    def _build_gaia_success_working_note(
        self,
        *,
        task_id: str,
        question: str,
        predicted: str,
        score: Any,
        reflection: dict[str, Any],
    ) -> str:
        checklist = reflection.get("correction_checklist") or []
        checklist_text = " | ".join(checklist) if checklist else reflection["lesson"]
        return (
            f"GAIA success reminder for task {task_id or 'unknown'}.\n"
            f"Question: {question}\n"
            f"Final answer: {predicted}\n"
            f"Score: {score}\n"
            f"Keep-check: {reflection['lesson']}\n"
            f"Checklist: {checklist_text}\n"
            f"Use when: {reflection['applicability']}"
        )

    def _build_gaia_failure_case_note(
        self,
        *,
        task_id: str,
        question: str,
        predicted: str,
        expected: str,
        exact_match: bool,
        partial_match: bool,
        score: Any,
        reflection: dict[str, Any],
    ) -> str:
        checklist = " | ".join(reflection.get("correction_checklist") or [])
        return (
            f"GAIA failure case for task {task_id or 'unknown'}.\n"
            f"Question: {question}\n"
            f"Predicted answer: {predicted}\n"
            f"Expected answer: {expected}\n"
            f"Exact match: {exact_match}\n"
            f"Partial match: {partial_match}\n"
            f"Score: {score}\n"
            f"Error type: {reflection['error_type']}\n"
            f"Failure mode: {reflection['failure_mode']}\n"
            f"Failure stage: {reflection['failure_stage']}\n"
            f"Severity: {reflection['severity']}\n"
            f"What went wrong: {reflection['failure_summary']}\n"
            f"Correction checklist: {checklist}"
        )

    def _build_gaia_lesson_note(
        self,
        *,
        task_id: str,
        question: str,
        predicted: str,
        expected: str,
        reflection: dict[str, Any],
    ) -> tuple[str, SemanticLesson]:
        semantic_lesson = build_semantic_lesson(
            question=question,
            task_id=task_id or "unknown",
            benchmark="GAIA",
            error_type=reflection["error_type"],
            lesson=reflection["lesson"],
            confidence=reflection["confidence"],
            metadata={
                "predicted": predicted,
                "expected": expected,
                "partial_match": reflection["severity"] == "partial",
            },
            failure_mode=reflection["failure_mode"],
            failure_stage=reflection["failure_stage"],
            severity=reflection["severity"],
            correction_checklist=reflection["correction_checklist"],
        )
        note = (
            "GAIA correction lesson\n"
            f"Observed mismatch: predicted '{predicted}' while expected '{expected}'.\n"
            f"{semantic_lesson.to_text()}"
        )
        return note, semantic_lesson

    def _build_gaia_reflection_record(
        self,
        *,
        question: str,
        predicted: str,
        expected: str,
        exact_match: bool,
        partial_match: bool,
    ) -> dict[str, Any]:
        stage2_outputs = getattr(self.agent, "last_stage2_outputs", []) or []
        error_type = classify_error_type(
            question=question,
            predicted=predicted,
            expected=expected,
            partial_match=partial_match,
        )
        failure_stage = self._infer_failure_stage(
            predicted=predicted,
            expected=expected,
            exact_match=exact_match,
            stage2_outputs=stage2_outputs,
        )
        candidate_collapse = self._has_candidate_collapse(
            stage2_outputs=stage2_outputs,
            expected=expected,
        )
        overrode_better_candidate = self._has_stage2_candidate(
            stage2_outputs=stage2_outputs,
            target_answer=expected,
        ) and self._answer_key(predicted) != self._answer_key(expected)
        failure_mode = classify_failure_mode(
            question=question,
            predicted=predicted,
            expected=expected,
            error_type=error_type,
            failure_stage=failure_stage,
            partial_match=partial_match,
            candidate_collapse=candidate_collapse,
            overrode_better_candidate=overrode_better_candidate,
        )

        if exact_match:
            success_checklist = [
                "Keep the successful reasoning pattern.",
                "Still verify units, scope, and final formatting before finalizing.",
                "Only change the answer if new evidence clearly improves it.",
            ]
            return {
                "error_type": "confirmed_reasoning",
                "failure_mode": "confirmed_reasoning",
                "failure_stage": "confirmed",
                "severity": "confirmed",
                "lesson": (
                    "Keep the successful reasoning pattern, but still verify units, scope, "
                    "and final formatting before finalizing."
                ),
                "tags": ["confirmed_reasoning", "verification"],
                "applicability": (
                    "Use this reminder for similar questions when the answer seems clear but "
                    "still benefits from a final consistency check."
                ),
                "failure_summary": "No failure. The reasoning path produced the expected answer.",
                "correction_checklist": success_checklist,
                "confidence": 0.55,
            }

        severity = "partial" if partial_match else "wrong"
        tags = build_tags(error_type, question, failure_mode=failure_mode)
        applicability = build_applicability(
            error_type,
            failure_mode=failure_mode,
            failure_stage=failure_stage,
        )
        correction_checklist = build_correction_checklist(
            error_type,
            failure_mode,
            partial_match=partial_match,
        )

        return {
            "error_type": error_type,
            "failure_mode": failure_mode,
            "failure_stage": failure_stage,
            "severity": severity,
            "lesson": correction_checklist[0],
            "tags": tags,
            "applicability": applicability,
            "failure_summary": self._build_failure_summary(
                error_type=error_type,
                failure_mode=failure_mode,
                failure_stage=failure_stage,
                predicted=predicted,
                expected=expected,
            ),
            "correction_checklist": correction_checklist,
            "confidence": 0.92 if partial_match else 0.97,
        }

    def _build_failure_summary(
        self,
        *,
        error_type: str,
        failure_mode: str,
        failure_stage: str,
        predicted: str,
        expected: str,
    ) -> str:
        summaries = {
            "unit_or_scale_mismatch": (
                "The answer likely stayed in the wrong unit, scale, or numeric format instead "
                "of matching the requested output representation."
            ),
            "format_or_rounding_slip": (
                "The reasoning was close, but the final formatting, rounding, or output string "
                "did not match the requested representation."
            ),
            "scope_filter_mismatch": (
                "The answer likely counted items outside the allowed range or included items "
                "that did not satisfy every requested filter."
            ),
            "boundary_condition_slip": (
                "The answer appears close, but the start or end boundary was likely handled "
                "incorrectly."
            ),
            "surface_form_shortcut": (
                "The answer appears to rely on surface wording or token appearance rather than "
                "explicit evidence from the task."
            ),
            "candidate_collapse": (
                "Stage2 candidates converged on an unsupported answer, so the final step did not "
                "receive a strong alternative to recover from."
            ),
            "final_overrode_better_candidate": (
                "A stronger stage2 candidate likely existed, but the final decision overrode it "
                "with a worse answer."
            ),
            "final_answer_selection": (
                "The final decision step likely chose or revised the answer poorly despite the "
                "available candidates."
            ),
            "insufficient_commitment": (
                "The system backed away from a resolvable answer instead of using the available "
                "evidence."
            ),
            "insufficient_verification": (
                "The answer appears to have been finalized before enough evidence or verification "
                "was collected."
            ),
        }
        summary = summaries.get(
            failure_mode,
            (
                f"The predicted answer '{predicted}' did not match the expected answer "
                f"'{expected}', which suggests that the final verification step was insufficient."
            ),
        )
        if failure_stage not in {"", "unknown"}:
            summary += f" Likely failure stage: {failure_stage}."
        if error_type:
            summary += f" Error type: {error_type}."
        return summary

    def _answer_key(self, answer: str) -> str:
        return normalize_text(answer).lower()

    def _successful_stage2_answers(self, stage2_outputs: list[dict[str, Any]]) -> list[str]:
        answers: list[str] = []
        for output in stage2_outputs or []:
            if not output.get("success", False):
                continue
            answer = self._answer_key(str(output.get("answer", "") or ""))
            if answer:
                answers.append(answer)
        return answers

    def _has_stage2_candidate(
        self,
        *,
        stage2_outputs: list[dict[str, Any]],
        target_answer: str,
    ) -> bool:
        target_key = self._answer_key(target_answer)
        if not target_key:
            return False
        return any(answer == target_key for answer in self._successful_stage2_answers(stage2_outputs))

    def _has_candidate_collapse(
        self,
        *,
        stage2_outputs: list[dict[str, Any]],
        expected: str,
    ) -> bool:
        answers = self._successful_stage2_answers(stage2_outputs)
        if not answers:
            return False
        unique_answers = set(answers)
        if len(unique_answers) != 1:
            return False
        return next(iter(unique_answers)) != self._answer_key(expected)

    def _infer_failure_stage(
        self,
        *,
        predicted: str,
        expected: str,
        exact_match: bool,
        stage2_outputs: list[dict[str, Any]],
    ) -> str:
        if exact_match:
            return "confirmed"

        predicted_key = self._answer_key(predicted)
        expected_key = self._answer_key(expected)
        stage1_key = self._answer_key(getattr(self.agent, "last_stage1_result", ""))
        stage2_answers = self._successful_stage2_answers(stage2_outputs)

        if stage2_answers:
            if expected_key and any(answer == expected_key for answer in stage2_answers):
                return "final"
            if stage1_key and stage1_key == expected_key and predicted_key != expected_key:
                return "final"
            if self._has_candidate_collapse(stage2_outputs=stage2_outputs, expected=expected):
                return "stage2"
            if stage1_key and stage1_key == predicted_key and predicted_key != expected_key:
                return "stage1"
            return "stage2"

        return "stage1"
