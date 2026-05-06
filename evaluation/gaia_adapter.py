from __future__ import annotations

from pathlib import Path
from typing import Any

from .benchmark_adapter import BaseBenchmarkAdapter


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


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

    def run_sample(self, prompt: str, sample: dict[str, Any]) -> str:
        normalized_prompt = self.normalize_question(prompt)
        context = {
            "benchmark": "GAIA",
            "task_id": str(sample.get("task_id", "") or ""),
            "level": sample.get("level"),
            "attachment": self._build_attachment_context(sample),
        }

        if self.use_two_stage:
            result = self.agent.forward_two_stage(normalized_prompt, context=context)
            final_answer = result.get("final_result", "")
            reasoning = result.get("stage1_result", "")
        else:
            final_answer, *_ = self.agent.forward(normalized_prompt)
            reasoning = ""

        if self.include_reasoning and reasoning:
            return f"REASONING: {reasoning}\nFINAL ANSWER: {final_answer}"

        return f"FINAL ANSWER: {final_answer}"

    def _build_attachment_context(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        file_path_text = str(
            sample.get("file_name") or sample.get("file_path") or ""
        ).strip()
        if not file_path_text:
            return None

        file_path = Path(file_path_text)
        return {
            "task_id": str(sample.get("task_id", "") or ""),
            "file_path": str(file_path),
            "file_name": file_path.name,
            "extension": file_path.suffix.lower(),
            "exists": file_path.exists(),
            "is_file": file_path.is_file(),
        }

    def record_evaluation_feedback(
        self,
        *,
        benchmark: str,
        sample: dict[str, Any],
        sample_result: dict[str, Any],
    ) -> None:
        if benchmark.upper() != "GAIA":
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
            self._record_graph_feedback(
                task_id=task_id,
                question=question,
                exact_match=exact_match,
                partial_match=partial_match,
                metadata={
                    **base_metadata,
                    "failure_summary": reflection["failure_summary"],
                    "correction_checklist": reflection["correction_checklist"],
                },
            )
        except Exception as exc:
            print(f"[WARN] GAIA graph feedback write failed: {exc}")

    def _record_graph_feedback(
        self,
        *,
        task_id: str,
        question: str,
        exact_match: bool,
        partial_match: bool,
        metadata: dict[str, Any],
    ) -> None:
        runtime = getattr(self.agent, "runtime", None)
        query_graph = getattr(runtime, "query_task_graph", None)
        insight_graph = getattr(runtime, "insight_graph", None)
        if runtime is None or query_graph is None or insight_graph is None:
            return

        context = getattr(runtime, "current_context", {}) or {}
        attachment = context.get("attachment") or {}
        attachment_type = str(attachment.get("extension", "") or "").strip().lower().lstrip(".") or None

        query_graph.register_task(
            task_id,
            question,
            metadata={
                "source": "gaia_feedback",
                "attachment_type": attachment_type,
                **metadata,
            },
        )
        classification = query_graph.classify_task(question, attachment_type=attachment_type)
        query_graph.link_task_signals(task_id, classification)
        insights = insight_graph.retrieve_insights(
            task_type=classification.task_type,
            trigger_terms=classification.trigger_terms,
            failure_modes=classification.failure_modes,
            limit=3,
        )
        stage2_changed_answer = self._stage2_changed_stage1_answer()
        for insight in insights:
            insight_id = str(insight.get("insight_id", "") or "").strip()
            if not insight_id:
                continue
            insight_graph.apply_feedback(
                insight_id,
                task_id,
                exact=exact_match,
                partial=partial_match,
                stage2_changed_answer=stage2_changed_answer,
                metadata=metadata,
            )
        runtime.record_memory_write(
            {
                "memory_type": "insight_graph",
                "source_stage": "gaia_feedback",
                "task_id": task_id,
                "task_type": classification.task_type,
                "insight_ids": [item.get("insight_id") for item in insights if isinstance(item, dict)],
                "exact_match": exact_match,
                "partial_match": partial_match,
            }
        )

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
        error_type = self._classify_error_type(
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
        failure_mode = self._classify_failure_mode(
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
        tags = self._build_tags(error_type, question, failure_mode=failure_mode)
        applicability = self._build_applicability(
            error_type,
            failure_mode=failure_mode,
            failure_stage=failure_stage,
        )
        correction_checklist = self._build_correction_checklist(
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

    def _classify_error_type(
        self,
        *,
        question: str,
        predicted: str,
        expected: str,
        partial_match: bool,
    ) -> str:
        text = normalize_text(question).lower()
        predicted_key = self._answer_key(predicted)
        expected_key = self._answer_key(expected)
        if partial_match:
            return "format_or_rounding_slip"
        if self._numeric_value(predicted_key) is not None and self._numeric_value(expected_key) is not None:
            return "unit_or_scale_mismatch"
        if any(marker in text for marker in ["how many", "count", "number of", "between", "during"]):
            return "scope_filter_mismatch"
        if any(marker in text for marker in ["probability", "random", "odds", "maximize", "state"]):
            return "reasoning_strategy_mismatch"
        if any(marker in text for marker in ["who", "when", "where", "published", "website", "source"]):
            return "insufficient_evidence"
        return "final_verification_failed"

    def _classify_failure_mode(
        self,
        *,
        question: str,
        predicted: str,
        expected: str,
        error_type: str,
        failure_stage: str,
        partial_match: bool,
        candidate_collapse: bool,
        overrode_better_candidate: bool,
    ) -> str:
        text = normalize_text(question).lower()
        if overrode_better_candidate:
            return "overrode_better_candidate"
        if candidate_collapse:
            return "candidate_collapse"
        if partial_match:
            return "answer_format_mismatch"
        if any(marker in text for marker in ["attachment", "spreadsheet", "excel", "image", "audio", "file"]):
            return "missed_attachment_evidence"
        if any(marker in text for marker in ["probability", "random", "odds", "maximize", "state"]):
            return "missing_state_transition_model"
        if error_type == "scope_filter_mismatch":
            return "scope_filter_mismatch"
        if failure_stage == "final":
            return "final_selection_error"
        return "insufficient_verification"

    def _build_tags(self, error_type: str, question: str, *, failure_mode: str) -> list[str]:
        tags = [error_type, failure_mode]
        text = normalize_text(question).lower()
        for marker, tag in [
            ("probability", "stochastic_process"),
            ("random", "stochastic_process"),
            ("excel", "spreadsheet"),
            ("image", "image"),
            ("audio", "audio"),
            ("website", "search"),
            ("published", "search"),
        ]:
            if marker in text:
                tags.append(tag)
        return sorted({tag for tag in tags if tag})

    def _build_applicability(self, error_type: str, *, failure_mode: str, failure_stage: str) -> str:
        return (
            f"Use this feedback for similar GAIA tasks with error_type={error_type}, "
            f"failure_mode={failure_mode}, or failure_stage={failure_stage}."
        )

    def _build_correction_checklist(
        self,
        error_type: str,
        failure_mode: str,
        *,
        partial_match: bool,
    ) -> list[str]:
        if partial_match or error_type == "format_or_rounding_slip":
            return [
                "Normalize the final answer format to exactly match the requested representation.",
                "Check rounding, units, and whether the answer should be a raw number or scaled phrase.",
            ]
        if failure_mode == "missing_state_transition_model":
            return [
                "Build explicit states and transitions before choosing the final answer.",
                "Use structured Python reasoning for enumeration, simulation, or dynamic programming when needed.",
            ]
        if failure_mode == "missed_attachment_evidence":
            return [
                "Read and summarize attachment evidence before answering.",
                "Use the attachment-derived evidence as the primary source for file-based questions.",
            ]
        if error_type == "insufficient_evidence":
            return [
                "Gather external evidence before finalizing factual claims.",
                "Prefer cited search evidence over memory-only guesses.",
            ]
        return [
            "Compare stage1, stage2, and expected answer constraints before final selection.",
            "Verify scope, units, and exact output format before finalizing.",
        ]

    def _numeric_value(self, value: str) -> float | None:
        normalized = normalize_text(value).replace(",", "")
        try:
            return float(normalized)
        except ValueError:
            return None

    def _stage2_changed_stage1_answer(self) -> bool:
        stage1_key = self._answer_key(getattr(self.agent, "last_stage1_result", ""))
        decision = getattr(self.agent, "last_final_decision", None) or {}
        final_key = self._answer_key(str(decision.get("final_result", "") or ""))
        return bool(stage1_key and final_key and stage1_key != final_key)

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
