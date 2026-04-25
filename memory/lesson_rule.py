"""Utilities for structured semantic lessons and lesson retrieval ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "between",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "to",
    "use",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _dedupe_preserve_case(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_text(value)
        if not normalized:
            continue
        folded = normalized.lower()
        if folded in seen:
            continue
        seen.add(folded)
        deduped.append(normalized)
    return deduped


def _tokenize_match_terms(text: Any) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9_]+", normalize_text(text).lower()):
        if token in _STOPWORDS:
            continue
        if len(token) <= 2 and not token.isdigit():
            continue
        tokens.append(token)
    return _dedupe_strings(tokens)


def _coerce_string_list(value: Any, *, preserve_case: bool = False) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values = [str(item) for item in value]
        return _dedupe_preserve_case(values) if preserve_case else _dedupe_strings(values)
    if isinstance(value, tuple):
        values = [str(item) for item in value]
        return _dedupe_preserve_case(values) if preserve_case else _dedupe_strings(values)
    text = normalize_text(value)
    if not text:
        return []
    if "|" in text:
        values = text.split("|")
        return _dedupe_preserve_case(values) if preserve_case else _dedupe_strings(values)
    values = text.split(",")
    return _dedupe_preserve_case(values) if preserve_case else _dedupe_strings(values)


def _safe_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_number(text: Any) -> float | None:
    normalized = normalize_text(text).lower().replace(",", "")
    if not normalized:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", normalized)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _is_scale_like_ratio(predicted: float | None, expected: float | None) -> bool:
    if predicted is None or expected is None or expected == 0:
        return False
    ratio = abs(predicted / expected)
    candidates = [0.001, 0.01, 0.1, 10.0, 100.0, 1000.0]
    return any(abs(ratio - candidate) <= max(0.05, candidate * 0.05) for candidate in candidates)


def _inferred_failure_modes(error_type: str, question: str) -> list[str]:
    lower_question = normalize_text(question).lower()
    if error_type == "unit_conversion":
        hints = ["unit_or_scale_mismatch", "format_or_rounding_slip"]
        if any(token in lower_question for token in ["round", "nearest", "comma", "format"]):
            hints.append("format_or_rounding_slip")
        return _dedupe_strings(hints)
    if error_type == "counting_scope":
        hints = ["scope_filter_mismatch", "boundary_condition_slip"]
        if any(token in lower_question for token in ["between", "from", "to", "during"]):
            hints.append("boundary_condition_slip")
        return _dedupe_strings(hints)
    if error_type == "surface_form_guess":
        return ["surface_form_shortcut"]
    return ["insufficient_verification"]


@dataclass(slots=True)
class SemanticLesson:
    """Structured semantic lesson stored in memory."""

    error_type: str
    lesson: str
    tags: list[str]
    applicability: str
    benchmark: str = "GAIA"
    source_task_id: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    failure_mode: str = "generic_verification"
    failure_stage: str = "unknown"
    severity: str = "wrong"
    correction_checklist: list[str] = field(default_factory=list)

    def selection_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.error_type,
            self.failure_mode,
            self.lesson,
            self.applicability,
            self.source_task_id,
        )

    def to_text(self) -> str:
        structured = self.to_dict()
        parts = [
            f"Error type: {self.error_type}",
            f"Failure mode: {self.failure_mode}",
            f"Failure stage: {self.failure_stage}",
            f"Severity: {self.severity}",
            f"Lesson: {self.lesson}",
            f"Tags: {', '.join(self.tags)}",
            f"Applicability: {self.applicability}",
            f"Benchmark: {self.benchmark}",
            f"Confidence: {self.confidence:.2f}",
        ]
        if self.correction_checklist:
            parts.append("Correction checklist: " + " | ".join(self.correction_checklist))
        if self.source_task_id:
            parts.append(f"Source task id: {self.source_task_id}")
        parts.append(
            "Structured fields: "
            + json.dumps(structured, ensure_ascii=True, sort_keys=True, default=str)
        )
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_summary(self) -> str:
        parts = [
            f"error_type={self.error_type}",
            f"failure_mode={self.failure_mode}",
            f"severity={self.severity}",
            f"lesson={self.lesson}",
        ]
        if self.applicability:
            parts.append(f"applicability={self.applicability}")
        if self.correction_checklist:
            parts.append(f"check={self.correction_checklist[0]}")
        return " | ".join(parts)


@dataclass(slots=True)
class LessonRetrievalProfile:
    """Signals used to rank retrieved semantic lessons."""

    question: str
    error_type: str
    tags: list[str]
    applicability_keywords: list[str]
    lesson_queries: list[str]
    case_queries: list[str]
    question_terms: list[str]
    inferred_failure_modes: list[str]


@dataclass(slots=True)
class LessonMatchResult:
    """Scoring details for matching a lesson to the current question."""

    score: float
    error_type_match: bool
    failure_mode_match: bool
    matched_tags: list[str]
    applicability_hits: list[str]
    matched_question_terms: list[str]


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def classify_error_type(
    question: str,
    *,
    predicted: str = "",
    expected: str = "",
    partial_match: bool | None = None,
) -> str:
    normalized_question = normalize_text(question)
    lower_question = normalized_question.lower()
    padded_question = f" {lower_question} "

    predicted_number = _parse_number(predicted)
    expected_number = _parse_number(expected)
    numeric_scale_mismatch = _is_scale_like_ratio(predicted_number, expected_number)

    if numeric_scale_mismatch and any(
        token in lower_question
        for token in [
            "round",
            "nearest",
            "thousand",
            "hour",
            "hours",
            "comma",
            "unit",
            "km",
            "meter",
            "meters",
        ]
    ):
        return "unit_conversion"

    if any(
        token in lower_question
        for token in [
            "round",
            "nearest",
            "thousand",
            "hour",
            "hours",
            "comma",
            "unit",
            "km",
            "meter",
            "meters",
        ]
    ):
        return "unit_conversion"

    if (
        "how many" in lower_question
        or "between" in lower_question
        or " during " in padded_question
        or " published " in padded_question
        or " album" in lower_question
        or " count" in lower_question
        or (" from " in padded_question and " to " in padded_question)
    ):
        return "counting_scope"

    if any(
        token in lower_question
        for token in ["hidden label", "token", "code", "identifier", "mapping"]
    ):
        return "surface_form_guess"

    if partial_match and predicted_number is not None and expected_number is not None:
        return "unit_conversion" if numeric_scale_mismatch else "counting_scope"

    return "insufficient_evidence_guess"


def classify_failure_mode(
    question: str,
    *,
    predicted: str = "",
    expected: str = "",
    error_type: str,
    failure_stage: str = "unknown",
    partial_match: bool = False,
    candidate_collapse: bool = False,
    overrode_better_candidate: bool = False,
) -> str:
    lower_predicted = normalize_text(predicted).lower()

    if overrode_better_candidate:
        return "final_overrode_better_candidate"
    if candidate_collapse:
        return "candidate_collapse"
    if "cannot determine" in lower_predicted or "not enough information" in lower_predicted:
        return "insufficient_commitment"

    if partial_match:
        if error_type == "unit_conversion":
            return "format_or_rounding_slip"
        if error_type == "counting_scope":
            return "boundary_condition_slip"
        return "partial_alignment_but_final_mismatch"

    if error_type == "unit_conversion":
        return "unit_or_scale_mismatch"
    if error_type == "counting_scope":
        return "scope_filter_mismatch"
    if error_type == "surface_form_guess":
        return "surface_form_shortcut"
    if failure_stage == "final":
        return "final_answer_selection"
    return "insufficient_verification"


def build_tags(error_type: str, question: str, *, failure_mode: str | None = None) -> list[str]:
    lower_question = normalize_text(question).lower()
    tags_by_type = {
        "unit_conversion": ["numeric", "unit", "conversion"],
        "counting_scope": ["counting", "scope", "filtering"],
        "surface_form_guess": ["label", "mapping", "surface_form"],
        "insufficient_evidence_guess": ["verification", "evidence"],
    }
    tags = list(tags_by_type.get(error_type, ["verification"]))

    if any(char.isdigit() for char in lower_question):
        tags.append("numeric")
    if "date" in lower_question or "between" in lower_question:
        tags.append("date_range")
    if "album" in lower_question:
        tags.append("album")
    if "round" in lower_question or "comma" in lower_question:
        tags.append("format")
    if "token" in lower_question or "label" in lower_question:
        tags.append("token")
    if "from " in lower_question or " to " in lower_question or "during" in lower_question:
        tags.append("boundary")
    if failure_mode:
        tags.append(failure_mode)

    return _dedupe_strings(tags)


def build_applicability(
    error_type: str,
    *,
    failure_mode: str = "",
    failure_stage: str = "",
) -> str:
    applicability_by_type = {
        "unit_conversion": (
            "Use for numeric questions mentioning units, rounding, comma formatting, "
            "scale words, or conversions."
        ),
        "counting_scope": (
            "Use for count or list questions with date ranges, filters, categories, "
            "or inclusion boundaries."
        ),
        "surface_form_guess": (
            "Use for token, label, code name, or hidden mapping questions where "
            "surface text may be misleading."
        ),
        "insufficient_evidence_guess": (
            "Use when the system is tempted to answer from intuition without enough evidence."
        ),
    }
    applicability = applicability_by_type.get(
        error_type,
        "Use this lesson for future questions with similar reasoning patterns.",
    )
    if failure_mode:
        applicability += f" Prioritize cases resembling {failure_mode.replace('_', ' ')}."
    if failure_stage and failure_stage not in {"", "unknown"}:
        applicability += f" It is especially relevant when the likely failure stage is {failure_stage}."
    return applicability


def build_correction_checklist(
    error_type: str,
    failure_mode: str,
    *,
    partial_match: bool = False,
) -> list[str]:
    checklist_by_mode = {
        "unit_or_scale_mismatch": [
            "Confirm the requested unit and rescale the numeric answer before finalizing.",
            "Check whether rounding or comma formatting changes the final string.",
            "Compare the final number against the question wording one more time.",
        ],
        "format_or_rounding_slip": [
            "Keep the reasoning but re-check the requested formatting and rounding rule.",
            "Verify whether commas, decimal places, or nearest-unit instructions change the answer.",
            "Do a final string-level check before committing the answer.",
        ],
        "scope_filter_mismatch": [
            "List the inclusion rules before counting any items.",
            "Exclude items outside the requested date range or category filters.",
            "Recount only the items that satisfy every constraint in the question.",
        ],
        "boundary_condition_slip": [
            "Re-check the start and end boundaries before counting.",
            "Confirm whether the endpoints should be included or excluded.",
            "Verify the final count against the filtered set, not the raw list.",
        ],
        "surface_form_shortcut": [
            "Do not trust a label or token based only on surface form.",
            "Look for explicit evidence that maps the label to the requested answer.",
            "If the mapping is not explicit, avoid guessing.",
        ],
        "candidate_collapse": [
            "Do not trust repeated candidates unless they cite the same verified evidence.",
            "Compare the top candidates against the task constraints before choosing one.",
            "Prefer the candidate that survives an explicit verification pass.",
        ],
        "final_overrode_better_candidate": [
            "Before the final selection, check whether any stage2 candidate already satisfies the constraints better.",
            "Do not revise away from the strongest verified candidate without stronger evidence.",
            "Use critiques to correct the solver only when they improve factual support.",
        ],
        "final_answer_selection": [
            "Review the best candidate answers before locking the final answer.",
            "Only override a candidate when the critique is clearly stronger than the current evidence.",
            "Run one last constraint check on the final answer string.",
        ],
        "insufficient_commitment": [
            "If the task is answerable, prefer a verified answer over an unnecessary refusal.",
            "Use the available evidence to narrow the answer before giving up.",
            "Only abstain when the evidence truly cannot resolve the question.",
        ],
        "insufficient_verification": [
            "Verify the answer against the task constraints before finalizing.",
            "Look for missing evidence instead of trusting the first plausible answer.",
            "If confidence is low, perform one more targeted check.",
        ],
        "partial_alignment_but_final_mismatch": [
            "Keep the good part of the reasoning, but fix the last-mile mismatch.",
            "Check formatting, boundary conditions, and output shape before finalizing.",
            "Ensure the final answer string matches the requested form exactly.",
        ],
    }

    if failure_mode in checklist_by_mode:
        return checklist_by_mode[failure_mode]

    if partial_match:
        return checklist_by_mode["partial_alignment_but_final_mismatch"]

    fallback_by_type = {
        "unit_conversion": checklist_by_mode["unit_or_scale_mismatch"],
        "counting_scope": checklist_by_mode["scope_filter_mismatch"],
        "surface_form_guess": checklist_by_mode["surface_form_shortcut"],
    }
    return fallback_by_type.get(error_type, checklist_by_mode["insufficient_verification"])


def build_retrieval_profile(question: str) -> LessonRetrievalProfile:
    normalized_question = normalize_text(question)
    error_type = classify_error_type(normalized_question)
    question_terms = _tokenize_match_terms(normalized_question)[:12]
    inferred_failure_modes = _inferred_failure_modes(error_type, normalized_question)
    tags = build_tags(error_type, normalized_question)
    applicability = build_applicability(error_type).lower()
    applicability_keywords = [
        token
        for token in re.findall(r"[a-z_]+", applicability)
        if len(token) > 3 and token not in {"use", "when", "with", "where", "this"}
    ]

    lesson_queries = [
        normalized_question,
        f"Error type: {error_type}",
        f"{error_type} lesson",
    ]
    if question_terms:
        lesson_queries.append(" ".join(question_terms[:6]))
    for failure_mode in inferred_failure_modes:
        lesson_queries.append(f"{failure_mode.replace('_', ' ')} lesson")

    case_queries = [normalized_question, f"{error_type} mistake case"]
    if error_type == "unit_conversion":
        case_queries.append("numeric conversion mistake case")
    elif error_type == "counting_scope":
        case_queries.append("counting scope mistake case")
    elif error_type == "surface_form_guess":
        case_queries.append("surface form guess mistake case")
    else:
        case_queries.append("insufficient evidence mistake case")

    return LessonRetrievalProfile(
        question=normalized_question,
        error_type=error_type,
        tags=tags,
        applicability_keywords=applicability_keywords,
        lesson_queries=_dedupe_strings(lesson_queries),
        case_queries=_dedupe_strings(case_queries),
        question_terms=question_terms,
        inferred_failure_modes=inferred_failure_modes,
    )


def build_semantic_lesson(
    *,
    question: str,
    task_id: str,
    lesson: str,
    benchmark: str = "GAIA",
    error_type: str | None = None,
    confidence: float = 1.0,
    metadata: dict[str, Any] | None = None,
    failure_mode: str = "generic_verification",
    failure_stage: str = "unknown",
    severity: str = "wrong",
    correction_checklist: list[str] | None = None,
) -> SemanticLesson:
    resolved_error_type = error_type or classify_error_type(question)
    enriched_metadata = dict(metadata or {})
    enriched_metadata.setdefault("question_terms", _tokenize_match_terms(question)[:12])
    enriched_metadata.setdefault("question_excerpt", normalize_text(question)[:220])
    enriched_metadata.setdefault("error_type", resolved_error_type)
    enriched_metadata.setdefault("failure_mode", failure_mode)
    enriched_metadata.setdefault("failure_stage", failure_stage)
    enriched_metadata.setdefault("severity", severity)
    return SemanticLesson(
        error_type=resolved_error_type,
        lesson=lesson,
        tags=build_tags(resolved_error_type, question, failure_mode=failure_mode),
        applicability=build_applicability(
            resolved_error_type,
            failure_mode=failure_mode,
            failure_stage=failure_stage,
        ),
        benchmark=benchmark,
        source_task_id=task_id,
        confidence=confidence,
        metadata=enriched_metadata,
        failure_mode=failure_mode,
        failure_stage=failure_stage,
        severity=severity,
        correction_checklist=_coerce_string_list(correction_checklist, preserve_case=True),
    )


def _semantic_lesson_from_payload(payload: dict[str, Any]) -> SemanticLesson | None:
    if not isinstance(payload, dict):
        return None
    error_type = normalize_text(payload.get("error_type"))
    lesson = normalize_text(payload.get("lesson"))
    if not error_type or not lesson:
        return None
    return SemanticLesson(
        error_type=error_type,
        lesson=lesson,
        tags=_coerce_string_list(payload.get("tags")),
        applicability=normalize_text(payload.get("applicability")),
        benchmark=normalize_text(payload.get("benchmark")) or "GAIA",
        source_task_id=normalize_text(payload.get("source_task_id")),
        confidence=_safe_float(payload.get("confidence"), 1.0),
        metadata=dict(payload.get("metadata") or {}),
        failure_mode=normalize_text(payload.get("failure_mode")) or "generic_verification",
        failure_stage=normalize_text(payload.get("failure_stage")) or "unknown",
        severity=normalize_text(payload.get("severity")) or "wrong",
        correction_checklist=_coerce_string_list(
            payload.get("correction_checklist"),
            preserve_case=True,
        ),
    )


def parse_semantic_lesson_text(content: str) -> SemanticLesson | None:
    raw_content = str(content or "").strip()
    if not raw_content:
        return None

    structured_match = re.search(
        r"Structured fields:\s*(\{.*\})\s*$",
        raw_content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if structured_match:
        try:
            payload = json.loads(structured_match.group(1))
            lesson = _semantic_lesson_from_payload(payload)
            if lesson is not None:
                return lesson
        except json.JSONDecodeError:
            pass

    normalized = normalize_text(raw_content)
    if not normalized:
        return None

    error_type_match = re.search(r"Error type:\s*([A-Za-z0-9_\-]+)", normalized, flags=re.IGNORECASE)
    lesson_match = re.search(
        r"Lesson:\s*(.+?)(?:\s+Tags:|\s+Applicability:|\s+Benchmark:|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not error_type_match or not lesson_match:
        return None

    tags_match = re.search(r"Tags:\s*(.+?)(?:\s+Applicability:|$)", normalized, flags=re.IGNORECASE)
    applicability_match = re.search(
        r"Applicability:\s*(.+?)(?:\s+Benchmark:|\s+Confidence:|\s+Source task id:|\s+Observed mismatch:|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    benchmark_match = re.search(r"Benchmark:\s*([A-Za-z0-9_\-]+)", normalized, flags=re.IGNORECASE)
    task_id_match = re.search(r"Source task id:\s*([A-Za-z0-9_\-]+)", normalized, flags=re.IGNORECASE)
    confidence_match = re.search(r"Confidence:\s*([0-9]+(?:\.[0-9]+)?)", normalized, flags=re.IGNORECASE)
    failure_mode_match = re.search(
        r"Failure mode:\s*([A-Za-z0-9_\-]+)",
        normalized,
        flags=re.IGNORECASE,
    )
    failure_stage_match = re.search(
        r"Failure stage:\s*([A-Za-z0-9_\-]+)",
        normalized,
        flags=re.IGNORECASE,
    )
    severity_match = re.search(r"Severity:\s*([A-Za-z0-9_\-]+)", normalized, flags=re.IGNORECASE)
    checklist_match = re.search(
        r"Correction checklist:\s*(.+?)(?:\s+Source task id:|\s+Structured fields:|$)",
        normalized,
        flags=re.IGNORECASE,
    )

    tags = _coerce_string_list(tags_match.group(1) if tags_match else [])
    correction_checklist = _coerce_string_list(
        checklist_match.group(1) if checklist_match else [],
        preserve_case=True,
    )

    return SemanticLesson(
        error_type=error_type_match.group(1).strip(),
        lesson=lesson_match.group(1).strip(),
        tags=tags,
        applicability=applicability_match.group(1).strip() if applicability_match else "",
        benchmark=benchmark_match.group(1).strip() if benchmark_match else "GAIA",
        source_task_id=task_id_match.group(1).strip() if task_id_match else "",
        confidence=_safe_float(confidence_match.group(1) if confidence_match else 1.0, 1.0),
        metadata={},
        failure_mode=failure_mode_match.group(1).strip() if failure_mode_match else "generic_verification",
        failure_stage=failure_stage_match.group(1).strip() if failure_stage_match else "unknown",
        severity=severity_match.group(1).strip() if severity_match else "wrong",
        correction_checklist=correction_checklist,
    )


def parse_semantic_lesson_memory(memory: Any) -> SemanticLesson | None:
    metadata = getattr(memory, "metadata", None)
    if isinstance(metadata, dict):
        semantic_payload = metadata.get("semantic_lesson")
        lesson = _semantic_lesson_from_payload(semantic_payload)
        if lesson is not None:
            return lesson

    lesson = parse_semantic_lesson_text(str(getattr(memory, "content", "") or ""))
    if lesson is None or not isinstance(metadata, dict):
        return lesson

    if not lesson.metadata:
        lesson.metadata = dict(metadata.get("semantic_lesson", {}).get("metadata", {}) or {})
    lesson.failure_mode = normalize_text(metadata.get("failure_mode")) or lesson.failure_mode
    lesson.failure_stage = normalize_text(metadata.get("failure_stage")) or lesson.failure_stage
    lesson.severity = normalize_text(metadata.get("severity")) or lesson.severity
    return lesson


def score_semantic_lesson_match(
    lesson: SemanticLesson,
    profile: LessonRetrievalProfile,
) -> LessonMatchResult:
    score = 0.0
    error_type_match = lesson.error_type == profile.error_type
    failure_mode_match = lesson.failure_mode in profile.inferred_failure_modes
    matched_tags: list[str] = []
    applicability_hits: list[str] = []

    if error_type_match:
        score += 2.5
    if failure_mode_match:
        score += 1.5

    lesson_tags = {tag.lower() for tag in lesson.tags}
    profile_tags = {tag.lower() for tag in profile.tags}
    matched_tags = sorted(lesson_tags & profile_tags)
    score += min(2.0, 0.75 * float(len(matched_tags)))

    applicability_text = lesson.applicability.lower()
    for keyword in profile.applicability_keywords:
        if keyword.lower() in applicability_text:
            applicability_hits.append(keyword)
    score += min(1.5, 0.35 * float(len(applicability_hits)))

    lesson_terms = set(_tokenize_match_terms(lesson.lesson))
    lesson_terms.update(_coerce_string_list(lesson.metadata.get("question_terms")))
    matched_question_terms = sorted(set(profile.question_terms) & lesson_terms)
    score += min(2.5, 0.5 * float(len(matched_question_terms)))

    if lesson.correction_checklist:
        score += 0.3
    if lesson.severity == "wrong":
        score += 0.25
    elif lesson.severity == "partial":
        score += 0.1

    score += min(0.75, max(0.0, lesson.confidence) * 0.5)

    return LessonMatchResult(
        score=score,
        error_type_match=error_type_match,
        failure_mode_match=failure_mode_match,
        matched_tags=matched_tags,
        applicability_hits=applicability_hits,
        matched_question_terms=matched_question_terms,
    )


def select_relevant_semantic_lessons(
    lessons: list[SemanticLesson],
    profile: LessonRetrievalProfile,
    *,
    min_score: float = 1.5,
    limit: int = 3,
) -> list[tuple[SemanticLesson, LessonMatchResult]]:
    deduped_lessons: dict[tuple[str, str, str, str, str], SemanticLesson] = {}
    for lesson in lessons:
        key = lesson.selection_key()
        current = deduped_lessons.get(key)
        if current is None or lesson.confidence > current.confidence:
            deduped_lessons[key] = lesson

    scored: list[tuple[SemanticLesson, LessonMatchResult]] = []
    for lesson in deduped_lessons.values():
        match = score_semantic_lesson_match(lesson, profile)
        if match.score >= min_score:
            scored.append((lesson, match))

    scored.sort(
        key=lambda item: (
            item[1].score,
            item[0].confidence,
            item[1].error_type_match,
            item[1].failure_mode_match,
            len(item[1].matched_question_terms),
            len(item[1].matched_tags),
            len(item[1].applicability_hits),
        ),
        reverse=True,
    )
    return scored[:limit]
