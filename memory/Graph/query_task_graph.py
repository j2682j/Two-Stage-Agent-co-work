from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _slug(value: Any, *, default: str = "unknown") -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"[^a-z0-9_./:-]+", "_", text)
    text = text.strip("_")
    return text or default


def _task_id_from_question(question: str) -> str:
    digest = hashlib.sha1(_clean_text(question).encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"gaia_task_{digest}"


def _tokenize(text: str) -> set[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "from",
        "what",
        "which",
        "who",
        "when",
        "where",
        "how",
        "is",
        "are",
        "was",
        "were",
        "this",
        "that",
        "please",
        "answer",
        "final",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9_./:-]+", _clean_text(text).lower())
        if len(token) > 2 and token not in stopwords
    }


def _lexical_similarity(a: str, b: str) -> float:
    left = _tokenize(a)
    right = _tokenize(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass(slots=True)
class TaskClassification:
    task_type: str
    trigger_terms: list[str] = field(default_factory=list)
    attachment_type: str | None = None
    failure_modes: list[str] = field(default_factory=list)
    tool_policy: dict[str, list[str]] = field(default_factory=dict)
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "trigger_terms": list(self.trigger_terms),
            "attachment_type": self.attachment_type,
            "failure_modes": list(self.failure_modes),
            "tool_policy": {
                "prefer": list(self.tool_policy.get("prefer", [])),
                "optional": list(self.tool_policy.get("optional", [])),
                "avoid": list(self.tool_policy.get("avoid", [])),
            },
            "confidence": self.confidence,
        }


class QueryTaskGraph:
    """Task/query graph for GAIA-style memory retrieval.

    The class writes GAIA task nodes and task-signal relationships to Neo4j
    when a driver is available. If Neo4j is unavailable, it keeps a small
    in-process fallback so callers can still build memory prompts.
    """

    def __init__(
        self,
        graph_store: Any | None = None,
        *,
        auto_connect: bool = True,
        namespace: str = "gaia",
    ):
        self.namespace = namespace
        self.graph_store = graph_store
        self._memory_tasks: dict[str, dict[str, Any]] = {}
        self._memory_edges: list[dict[str, Any]] = []
        if self.graph_store is None and auto_connect:
            self.graph_store = self._create_graph_store()

    @property
    def available(self) -> bool:
        return bool(getattr(self.graph_store, "driver", None))

    def _create_graph_store(self) -> Any | None:
        try:
            from core.database_config import get_database_config
            from memory.storage.neo4j_store import Neo4jGraphStore

            store = Neo4jGraphStore(**get_database_config().get_neo4j_config())
            if not store.health_check():
                return None
            self._create_indexes(store)
            return store
        except Exception as exc:
            logger.warning("QueryTaskGraph Neo4j unavailable; using memory fallback: %s", exc)
            return None

    def _create_indexes(self, store: Any | None = None) -> None:
        store = store or self.graph_store
        if not getattr(store, "driver", None):
            return
        queries = [
            "CREATE INDEX gaia_task_id_index IF NOT EXISTS FOR (t:GaiaTask) ON (t.id)",
            "CREATE INDEX gaia_task_type_index IF NOT EXISTS FOR (t:GaiaTaskType) ON (t.name)",
            "CREATE INDEX gaia_trigger_index IF NOT EXISTS FOR (t:GaiaTriggerTerm) ON (t.name)",
            "CREATE INDEX gaia_failure_index IF NOT EXISTS FOR (f:GaiaFailureMode) ON (f.name)",
            "CREATE INDEX gaia_policy_index IF NOT EXISTS FOR (p:GaiaToolPolicy) ON (p.name)",
        ]
        with store.driver.session(database=store.database) as session:
            for query in queries:
                session.run(query)

    def _write(self, query: str, **params: Any) -> bool:
        if not self.available:
            return False
        try:
            with self.graph_store.driver.session(database=self.graph_store.database) as session:
                session.run(query, **params)
            return True
        except Exception as exc:
            logger.warning("QueryTaskGraph write failed: %s", exc)
            return False

    def _read(self, query: str, **params: Any) -> list[dict[str, Any]]:
        if not self.available:
            return []
        try:
            with self.graph_store.driver.session(database=self.graph_store.database) as session:
                result = session.run(query, **params)
                return [dict(record) for record in result]
        except Exception as exc:
            logger.warning("QueryTaskGraph read failed: %s", exc)
            return []

    def register_task(
        self,
        task_id: str | None,
        question: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        resolved_id = _clean_text(task_id) or _task_id_from_question(question)
        payload = dict(metadata or {})
        payload.setdefault("created_at", _now_iso())
        payload.setdefault("namespace", self.namespace)
        task = {
            "id": resolved_id,
            "question": _clean_text(question),
            "metadata": payload,
        }
        self._memory_tasks.setdefault(resolved_id, {}).update(task)
        self._write(
            """
            MERGE (t:GaiaTask {id: $task_id})
            SET t.question = $question,
                t.namespace = $namespace,
                t.updated_at = $updated_at,
                t += $metadata
            """,
            task_id=resolved_id,
            question=task["question"],
            namespace=self.namespace,
            updated_at=_now_iso(),
            metadata=payload,
        )
        return resolved_id

    def classify_task(
        self,
        question: str,
        attachment_type: str | None = None,
    ) -> TaskClassification:
        text = _clean_text(question).lower()
        ext = _slug(attachment_type or "", default="")

        rules: list[tuple[str, list[str], list[str], dict[str, list[str]], float]] = [
            (
                "stochastic_process",
                ["random", "randomly", "probability", "odds", "maximize", "position", "advance"],
                ["missing_state_transition_model", "surface_numeric_guess"],
                {
                    "prefer": ["python_solver"],
                    "optional": ["search"],
                    "avoid": ["calculator_on_raw_question"],
                },
                0.82,
            ),
            (
                "spreadsheet_reasoning",
                ["spreadsheet", "excel", "xlsx", "xls", "sheet", "cell", "row", "column", "color"],
                ["table_scope_mismatch", "missed_attachment_evidence"],
                {
                    "prefer": ["attachment_reader", "pandas_excel"],
                    "optional": ["python_solver"],
                    "avoid": ["raw_text_guess"],
                },
                0.86,
            ),
            (
                "image_understanding",
                ["image", "png", "jpg", "jpeg", "screenshot", "photo", "visual", "picture"],
                ["missed_visual_evidence", "weak_ocr_or_caption"],
                {
                    "prefer": ["attachment_reader", "vision_model"],
                    "optional": ["search"],
                    "avoid": ["text_only_guess"],
                },
                0.82,
            ),
            (
                "audio_understanding",
                ["audio", "mp3", "listen", "transcribe", "recording", "sound"],
                ["missed_audio_evidence", "transcription_error"],
                {
                    "prefer": ["attachment_reader", "audio_transcription"],
                    "optional": ["search"],
                    "avoid": ["text_only_guess"],
                },
                0.82,
            ),
            (
                "counting_scope",
                ["how many", "count", "number of", "total", "list all", "between", "during"],
                ["scope_filter_mismatch", "boundary_condition_slip"],
                {
                    "prefer": ["search"],
                    "optional": ["python_solver"],
                    "avoid": ["candidate_collapse"],
                },
                0.72,
            ),
            (
                "unit_conversion",
                ["unit", "convert", "nearest", "round", "km", "mile", "meter", "kg", "percent"],
                ["unit_or_scale_mismatch", "format_or_rounding_slip"],
                {
                    "prefer": ["python_solver"],
                    "optional": ["calculator"],
                    "avoid": ["unverified_mental_math"],
                },
                0.7,
            ),
            (
                "factual_search",
                ["who", "when", "where", "website", "source", "latest", "current", "published", "released"],
                ["insufficient_evidence", "outdated_fact"],
                {
                    "prefer": ["search"],
                    "optional": ["rag"],
                    "avoid": ["memory_as_answer_lookup"],
                },
                0.68,
            ),
        ]

        if ext in {"xlsx", "xls", "csv"}:
            return TaskClassification(
                task_type="spreadsheet_reasoning",
                trigger_terms=[ext, "spreadsheet"],
                attachment_type=ext,
                failure_modes=["table_scope_mismatch", "missed_attachment_evidence"],
                tool_policy={"prefer": ["attachment_reader", "pandas_excel"], "optional": ["python_solver"], "avoid": ["raw_text_guess"]},
                confidence=0.9,
            )
        if ext in {"png", "jpg", "jpeg", "webp"}:
            return TaskClassification(
                task_type="image_understanding",
                trigger_terms=[ext, "image"],
                attachment_type=ext,
                failure_modes=["missed_visual_evidence", "weak_ocr_or_caption"],
                tool_policy={"prefer": ["attachment_reader", "vision_model"], "optional": ["search"], "avoid": ["text_only_guess"]},
                confidence=0.9,
            )
        if ext in {"mp3", "wav", "m4a"}:
            return TaskClassification(
                task_type="audio_understanding",
                trigger_terms=[ext, "audio"],
                attachment_type=ext,
                failure_modes=["missed_audio_evidence", "transcription_error"],
                tool_policy={"prefer": ["attachment_reader", "audio_transcription"], "optional": ["search"], "avoid": ["text_only_guess"]},
                confidence=0.9,
            )

        best: TaskClassification | None = None
        for task_type, terms, failures, policy, confidence in rules:
            matched = [term for term in terms if term in text]
            if not matched:
                continue
            score = confidence + min(0.12, 0.02 * len(matched))
            if best is None or score > best.confidence:
                best = TaskClassification(
                    task_type=task_type,
                    trigger_terms=matched,
                    attachment_type=ext or None,
                    failure_modes=failures,
                    tool_policy=policy,
                    confidence=min(score, 0.98),
                )

        if best is not None:
            return best

        return TaskClassification(
            task_type="general_reasoning",
            trigger_terms=sorted(_tokenize(question))[:8],
            attachment_type=ext or None,
            failure_modes=["insufficient_verification"],
            tool_policy={"prefer": [], "optional": ["search"], "avoid": ["memory_as_answer_lookup"]},
            confidence=0.45,
        )

    def link_task_signals(self, task_id: str, classification: TaskClassification | dict[str, Any]) -> None:
        data = classification.to_dict() if isinstance(classification, TaskClassification) else dict(classification)
        task_type = _slug(data.get("task_type"))
        trigger_terms = [_slug(term) for term in data.get("trigger_terms", []) if _slug(term)]
        attachment_type = _slug(data.get("attachment_type") or "", default="")
        failure_modes = [_slug(mode) for mode in data.get("failure_modes", []) if _slug(mode)]
        policy = data.get("tool_policy") or {}

        task = self._memory_tasks.setdefault(task_id, {"id": task_id})
        task["classification"] = data
        task["task_type"] = task_type
        task["trigger_terms"] = trigger_terms
        self._memory_edges.extend(
            {"from": task_id, "to": value, "type": edge_type}
            for edge_type, values in [
                ("CLASSIFIED_AS", [task_type]),
                ("HAS_TRIGGER", trigger_terms),
                ("HAS_ATTACHMENT_TYPE", [attachment_type] if attachment_type else []),
                ("FAILED_WITH", failure_modes),
            ]
            for value in values
        )

        self._write(
            """
            MATCH (task:GaiaTask {id: $task_id})
            MERGE (type:GaiaTaskType {name: $task_type})
            MERGE (task)-[:CLASSIFIED_AS]->(type)
            SET task.task_type = $task_type,
                task.classification_confidence = $confidence
            """,
            task_id=task_id,
            task_type=task_type,
            confidence=float(data.get("confidence", 0.0) or 0.0),
        )

        for term in trigger_terms:
            self._write(
                """
                MATCH (task:GaiaTask {id: $task_id})
                MERGE (term:GaiaTriggerTerm {name: $term})
                MERGE (task)-[:HAS_TRIGGER]->(term)
                """,
                task_id=task_id,
                term=term,
            )
        if attachment_type:
            self._write(
                """
                MATCH (task:GaiaTask {id: $task_id})
                MERGE (attachment:GaiaAttachmentType {name: $attachment_type})
                MERGE (task)-[:HAS_ATTACHMENT_TYPE]->(attachment)
                """,
                task_id=task_id,
                attachment_type=attachment_type,
            )
        for mode in failure_modes:
            self._write(
                """
                MATCH (task:GaiaTask {id: $task_id})
                MERGE (mode:GaiaFailureMode {name: $mode})
                MERGE (task)-[:HAS_POSSIBLE_FAILURE]->(mode)
                """,
                task_id=task_id,
                mode=mode,
            )
        for relation, tools in [("PREFERS_TOOL", policy.get("prefer", [])), ("OPTIONAL_TOOL", policy.get("optional", [])), ("AVOIDS_TOOL", policy.get("avoid", []))]:
            for tool in [_slug(item) for item in tools if _slug(item)]:
                self._write(
                    f"""
                    MATCH (type:GaiaTaskType {{name: $task_type}})
                    MERGE (tool:GaiaToolPolicy {{name: $tool}})
                    MERGE (type)-[:{relation}]->(tool)
                    """,
                    task_type=task_type,
                    tool=tool,
                )

    def link_similar_tasks(
        self,
        task_id: str,
        question: str,
        *,
        top_k: int = 5,
        min_weight: float = 0.20,
    ) -> list[dict[str, Any]]:
        candidates = []
        current = self._memory_tasks.get(task_id, {})
        current_type = current.get("task_type")
        current_terms = set(current.get("trigger_terms") or [])

        for other_id, other in self._memory_tasks.items():
            if other_id == task_id:
                continue
            other_question = str(other.get("question", "") or "")
            score = _lexical_similarity(question, other_question)
            if current_type and other.get("task_type") == current_type:
                score += 0.25
            other_terms = set(other.get("trigger_terms") or [])
            if current_terms or other_terms:
                score += 0.20 * (len(current_terms & other_terms) / max(1, len(current_terms | other_terms)))
            if score >= min_weight:
                candidates.append({"task_id": other_id, "weight": min(score, 1.0), "question": other_question})

        if self.available:
            rows = self._read(
                """
                MATCH (current:GaiaTask {id: $task_id})
                MATCH (other:GaiaTask)
                WHERE other.id <> $task_id
                OPTIONAL MATCH (current)-[:CLASSIFIED_AS]->(ct:GaiaTaskType)<-[:CLASSIFIED_AS]-(other)
                OPTIONAL MATCH (current)-[:HAS_TRIGGER]->(trig:GaiaTriggerTerm)<-[:HAS_TRIGGER]-(other)
                WITH other, count(DISTINCT ct) AS type_hits, count(DISTINCT trig) AS trigger_hits
                RETURN other.id AS task_id, other.question AS question,
                       (0.25 * type_hits + 0.08 * trigger_hits) AS graph_weight
                ORDER BY graph_weight DESC
                LIMIT $limit
                """,
                task_id=task_id,
                limit=max(top_k * 3, 10),
            )
            for row in rows:
                weight = float(row.get("graph_weight", 0.0) or 0.0)
                if weight >= min_weight:
                    candidates.append(
                        {
                            "task_id": row.get("task_id"),
                            "weight": min(weight, 1.0),
                            "question": row.get("question", ""),
                        }
                    )

        deduped: dict[str, dict[str, Any]] = {}
        for item in candidates:
            oid = str(item.get("task_id") or "")
            if not oid:
                continue
            if oid not in deduped or item["weight"] > deduped[oid]["weight"]:
                deduped[oid] = item
        selected = sorted(deduped.values(), key=lambda item: item["weight"], reverse=True)[:top_k]

        for item in selected:
            self._memory_edges.append({"from": task_id, "to": item["task_id"], "type": "SIMILAR_TO", "weight": item["weight"]})
            self._write(
                """
                MATCH (a:GaiaTask {id: $task_id})
                MATCH (b:GaiaTask {id: $other_id})
                MERGE (a)-[r:SIMILAR_TO]->(b)
                SET r.weight = $weight,
                    r.updated_at = $updated_at
                """,
                task_id=task_id,
                other_id=item["task_id"],
                weight=float(item["weight"]),
                updated_at=_now_iso(),
            )
        return selected

    def update_task_outcome(
        self,
        task_id: str,
        *,
        stage1_result: str | None = None,
        final_result: str | None = None,
        expected: str | None = None,
        exact: bool | None = None,
        partial: bool | None = None,
        failure_mode: str | None = None,
        tools_used: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        task = self._memory_tasks.setdefault(task_id, {"id": task_id})
        task.update(
            {
                "stage1_result": stage1_result,
                "final_result": final_result,
                "expected": expected,
                "exact": exact,
                "partial": partial,
                "failure_mode": _slug(failure_mode or "", default="") or None,
                "tools_used": [_slug(tool) for tool in (tools_used or [])],
                "outcome_metadata": dict(metadata or {}),
            }
        )
        self._write(
            """
            MATCH (task:GaiaTask {id: $task_id})
            SET task.stage1_result = $stage1_result,
                task.final_result = $final_result,
                task.expected = $expected,
                task.exact = $exact,
                task.partial = $partial,
                task.failure_mode = $failure_mode,
                task.tools_used = $tools_used,
                task.updated_at = $updated_at
            """,
            task_id=task_id,
            stage1_result=stage1_result,
            final_result=final_result,
            expected=expected,
            exact=exact,
            partial=partial,
            failure_mode=_slug(failure_mode or "", default=""),
            tools_used=[_slug(tool) for tool in (tools_used or [])],
            updated_at=_now_iso(),
        )
        if failure_mode:
            self._write(
                """
                MATCH (task:GaiaTask {id: $task_id})
                MERGE (mode:GaiaFailureMode {name: $mode})
                MERGE (task)-[:FAILED_WITH]->(mode)
                """,
                task_id=task_id,
                mode=_slug(failure_mode),
            )

    def retrieve_for_stage1_round0(
        self,
        task_id: str | None,
        question: str,
        *,
        limit: int = 5,
    ) -> dict[str, Any]:
        resolved_id = _clean_text(task_id) or _task_id_from_question(question)
        task = self._memory_tasks.get(resolved_id)
        if task is None:
            resolved_id = self.register_task(resolved_id, question)
            classification = self.classify_task(question)
            self.link_task_signals(resolved_id, classification)
            task = self._memory_tasks.get(resolved_id, {})

        similar = self.link_similar_tasks(resolved_id, question, top_k=limit)
        similar_failures = []
        for item in similar:
            other = self._memory_tasks.get(str(item.get("task_id")), {})
            failure_mode = other.get("failure_mode")
            if failure_mode or other.get("exact") is False:
                similar_failures.append(
                    {
                        "task_id": item.get("task_id"),
                        "similarity": item.get("weight"),
                        "failure_mode": failure_mode or "previous_wrong_answer",
                        "summary": self._summarize_similar_task(other),
                    }
                )

        if self.available:
            rows = self._read(
                """
                MATCH (task:GaiaTask {id: $task_id})-[sim:SIMILAR_TO]->(other:GaiaTask)
                WHERE coalesce(other.exact, false) = false OR other.failure_mode IS NOT NULL
                RETURN other.id AS task_id, sim.weight AS similarity,
                       other.failure_mode AS failure_mode,
                       other.stage1_result AS stage1_result,
                       other.final_result AS final_result,
                       other.expected AS expected
                ORDER BY sim.weight DESC
                LIMIT $limit
                """,
                task_id=resolved_id,
                limit=limit,
            )
            for row in rows:
                similar_failures.append(
                    {
                        "task_id": row.get("task_id"),
                        "similarity": row.get("similarity"),
                        "failure_mode": row.get("failure_mode") or "previous_wrong_answer",
                        "summary": self._summarize_similar_task(row),
                    }
                )

        classification = task.get("classification") or self.classify_task(question).to_dict()
        return {
            "task_id": resolved_id,
            "task_type": classification.get("task_type", "general_reasoning"),
            "trigger_terms": list(classification.get("trigger_terms", []))[:8],
            "attachment_type": classification.get("attachment_type"),
            "failure_modes": list(classification.get("failure_modes", []))[:5],
            "tool_policy": classification.get("tool_policy", {}),
            "similar_failures": similar_failures[:limit],
        }

    def _summarize_similar_task(self, task: dict[str, Any]) -> str:
        failure_mode = task.get("failure_mode") or task.get("failure_mode".upper()) or "unknown_failure"
        stage1 = task.get("stage1_result")
        final = task.get("final_result")
        expected = task.get("expected")
        pieces = [f"failure_mode={failure_mode}"]
        if stage1:
            pieces.append(f"stage1={stage1}")
        if final:
            pieces.append(f"final={final}")
        if expected:
            pieces.append("expected was different")
        return "; ".join(pieces)

    def build_stage1_guidance_prompt(
        self,
        retrieval: dict[str, Any],
        *,
        insights: list[dict[str, Any]] | None = None,
        max_failures: int = 1,
    ) -> str:
        lines = ["Relevant Memory Guidance:"]
        task_type = retrieval.get("task_type")
        if task_type:
            lines.append(f"- Task type: {task_type}")
        terms = retrieval.get("trigger_terms") or []
        if terms:
            lines.append(f"- Trigger terms: {', '.join(map(str, terms[:8]))}")

        for insight in (insights or [])[:3]:
            strategy = _clean_text(insight.get("strategy", ""))
            if strategy:
                lines.append(f"- Strategy: {strategy}")
            checklist = insight.get("checklist") or []
            if checklist:
                compact = "; ".join(_clean_text(item) for item in checklist[:4] if _clean_text(item))
                if compact:
                    lines.append(f"- Checklist: {compact}")

        for failure in (retrieval.get("similar_failures") or [])[:max_failures]:
            summary = _clean_text(failure.get("summary", ""))
            if summary:
                lines.append(f"- Similar failure warning: {summary}")

        policy = retrieval.get("tool_policy") or {}
        prefer = policy.get("prefer") or []
        avoid = policy.get("avoid") or []
        policy_bits = []
        if prefer:
            policy_bits.append(f"prefer {', '.join(map(str, prefer[:4]))}")
        if avoid:
            policy_bits.append(f"avoid {', '.join(map(str, avoid[:4]))}")
        if policy_bits:
            lines.append(f"- Tool policy for later repair: {'; '.join(policy_bits)}")

        return "\n".join(lines).strip()
