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


def _dedupe(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _slug(value, default="")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _make_insight_id(task_type: str, strategy: str) -> str:
    digest = hashlib.sha1(
        f"{_slug(task_type)}|{_clean_text(strategy).lower()}".encode("utf-8", errors="ignore")
    ).hexdigest()[:12]
    return f"gaia_insight_{_slug(task_type)}_{digest}"


@dataclass(slots=True)
class InsightRecord:
    insight_id: str
    task_type: str
    strategy: str
    trigger_terms: list[str] = field(default_factory=list)
    checklist: list[str] = field(default_factory=list)
    tool_policy: dict[str, list[str]] = field(default_factory=dict)
    failure_modes: list[str] = field(default_factory=list)
    score: float = 2.0
    support_count: int = 0
    contradiction_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InsightRecord":
        task_type = _slug(data.get("task_type"))
        strategy = _clean_text(data.get("strategy"))
        insight_id = _clean_text(data.get("insight_id")) or _make_insight_id(task_type, strategy)
        policy = data.get("tool_policy") or {}
        return cls(
            insight_id=insight_id,
            task_type=task_type,
            strategy=strategy,
            trigger_terms=_dedupe(list(data.get("trigger_terms") or [])),
            checklist=[_clean_text(item) for item in list(data.get("checklist") or []) if _clean_text(item)],
            tool_policy={
                "prefer": _dedupe(list(policy.get("prefer") or [])),
                "optional": _dedupe(list(policy.get("optional") or [])),
                "avoid": _dedupe(list(policy.get("avoid") or [])),
            },
            failure_modes=_dedupe(list(data.get("failure_modes") or [])),
            score=float(data.get("score", 2.0) or 2.0),
            support_count=int(data.get("support_count", 0) or 0),
            contradiction_count=int(data.get("contradiction_count", 0) or 0),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "task_type": self.task_type,
            "strategy": self.strategy,
            "trigger_terms": list(self.trigger_terms),
            "checklist": list(self.checklist),
            "tool_policy": {
                "prefer": list(self.tool_policy.get("prefer", [])),
                "optional": list(self.tool_policy.get("optional", [])),
                "avoid": list(self.tool_policy.get("avoid", [])),
            },
            "failure_modes": list(self.failure_modes),
            "score": self.score,
            "support_count": self.support_count,
            "contradiction_count": self.contradiction_count,
            "metadata": dict(self.metadata),
        }


class InsightGraph:
    """Reusable strategy insight graph for GAIA tasks."""

    def __init__(
        self,
        graph_store: Any | None = None,
        *,
        auto_connect: bool = True,
        seed_defaults: bool = True,
        namespace: str = "gaia",
    ):
        self.namespace = namespace
        self.graph_store = graph_store
        self._memory_insights: dict[str, InsightRecord] = {}
        if self.graph_store is None and auto_connect:
            self.graph_store = self._create_graph_store()
        if seed_defaults:
            self.seed_default_insights()

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
            logger.warning("InsightGraph Neo4j unavailable; using memory fallback: %s", exc)
            return None

    def _create_indexes(self, store: Any | None = None) -> None:
        store = store or self.graph_store
        if not getattr(store, "driver", None):
            return
        queries = [
            "CREATE INDEX gaia_insight_id_index IF NOT EXISTS FOR (i:GaiaInsight) ON (i.id)",
            "CREATE INDEX gaia_insight_task_type_index IF NOT EXISTS FOR (i:GaiaInsight) ON (i.task_type)",
            "CREATE INDEX gaia_insight_score_index IF NOT EXISTS FOR (i:GaiaInsight) ON (i.score)",
            "CREATE INDEX gaia_checklist_id_index IF NOT EXISTS FOR (c:GaiaChecklistItem) ON (c.id)",
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
            logger.warning("InsightGraph write failed: %s", exc)
            return False

    def _read(self, query: str, **params: Any) -> list[dict[str, Any]]:
        if not self.available:
            return []
        try:
            with self.graph_store.driver.session(database=self.graph_store.database) as session:
                result = session.run(query, **params)
                return [dict(record) for record in result]
        except Exception as exc:
            logger.warning("InsightGraph read failed: %s", exc)
            return []

    def seed_default_insights(self) -> None:
        for insight in self.default_insights():
            self.upsert_insight(insight)

    def default_insights(self) -> list[dict[str, Any]]:
        return [
            {
                "insight_id": "gaia_insight_stochastic_process_state_transition",
                "task_type": "stochastic_process",
                "trigger_terms": ["random", "randomly", "probability", "odds", "position", "advance"],
                "strategy": "For stochastic process puzzles, define states and transition rules before selecting a numeric answer.",
                "checklist": [
                    "Identify all state variables.",
                    "Define transition probabilities.",
                    "Compare candidate outcomes with a model.",
                    "Use simulation or dynamic programming if manual reasoning is uncertain.",
                ],
                "tool_policy": {
                    "prefer": ["python_solver"],
                    "optional": ["search"],
                    "avoid": ["calculator_on_raw_question"],
                },
                "failure_modes": ["surface_numeric_guess", "missing_state_transition_model"],
                "score": 3.0,
            },
            {
                "insight_id": "gaia_insight_counting_scope_boundaries",
                "task_type": "counting_scope",
                "trigger_terms": ["count", "how many", "number of", "between", "during"],
                "strategy": "For counting tasks, define the inclusion scope before trusting repeated candidate answers.",
                "checklist": [
                    "List inclusion and exclusion criteria.",
                    "Check boundaries such as dates, ranges, and aliases.",
                    "Verify the count against evidence rather than majority guesses.",
                ],
                "tool_policy": {
                    "prefer": ["search"],
                    "optional": ["python_solver"],
                    "avoid": ["candidate_collapse"],
                },
                "failure_modes": ["scope_filter_mismatch", "boundary_condition_slip"],
                "score": 2.5,
            },
            {
                "insight_id": "gaia_insight_spreadsheet_attachment_first",
                "task_type": "spreadsheet_reasoning",
                "trigger_terms": ["xlsx", "xls", "spreadsheet", "sheet", "cell", "color"],
                "strategy": "For spreadsheet tasks, inspect the attachment structure before answering from surface text.",
                "checklist": [
                    "Read workbook sheets and dimensions.",
                    "Route color-grid questions to compact color summaries.",
                    "Use pandas for .xls and data-heavy sheets.",
                    "Keep row/column references explicit.",
                ],
                "tool_policy": {
                    "prefer": ["attachment_reader", "pandas_excel"],
                    "optional": ["python_solver"],
                    "avoid": ["raw_text_guess"],
                },
                "failure_modes": ["table_scope_mismatch", "missed_attachment_evidence"],
                "score": 2.5,
            },
            {
                "insight_id": "gaia_insight_factual_search_evidence",
                "task_type": "factual_search",
                "trigger_terms": ["who", "when", "where", "website", "latest", "published"],
                "strategy": "For factual lookup tasks, base the answer on structured search evidence and cite the specific fact used.",
                "checklist": [
                    "Search for primary or high-quality sources.",
                    "Extract the exact entity/date/value requested.",
                    "Avoid copying old memory answers without current evidence.",
                ],
                "tool_policy": {
                    "prefer": ["search"],
                    "optional": ["rag"],
                    "avoid": ["memory_as_answer_lookup"],
                },
                "failure_modes": ["insufficient_evidence", "outdated_fact"],
                "score": 2.2,
            },
            {
                "insight_id": "gaia_insight_audio_attachment_transcribe",
                "task_type": "audio_understanding",
                "trigger_terms": ["audio", "mp3", "transcribe", "listen"],
                "strategy": "For audio tasks, transcribe the attachment first and answer only after checking the transcript against the question.",
                "checklist": [
                    "Run audio transcription.",
                    "Keep uncertain words marked as uncertain.",
                    "Search only if the transcript refers to external facts.",
                ],
                "tool_policy": {
                    "prefer": ["attachment_reader", "audio_transcription"],
                    "optional": ["search"],
                    "avoid": ["text_only_guess"],
                },
                "failure_modes": ["missed_audio_evidence", "transcription_error"],
                "score": 2.2,
            },
            {
                "insight_id": "gaia_insight_image_attachment_vision",
                "task_type": "image_understanding",
                "trigger_terms": ["image", "png", "jpg", "screenshot", "visual"],
                "strategy": "For image tasks, convert visual evidence into concise text before reasoning.",
                "checklist": [
                    "Use a vision model or OCR-capable reader.",
                    "Separate observed text from visual inference.",
                    "Do not answer from filename or question text alone.",
                ],
                "tool_policy": {
                    "prefer": ["attachment_reader", "vision_model"],
                    "optional": ["search"],
                    "avoid": ["text_only_guess"],
                },
                "failure_modes": ["missed_visual_evidence", "weak_ocr_or_caption"],
                "score": 2.2,
            },
        ]

    def upsert_insight(self, insight: dict[str, Any] | InsightRecord) -> InsightRecord:
        record = insight if isinstance(insight, InsightRecord) else InsightRecord.from_dict(insight)
        self._memory_insights[record.insight_id] = record
        self._write(
            """
            MERGE (insight:GaiaInsight {id: $insight_id})
            SET insight.task_type = $task_type,
                insight.strategy = $strategy,
                insight.trigger_terms = $trigger_terms,
                insight.checklist = $checklist,
                insight.failure_modes = $failure_modes,
                insight.score = $score,
                insight.support_count = $support_count,
                insight.contradiction_count = $contradiction_count,
                insight.namespace = $namespace,
                insight.updated_at = $updated_at
            MERGE (type:GaiaTaskType {name: $task_type})
            MERGE (insight)-[:APPLIES_TO]->(type)
            """,
            insight_id=record.insight_id,
            task_type=record.task_type,
            strategy=record.strategy,
            trigger_terms=record.trigger_terms,
            checklist=record.checklist,
            failure_modes=record.failure_modes,
            score=record.score,
            support_count=record.support_count,
            contradiction_count=record.contradiction_count,
            namespace=self.namespace,
            updated_at=_now_iso(),
        )
        self._link_insight_signals(record)
        return record

    def _link_insight_signals(self, record: InsightRecord) -> None:
        for term in record.trigger_terms:
            self._write(
                """
                MATCH (insight:GaiaInsight {id: $insight_id})
                MERGE (term:GaiaTriggerTerm {name: $term})
                MERGE (insight)-[:TRIGGERED_BY]->(term)
                """,
                insight_id=record.insight_id,
                term=term,
            )
        for mode in record.failure_modes:
            self._write(
                """
                MATCH (insight:GaiaInsight {id: $insight_id})
                MERGE (mode:GaiaFailureMode {name: $mode})
                MERGE (insight)-[:AVOIDS_FAILURE]->(mode)
                """,
                insight_id=record.insight_id,
                mode=mode,
            )
        for idx, item in enumerate(record.checklist):
            item_id = f"{record.insight_id}_check_{idx}"
            self._write(
                """
                MATCH (insight:GaiaInsight {id: $insight_id})
                MERGE (check:GaiaChecklistItem {id: $item_id})
                SET check.text = $text,
                    check.order = $order
                MERGE (insight)-[:HAS_CHECKLIST]->(check)
                """,
                insight_id=record.insight_id,
                item_id=item_id,
                text=item,
                order=idx,
            )
        for relation, tools in [
            ("PREFERS_TOOL", record.tool_policy.get("prefer", [])),
            ("OPTIONAL_TOOL", record.tool_policy.get("optional", [])),
            ("AVOIDS_TOOL", record.tool_policy.get("avoid", [])),
        ]:
            for tool in tools:
                self._write(
                    f"""
                    MATCH (insight:GaiaInsight {{id: $insight_id}})
                    MERGE (tool:GaiaToolPolicy {{name: $tool}})
                    MERGE (insight)-[:{relation}]->(tool)
                    """,
                    insight_id=record.insight_id,
                    tool=tool,
                )

    def retrieve_insights(
        self,
        task_type: str,
        trigger_terms: list[str] | None = None,
        failure_modes: list[str] | None = None,
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        task_type_slug = _slug(task_type)
        triggers = set(_dedupe(trigger_terms or []))
        failures = set(_dedupe(failure_modes or []))

        candidates: list[tuple[float, InsightRecord]] = []
        for record in self._memory_insights.values():
            score = float(record.score)
            if record.task_type == task_type_slug:
                score += 2.0
            trigger_overlap = len(triggers & set(record.trigger_terms))
            failure_overlap = len(failures & set(record.failure_modes))
            score += 0.35 * trigger_overlap + 0.45 * failure_overlap
            if record.task_type == task_type_slug or trigger_overlap or failure_overlap:
                candidates.append((score, record))

        if self.available:
            rows = self._read(
                """
                MATCH (insight:GaiaInsight)
                WHERE insight.task_type = $task_type
                   OR any(term IN coalesce(insight.trigger_terms, []) WHERE term IN $trigger_terms)
                   OR any(mode IN coalesce(insight.failure_modes, []) WHERE mode IN $failure_modes)
                RETURN insight.id AS insight_id,
                       insight.task_type AS task_type,
                       insight.strategy AS strategy,
                       insight.trigger_terms AS trigger_terms,
                       insight.checklist AS checklist,
                       insight.failure_modes AS failure_modes,
                       insight.score AS score,
                       insight.support_count AS support_count,
                       insight.contradiction_count AS contradiction_count
                ORDER BY insight.score DESC
                LIMIT $limit
                """,
                task_type=task_type_slug,
                trigger_terms=list(triggers),
                failure_modes=list(failures),
                limit=max(limit * 2, 6),
            )
            for row in rows:
                record = self._memory_insights.get(str(row.get("insight_id") or ""))
                if record is None:
                    record = InsightRecord.from_dict(
                        {
                            "insight_id": row.get("insight_id"),
                            "task_type": row.get("task_type"),
                            "strategy": row.get("strategy"),
                            "trigger_terms": row.get("trigger_terms") or [],
                            "checklist": row.get("checklist") or [],
                            "failure_modes": row.get("failure_modes") or [],
                            "score": row.get("score") or 0,
                            "support_count": row.get("support_count") or 0,
                            "contradiction_count": row.get("contradiction_count") or 0,
                        }
                    )
                score = float(record.score)
                if record.task_type == task_type_slug:
                    score += 2.0
                score += 0.35 * len(triggers & set(record.trigger_terms))
                score += 0.45 * len(failures & set(record.failure_modes))
                candidates.append((score, record))

        deduped: dict[str, tuple[float, InsightRecord]] = {}
        for score, record in candidates:
            if record.insight_id not in deduped or score > deduped[record.insight_id][0]:
                deduped[record.insight_id] = (score, record)

        selected = sorted(deduped.values(), key=lambda item: item[0], reverse=True)[:limit]
        return [record.to_dict() | {"match_score": score} for score, record in selected]

    def apply_feedback(
        self,
        insight_id: str,
        task_id: str,
        *,
        exact: bool,
        partial: bool = False,
        stage2_changed_answer: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        record = self._memory_insights.get(insight_id)
        reward = 1.0 if exact else (0.35 if partial else -0.75)
        if stage2_changed_answer and exact:
            reward += 0.35
        if record is not None:
            record.score = max(0.0, record.score + reward)
            if exact or partial:
                record.support_count += 1
            else:
                record.contradiction_count += 1
        relation = "SUPPORTED_BY" if exact or partial else "CONTRADICTED_BY"
        self._write(
            f"""
            MATCH (insight:GaiaInsight {{id: $insight_id}})
            MERGE (task:GaiaTask {{id: $task_id}})
            MERGE (insight)-[r:{relation}]->(task)
            SET r.reward = $reward,
                r.metadata = $metadata,
                r.updated_at = $updated_at
            SET insight.score = coalesce(insight.score, 0.0) + $reward,
                insight.support_count = coalesce(insight.support_count, 0) + $support_delta,
                insight.contradiction_count = coalesce(insight.contradiction_count, 0) + $contradiction_delta,
                insight.updated_at = $updated_at
            """,
            insight_id=insight_id,
            task_id=task_id,
            reward=reward,
            metadata=str(dict(metadata or {})),
            updated_at=_now_iso(),
            support_delta=1 if exact or partial else 0,
            contradiction_delta=0 if exact or partial else 1,
        )

    def should_generate_insights(self, finished_task_count: int, interval: int = 5) -> bool:
        return finished_task_count > 0 and finished_task_count % max(1, interval) == 0

    def generate_insight_candidates(
        self,
        success_cases: list[dict[str, Any]],
        failure_cases: list[dict[str, Any]],
        *,
        llm_callable: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Generate insight candidates.

        First version is deliberately conservative. If no LLM callable is
        provided, it returns heuristic candidates from observed failure modes.
        """
        if llm_callable is None:
            candidates: list[dict[str, Any]] = []
            grouped: dict[str, set[str]] = {}
            for case in failure_cases:
                task_type = _slug(case.get("task_type") or "general_reasoning")
                failure = _slug(case.get("failure_mode") or "insufficient_verification")
                grouped.setdefault(task_type, set()).add(failure)
            for task_type, failures in grouped.items():
                candidates.append(
                    {
                        "task_type": task_type,
                        "trigger_terms": [],
                        "strategy": f"For {task_type} tasks, explicitly check for {', '.join(sorted(failures))} before finalizing the answer.",
                        "checklist": [f"Check failure mode: {failure}" for failure in sorted(failures)[:4]],
                        "tool_policy": {"prefer": [], "optional": ["search"], "avoid": ["memory_as_answer_lookup"]},
                        "failure_modes": sorted(failures),
                        "score": 1.5,
                    }
                )
            return candidates

        prompt = self._build_generation_prompt(success_cases, failure_cases)
        raw = llm_callable(prompt)
        return self._parse_json_list(raw)

    def _build_generation_prompt(
        self,
        success_cases: list[dict[str, Any]],
        failure_cases: list[dict[str, Any]],
    ) -> str:
        return (
            "Compare the successful and failed GAIA task cases. "
            "Return JSON list of reusable strategy insights. "
            "Each item must contain task_type, trigger_terms, strategy, checklist, "
            "tool_policy, and failure_modes.\n\n"
            f"Success cases:\n{success_cases}\n\nFailure cases:\n{failure_cases}"
        )

    def _parse_json_list(self, raw: Any) -> list[dict[str, Any]]:
        import json

        text = _clean_text(raw)
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            match = re.search(r"\[.*\]", text, flags=re.DOTALL)
            if not match:
                return []
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []

    def build_strategy_prompt(self, insights: list[dict[str, Any]], *, limit: int = 3) -> str:
        lines: list[str] = []
        for insight in insights[:limit]:
            strategy = _clean_text(insight.get("strategy", ""))
            if strategy:
                lines.append(f"- Strategy: {strategy}")
            checklist = insight.get("checklist") or []
            if checklist:
                compact = "; ".join(_clean_text(item) for item in checklist[:4] if _clean_text(item))
                if compact:
                    lines.append(f"  Checklist: {compact}")
            policy = insight.get("tool_policy") or {}
            prefer = policy.get("prefer") or []
            avoid = policy.get("avoid") or []
            bits = []
            if prefer:
                bits.append(f"prefer {', '.join(prefer[:4])}")
            if avoid:
                bits.append(f"avoid {', '.join(avoid[:4])}")
            if bits:
                lines.append(f"  Tool policy: {'; '.join(bits)}")
        return "\n".join(lines).strip()
