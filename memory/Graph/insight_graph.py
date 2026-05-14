from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """
    負責執行 memory.graph.insight_graph 中的 _now_iso 流程，依照 memory.graph.insight_graph 的流程需求處理 _now_iso 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return datetime.now().isoformat(timespec="seconds")


def _clean_text(value: Any) -> str:
    """
    負責執行 memory.graph.insight_graph 中的 _clean_text 流程，依照 memory.graph.insight_graph 的流程需求處理 _clean_text 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        value: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _slug(value: Any, *, default: str = "unknown") -> str:
    """
    負責執行 memory.graph.insight_graph 中的 _slug 流程，依照 memory.graph.insight_graph 的流程需求處理 _slug 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        value: 記憶系統提供的檢索結果、寫入資料或操作介面。
        default: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    text = _clean_text(value).lower()
    text = re.sub(r"[^a-z0-9_./:-]+", "_", text)
    text = text.strip("_")
    return text or default


def _dedupe(values: list[Any]) -> list[str]:
    """
    負責執行 memory.graph.insight_graph 中的 _dedupe 流程，依照 memory.graph.insight_graph 的流程需求處理 _dedupe 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        values: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 list[str]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _slug(value, default="")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _dedupe_text(values: list[Any]) -> list[str]:
    """
    負責執行 memory.graph.insight_graph 中的 _dedupe_text 流程，依照 memory.graph.insight_graph 的流程需求處理 _dedupe_text 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        values: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 list[str]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _tokenize(text: str) -> set[str]:
    """
    負責執行 memory.graph.insight_graph 中的 _tokenize 流程，依照 memory.graph.insight_graph 的流程需求處理 _tokenize 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        text: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 set[str]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
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
        "before",
        "after",
        "answer",
        "task",
        "tasks",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9_./:-]+", _clean_text(text).lower())
        if len(token) > 2 and token not in stopwords
    }


def _lexical_similarity(a: str, b: str) -> float:
    """
    負責執行 memory.graph.insight_graph 中的 _lexical_similarity 流程，依照 memory.graph.insight_graph 的流程需求處理 _lexical_similarity 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        a: 記憶系統提供的檢索結果、寫入資料或操作介面。
        b: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 float。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    left = _tokenize(a)
    right = _tokenize(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _make_insight_id(task_type: str, strategy: str) -> str:
    """
    負責執行 memory.graph.insight_graph 中的 _make_insight_id 流程，依照 memory.graph.insight_graph 的流程需求處理 _make_insight_id 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        task_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
        strategy: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    digest = hashlib.sha1(
        f"{_slug(task_type)}|{_clean_text(strategy).lower()}".encode("utf-8", errors="ignore")
    ).hexdigest()[:12]
    return f"gaia_insight_{_slug(task_type)}_{digest}"


@dataclass(slots=True)
class InsightRecord:
    """
    負責在 memory.graph.insight_graph 中封裝 InsightRecord，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    insight_id: str
    rule: str
    task_type: str = "general_reasoning"
    strategy: str = ""
    trigger_terms: list[str] = field(default_factory=list)
    checklist: list[str] = field(default_factory=list)
    tool_policy: dict[str, list[str]] = field(default_factory=dict)
    failure_modes: list[str] = field(default_factory=list)
    score: float = 2.0
    positive_correlation_tasks: list[str] = field(default_factory=list)
    negative_correlation_tasks: list[str] = field(default_factory=list)
    support_count: int = 0
    contradiction_count: int = 0
    status: str = "active"
    confidence: float = 0.5
    created_from: str = "manual_seed"
    evidence_task_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InsightRecord":
        """
        負責執行 InsightRecord 中的 from_dict 流程，依照 InsightRecord 的流程需求處理 from_dict 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            data: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 'InsightRecord'。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        task_type = _slug(data.get("task_type"))
        rule = _clean_text(data.get("rule") or data.get("strategy"))
        strategy = _clean_text(data.get("strategy") or rule)
        insight_id = _clean_text(data.get("insight_id")) or _make_insight_id(task_type, rule or strategy)
        policy = data.get("tool_policy") or {}
        return cls(
            insight_id=insight_id,
            rule=rule,
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
            positive_correlation_tasks=_dedupe(list(data.get("positive_correlation_tasks") or [])),
            negative_correlation_tasks=_dedupe(list(data.get("negative_correlation_tasks") or [])),
            support_count=int(data.get("support_count", 0) or 0),
            contradiction_count=int(data.get("contradiction_count", 0) or 0),
            status=_slug(data.get("status") or "active"),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5) or 0.5))),
            created_from=_slug(data.get("created_from") or "manual_seed"),
            evidence_task_ids=_dedupe(list(data.get("evidence_task_ids") or [])),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        負責執行 InsightRecord 中的 to_dict 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return {
            "insight_id": self.insight_id,
            "rule": self.rule,
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
            "positive_correlation_tasks": list(self.positive_correlation_tasks),
            "negative_correlation_tasks": list(self.negative_correlation_tasks),
            "support_count": self.support_count,
            "contradiction_count": self.contradiction_count,
            "status": self.status,
            "confidence": self.confidence,
            "created_from": self.created_from,
            "evidence_task_ids": list(self.evidence_task_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class InsightCandidateEvent:
    """
    負責在 memory.graph.insight_graph 中封裝 InsightCandidateEvent，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    task_id: str
    question: str
    task_type: str = "general_reasoning"
    failure_mode: str = "insufficient_verification"
    predicted: str = ""
    expected: str = ""
    related_task_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InsightCandidate:
    """
    負責在 memory.graph.insight_graph 中封裝 InsightCandidate，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    rule: str
    failure_type: str = "insufficient_verification"
    applies_when: list[str] = field(default_factory=list)
    avoid_when: list[str] = field(default_factory=list)
    recommended_action: str = ""
    evidence_task_ids: list[str] = field(default_factory=list)
    confidence: float = 0.55
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InsightCandidate":
        """
        負責執行 InsightCandidate 中的 from_dict 流程，依照 InsightCandidate 的流程需求處理 from_dict 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            data: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 'InsightCandidate'。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return cls(
            rule=_clean_text(data.get("rule")),
            failure_type=_slug(data.get("failure_type") or data.get("failure_mode") or "insufficient_verification"),
            applies_when=_dedupe(list(data.get("applies_when") or data.get("trigger_terms") or [])),
            avoid_when=_dedupe(list(data.get("avoid_when") or [])),
            recommended_action=_slug(data.get("recommended_action") or ""),
            evidence_task_ids=_dedupe(list(data.get("evidence_task_ids") or [])),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.55) or 0.55))),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_insight_dict(self, *, task_type: str, source_task_id: str | None = None) -> dict[str, Any]:
        """
        負責執行 InsightCandidate 中的 to_insight_dict 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            task_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            source_task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        evidence = list(self.evidence_task_ids)
        if source_task_id:
            evidence.append(source_task_id)
        checklist = []
        if self.recommended_action:
            checklist.append(f"Apply action: {self.recommended_action}")
        checklist.append(f"Check failure mode: {self.failure_type}")
        return {
            "rule": self.rule,
            "task_type": task_type,
            "strategy": self.rule,
            "trigger_terms": self.applies_when,
            "checklist": checklist,
            "tool_policy": {"prefer": [], "optional": ["search"], "avoid": ["memory_as_answer_lookup"]},
            "failure_modes": [self.failure_type],
            "score": max(1.0, 1.0 + self.confidence),
            "confidence": self.confidence,
            "created_from": "llm_candidate" if self.metadata.get("llm_generated") else "heuristic_candidate",
            "positive_correlation_tasks": _dedupe(evidence),
            "evidence_task_ids": _dedupe(evidence),
            "metadata": {"candidate": self.metadata, "candidate_rules": [self.rule]},
        }


class InsightGraph:
    """
    負責在 memory.graph.insight_graph 中封裝 InsightGraph，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        graph_store: 記憶系統提供的檢索結果、寫入資料或操作介面。
        auto_connect: 記憶系統提供的檢索結果、寫入資料或操作介面。
        seed_defaults: 記憶系統提供的檢索結果、寫入資料或操作介面。
        namespace: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(
        self,
        graph_store: Any | None = None,
        *,
        auto_connect: bool = True,
        seed_defaults: bool = True,
        namespace: str = "gaia",
    ):
        """
        負責執行 InsightGraph 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            graph_store: 記憶系統提供的檢索結果、寫入資料或操作介面。
            auto_connect: 記憶系統提供的檢索結果、寫入資料或操作介面。
            seed_defaults: 記憶系統提供的檢索結果、寫入資料或操作介面。
            namespace: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.namespace = namespace
        self.graph_store = graph_store
        self._memory_insights: dict[str, InsightRecord] = {}
        if self.graph_store is None and auto_connect:
            self.graph_store = self._create_graph_store()
        if seed_defaults:
            self.seed_default_insights()

    @property
    def available(self) -> bool:
        """
        負責執行 InsightGraph 中的 available 流程，依照 InsightGraph 的流程需求處理 available 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return bool(getattr(self.graph_store, "driver", None))

    def _create_graph_store(self) -> Any | None:
        """
        負責執行 InsightGraph 中的 _create_graph_store 流程，依照 InsightGraph 的流程需求處理 _create_graph_store 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Any | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 InsightGraph 中的 _create_indexes 流程，依照 InsightGraph 的流程需求處理 _create_indexes 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            store: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        store = store or self.graph_store
        if not getattr(store, "driver", None):
            return
        migrations = [
            "MATCH (n:GaiaInsight) SET n:MemoryInsight",
            "MATCH (n:GaiaTask) SET n:MemoryTask",
            "MATCH (n:GaiaTaskType) SET n:MemoryTaskType",
            "MATCH (n:GaiaTriggerTerm) SET n:MemoryTriggerTerm",
            "MATCH (n:GaiaFailureMode) SET n:MemoryFailureMode",
            "MATCH (n:GaiaToolPolicy) SET n:MemoryToolPolicy",
            "MATCH (n:GaiaChecklistItem) SET n:MemoryChecklistItem",
        ]
        queries = [
            "CREATE INDEX memory_insight_id_index IF NOT EXISTS FOR (i:MemoryInsight) ON (i.id)",
            "CREATE INDEX memory_insight_task_type_index IF NOT EXISTS FOR (i:MemoryInsight) ON (i.task_type)",
            "CREATE INDEX memory_insight_score_index IF NOT EXISTS FOR (i:MemoryInsight) ON (i.score)",
            "CREATE INDEX memory_insight_rule_index IF NOT EXISTS FOR (i:MemoryInsight) ON (i.rule)",
            "CREATE INDEX memory_insight_status_index IF NOT EXISTS FOR (i:MemoryInsight) ON (i.status)",
            "CREATE INDEX memory_checklist_id_index IF NOT EXISTS FOR (c:MemoryChecklistItem) ON (c.id)",
        ]
        with store.driver.session(database=store.database) as session:
            for query in migrations:
                session.run(query)
            for query in queries:
                session.run(query)

    def _write(self, query: str, **params: Any) -> bool:
        """
        負責執行 InsightGraph 中的 _write 流程，依照 InsightGraph 的流程需求處理 _write 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            query: 目前要處理的任務、問題或查詢文字。
            **params: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 InsightGraph 中的 _read 流程，依照 InsightGraph 的流程需求處理 _read 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            query: 目前要處理的任務、問題或查詢文字。
            **params: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 InsightGraph 中的 seed_default_insights 流程，依照 InsightGraph 的流程需求處理 seed_default_insights 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        for insight in self.default_insights():
            self.upsert_insight(insight)

    def default_insights(self) -> list[dict[str, Any]]:
        """
        負責執行 InsightGraph 中的 default_insights 流程，依照 InsightGraph 的流程需求處理 default_insights 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return [
            {
                "insight_id": "gaia_insight_stochastic_process_state_transition",
                "rule": "For stochastic process puzzles, define states and transition rules before selecting a numeric answer.",
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
                "rule": "For counting tasks, define the inclusion scope before trusting repeated candidate answers.",
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
                "rule": "For spreadsheet tasks, inspect the attachment structure before answering from surface text.",
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
                "rule": "For factual lookup tasks, base the answer on structured search evidence and cite the specific fact used.",
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
                "rule": "For audio tasks, transcribe the attachment first and answer only after checking the transcript against the question.",
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
                "rule": "For image tasks, convert visual evidence into concise text before reasoning.",
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
        """
        負責執行 InsightGraph 中的 upsert_insight 流程，依照 InsightGraph 的流程需求處理 upsert_insight 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            insight: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 InsightRecord。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        record = insight if isinstance(insight, InsightRecord) else InsightRecord.from_dict(insight)
        self._memory_insights[record.insight_id] = record
        self._write(
            """
            MERGE (insight:MemoryInsight {id: $insight_id})
            SET insight.rule = $rule,
                insight.task_type = $task_type,
                insight.strategy = $strategy,
                insight.trigger_terms = $trigger_terms,
                insight.checklist = $checklist,
                insight.failure_modes = $failure_modes,
                insight.score = $score,
                insight.positive_correlation_tasks = $positive_correlation_tasks,
                insight.negative_correlation_tasks = $negative_correlation_tasks,
                insight.support_count = $support_count,
                insight.contradiction_count = $contradiction_count,
                insight.status = $status,
                insight.confidence = $confidence,
                insight.created_from = $created_from,
                insight.evidence_task_ids = $evidence_task_ids,
                insight.namespace = $namespace,
                insight.updated_at = $updated_at
            MERGE (type:MemoryTaskType {name: $task_type})
            MERGE (insight)-[:APPLIES_TO]->(type)
            """,
            insight_id=record.insight_id,
            rule=record.rule,
            task_type=record.task_type,
            strategy=record.strategy,
            trigger_terms=record.trigger_terms,
            checklist=record.checklist,
            failure_modes=record.failure_modes,
            score=record.score,
            positive_correlation_tasks=record.positive_correlation_tasks,
            negative_correlation_tasks=record.negative_correlation_tasks,
            support_count=record.support_count,
            contradiction_count=record.contradiction_count,
            status=record.status,
            confidence=record.confidence,
            created_from=record.created_from,
            evidence_task_ids=record.evidence_task_ids,
            namespace=self.namespace,
            updated_at=_now_iso(),
        )
        self._link_insight_signals(record)
        self._link_correlated_tasks(record)
        return record

    def _link_correlated_tasks(self, record: InsightRecord) -> None:
        """
        負責執行 InsightGraph 中的 _link_correlated_tasks 流程，依照 InsightGraph 的流程需求處理 _link_correlated_tasks 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            record: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        for task_id in record.positive_correlation_tasks:
            self._write(
                """
                MATCH (insight:MemoryInsight {id: $insight_id})
                MERGE (task:MemoryTask {id: $task_id})
                MERGE (insight)-[r:POSITIVE_CORRELATION]->(task)
                SET r.source = coalesce(r.source, 'upsert'),
                    r.updated_at = $updated_at
                """,
                insight_id=record.insight_id,
                task_id=task_id,
                updated_at=_now_iso(),
            )
        for task_id in record.negative_correlation_tasks:
            self._write(
                """
                MATCH (insight:MemoryInsight {id: $insight_id})
                MERGE (task:MemoryTask {id: $task_id})
                MERGE (insight)-[r:NEGATIVE_CORRELATION]->(task)
                SET r.source = coalesce(r.source, 'upsert'),
                    r.updated_at = $updated_at
                """,
                insight_id=record.insight_id,
                task_id=task_id,
                updated_at=_now_iso(),
            )

    def _link_insight_signals(self, record: InsightRecord) -> None:
        """
        負責執行 InsightGraph 中的 _link_insight_signals 流程，依照 InsightGraph 的流程需求處理 _link_insight_signals 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            record: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        for term in record.trigger_terms:
            self._write(
                """
                MATCH (insight:MemoryInsight {id: $insight_id})
                MERGE (term:MemoryTriggerTerm {name: $term})
                MERGE (insight)-[:TRIGGERED_BY]->(term)
                """,
                insight_id=record.insight_id,
                term=term,
            )
        for mode in record.failure_modes:
            self._write(
                """
                MATCH (insight:MemoryInsight {id: $insight_id})
                MERGE (mode:MemoryFailureMode {name: $mode})
                MERGE (insight)-[:AVOIDS_FAILURE]->(mode)
                """,
                insight_id=record.insight_id,
                mode=mode,
            )
        for idx, item in enumerate(record.checklist):
            item_id = f"{record.insight_id}_check_{idx}"
            self._write(
                """
                MATCH (insight:MemoryInsight {id: $insight_id})
                MERGE (check:MemoryChecklistItem {id: $item_id})
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
                    MATCH (insight:MemoryInsight {{id: $insight_id}})
                    MERGE (tool:MemoryToolPolicy {{name: $tool}})
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
        """
        負責執行 InsightGraph 中的 retrieve_insights 流程，從記憶圖、向量索引或任務關聯中取回相關案例與策略提醒。
        
        Args:
            task_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            trigger_terms: 記憶系統提供的檢索結果、寫入資料或操作介面。
            failure_modes: 記憶系統提供的檢索結果、寫入資料或操作介面。
            limit: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        task_type_slug = _slug(task_type)
        triggers = set(_dedupe(trigger_terms or []))
        failures = set(_dedupe(failure_modes or []))

        candidates: list[tuple[float, InsightRecord]] = []
        for record in self._memory_insights.values():
            if record.status != "active" or record.score <= 0:
                continue
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
                MATCH (insight:MemoryInsight)
                WHERE coalesce(insight.status, 'active') = 'active'
                  AND coalesce(insight.score, 0.0) > 0
                  AND (
                    insight.task_type = $task_type
                    OR any(term IN coalesce(insight.trigger_terms, []) WHERE term IN $trigger_terms)
                    OR any(mode IN coalesce(insight.failure_modes, []) WHERE mode IN $failure_modes)
                  )
                RETURN insight.id AS insight_id,
                       insight.rule AS rule,
                       insight.task_type AS task_type,
                       insight.strategy AS strategy,
                       insight.trigger_terms AS trigger_terms,
                       insight.checklist AS checklist,
                       insight.failure_modes AS failure_modes,
                       insight.score AS score,
                       insight.status AS status,
                       insight.confidence AS confidence,
                       insight.created_from AS created_from,
                       insight.evidence_task_ids AS evidence_task_ids,
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
                            "rule": row.get("rule"),
                            "task_type": row.get("task_type"),
                            "strategy": row.get("strategy"),
                            "trigger_terms": row.get("trigger_terms") or [],
                            "checklist": row.get("checklist") or [],
                            "failure_modes": row.get("failure_modes") or [],
                            "score": row.get("score") or 0,
                            "status": row.get("status") or "active",
                            "confidence": row.get("confidence") or 0.5,
                            "created_from": row.get("created_from") or "neo4j",
                            "evidence_task_ids": row.get("evidence_task_ids") or [],
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

    def retrieve_insights_for_tasks(
        self,
        task_ids: list[str],
        *,
        limit: int = 3,
        fallback_task_type: str | None = None,
        fallback_trigger_terms: list[str] | None = None,
        fallback_failure_modes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        負責執行 InsightGraph 中的 retrieve_insights_for_tasks 流程，從記憶圖、向量索引或任務關聯中取回相關案例與策略提醒。
        
        Args:
            task_ids: 記憶系統提供的檢索結果、寫入資料或操作介面。
            limit: 控制檢索、篩選或輸出數量的數值參數。
            fallback_task_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            fallback_trigger_terms: 記憶系統提供的檢索結果、寫入資料或操作介面。
            fallback_failure_modes: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        related = set(_dedupe(task_ids or []))
        candidates: list[tuple[float, InsightRecord]] = []
        for record in self._memory_insights.values():
            if record.status != "active" or record.score <= 0:
                continue
            positives = set(record.positive_correlation_tasks)
            negatives = set(record.negative_correlation_tasks)
            positive_hits = len(related & positives)
            negative_hits = len(related & negatives)
            if positive_hits:
                score = float(record.score) + (1.0 * positive_hits) - (1.5 * negative_hits)
                if score > 0:
                    candidates.append((score, record))

        if self.available and related:
            rows = self._read(
                """
                MATCH (insight:MemoryInsight)
                OPTIONAL MATCH (insight)-[pos:POSITIVE_CORRELATION]->(ptask:MemoryTask)
                WHERE ptask.id IN $task_ids
                WITH insight, count(DISTINCT ptask) AS positive_hits
                OPTIONAL MATCH (insight)-[neg:NEGATIVE_CORRELATION]->(ntask:MemoryTask)
                WHERE ntask.id IN $task_ids
                WITH insight, positive_hits, count(DISTINCT ntask) AS negative_hits
                WHERE positive_hits > 0
                  AND coalesce(insight.status, 'active') = 'active'
                  AND coalesce(insight.score, 0.0) > 0
                RETURN insight.id AS insight_id,
                       insight.rule AS rule,
                       insight.task_type AS task_type,
                       insight.strategy AS strategy,
                       insight.trigger_terms AS trigger_terms,
                       insight.checklist AS checklist,
                       insight.failure_modes AS failure_modes,
                       insight.score AS score,
                       insight.status AS status,
                       insight.confidence AS confidence,
                       insight.created_from AS created_from,
                       insight.evidence_task_ids AS evidence_task_ids,
                       insight.positive_correlation_tasks AS positive_correlation_tasks,
                       insight.negative_correlation_tasks AS negative_correlation_tasks,
                       insight.support_count AS support_count,
                       insight.contradiction_count AS contradiction_count,
                       positive_hits,
                       negative_hits
                ORDER BY (coalesce(insight.score, 0.0) + positive_hits - (1.5 * negative_hits)) DESC
                LIMIT $limit
                """,
                task_ids=list(related),
                limit=max(limit * 2, 6),
            )
            for row in rows:
                record = self._memory_insights.get(str(row.get("insight_id") or ""))
                if record is None:
                    record = InsightRecord.from_dict(
                        {
                            "insight_id": row.get("insight_id"),
                            "rule": row.get("rule"),
                            "task_type": row.get("task_type"),
                            "strategy": row.get("strategy"),
                            "trigger_terms": row.get("trigger_terms") or [],
                            "checklist": row.get("checklist") or [],
                            "failure_modes": row.get("failure_modes") or [],
                            "score": row.get("score") or 0,
                            "status": row.get("status") or "active",
                            "confidence": row.get("confidence") or 0.5,
                            "created_from": row.get("created_from") or "neo4j",
                            "evidence_task_ids": row.get("evidence_task_ids") or [],
                            "positive_correlation_tasks": row.get("positive_correlation_tasks") or [],
                            "negative_correlation_tasks": row.get("negative_correlation_tasks") or [],
                            "support_count": row.get("support_count") or 0,
                            "contradiction_count": row.get("contradiction_count") or 0,
                        }
                    )
                    self._memory_insights[record.insight_id] = record
                score = float(record.score) + float(row.get("positive_hits", 0) or 0) - 1.5 * float(row.get("negative_hits", 0) or 0)
                if score > 0:
                    candidates.append((score, record))

        if not candidates and fallback_task_type:
            return self.retrieve_insights(
                task_type=fallback_task_type,
                trigger_terms=fallback_trigger_terms,
                failure_modes=fallback_failure_modes,
                limit=limit,
            )

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
        """
        負責執行 InsightGraph 中的 apply_feedback 流程，依照 InsightGraph 的流程需求處理 apply_feedback 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            insight_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            exact: 記憶系統提供的檢索結果、寫入資料或操作介面。
            partial: 記憶系統提供的檢索結果、寫入資料或操作介面。
            stage2_changed_answer: 記憶系統提供的檢索結果、寫入資料或操作介面。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        record = self._memory_insights.get(insight_id)
        reward = 1.0 if exact else (0.35 if partial else -0.75)
        if stage2_changed_answer and exact:
            reward += 0.35
        if record is not None:
            record.score = max(0.0, record.score + reward)
            if exact or partial:
                record.support_count += 1
                record.positive_correlation_tasks = _dedupe(record.positive_correlation_tasks + [task_id])
            else:
                record.contradiction_count += 1
                record.negative_correlation_tasks = _dedupe(record.negative_correlation_tasks + [task_id])
            if record.score <= 0:
                record.status = "inactive"
        relation = "POSITIVE_CORRELATION" if exact or partial else "NEGATIVE_CORRELATION"
        self._write(
            f"""
            MATCH (insight:MemoryInsight {{id: $insight_id}})
            MERGE (task:MemoryTask {{id: $task_id}})
            MERGE (insight)-[r:{relation}]->(task)
            SET r.reward = $reward,
                r.metadata = $metadata,
                r.updated_at = $updated_at
            SET insight.score = coalesce(insight.score, 0.0) + $reward,
                insight.positive_correlation_tasks = CASE
                    WHEN $is_positive AND NOT $task_id IN coalesce(insight.positive_correlation_tasks, [])
                    THEN coalesce(insight.positive_correlation_tasks, []) + [$task_id]
                    ELSE coalesce(insight.positive_correlation_tasks, [])
                END,
                insight.negative_correlation_tasks = CASE
                    WHEN NOT $is_positive AND NOT $task_id IN coalesce(insight.negative_correlation_tasks, [])
                    THEN coalesce(insight.negative_correlation_tasks, []) + [$task_id]
                    ELSE coalesce(insight.negative_correlation_tasks, [])
                END,
                insight.support_count = coalesce(insight.support_count, 0) + $support_delta,
                insight.contradiction_count = coalesce(insight.contradiction_count, 0) + $contradiction_delta,
                insight.status = CASE
                    WHEN coalesce(insight.score, 0.0) + $reward <= 0 THEN 'inactive'
                    ELSE coalesce(insight.status, 'active')
                END,
                insight.updated_at = $updated_at
            """,
            insight_id=insight_id,
            task_id=task_id,
            reward=reward,
            metadata=str(dict(metadata or {})),
            updated_at=_now_iso(),
            is_positive=bool(exact or partial),
            support_delta=1 if exact or partial else 0,
            contradiction_delta=0 if exact or partial else 1,
        )

    def build_candidate_event(
        self,
        *,
        task_id: str,
        question: str,
        task_type: str = "general_reasoning",
        failure_mode: str = "insufficient_verification",
        predicted: str = "",
        expected: str = "",
        related_task_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InsightCandidateEvent:
        """
        負責執行 InsightGraph 中的 build_candidate_event 流程，建立記憶圖或任務記錄結構，供後續檢索、寫入與提示注入使用。
        
        Args:
            task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            question: 目前要處理的任務、問題或查詢文字。
            task_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            failure_mode: 記憶系統提供的檢索結果、寫入資料或操作介面。
            predicted: 記憶系統提供的檢索結果、寫入資料或操作介面。
            expected: 記憶系統提供的檢索結果、寫入資料或操作介面。
            related_task_ids: 記憶系統提供的檢索結果、寫入資料或操作介面。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 InsightCandidateEvent。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return InsightCandidateEvent(
            task_id=_slug(task_id, default="unknown_task"),
            question=_clean_text(question),
            task_type=_slug(task_type),
            failure_mode=_slug(failure_mode or "insufficient_verification"),
            predicted=_clean_text(predicted),
            expected=_clean_text(expected),
            related_task_ids=_dedupe(related_task_ids or []),
            metadata=dict(metadata or {}),
        )

    def generate_candidates_from_event(
        self,
        event: InsightCandidateEvent | dict[str, Any],
        *,
        llm_callable: Any | None = None,
    ) -> list[dict[str, Any]]:
        """
        負責執行 InsightGraph 中的 generate_candidates_from_event 流程，依照 InsightGraph 的流程需求處理 generate_candidates_from_event 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            event: 記憶系統提供的檢索結果、寫入資料或操作介面。
            llm_callable: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        evt = event if isinstance(event, InsightCandidateEvent) else InsightCandidateEvent(
            task_id=_slug(event.get("task_id"), default="unknown_task"),
            question=_clean_text(event.get("question")),
            task_type=_slug(event.get("task_type") or "general_reasoning"),
            failure_mode=_slug(event.get("failure_mode") or "insufficient_verification"),
            predicted=_clean_text(event.get("predicted")),
            expected=_clean_text(event.get("expected")),
            related_task_ids=_dedupe(list(event.get("related_task_ids") or [])),
            metadata=dict(event.get("metadata") or {}),
        )
        if llm_callable is not None:
            raw = llm_callable(self._build_candidate_prompt(evt))
            parsed = self._parse_json_list(raw)
            candidates = []
            for item in parsed:
                if isinstance(item, dict):
                    item.setdefault("evidence_task_ids", [evt.task_id])
                    item.setdefault("failure_type", evt.failure_mode)
                    item.setdefault("metadata", {})
                    item["metadata"]["llm_generated"] = True
                    candidates.append(item)
            if candidates:
                return candidates

        terms = self._event_terms(evt)
        rule = self._heuristic_rule_from_event(evt, terms)
        return [
            {
                "rule": rule,
                "failure_type": evt.failure_mode,
                "applies_when": terms,
                "avoid_when": [],
                "recommended_action": self._recommended_action(evt.failure_mode),
                "evidence_task_ids": _dedupe([evt.task_id] + evt.related_task_ids[:2]),
                "confidence": 0.62,
                "metadata": {
                    "source": "heuristic_candidate",
                    "question_excerpt": evt.question[:220],
                    "predicted": evt.predicted,
                    "expected": evt.expected,
                    **evt.metadata,
                },
            }
        ]

    def apply_insight_candidates(
        self,
        candidates: list[dict[str, Any]],
        *,
        task_type: str = "general_reasoning",
        source_task_id: str | None = None,
        similarity_threshold: float = 0.62,
    ) -> list[dict[str, Any]]:
        """
        負責執行 InsightGraph 中的 apply_insight_candidates 流程，依照 InsightGraph 的流程需求處理 apply_insight_candidates 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            candidates: 記憶系統提供的檢索結果、寫入資料或操作介面。
            task_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            source_task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            similarity_threshold: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        updates: list[dict[str, Any]] = []
        for raw in candidates:
            if not isinstance(raw, dict):
                continue
            candidate = InsightCandidate.from_dict(raw)
            if not candidate.rule:
                continue
            existing = self._find_similar_insight(candidate.rule, threshold=similarity_threshold)
            if existing is None:
                record = self.upsert_insight(
                    candidate.to_insight_dict(task_type=task_type, source_task_id=source_task_id)
                )
                updates.append({"operation": "add", "insight_id": record.insight_id, "rule": record.rule})
                continue
            self._merge_candidate_into_record(existing, candidate, source_task_id=source_task_id)
            self.upsert_insight(existing)
            updates.append({"operation": "merge", "insight_id": existing.insight_id, "rule": existing.rule})
        return updates

    def _find_similar_insight(self, rule: str, *, threshold: float) -> InsightRecord | None:
        """
        負責執行 InsightGraph 中的 _find_similar_insight 流程，依照 InsightGraph 的流程需求處理 _find_similar_insight 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            rule: 記憶系統提供的檢索結果、寫入資料或操作介面。
            threshold: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 InsightRecord | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        best: tuple[float, InsightRecord] | None = None
        for record in self._memory_insights.values():
            if record.status == "archived":
                continue
            score = _lexical_similarity(rule, record.rule or record.strategy)
            if score >= threshold and (best is None or score > best[0]):
                best = (score, record)
        return best[1] if best else None

    def _merge_candidate_into_record(
        self,
        record: InsightRecord,
        candidate: InsightCandidate,
        *,
        source_task_id: str | None = None,
    ) -> None:
        """
        負責執行 InsightGraph 中的 _merge_candidate_into_record 流程，依照 InsightGraph 的流程需求處理 _merge_candidate_into_record 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            record: 記憶系統提供的檢索結果、寫入資料或操作介面。
            candidate: 模型、節點或工具產生的候選回覆內容。
            source_task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        evidence = list(candidate.evidence_task_ids)
        if source_task_id:
            evidence.append(source_task_id)
        record.status = "active"
        record.confidence = max(record.confidence, candidate.confidence)
        record.score = max(0.0, record.score) + max(0.2, candidate.confidence)
        record.trigger_terms = _dedupe(record.trigger_terms + candidate.applies_when)
        record.failure_modes = _dedupe(record.failure_modes + [candidate.failure_type])
        record.positive_correlation_tasks = _dedupe(record.positive_correlation_tasks + evidence)
        record.evidence_task_ids = _dedupe(record.evidence_task_ids + evidence)
        if candidate.recommended_action:
            record.checklist = _dedupe_text(record.checklist + [f"Apply action: {candidate.recommended_action}"])
        record.checklist = _dedupe_text(record.checklist + [f"Check failure mode: {candidate.failure_type}"])
        metadata = dict(record.metadata or {})
        candidate_rules = list(metadata.get("candidate_rules") or [])
        candidate_rules.append(candidate.rule)
        metadata["candidate_rules"] = _dedupe_text(candidate_rules)[-8:]
        metadata["last_candidate_update"] = _now_iso()
        record.metadata = metadata

    def _build_candidate_prompt(self, event: InsightCandidateEvent) -> str:
        """
        負責執行 InsightGraph 中的 _build_candidate_prompt 流程，依照 InsightGraph 的流程需求處理 _build_candidate_prompt 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            event: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return (
            "Given one failed GAIA task, propose reusable memory insight candidates. "
            "Return JSON list only. Each item must contain rule, failure_type, "
            "applies_when, avoid_when, recommended_action, evidence_task_ids, confidence.\n\n"
            f"Task id: {event.task_id}\n"
            f"Task type: {event.task_type}\n"
            f"Failure mode: {event.failure_mode}\n"
            f"Question: {event.question}\n"
            f"Predicted: {event.predicted}\n"
            f"Expected: {event.expected}\n"
        )

    def _event_terms(self, event: InsightCandidateEvent) -> list[str]:
        """
        負責執行 InsightGraph 中的 _event_terms 流程，依照 InsightGraph 的流程需求處理 _event_terms 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            event: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        terms = sorted(_tokenize(event.question))
        prioritized = [term for term in terms if term in {"thousand", "hours", "random", "probability", "count", "image", "audio", "spreadsheet", "xls", "xlsx"}]
        return _dedupe((prioritized + terms)[:8])

    def _recommended_action(self, failure_mode: str) -> str:
        """
        負責執行 InsightGraph 中的 _recommended_action 流程，依照 InsightGraph 的流程需求處理 _recommended_action 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            failure_mode: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        failure = _slug(failure_mode)
        mapping = {
            "output_unit_mismatch": "check_output_unit",
            "missed_attachment_evidence": "read_attachment_first",
            "insufficient_evidence": "gather_structured_evidence",
            "outdated_fact": "search_current_sources",
            "surface_numeric_guess": "build_explicit_model",
            "missing_state_transition_model": "use_python_solver_or_dp",
        }
        return mapping.get(failure, "verify_against_question_constraints")

    def _heuristic_rule_from_event(self, event: InsightCandidateEvent, terms: list[str]) -> str:
        """
        負責執行 InsightGraph 中的 _heuristic_rule_from_event 流程，依照 InsightGraph 的流程需求處理 _heuristic_rule_from_event 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            event: 記憶系統提供的檢索結果、寫入資料或操作介面。
            terms: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        failure = event.failure_mode.replace("_", " ")
        if event.failure_mode == "output_unit_mismatch":
            return "When the question requests a specific output unit or format, convert the intermediate value before finalizing the answer."
        if "attachment" in event.failure_mode:
            return "When a GAIA task includes an attachment, read and summarize the attachment evidence before relying on text-only reasoning."
        if event.task_type == "stochastic_process":
            return "For probability or random-process tasks, define the state transition model before choosing a numeric answer."
        term_text = ", ".join(terms[:4]) if terms else event.task_type
        return f"For similar {event.task_type} tasks involving {term_text}, explicitly check for {failure} before finalizing."

    def _parse_json_list(self, raw: Any) -> list[dict[str, Any]]:
        """
        負責執行 InsightGraph 中的 _parse_json_list 流程，依照 InsightGraph 的流程需求處理 _parse_json_list 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            raw: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
