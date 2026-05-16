from __future__ import annotations

from datetime import datetime
import hashlib
import logging
from pathlib import Path
import pickle
import re
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """
    負責執行 memory.graph.query_task_graph 中的 _now_iso 流程，依照 memory.graph.query_task_graph 的流程需求處理 _now_iso 對應的資料轉換、狀態操作或結果產生。
    
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
    負責執行 memory.graph.query_task_graph 中的 _clean_text 流程，依照 memory.graph.query_task_graph 的流程需求處理 _clean_text 對應的資料轉換、狀態操作或結果產生。
    
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
    負責執行 memory.graph.query_task_graph 中的 _slug 流程，依照 memory.graph.query_task_graph 的流程需求處理 _slug 對應的資料轉換、狀態操作或結果產生。
    
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


def _task_id_from_question(question: str) -> str:
    """
    負責執行 memory.graph.query_task_graph 中的 _task_id_from_question 流程，依照 memory.graph.query_task_graph 的流程需求處理 _task_id_from_question 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        question: 目前要處理的任務、問題或查詢文字。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    digest = hashlib.sha1(_clean_text(question).encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"gaia_task_{digest}"


def _tokenize(text: str) -> set[str]:
    """
    負責執行 memory.graph.query_task_graph 中的 _tokenize 流程，依照 memory.graph.query_task_graph 的流程需求處理 _tokenize 對應的資料轉換、狀態操作或結果產生。
    
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
    """
    負責執行 memory.graph.query_task_graph 中的 _lexical_similarity 流程，依照 memory.graph.query_task_graph 的流程需求處理 _lexical_similarity 對應的資料轉換、狀態操作或結果產生。
    
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


class QueryTaskGraph:
    """
    負責在 memory.graph.query_task_graph 中封裝 QueryTaskGraph，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        graph_store: 記憶系統提供的檢索結果、寫入資料或操作介面。
        auto_connect: 記憶系統提供的檢索結果、寫入資料或操作介面。
        namespace: 記憶系統提供的檢索結果、寫入資料或操作介面。
        task_vector_index: 記憶系統提供的檢索結果、寫入資料或操作介面。
        similarity_threshold: 控制檢索、篩選或輸出數量的數值參數。
        default_hop: 記憶系統提供的檢索結果、寫入資料或操作介面。
        graph_persist_path: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
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
        namespace: str = "gaia",
        task_vector_index: Any | None = None,
        similarity_threshold: float = 0.70,
        default_hop: int = 1,
        graph_persist_path: str | Path | None = None,
    ):
        """
        負責執行 QueryTaskGraph 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            graph_store: 記憶系統提供的檢索結果、寫入資料或操作介面。
            auto_connect: 記憶系統提供的檢索結果、寫入資料或操作介面。
            namespace: 記憶系統提供的檢索結果、寫入資料或操作介面。
            task_vector_index: 記憶系統提供的檢索結果、寫入資料或操作介面。
            similarity_threshold: 控制檢索、篩選或輸出數量的數值參數。
            default_hop: 記憶系統提供的檢索結果、寫入資料或操作介面。
            graph_persist_path: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.namespace = namespace
        self.graph_store = graph_store
        self.task_vector_index = task_vector_index
        self.similarity_threshold = similarity_threshold
        self.default_hop = default_hop
        self.graph_persist_path = Path(graph_persist_path) if graph_persist_path else (
            Path("memory") / "storage" / "graph" / f"query_task_graph_{self.namespace}.pkl"
        )
        self.graph = nx.Graph()
        self._memory_tasks: dict[str, dict[str, Any]] = {}
        self._memory_edges: list[dict[str, Any]] = []
        self._neo4j_hydrated = False
        self._load_graph_snapshot()
        if self.graph_store is None and auto_connect:
            self.graph_store = self._create_graph_store()
        self._hydrate_from_neo4j()

    @property
    def available(self) -> bool:
        """
        負責執行 QueryTaskGraph 中的 available 流程，依照 QueryTaskGraph 的流程需求處理 available 對應的資料轉換、狀態操作或結果產生。
        
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
        負責執行 QueryTaskGraph 中的 _create_graph_store 流程，依照 QueryTaskGraph 的流程需求處理 _create_graph_store 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Any | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            from memory.database_config import get_database_config
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
        """
        負責執行 QueryTaskGraph 中的 _create_indexes 流程，依照 QueryTaskGraph 的流程需求處理 _create_indexes 對應的資料轉換、狀態操作或結果產生。
        
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
            "MATCH (n:GaiaTask) SET n:MemoryTask",
            "MATCH (n:GaiaTaskType) SET n:MemoryTaskType",
            "MATCH (n:GaiaTriggerTerm) SET n:MemoryTriggerTerm",
            "MATCH (n:GaiaFailureMode) SET n:MemoryFailureMode",
            "MATCH (n:GaiaToolPolicy) SET n:MemoryToolPolicy",
            "MATCH (n:GaiaAttachmentType) SET n:MemoryAttachmentType",
        ]
        queries = [
            "CREATE INDEX memory_task_id_index IF NOT EXISTS FOR (t:MemoryTask) ON (t.id)",
            "CREATE INDEX memory_task_namespace_index IF NOT EXISTS FOR (t:MemoryTask) ON (t.namespace)",
            "CREATE INDEX memory_task_type_index IF NOT EXISTS FOR (t:MemoryTaskType) ON (t.name)",
            "CREATE INDEX memory_trigger_index IF NOT EXISTS FOR (t:MemoryTriggerTerm) ON (t.name)",
            "CREATE INDEX memory_failure_index IF NOT EXISTS FOR (f:MemoryFailureMode) ON (f.name)",
            "CREATE INDEX memory_policy_index IF NOT EXISTS FOR (p:MemoryToolPolicy) ON (p.name)",
            "CREATE INDEX memory_task_main_index IF NOT EXISTS FOR (t:MemoryTask) ON (t.task_main)",
        ]
        with store.driver.session(database=store.database) as session:
            for query in migrations:
                session.run(query)
            for query in queries:
                session.run(query)

    def _write(self, query: str, **params: Any) -> bool:
        """
        負責執行 QueryTaskGraph 中的 _write 流程，依照 QueryTaskGraph 的流程需求處理 _write 對應的資料轉換、狀態操作或結果產生。
        
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
            logger.warning("QueryTaskGraph write failed: %s", exc)
            return False

    def _read(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """
        負責執行 QueryTaskGraph 中的 _read 流程，依照 QueryTaskGraph 的流程需求處理 _read 對應的資料轉換、狀態操作或結果產生。
        
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
            logger.warning("QueryTaskGraph read failed: %s", exc)
            return []

    def _load_graph_snapshot(self) -> None:
        """
        負責執行 QueryTaskGraph 中的 _load_graph_snapshot 流程，依照 QueryTaskGraph 的流程需求處理 _load_graph_snapshot 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.graph_persist_path.exists():
            return
        try:
            with self.graph_persist_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            logger.warning("QueryTaskGraph graph snapshot load failed: %s", exc)
            return
        try:
            graph = payload.get("graph") if isinstance(payload, dict) else payload
            if isinstance(graph, nx.Graph):
                self.graph = graph
            tasks = payload.get("memory_tasks", {}) if isinstance(payload, dict) else {}
            if isinstance(tasks, dict):
                self._memory_tasks.update(tasks)
            edges = payload.get("memory_edges", []) if isinstance(payload, dict) else []
            if isinstance(edges, list):
                self._memory_edges.extend(edge for edge in edges if isinstance(edge, dict))
        except Exception as exc:
            logger.warning("QueryTaskGraph graph snapshot restore failed: %s", exc)

    def _persist_graph_snapshot(self) -> None:
        """
        負責執行 QueryTaskGraph 中的 _persist_graph_snapshot 流程，依照 QueryTaskGraph 的流程需求處理 _persist_graph_snapshot 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            self.graph_persist_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "namespace": self.namespace,
                "graph": self.graph,
                "memory_tasks": self._memory_tasks,
                "memory_edges": self._memory_edges[-5000:],
                "updated_at": _now_iso(),
            }
            with self.graph_persist_path.open("wb") as handle:
                pickle.dump(payload, handle)
        except Exception as exc:
            logger.warning("QueryTaskGraph graph snapshot persist failed: %s", exc)

    def _hydrate_from_neo4j(self) -> None:
        """
        負責執行 QueryTaskGraph 中的 _hydrate_from_neo4j 流程，依照 QueryTaskGraph 的流程需求處理 _hydrate_from_neo4j 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self._neo4j_hydrated or not self.available:
            return
        self._neo4j_hydrated = True
        task_rows = self._read(
            """
            MATCH (task:MemoryTask)
            WHERE coalesce(task.namespace, $namespace) = $namespace
            RETURN task.id AS task_id,
                   task.question AS question,
                   task.task_main AS task_main,
                   task.task_description AS task_description,
                   task.task_type AS task_type,
                   task.label AS label,
                   task.has_interaction_record AS has_interaction_record
            """,
            namespace=self.namespace,
        )
        for row in task_rows:
            task_id = _clean_text(row.get("task_id"))
            if not task_id:
                continue
            task = self._memory_tasks.setdefault(task_id, {"id": task_id})
            task.update(
                {
                    "id": task_id,
                    "question": _clean_text(row.get("question")),
                    "task_main": _clean_text(row.get("task_main")) or task.get("task_main") or task_id,
                    "task_description": _clean_text(row.get("task_description")),
                    "task_type": _clean_text(row.get("task_type")),
                    "label": _clean_text(row.get("label")),
                    "has_interaction_record": bool(row.get("has_interaction_record")),
                }
            )
            self._add_or_update_graph_node(task_id, persist=False)

        edge_rows = self._read(
            """
            MATCH (a:MemoryTask)-[rel:SIMILAR_TO]-(b:MemoryTask)
            WHERE coalesce(a.namespace, $namespace) = $namespace
              AND coalesce(b.namespace, $namespace) = $namespace
            RETURN a.id AS left_task_id,
                   b.id AS right_task_id,
                   coalesce(rel.weight, 0.0) AS weight
            """,
            namespace=self.namespace,
        )
        for row in edge_rows:
            left = _clean_text(row.get("left_task_id"))
            right = _clean_text(row.get("right_task_id"))
            if not left or not right or left == right:
                continue
            self._add_similarity_edge(left, right, float(row.get("weight", 0.0) or 0.0), persist=False)
        if task_rows or edge_rows:
            self._persist_graph_snapshot()

    def update_task_node(
        self,
        task_id: str | None,
        question: str | None = None,
        metadata: dict[str, Any] | None = None,
        **properties: Any,
    ) -> str:
        """
        負責執行 QueryTaskGraph 中的 update_task_node 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            question: 目前要處理的任務、問題或查詢文字。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
            **properties: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        resolved_id = _clean_text(task_id) or _task_id_from_question(question or "")
        payload = dict(metadata or {})
        payload.setdefault("created_at", _now_iso())
        payload.setdefault("namespace", self.namespace)
        task = self._memory_tasks.setdefault(resolved_id, {"id": resolved_id})
        if question is not None:
            task["question"] = _clean_text(question)
        task["metadata"] = {**dict(task.get("metadata") or {}), **payload}
        for key, value in properties.items():
            if value is not None:
                task[key] = value
        self._memory_tasks.setdefault(resolved_id, {}).update(task)
        self._add_or_update_graph_node(resolved_id)
        self._write(
            """
            MERGE (t:MemoryTask {id: $task_id})
            SET t.question = $question,
                t.namespace = $namespace,
                t.updated_at = $updated_at,
                t += $metadata
            """,
            task_id=resolved_id,
            question=task.get("question", ""),
            namespace=self.namespace,
            updated_at=_now_iso(),
            metadata={**payload, **{key: value for key, value in properties.items() if value is not None}},
        )
        self._persist_graph_snapshot()
        return resolved_id

    def _add_or_update_graph_node(self, task_id: str, *, persist: bool = False) -> None:
        """
        負責執行 QueryTaskGraph 中的 _add_or_update_graph_node 流程，依照 QueryTaskGraph 的流程需求處理 _add_or_update_graph_node 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            persist: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        task = self._memory_tasks.get(task_id, {})
        metadata = task.get("metadata") or {}
        self.graph.add_node(
            task_id,
            task_id=task_id,
            task_main=task.get("task_main") or metadata.get("task_main") or task_id,
            question=task.get("question") or task.get("task_description") or "",
            task_type=task.get("task_type") or metadata.get("task_type"),
            trigger_terms=list(task.get("trigger_terms") or metadata.get("trigger_terms") or []),
            label=task.get("label") or metadata.get("label"),
            has_interaction_record=bool(task.get("has_interaction_record") or metadata.get("has_interaction_record")),
            updated_at=_now_iso(),
        )
        if persist:
            self._persist_graph_snapshot()

    def _add_similarity_edge(self, left_task_id: str, right_task_id: str, weight: float, *, persist: bool = False) -> None:
        """
        負責執行 QueryTaskGraph 中的 _add_similarity_edge 流程，依照 QueryTaskGraph 的流程需求處理 _add_similarity_edge 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            left_task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            right_task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            weight: 記憶系統提供的檢索結果、寫入資料或操作介面。
            persist: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not left_task_id or not right_task_id or left_task_id == right_task_id:
            return
        self._add_or_update_graph_node(left_task_id)
        self._add_or_update_graph_node(right_task_id)
        existing = self.graph.get_edge_data(left_task_id, right_task_id, default={})
        resolved_weight = max(float(weight or 0.0), float(existing.get("weight", 0.0) or 0.0))
        self.graph.add_edge(
            left_task_id,
            right_task_id,
            weight=resolved_weight,
            edge_type="SIMILAR_TO",
            updated_at=_now_iso(),
        )
        if persist:
            self._persist_graph_snapshot()

    def _task_search_text(self, task: dict[str, Any]) -> str:
        """
        負責執行 QueryTaskGraph 中的 _task_search_text 流程，依照 QueryTaskGraph 的流程需求處理 _task_search_text 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            task: 目前要處理的任務、問題或查詢文字。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        metadata = task.get("metadata") or {}
        parts = [
            task.get("task_main"),
            task.get("task_description"),
            task.get("question"),
            metadata.get("task_main"),
            metadata.get("question_excerpt"),
            task.get("failure_mode"),
            " ".join(task.get("trigger_terms") or []),
            task.get("task_type"),
        ]
        return _clean_text(" ".join(str(part or "") for part in parts if part is not None))

    def _retrieve_seed_tasks_from_vector_index(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        """
        負責執行 QueryTaskGraph 中的 _retrieve_seed_tasks_from_vector_index 流程，依照 QueryTaskGraph 的流程需求處理 _retrieve_seed_tasks_from_vector_index 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            query: 目前要處理的任務、問題或查詢文字。
            top_k: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        index = self.task_vector_index
        if index is None:
            return []
        try:
            if hasattr(index, "search_similar_tasks"):
                raw_results = index.search_similar_tasks(query, k=top_k)
            elif hasattr(index, "similarity_search_with_score"):
                raw_results = index.similarity_search_with_score(query, k=top_k)
            elif hasattr(index, "similarity_search"):
                raw_results = index.similarity_search(query, k=top_k)
            else:
                return []
        except Exception as exc:
            logger.warning("QueryTaskGraph vector seed search failed: %s", exc)
            return []

        seeds: list[dict[str, Any]] = []
        for item in raw_results or []:
            seeds.extend(self._normalize_vector_seed(item))
        return sorted(seeds, key=lambda row: float(row.get("weight", 0.0) or 0.0), reverse=True)[:top_k]

    def _normalize_vector_seed(self, item: Any) -> list[dict[str, Any]]:
        """
        負責執行 QueryTaskGraph 中的 _normalize_vector_seed 流程，依照 QueryTaskGraph 的流程需求處理 _normalize_vector_seed 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            item: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if isinstance(item, tuple) and len(item) >= 2:
            doc, score = item[0], item[1]
            metadata = getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else {}) or {}
            content = getattr(doc, "page_content", None) or (doc.get("page_content") if isinstance(doc, dict) else "") or ""
            task_id = metadata.get("task_id") or metadata.get("id")
            weight = self._distance_to_similarity(score)
            return [{"task_id": task_id, "weight": weight, "question": content, "source": "vector_index"}] if task_id else []
        if isinstance(item, dict):
            metadata = item.get("metadata") or {}
            task_id = item.get("task_id") or metadata.get("task_id") or item.get("id")
            if "distance" in item:
                weight = self._distance_to_similarity(item.get("distance"))
            else:
                score = item.get("similarity", item.get("score", item.get("weight")))
                weight = self._score_to_similarity(score)
            return [
                {
                    "task_id": task_id,
                    "weight": weight,
                    "question": item.get("question") or item.get("page_content") or item.get("content") or "",
                    "source": item.get("source", "vector_index"),
                }
            ] if task_id else []
        return []

    def _score_to_similarity(self, value: Any) -> float:
        """
        負責執行 QueryTaskGraph 中的 _score_to_similarity 流程，依照 QueryTaskGraph 的流程需求處理 _score_to_similarity 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            value: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 float。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if score < 0:
            return max(0.0, 1.0 + score)
        if score <= 1.0:
            return max(0.0, min(1.0, score))
        return max(0.0, min(1.0, 1.0 - score))

    def _distance_to_similarity(self, value: Any) -> float:
        """
        負責執行 QueryTaskGraph 中的 _distance_to_similarity 流程，依照 QueryTaskGraph 的流程需求處理 _distance_to_similarity 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            value: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 float。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            distance = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, 1.0 - distance))

    def _best_path_edge_weight(self, source_task_id: str, target_task_id: str) -> float:
        """
        負責執行 QueryTaskGraph 中的 _best_path_edge_weight 流程，依照 QueryTaskGraph 的流程需求處理 _best_path_edge_weight 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            source_task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            target_task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 float。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            path = nx.shortest_path(self.graph, source_task_id, target_task_id)
        except Exception:
            return 0.0
        if len(path) < 2:
            return 1.0
        weights = []
        for left, right in zip(path, path[1:]):
            data = self.graph.get_edge_data(left, right, default={})
            weights.append(float(data.get("weight", 0.0) or 0.0))
        return min(weights) if weights else 0.0

    def link_task_signals(self, task_id: str, classification: dict[str, Any]) -> None:
        """
        負責執行 QueryTaskGraph 中的 link_task_signals 流程，依照 QueryTaskGraph 的流程需求處理 link_task_signals 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            classification: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        data = dict(classification or {})
        task_type = _slug(data.get("task_type"))
        trigger_terms = [_slug(term) for term in data.get("trigger_terms", []) if _slug(term)]
        attachment_type = _slug(data.get("attachment_type") or "", default="")
        failure_modes = [_slug(mode) for mode in data.get("failure_modes", []) if _slug(mode)]
        policy = data.get("tool_policy") or {}

        task = self._memory_tasks.setdefault(task_id, {"id": task_id})
        task["classification"] = data
        task["task_type"] = task_type
        task["trigger_terms"] = trigger_terms
        self._add_or_update_graph_node(task_id)
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
            MATCH (task:MemoryTask {id: $task_id})
            MERGE (type:MemoryTaskType {name: $task_type})
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
                MATCH (task:MemoryTask {id: $task_id})
                MERGE (term:MemoryTriggerTerm {name: $term})
                MERGE (task)-[:HAS_TRIGGER]->(term)
                """,
                task_id=task_id,
                term=term,
            )
        if attachment_type:
            self._write(
                """
                MATCH (task:MemoryTask {id: $task_id})
                MERGE (attachment:MemoryAttachmentType {name: $attachment_type})
                MERGE (task)-[:HAS_ATTACHMENT_TYPE]->(attachment)
                """,
                task_id=task_id,
                attachment_type=attachment_type,
            )
        for mode in failure_modes:
            self._write(
                """
                MATCH (task:MemoryTask {id: $task_id})
                MERGE (mode:MemoryFailureMode {name: $mode})
                MERGE (task)-[:HAS_POSSIBLE_FAILURE]->(mode)
                """,
                task_id=task_id,
                mode=mode,
            )
        for relation, tools in [("PREFERS_TOOL", policy.get("prefer", [])), ("OPTIONAL_TOOL", policy.get("optional", [])), ("AVOIDS_TOOL", policy.get("avoid", []))]:
            for tool in [_slug(item) for item in tools if _slug(item)]:
                self._write(
                    f"""
                    MATCH (type:MemoryTaskType {{name: $task_type}})
                    MERGE (tool:MemoryToolPolicy {{name: $tool}})
                    MERGE (type)-[:{relation}]->(tool)
                    """,
                    task_type=task_type,
                    tool=tool,
                )
        self._persist_graph_snapshot()

    def link_similar_tasks(
        self,
        task_id: str,
        question: str,
        *,
        top_k: int = 5,
        min_weight: float = 0.20,
    ) -> list[dict[str, Any]]:
        """
        負責執行 QueryTaskGraph 中的 link_similar_tasks 流程，依照 QueryTaskGraph 的流程需求處理 link_similar_tasks 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            question: 目前要處理的任務、問題或查詢文字。
            top_k: 控制檢索、篩選或輸出數量的數值參數。
            min_weight: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        candidates = []
        for item in self.retrieve_seed_tasks_by_embedding(
            question,
            top_k=max(top_k * 2, 10),
            exclude_task_id=task_id,
        ):
            weight = float(item.get("weight", 0.0) or 0.0)
            if weight >= min_weight:
                candidates.append(
                    {
                        "task_id": item.get("task_id"),
                        "weight": min(weight, 1.0),
                        "question": item.get("question", ""),
                        "source": item.get("source", "seed_retrieval"),
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
            self._add_similarity_edge(task_id, str(item["task_id"]), float(item["weight"]))
            self._write(
                """
                MATCH (a:MemoryTask {id: $task_id})
                MATCH (b:MemoryTask {id: $other_id})
                MERGE (a)-[r:SIMILAR_TO]->(b)
                SET r.weight = $weight,
                    r.updated_at = $updated_at
                """,
                task_id=task_id,
                other_id=item["task_id"],
                weight=float(item["weight"]),
                updated_at=_now_iso(),
            )
        if selected:
            self._persist_graph_snapshot()
        return selected

    def retrieve_seed_tasks_by_embedding(
        self,
        query: str,
        *,
        top_k: int = 10,
        exclude_task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        負責執行 QueryTaskGraph 中的 retrieve_seed_tasks_by_embedding 流程，從記憶圖、向量索引或任務關聯中取回相關案例與策略提醒。
        
        Args:
            query: 目前要處理的任務、問題或查詢文字。
            top_k: 控制檢索、篩選或輸出數量的數值參數。
            exclude_task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self._hydrate_from_neo4j()
        seeds = self._retrieve_seed_tasks_from_vector_index(query, top_k=top_k)
        if seeds:
            return [
                item
                for item in seeds
                if str(item.get("task_id") or "") and str(item.get("task_id")) != str(exclude_task_id or "")
            ][:top_k]

        candidates = []
        for other_id, task in self._memory_tasks.items():
            if other_id == exclude_task_id:
                continue
            text = self._task_search_text(task)
            score = _lexical_similarity(query, text)
            if score <= 0:
                continue
            candidates.append(
                {
                    "task_id": other_id,
                    "weight": min(score, 1.0),
                    "question": text,
                    "source": "lexical_seed",
                }
            )
        return sorted(candidates, key=lambda item: item["weight"], reverse=True)[:top_k]

    def expand_related_tasks(
        self,
        seed_task_ids: list[str],
        *,
        hop: int | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        負責執行 QueryTaskGraph 中的 expand_related_tasks 流程，依照 QueryTaskGraph 的流程需求處理 expand_related_tasks 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            seed_task_ids: 記憶系統提供的檢索結果、寫入資料或操作介面。
            hop: 記憶系統提供的檢索結果、寫入資料或操作介面。
            limit: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self._hydrate_from_neo4j()
        max_hop = self.default_hop if hop is None else max(0, int(hop))
        scores: dict[str, float] = {}
        for seed_id in seed_task_ids:
            if not seed_id or not self.graph.has_node(seed_id):
                continue
            scores.setdefault(seed_id, 1.0)
            lengths = nx.single_source_shortest_path_length(self.graph, seed_id, cutoff=max_hop)
            for node_id, distance in lengths.items():
                if node_id == seed_id:
                    continue
                edge_weight = self._best_path_edge_weight(seed_id, node_id)
                hop_penalty = 1.0 / max(1, distance + 1)
                score = edge_weight * hop_penalty
                if score > scores.get(str(node_id), 0.0):
                    scores[str(node_id)] = score
        ranked = [
            {
                "task_id": task_id,
                "weight": weight,
                "task_main": self._memory_tasks.get(task_id, {}).get("task_main"),
            }
            for task_id, weight in scores.items()
        ]
        merged: dict[str, dict[str, Any]] = {
            str(item["task_id"]): item
            for item in ranked
            if item.get("task_id")
        }
        if len(merged) < limit and self.available:
            for item in self._expand_related_tasks_neo4j(seed_task_ids, hop=max_hop, limit=limit):
                oid = str(item.get("task_id") or "")
                if not oid:
                    continue
                if oid not in merged or float(item.get("weight", 0.0) or 0.0) > float(merged[oid].get("weight", 0.0) or 0.0):
                    merged[oid] = item
        return sorted(merged.values(), key=lambda item: float(item.get("weight", 0.0) or 0.0), reverse=True)[:limit]

    def _expand_related_tasks_neo4j(
        self,
        seed_task_ids: list[str],
        *,
        hop: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        負責執行 QueryTaskGraph 中的 _expand_related_tasks_neo4j 流程，依照 QueryTaskGraph 的流程需求處理 _expand_related_tasks_neo4j 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            seed_task_ids: 記憶系統提供的檢索結果、寫入資料或操作介面。
            hop: 記憶系統提供的檢索結果、寫入資料或操作介面。
            limit: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        max_hop = max(0, int(hop))
        if max_hop <= 0 or not seed_task_ids:
            return []
        results: dict[str, dict[str, Any]] = {}
        for seed_id in [_clean_text(item) for item in seed_task_ids if _clean_text(item)]:
            rows = self._read(
                f"""
                MATCH path = (seed:MemoryTask {{id: $seed_id}})-[:SIMILAR_TO*1..{max_hop}]-(related:MemoryTask)
                WHERE coalesce(seed.namespace, $namespace) = $namespace
                  AND coalesce(related.namespace, $namespace) = $namespace
                WITH related, path,
                     reduce(w = 1.0, rel IN relationships(path) | w * coalesce(rel.weight, 0.0)) AS path_weight,
                     length(path) AS hop_count
                RETURN related.id AS task_id,
                       related.task_main AS task_main,
                       path_weight / CASE WHEN hop_count = 0 THEN 1 ELSE hop_count END AS weight
                ORDER BY weight DESC
                LIMIT $limit
                """,
                seed_id=seed_id,
                namespace=self.namespace,
                limit=max(limit, 1),
            )
            for row in rows:
                task_id = _clean_text(row.get("task_id"))
                if not task_id or task_id == seed_id:
                    continue
                weight = float(row.get("weight", 0.0) or 0.0)
                task = self._memory_tasks.setdefault(task_id, {"id": task_id})
                if row.get("task_main"):
                    task["task_main"] = _clean_text(row.get("task_main"))
                self._add_or_update_graph_node(task_id)
                self._add_similarity_edge(seed_id, task_id, weight)
                if task_id not in results or weight > float(results[task_id].get("weight", 0.0) or 0.0):
                    results[task_id] = {
                        "task_id": task_id,
                        "weight": weight,
                        "task_main": task.get("task_main"),
                        "source": "neo4j_k_hop",
                    }
        if results:
            self._persist_graph_snapshot()
        return sorted(results.values(), key=lambda item: float(item.get("weight", 0.0) or 0.0), reverse=True)[:limit]
