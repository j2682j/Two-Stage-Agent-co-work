from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import os

from memory.embedding import get_dimension, get_text_embedder
from memory.storage.qdrant_store import QdrantConnectionManager

from .builder import InteractionGraphBuilder
from .insight_graph import InsightGraph
from .interaction_graph import InteractionGraph, TaskMetadata, classify_task_metadata
from .memory_prompt_builder import MemoryPromptBuilder
from .memory_base import GraphMemoryBase, _clean_text, _task_id_from_text
from .query_task_graph import QueryTaskGraph
from .task_record import TaskRecord
from .task_vector_index import LocalTaskVectorIndex


class QdrantTaskVectorIndex:
    """
    負責在 memory.graph.network_memory 中封裝 QdrantTaskVectorIndex，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        namespace: 記憶系統提供的檢索結果、寫入資料或操作介面。
        collection_name: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, *, namespace: str = "system", collection_name: str | None = None) -> None:
        """
        負責執行 QdrantTaskVectorIndex 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            namespace: 記憶系統提供的檢索結果、寫入資料或操作介面。
            collection_name: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.namespace = _clean_text(namespace) or "system"
        self.embedder = get_text_embedder()
        self.dimension = get_dimension(384)
        self.store = QdrantConnectionManager.get_instance(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=collection_name or os.getenv("GRAPH_MEMORY_QDRANT_COLLECTION", "task_memory_vectors"),
            vector_size=self.dimension,
            distance="cosine",
        )

    def add_task(self, *, task_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        """
        負責執行 QdrantTaskVectorIndex 中的 add_task 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            text: 記憶系統提供的檢索結果、寫入資料或操作介面。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        resolved_id = _clean_text(task_id)
        content = _clean_text(text)
        if not resolved_id or not content:
            return
        vector = self._embed(content)
        payload = {
            **dict(metadata or {}),
            "task_id": resolved_id,
            "content": content,
            "namespace": self.namespace,
            "memory_type": "graph_task",
            "is_graph_memory": True,
        }
        self.store.add_vectors(vectors=[vector], metadata=[payload])

    def add_document(self, page_content: str, metadata: dict[str, Any] | None = None) -> None:
        """
        負責執行 QdrantTaskVectorIndex 中的 add_document 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            page_content: 記憶系統提供的檢索結果、寫入資料或操作介面。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        data = dict(metadata or {})
        task_id = _clean_text(data.get("task_id") or data.get("id"))
        if task_id:
            self.add_task(task_id=task_id, text=page_content, metadata=data)

    def add_documents(self, documents: list[Any]) -> None:
        """
        負責執行 QdrantTaskVectorIndex 中的 add_documents 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            documents: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        for document in documents or []:
            if isinstance(document, dict):
                content = document.get("page_content") or document.get("content") or document.get("text") or ""
                metadata = document.get("metadata") or {}
            else:
                content = getattr(document, "page_content", "") or ""
                metadata = getattr(document, "metadata", {}) or {}
            self.add_document(content, metadata=dict(metadata or {}))

    def search_similar_tasks(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """
        負責執行 QdrantTaskVectorIndex 中的 search_similar_tasks 流程，從記憶圖、向量索引或任務關聯中取回相關案例與策略提醒。
        
        Args:
            query: 目前要處理的任務、問題或查詢文字。
            k: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        content = _clean_text(query)
        if not content:
            return []
        hits = self.store.search_similar(
            query_vector=self._embed(content),
            limit=max(1, int(k)),
            where={"namespace": self.namespace, "memory_type": "graph_task"},
        )
        results = []
        for hit in hits or []:
            metadata = dict(hit.get("metadata") or {})
            task_id = _clean_text(metadata.get("task_id") or hit.get("id"))
            if not task_id:
                continue
            score = float(hit.get("score", 0.0) or 0.0)
            results.append(
                {
                    "task_id": task_id,
                    "similarity": score,
                    "weight": score,
                    "question": metadata.get("content", ""),
                    "metadata": metadata,
                    "source": "qdrant_task_vector_index",
                }
            )
        return sorted(results, key=lambda item: float(item.get("weight", 0.0) or 0.0), reverse=True)[: max(1, int(k))]

    def similarity_search_with_score(self, query: str, k: int = 10) -> list[tuple[dict[str, Any], float]]:
        """
        負責執行 QdrantTaskVectorIndex 中的 similarity_search_with_score 流程，依照 QdrantTaskVectorIndex 的流程需求處理 similarity_search_with_score 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            query: 目前要處理的任務、問題或查詢文字。
            k: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[tuple[dict[str, Any], float]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        pairs = []
        for item in self.search_similar_tasks(query, k=k):
            doc = {
                "page_content": item.get("question", ""),
                "metadata": item.get("metadata", {}),
            }
            pairs.append((doc, 1.0 - float(item.get("weight", 0.0) or 0.0)))
        return pairs

    def _embed(self, text: str) -> list[float]:
        """
        負責執行 QdrantTaskVectorIndex 中的 _embed 流程，依照 QdrantTaskVectorIndex 的流程需求處理 _embed 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[float]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        vector = self.embedder.encode(text)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if vector and isinstance(vector[0], (list, tuple)):
            vector = list(vector[0])
        return [float(value) for value in vector]


@dataclass
class NetworkMemory(GraphMemoryBase):
    """
    負責在 memory.graph.network_memory 中封裝 NetworkMemory，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    auto_connect: bool = False
    use_qdrant: bool = True
    query_task_graph: QueryTaskGraph | None = None
    insight_graph: InsightGraph | None = None
    task_vector_index: Any | None = None
    interaction_builder: InteractionGraphBuilder = field(default_factory=InteractionGraphBuilder, init=False)
    prompt_builder: MemoryPromptBuilder = field(default_factory=MemoryPromptBuilder, init=False)
    insights_cache: list[dict[str, Any]] = field(default_factory=list, init=False)
    last_retrieval: dict[str, Any] = field(default_factory=dict, init=False)
    last_retrieval_debug: dict[str, Any] = field(default_factory=dict, init=False)
    task_record_store_path: Path = field(default_factory=Path, init=False)
    _task_records: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """
        負責執行 NetworkMemory 中的 __post_init__ 流程，依照 NetworkMemory 的流程需求處理 __post_init__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        super().__post_init__()
        if self.task_vector_index is None:
            self.task_vector_index = self._create_task_vector_index()
        if self.query_task_graph is None:
            self.query_task_graph = QueryTaskGraph(
                auto_connect=self.auto_connect,
                namespace=self.namespace,
                task_vector_index=self.task_vector_index,
                default_hop=int(self.global_config.get("hop", 1) if isinstance(self.global_config, dict) else 1),
                graph_persist_path=Path(self.persist_dir) / "query_task_graph.pkl",
            )
        if self.insight_graph is None:
            self.insight_graph = InsightGraph(
                auto_connect=self.auto_connect,
                namespace=self.namespace,
            )
        self.task_record_store_path = Path(self.persist_dir) / "task_records.json"
        self._load_task_records()

    def _create_task_vector_index(self) -> Any:
        """
        負責執行 NetworkMemory 中的 _create_task_vector_index 流程，依照 NetworkMemory 的流程需求處理 _create_task_vector_index 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Any。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self.use_qdrant:
            try:
                return QdrantTaskVectorIndex(namespace=self.namespace)
            except Exception as exc:
                print(f"[WARN] Qdrant graph task index unavailable; using local fallback: {exc}")
        return LocalTaskVectorIndex(
            namespace=self.namespace,
            persist_path=Path(self.persist_dir) / "task_vector_index.json",
        )

    def _add_task_record(self, mas_message: InteractionGraph | dict[str, Any]) -> str:
        """
        負責執行 NetworkMemory 中的 _add_task_record 流程，依照 NetworkMemory 的流程需求處理 _add_task_record 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            mas_message: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        record = mas_message.to_mas_message() if isinstance(mas_message, InteractionGraph) else dict(mas_message or {})
        task_id = _clean_text(record.get("task_id")) or _task_id_from_text(
            record.get("task_description") or record.get("task_main") or ""
        )
        record["task_id"] = task_id
        extra_fields = record.setdefault("extra_fields", {})
        task_metadata = extra_fields.get("task_metadata") if isinstance(extra_fields, dict) else None
        if isinstance(task_metadata, dict):
            for key in ("task_type", "trigger_terms", "failure_modes", "tool_policy"):
                extra_fields.setdefault(key, task_metadata.get(key))
        serialized = json.dumps(record, ensure_ascii=False, default=str)
        self._task_records[task_id] = record
        self._persist_task_records()

        task_main = _clean_text(record.get("task_main")) or task_id
        task_description = _clean_text(record.get("task_description"))
        self.query_task_graph.update_task_node(
            task_id,
            task_description or task_main,
            metadata={
                "task_main": task_main,
                "label": _clean_text(record.get("label")),
                "has_interaction_record": True,
                "record_updated_at": self._now_marker(),
                "expected": _clean_text(extra_fields.get("expected")),
                "stage1_result": _clean_text(extra_fields.get("stage1_result")),
                "final_result": _clean_text(extra_fields.get("final_result")),
                "predicted": _clean_text(extra_fields.get("predicted")),
                "score": extra_fields.get("score"),
                "exact_match": extra_fields.get("exact_match"),
                "partial_match": extra_fields.get("partial_match"),
            },
            task_main=task_main,
            task_description=task_description,
            label=_clean_text(record.get("label")),
            has_interaction_record=True,
        )
        self._add_task_record_to_vector_index(task_id, task_main, task_description, record, serialized)
        self.query_task_graph.link_similar_tasks(
            task_id,
            self._task_search_text(record),
            top_k=10,
            min_weight=self.query_task_graph.similarity_threshold,
        )
        return task_id

    def _load_task_record(self, task_id: str) -> dict[str, Any] | None:
        """
        負責執行 NetworkMemory 中的 _load_task_record 流程，依照 NetworkMemory 的流程需求處理 _load_task_record 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any] | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self._task_records.get(_clean_text(task_id))

    def _load_task_records(self) -> None:
        """
        負責執行 NetworkMemory 中的 _load_task_records 流程，依照 NetworkMemory 的流程需求處理 _load_task_records 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.task_record_store_path.exists():
            return
        try:
            payload = json.loads(self.task_record_store_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(payload, dict):
            records = payload.get("records", payload)
            if isinstance(records, dict):
                self._task_records = {
                    _clean_text(task_id): dict(record)
                    for task_id, record in records.items()
                    if _clean_text(task_id) and isinstance(record, dict)
                }

    def _persist_task_records(self) -> None:
        """
        負責執行 NetworkMemory 中的 _persist_task_records 流程，依照 NetworkMemory 的流程需求處理 _persist_task_records 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            self.task_record_store_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "namespace": self.namespace,
                "records": self._task_records,
            }
            self.task_record_store_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            print(f"[WARN] NetworkMemory task record persist failed: {exc}")

    def _add_task_record_to_vector_index(
        self,
        task_id: str,
        task_main: str,
        task_description: str,
        record: dict[str, Any],
        serialized: str,
    ) -> None:
        """
        負責執行 NetworkMemory 中的 _add_task_record_to_vector_index 流程，依照 NetworkMemory 的流程需求處理 _add_task_record_to_vector_index 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            task_main: 目前要處理的任務、問題或查詢文字。
            task_description: 目前要處理的任務、問題或查詢文字。
            record: 記憶系統提供的檢索結果、寫入資料或操作介面。
            serialized: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self.task_vector_index is None:
            return
        content = _clean_text(f"{task_main}\n{task_description}")
        if not content:
            return
        extra_fields = dict(record.get("extra_fields") or {})
        metadata = {
            "task_id": task_id,
            "task_main": task_main,
            "label": record.get("label"),
            "task_type": extra_fields.get("task_type"),
            "failure_mode": extra_fields.get("failure_mode"),
            "source": extra_fields.get("source"),
            "task_record_json": serialized,
        }
        self.task_vector_index.add_document(content, metadata=metadata)

    def _task_search_text(self, record: dict[str, Any]) -> str:
        """
        負責執行 NetworkMemory 中的 _task_search_text 流程，依照 NetworkMemory 的流程需求處理 _task_search_text 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            record: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        extra_fields = dict(record.get("extra_fields") or {})
        parts = [
            record.get("task_main"),
            record.get("task_description"),
            extra_fields.get("input_text"),
            extra_fields.get("failure_mode"),
            " ".join(extra_fields.get("trigger_terms") or []),
            extra_fields.get("task_type"),
        ]
        return _clean_text(" ".join(str(part or "") for part in parts if part is not None))

    def _now_marker(self) -> str:
        """
        負責執行 NetworkMemory 中的 _now_marker 流程，依照 NetworkMemory 的流程需求處理 _now_marker 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        from datetime import datetime

        return datetime.now().isoformat(timespec="seconds")

    def add_memory(self, mas_message: InteractionGraph) -> None:
        """
        負責執行 NetworkMemory 中的 add_memory 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            mas_message: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        classification = classify_task_metadata(
            mas_message.task_description or mas_message.task_main or ""
        )
        mas_message.set_task_metadata(classification)
        self._prepare_completed_task_record(mas_message, classification)
        task_id = self._add_task_record(mas_message)
        self.query_task_graph.link_task_signals(task_id, classification.to_dict())
        self._maybe_update_insights_from_completed_task(mas_message, classification)

    def _prepare_completed_task_record(self, graph: InteractionGraph, classification: Any) -> None:
        """
        負責執行 NetworkMemory 中的 _prepare_completed_task_record 流程，依照 NetworkMemory 的流程需求處理 _prepare_completed_task_record 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            graph: 記憶系統提供的檢索結果、寫入資料或操作介面。
            classification: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self._sparsify_state_chain(graph)
        key_steps = self._extract_key_steps(graph)
        graph.add_extra_field("key_steps", key_steps)
        graph.add_extra_field("clean_traj", self._format_key_steps(key_steps))
        label = _clean_text(graph.label).lower()
        if label in {"failed", "failure", "false", "wrong", "incorrect"}:
            failure_reason = self._detect_failure_reason(graph, classification)
            graph.add_extra_field("fail_reason", failure_reason)
            graph.add_extra_field("failure_mode", failure_reason)

    def _sparsify_state_chain(self, graph: InteractionGraph) -> None:
        """
        負責執行 NetworkMemory 中的 _sparsify_state_chain 流程，依照 NetworkMemory 的流程需求處理 _sparsify_state_chain 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            graph: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        state_chain = graph.chain_of_states
        states = []
        for state in state_chain.chain_of_states:
            reward = state.graph.get("reward")
            try:
                reward_value = float(reward)
            except (TypeError, ValueError):
                reward_value = None
            if reward_value is not None and reward_value < 0:
                continue
            states.append(state)
        state_chain._states = states
        state_chain._current_state = None

    def _extract_key_steps(self, graph: InteractionGraph) -> list[dict[str, Any]]:
        """
        負責執行 NetworkMemory 中的 _extract_key_steps 流程，依照 NetworkMemory 的流程需求處理 _extract_key_steps 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            graph: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        key_steps: list[dict[str, Any]] = []
        for idx, state in enumerate(graph.chain_of_states.chain_of_states):
            action = _clean_text(state.graph.get("action"))
            observation = _clean_text(state.graph.get("observation"))
            reward = state.graph.get("reward")
            if not action and not observation:
                continue
            key_steps.append(
                {
                    "index": idx,
                    "state_id": state.graph.get("state_id"),
                    "stage": state.graph.get("stage"),
                    "state_type": state.graph.get("state_type"),
                    "action": action,
                    "observation": observation,
                    "reward": reward,
                    "node_count": state.number_of_nodes(),
                }
            )
        return key_steps

    def _format_key_steps(self, key_steps: list[dict[str, Any]]) -> str:
        """
        負責執行 NetworkMemory 中的 _format_key_steps 流程，依照 NetworkMemory 的流程需求處理 _format_key_steps 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            key_steps: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        lines = []
        for step in key_steps:
            bits = [f"{step.get('stage') or step.get('state_type') or 'state'}"]
            if step.get("action"):
                bits.append(f"action={step['action']}")
            if step.get("observation"):
                bits.append(f"observation={step['observation']}")
            if step.get("reward") is not None:
                bits.append(f"reward={step['reward']}")
            lines.append("; ".join(bits))
        return "\n".join(lines)

    def _detect_failure_reason(self, graph: InteractionGraph, classification: Any) -> str:
        """
        負責執行 NetworkMemory 中的 _detect_failure_reason 流程，依照 NetworkMemory 的流程需求處理 _detect_failure_reason 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            graph: 記憶系統提供的檢索結果、寫入資料或操作介面。
            classification: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        existing = _clean_text(graph.extra_fields.get("failure_mode") or graph.extra_fields.get("fail_reason"))
        if existing:
            return existing
        text = _clean_text(
            f"{graph.task_description or ''} {graph.task_trajectory or ''} {graph.extra_fields.get('clean_traj') or ''}"
        ).lower()
        if "format" in text or "unit" in text:
            return "output_format_or_unit_mismatch"
        if "attachment" in text or classification.task_type in {"spreadsheet_reasoning", "image_understanding", "audio_understanding"}:
            return "missed_attachment_evidence"
        if "state" in text or "probability" in text or classification.task_type == "stochastic_process":
            return "missing_state_transition_model"
        if "search" in text or "source" in text or classification.task_type == "factual_search":
            return "insufficient_evidence"
        return "insufficient_verification"

    def _maybe_update_insights_from_completed_task(self, graph: InteractionGraph, classification: Any) -> list[dict[str, Any]]:
        """
        負責執行 NetworkMemory 中的 _maybe_update_insights_from_completed_task 流程，依照 NetworkMemory 的流程需求處理 _maybe_update_insights_from_completed_task 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            graph: 記憶系統提供的檢索結果、寫入資料或操作介面。
            classification: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        label = _clean_text(graph.label).lower()
        if label not in {"failed", "failure", "false", "wrong", "incorrect"}:
            return []
        event = self.insight_graph.build_candidate_event(
            task_id=graph.task_id,
            question=graph.task_description or graph.task_main,
            task_type=classification.task_type,
            failure_mode=str(graph.extra_fields.get("failure_mode") or "insufficient_verification"),
            predicted=str(graph.extra_fields.get("final_answer") or graph.extra_fields.get("predicted") or ""),
            expected=str(graph.extra_fields.get("expected_answer") or graph.extra_fields.get("expected") or ""),
            related_task_ids=[
                str(item.get("task_id") or "")
                for item in self.last_retrieval.get("retrieval", {}).get("similar_task_records", [])
                if isinstance(item, dict) and item.get("task_id")
            ],
            metadata={"source": "completed_task_lifecycle"},
        )
        candidates = self.insight_graph.generate_candidates_from_event(event)
        return self.insight_graph.apply_insight_candidates(
            candidates,
            task_type=classification.task_type,
            source_task_id=graph.task_id,
        )

    def _upsert_query_task(
        self,
        task_id: str,
        question: str,
        *,
        classification: Any,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        負責執行 NetworkMemory 中的 _upsert_query_task 流程，依照 NetworkMemory 的流程需求處理 _upsert_query_task 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            question: 目前要處理的任務、問題或查詢文字。
            classification: 記憶系統提供的檢索結果、寫入資料或操作介面。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        resolved_id = self.query_task_graph.update_task_node(
            task_id,
            question,
            metadata=dict(metadata or {}),
        )
        self.query_task_graph.link_task_signals(resolved_id, classification.to_dict())
        return resolved_id

    def _collect_similar_records(
        self,
        task_id: str,
        question: str,
        *,
        limit: int = 3,
        min_weight: float = 0.20,
        hop: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        負責執行 NetworkMemory 中的 _collect_similar_records 流程，依照 NetworkMemory 的流程需求處理 _collect_similar_records 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            question: 目前要處理的任務、問題或查詢文字。
            limit: 控制檢索、篩選或輸出數量的數值參數。
            min_weight: 控制檢索、篩選或輸出數量的數值參數。
            hop: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        seed_tasks = self.query_task_graph.link_similar_tasks(
            task_id,
            question,
            top_k=max(limit * 2, 4),
            min_weight=min_weight,
        )
        expanded = self.query_task_graph.expand_related_tasks(
            [str(item.get("task_id")) for item in seed_tasks if item.get("task_id")],
            hop=hop,
            limit=max(limit * 3, limit),
        )
        self.last_retrieval_debug = {
            "seed_task_hits": [
                {
                    "task_id": str(item.get("task_id") or ""),
                    "weight": item.get("weight"),
                    "similarity": item.get("similarity", item.get("weight")),
                    "source": item.get("source", ""),
                    "question": item.get("question", ""),
                }
                for item in seed_tasks
                if isinstance(item, dict) and item.get("task_id")
            ],
            "expanded_task_hits": [
                {
                    "task_id": str(item.get("task_id") or ""),
                    "weight": item.get("weight"),
                    "source": item.get("source", "query_graph_hop"),
                }
                for item in expanded
                if isinstance(item, dict) and item.get("task_id")
            ],
        }
        merged: dict[str, dict[str, Any]] = {}
        for item in seed_tasks + expanded:
            oid = str(item.get("task_id") or "")
            if not oid or oid == task_id:
                continue
            if oid not in merged or float(item.get("weight", 0.0) or 0.0) > float(merged[oid].get("weight", 0.0) or 0.0):
                merged[oid] = item

        records: list[dict[str, Any]] = []
        for item in sorted(merged.values(), key=lambda row: float(row.get("weight", 0.0) or 0.0), reverse=True):
            other_id = str(item.get("task_id") or "")
            record = self._load_task_record(other_id)
            if not record:
                continue
            records.append(
                {
                    "task_id": other_id,
                    "similarity": item.get("weight"),
                    "task_main": record.get("task_main"),
                    "label": record.get("label"),
                    "summary": self._summarize_task_record(record),
                    "record": record,
                }
            )
        return records[:limit]

    def _build_retrieval_context(
        self,
        *,
        task_id: str,
        question: str,
        classification: Any,
        limit: int,
        ) -> dict[str, Any]:
        """
        負責執行 NetworkMemory 中的 _build_retrieval_context 流程，依照 NetworkMemory 的流程需求處理 _build_retrieval_context 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            question: 目前要處理的任務、問題或查詢文字。
            classification: 記憶系統提供的檢索結果、寫入資料或操作介面。
            limit: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        similar_task_records = self._collect_similar_records(
            task_id,
            question,
            limit=limit,
            min_weight=0.20,
            hop=self.query_task_graph.default_hop,
        )
        similar_successes = self._rerank_success_cases(similar_task_records)
        similar_failures = []
        for item in similar_task_records:
            record = item.get("record") or {}
            label = _clean_text(record.get("label")).lower()
            extra = record.get("extra_fields") or {}
            failure_mode = extra.get("failure_mode")
            if failure_mode or label in {"failed", "failure", "wrong", "incorrect"}:
                similar_failures.append(
                    {
                        "task_id": item.get("task_id"),
                        "similarity": item.get("similarity"),
                        "failure_mode": failure_mode or "previous_wrong_answer",
                        "summary": item.get("summary") or self._summarize_task_record(record),
                    }
                )
        return {
            "task_id": task_id,
            "task_type": classification.task_type,
            "trigger_terms": list(classification.trigger_terms)[:8],
            "attachment_type": classification.attachment_type,
            "failure_modes": list(classification.failure_modes)[:5],
            "tool_policy": classification.tool_policy,
            "similar_failures": similar_failures[:limit],
            "similar_successes": similar_successes[:limit],
            "similar_task_records": [
                {key: value for key, value in item.items() if key != "record"}
                for item in similar_task_records[:limit]
            ],
            "seed_task_hits": list(self.last_retrieval_debug.get("seed_task_hits", [])),
            "expanded_task_hits": list(self.last_retrieval_debug.get("expanded_task_hits", [])),
        }

    def _rerank_success_cases(self, similar_task_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        負責執行 NetworkMemory 中的 _rerank_success_cases 流程，依照 NetworkMemory 的流程需求處理 _rerank_success_cases 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            similar_task_records: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        successes = []
        for item in similar_task_records:
            record = item.get("record") or {}
            label = _clean_text(record.get("label")).lower()
            if label not in {"successful", "success", "true", "exact"}:
                continue
            extra = record.get("extra_fields") or {}
            score = extra.get("score")
            try:
                outcome_score = float(score)
            except (TypeError, ValueError):
                outcome_score = 1.0
            successes.append(
                {
                    "task_id": item.get("task_id"),
                    "similarity": float(item.get("similarity", 0.0) or 0.0),
                    "outcome_score": outcome_score,
                    "summary": item.get("summary") or self._summarize_task_record(record),
                }
            )
        return sorted(
            successes,
            key=lambda row: (float(row.get("outcome_score", 0.0)), float(row.get("similarity", 0.0))),
            reverse=True,
        )

    def _summarize_task_record(self, record: dict[str, Any]) -> str:
        """
        負責執行 NetworkMemory 中的 _summarize_task_record 流程，依照 NetworkMemory 的流程需求處理 _summarize_task_record 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            record: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        extra = record.get("extra_fields") or {}
        pieces = []
        label = record.get("label")
        if label:
            pieces.append(f"label={label}")
        task_type = extra.get("task_type")
        if task_type:
            pieces.append(f"task_type={task_type}")
        expected = extra.get("expected") or extra.get("expected_answer")
        stage1 = extra.get("stage1_result")
        final = extra.get("final_result") or extra.get("final_answer")
        if stage1:
            pieces.append(f"stage1={stage1}")
        if final:
            pieces.append(f"final={final}")
        if expected:
            pieces.append("expected available")
        trajectory = _clean_text(record.get("task_trajectory", ""))
        if trajectory:
            pieces.append(f"trajectory={trajectory[:180]}")
        return "; ".join(pieces)

    def retrieve_memory(
        self,
        query_task: str,
        successful_topk: int = 2,
        failed_topk: int = 1,
        insight_topk: int = 3,
        threshold: float = 0.20,
        **kwargs: Any,
    ) -> tuple[list, list, list]:
        """
        負責執行 NetworkMemory 中的 retrieve_memory 流程，從記憶圖、向量索引或任務關聯中取回相關案例與策略提醒。
        
        Args:
            query_task: 目前要處理的任務、問題或查詢文字。
            successful_topk: 控制檢索、篩選或輸出數量的數值參數。
            failed_topk: 控制檢索、篩選或輸出數量的數值參數。
            insight_topk: 控制檢索、篩選或輸出數量的數值參數。
            threshold: 控制檢索、篩選或輸出數量的數值參數。
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 tuple[list, list, list]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        task_id = _clean_text(kwargs.get("task_id")) or _task_id_from_text(query_task)
        classification = classify_task_metadata(query_task, attachment_type=kwargs.get("attachment_type"))
        self._upsert_query_task(task_id, query_task, classification=classification)
        retrieval = self._build_retrieval_context(
            task_id=task_id,
            question=query_task,
            classification=classification,
            limit=max(successful_topk + failed_topk, 3),
        )
        related_records = retrieval.get("similar_task_records", []) or []
        successful: list[InteractionGraph] = []
        failed: list[InteractionGraph] = []
        related_task_ids: list[str] = []

        for item in related_records:
            if float(item.get("similarity", 0.0) or 0.0) < threshold:
                continue
            related_task_ids.append(str(item.get("task_id") or ""))
            record = item.get("record") or self._load_task_record(str(item.get("task_id") or ""))
            if not record:
                continue
            graph = InteractionGraph.from_mas_message(record)
            label = _clean_text(record.get("label")).lower()
            if label in {"successful", "success", "true", "exact"}:
                successful.append(graph)
            elif label in {"failed", "failure", "false", "wrong", "incorrect"}:
                failed.append(graph)
            else:
                failed.append(graph)

        insights = self.insight_graph.retrieve_insights_for_tasks(
            [task_id for task_id in related_task_ids if task_id],
            fallback_task_type=retrieval.get("task_type", "general_reasoning"),
            fallback_trigger_terms=retrieval.get("trigger_terms", []),
            fallback_failure_modes=retrieval.get("failure_modes", []),
            limit=insight_topk,
        )
        self.insights_cache = [item for item in insights if isinstance(item, dict)]
        self.last_retrieval = {
            "task_id": task_id,
            "query_task": query_task,
            "retrieval": retrieval,
            "related_task_ids": related_task_ids,
            "insights": insights,
            "seed_task_hits": retrieval.get("seed_task_hits", []),
            "expanded_task_hits": retrieval.get("expanded_task_hits", []),
        }
        return successful[:successful_topk], failed[:failed_topk], insights[:insight_topk]

    def retrieve_context(
        self,
        *,
        task_id: str | None,
        input_text: str,
        source: str = "system",
        benchmark: str | None = None,
        attachment_type: str | None = None,
        limit: int = 3,
        injection_target: str = "generic",
    ) -> dict[str, Any]:
        """
        負責執行 NetworkMemory 中的 retrieve_context 流程，從記憶圖、向量索引或任務關聯中取回相關案例與策略提醒。
        
        Args:
            task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            input_text: 記憶系統提供的檢索結果、寫入資料或操作介面。
            source: 記憶系統提供的檢索結果、寫入資料或操作介面。
            attachment_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            limit: 控制檢索、篩選或輸出數量的數值參數。
            injection_target: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        normalized = _clean_text(input_text)
        resolved_id = _clean_text(task_id) or _task_id_from_text(normalized)
        classification = classify_task_metadata(normalized, attachment_type=attachment_type)
        benchmark_value = _clean_text(benchmark)
        if benchmark_value.upper() == "BFCL":
            classification = TaskMetadata(
                task_type="function_calling",
                trigger_terms=["function", "parameters", "arguments", "json"],
                attachment_type=attachment_type,
                failure_modes=["wrong_function_name", "argument_schema_mismatch", "invalid_json"],
                tool_policy={"prefer": [], "optional": [], "avoid": ["search", "memory_as_answer_lookup"]},
                confidence=0.95,
            )
        resolved_id = self._upsert_query_task(
            resolved_id,
            normalized,
            classification=classification,
            metadata={"source": source, "benchmark": benchmark_value, "attachment_type": attachment_type},
        )
        retrieval = self._build_retrieval_context(
            task_id=resolved_id,
            question=normalized,
            classification=classification,
            limit=max(limit, 3),
        )
        related_task_ids = [
            str(item.get("task_id") or "")
            for item in retrieval.get("similar_task_records", [])
            if isinstance(item, dict) and item.get("task_id")
        ]
        insights = self.insight_graph.retrieve_insights_for_tasks(
            related_task_ids,
            fallback_task_type=retrieval.get("task_type", "general_reasoning"),
            fallback_trigger_terms=retrieval.get("trigger_terms", []),
            fallback_failure_modes=retrieval.get("failure_modes", []),
            limit=max(limit, 3),
        )
        guidance = self.prompt_builder.build_guidance_prompt(
            retrieval,
            insights=insights,
            max_failures=1,
            injection_target=injection_target,
        )
        self.insights_cache = [item for item in insights if isinstance(item, dict)]
        self.last_retrieval = {
            "task_id": resolved_id,
            "query_task": normalized,
            "retrieval": retrieval,
            "related_task_ids": related_task_ids,
            "insights": insights,
            "seed_task_hits": retrieval.get("seed_task_hits", []),
            "expanded_task_hits": retrieval.get("expanded_task_hits", []),
        }
        return {
            "task_id": resolved_id,
            "source": source,
            "benchmark": benchmark_value,
            "classification": classification.to_dict() if hasattr(classification, "to_dict") else dict(classification),
            "retrieval": retrieval,
            "related_task_ids": related_task_ids,
            "insights": insights,
            "seed_task_hits": retrieval.get("seed_task_hits", []),
            "expanded_task_hits": retrieval.get("expanded_task_hits", []),
            "guidance": guidance,
            "injection_target": injection_target,
        }

    def record_interaction_from_network(
        self,
        *,
        network: Any,
        sample: dict[str, Any] | None = None,
        sample_result: dict[str, Any] | None = None,
        source: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord:
        """
        負責執行 NetworkMemory 中的 record_interaction_from_network 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            network: 記憶系統提供的檢索結果、寫入資料或操作介面。
            sample: 記憶系統提供的檢索結果、寫入資料或操作介面。
            sample_result: 記憶系統提供的檢索結果、寫入資料或操作介面。
            source: 記憶系統提供的檢索結果、寫入資料或操作介面。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 TaskRecord。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        trace = self.interaction_builder.build_from_network(
            network=network,
            sample=sample or {},
            sample_result=sample_result or {},
        )
        trace.extra_fields.setdefault("source", source)
        trace.extra_fields.update(dict(metadata or {}))
        self.add_memory(trace)
        return TaskRecord.from_interaction_graph(trace, source=source)

    def record_task_result(
        self,
        *,
        task_id: str | None,
        input_text: str,
        source: str = "system",
        benchmark: str | None = None,
        predicted: str = "",
        expected: str = "",
        exact_match: bool = False,
        partial_match: bool = False,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
        network: Any | None = None,
        sample: dict[str, Any] | None = None,
        sample_result: dict[str, Any] | None = None,
        attachment_type: str | None = None,
    ) -> dict[str, Any]:
        """
        負責執行 NetworkMemory 中的 record_task_result 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            input_text: 記憶系統提供的檢索結果、寫入資料或操作介面。
            source: 記憶系統提供的檢索結果、寫入資料或操作介面。
            benchmark: 記憶系統提供的檢索結果、寫入資料或操作介面。
            predicted: 記憶系統提供的檢索結果、寫入資料或操作介面。
            expected: 記憶系統提供的檢索結果、寫入資料或操作介面。
            exact_match: 記憶系統提供的檢索結果、寫入資料或操作介面。
            partial_match: 記憶系統提供的檢索結果、寫入資料或操作介面。
            score: 記憶系統提供的檢索結果、寫入資料或操作介面。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
            network: 記憶系統提供的檢索結果、寫入資料或操作介面。
            sample: 記憶系統提供的檢索結果、寫入資料或操作介面。
            sample_result: 記憶系統提供的檢索結果、寫入資料或操作介面。
            attachment_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        normalized = _clean_text(input_text)
        resolved_id = _clean_text(task_id) or _task_id_from_text(normalized)
        meta = dict(metadata or {})
        label = "successful" if exact_match else ("partial" if partial_match else "failed")
        classification = classify_task_metadata(normalized, attachment_type=attachment_type)

        task_record: TaskRecord | None
        if network is not None:
            task_record = self.record_interaction_from_network(
                network=network,
                sample=sample or {},
                sample_result=sample_result or {},
                source=source,
                metadata={
                    **meta,
                    "input_text": normalized,
                    "final_answer": predicted,
                    "expected_answer": expected,
                    "error_type": meta.get("error_type"),
                    "failure_mode": meta.get("failure_mode"),
                    "score": score,
                },
            )
            resolved_id = task_record.task_id or resolved_id
        else:
            task_record = TaskRecord(
                task_id=resolved_id,
                source=source,
                benchmark=benchmark,
                task_main=benchmark or source or "system_task",
                task_description=normalized,
                input_text=normalized,
                final_answer=predicted,
                expected_answer=expected,
                label=label,
                score=score,
                error_type=meta.get("error_type"),
                failure_mode=meta.get("failure_mode"),
                metadata={
                    **meta,
                    "task_metadata": classification.to_dict(),
                    "task_type": classification.task_type,
                    "trigger_terms": classification.trigger_terms,
                    "failure_modes": classification.failure_modes,
                    "tool_policy": classification.tool_policy,
                },
            )
            self._add_task_record(task_record.to_task_record_message())

        self._upsert_query_task(
            resolved_id,
            normalized,
            classification=classification,
            metadata={
                "source": source,
                "benchmark": benchmark,
                "attachment_type": attachment_type,
                "predicted": predicted,
                "expected": expected,
                "exact_match": exact_match,
                "partial_match": partial_match,
                "score": score,
                **meta,
            },
        )
        related_task_records = self._collect_similar_records(
            resolved_id,
            normalized,
            limit=5,
            min_weight=0.20,
        )
        related_task_ids = [
            str(item.get("task_id") or "")
            for item in related_task_records
            if isinstance(item, dict) and item.get("task_id")
        ]
        insights = self.insight_graph.retrieve_insights_for_tasks(
            related_task_ids,
            fallback_task_type=classification.task_type,
            fallback_trigger_terms=classification.trigger_terms,
            fallback_failure_modes=classification.failure_modes,
            limit=3,
        )
        stage2_changed_answer = self._stage2_changed_stage1_answer(network)
        for insight in insights:
            insight_id = _clean_text(insight.get("insight_id"))
            if insight_id:
                self.insight_graph.apply_feedback(
                    insight_id,
                    resolved_id,
                    exact=exact_match,
                    partial=partial_match,
                    stage2_changed_answer=stage2_changed_answer,
                    metadata=meta,
                )

        candidate_updates: list[dict[str, Any]] = []
        if not exact_match and not partial_match:
            event = self.insight_graph.build_candidate_event(
                task_id=resolved_id,
                question=normalized,
                task_type=classification.task_type,
                failure_mode=str(meta.get("failure_mode") or "insufficient_verification"),
                predicted=predicted,
                expected=expected,
                related_task_ids=related_task_ids,
                metadata=meta,
            )
            candidates = self.insight_graph.generate_candidates_from_event(event)
            candidate_updates = self.insight_graph.apply_insight_candidates(
                candidates,
                task_type=classification.task_type,
                source_task_id=resolved_id,
            )

        return {
            "task_id": resolved_id,
            "source": source,
            "label": label,
            "task_record": task_record.to_dict() if task_record else None,
            "task_type": classification.task_type,
            "related_task_ids": related_task_ids,
            "insight_ids": [item.get("insight_id") for item in insights if isinstance(item, dict)],
            "candidate_updates": candidate_updates,
            "exact_match": exact_match,
            "partial_match": partial_match,
        }

    def backward(self, reward: Any, **kwargs: Any) -> None:
        """
        負責執行 NetworkMemory 中的 backward 流程，根據結果、評分或回饋更新節點狀態與權重資訊。
        
        Args:
            reward: 記憶系統提供的檢索結果、寫入資料或操作介面。
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        exact = bool(reward is True or reward == 1 or str(reward).lower() in {"true", "success", "successful"})
        partial = bool(kwargs.get("partial", False))
        task_id = _clean_text(kwargs.get("task_id") or self.last_retrieval.get("task_id"))
        if not task_id:
            self.insights_cache = []
            return
        for insight in self.insights_cache:
            insight_id = _clean_text(insight.get("insight_id"))
            if insight_id:
                self.insight_graph.apply_feedback(
                    insight_id,
                    task_id,
                    exact=exact,
                    partial=partial,
                    stage2_changed_answer=bool(kwargs.get("stage2_changed_answer", False)),
                    metadata=dict(kwargs.get("metadata") or {}),
                )
        self.insights_cache = []

    def update_memory(self, query: str, **kwargs: Any) -> None:
        """
        負責執行 NetworkMemory 中的 update_memory 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            query: 目前要處理的任務、問題或查詢文字。
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        task_id = _clean_text(kwargs.get("task_id")) or _task_id_from_text(query)
        metadata = dict(kwargs.get("metadata") or {})
        failure_mode = _clean_text(kwargs.get("failure_mode"))
        tools_used = [_clean_text(tool) for tool in (kwargs.get("tools_used") or []) if _clean_text(tool)]
        self.query_task_graph.update_task_node(
            task_id,
            query,
            metadata={
                **metadata,
                "outcome_updated_at": self._now_marker(),
            },
            stage1_result=kwargs.get("stage1_result"),
            final_result=kwargs.get("final_result"),
            expected=kwargs.get("expected"),
            exact=kwargs.get("exact"),
            partial=kwargs.get("partial"),
            failure_mode=failure_mode or None,
            tools_used=tools_used,
            outcome_metadata=metadata,
        )
        record = self._task_records.get(task_id)
        if isinstance(record, dict):
            extra = record.setdefault("extra_fields", {})
            extra.update(
                {
                    "stage1_result": kwargs.get("stage1_result"),
                    "final_result": kwargs.get("final_result"),
                    "expected": kwargs.get("expected"),
                    "exact": kwargs.get("exact"),
                    "partial": kwargs.get("partial"),
                    "failure_mode": failure_mode or extra.get("failure_mode"),
                    "tools_used": tools_used,
                    "outcome_metadata": metadata,
                }
            )
            if failure_mode:
                record["label"] = record.get("label") or "failed"
            self._task_records[task_id] = record
            self._persist_task_records()
        if failure_mode:
            classification = classify_task_metadata(query, attachment_type=kwargs.get("attachment_type"))
            data = classification.to_dict()
            modes = list(data.get("failure_modes") or [])
            if failure_mode not in modes:
                modes.append(failure_mode)
            data["failure_modes"] = modes
            self.query_task_graph.link_task_signals(task_id, data)

    def _stage2_changed_stage1_answer(self, network: Any | None) -> bool:
        """
        負責執行 NetworkMemory 中的 _stage2_changed_stage1_answer 流程，依照 NetworkMemory 的流程需求處理 _stage2_changed_stage1_answer 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            network: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if network is None:
            return False
        stage1 = _clean_text(getattr(network, "last_stage1_result", ""))
        final = _clean_text(
            (getattr(network, "last_final_decision", {}) or {}).get("final_result", "")
            if isinstance(getattr(network, "last_final_decision", {}), dict)
            else ""
        )
        return bool(stage1 and final and stage1 != final)

    @property
    def memory_size(self) -> int:
        """
        負責執行 NetworkMemory 中的 memory_size 流程，依照 NetworkMemory 的流程需求處理 memory_size 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 int。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return len(getattr(self.query_task_graph, "_memory_tasks", {}) or {})
