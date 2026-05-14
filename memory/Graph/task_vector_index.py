from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import re
from pathlib import Path
from typing import Any


def _clean_text(value: Any) -> str:
    """
    負責執行 memory.graph.task_vector_index 中的 _clean_text 流程，依照 memory.graph.task_vector_index 的流程需求處理 _clean_text 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        value: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _now_iso() -> str:
    """
    負責執行 memory.graph.task_vector_index 中的 _now_iso 流程，依照 memory.graph.task_vector_index 的流程需求處理 _now_iso 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return datetime.now().isoformat(timespec="seconds")


def _tokenize(text: str) -> list[str]:
    """
    負責執行 memory.graph.task_vector_index 中的 _tokenize 流程，依照 memory.graph.task_vector_index 的流程需求處理 _tokenize 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        text: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 list[str]。
    
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
    return [
        token
        for token in re.findall(r"[a-z0-9_./:-]+", _clean_text(text).lower())
        if len(token) > 2 and token not in stopwords
    ]


def _stable_bucket(token: str, dimensions: int) -> int:
    """
    負責執行 memory.graph.task_vector_index 中的 _stable_bucket 流程，依照 memory.graph.task_vector_index 的流程需求處理 _stable_bucket 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        token: 記憶系統提供的檢索結果、寫入資料或操作介面。
        dimensions: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 int。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    value = 2166136261
    for char in token:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value % dimensions


def _cosine(left: list[float], right: list[float]) -> float:
    """
    負責執行 memory.graph.task_vector_index 中的 _cosine 流程，依照 memory.graph.task_vector_index 的流程需求處理 _cosine 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        left: 記憶系統提供的檢索結果、寫入資料或操作介面。
        right: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 float。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


@dataclass(slots=True)
class TaskVectorRecord:
    """
    負責在 memory.graph.task_vector_index 中封裝 TaskVectorRecord，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    task_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        """
        負責執行 TaskVectorRecord 中的 to_dict 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return {
            "task_id": self.task_id,
            "text": self.text,
            "metadata": dict(self.metadata),
            "embedding": list(self.embedding),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskVectorRecord":
        """
        負責執行 TaskVectorRecord 中的 from_dict 流程，依照 TaskVectorRecord 的流程需求處理 from_dict 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            data: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 'TaskVectorRecord'。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return cls(
            task_id=_clean_text(data.get("task_id")),
            text=_clean_text(data.get("text")),
            metadata=dict(data.get("metadata") or {}),
            embedding=[float(value) for value in (data.get("embedding") or [])],
            updated_at=_clean_text(data.get("updated_at")) or _now_iso(),
        )


class LocalTaskVectorIndex:
    """
    負責在 memory.graph.task_vector_index 中封裝 LocalTaskVectorIndex，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        namespace: 記憶系統提供的檢索結果、寫入資料或操作介面。
        persist_path: 記憶系統提供的檢索結果、寫入資料或操作介面。
        dimensions: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(
        self,
        *,
        namespace: str = "system",
        persist_path: str | Path | None = None,
        dimensions: int = 256,
    ) -> None:
        """
        負責執行 LocalTaskVectorIndex 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            namespace: 記憶系統提供的檢索結果、寫入資料或操作介面。
            persist_path: 記憶系統提供的檢索結果、寫入資料或操作介面。
            dimensions: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.namespace = _clean_text(namespace) or "system"
        self.dimensions = max(32, int(dimensions))
        self.persist_path = Path(persist_path) if persist_path else (
            Path("memory") / "storage" / "graph" / f"task_vector_index_{self.namespace}.json"
        )
        self.records: dict[str, TaskVectorRecord] = {}
        self._load()

    def add_task(self, *, task_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        """
        負責執行 LocalTaskVectorIndex 中的 add_task 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
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
        self.records[resolved_id] = TaskVectorRecord(
            task_id=resolved_id,
            text=content,
            metadata=dict(metadata or {}),
            embedding=self.embed_text(content),
        )
        self._persist()

    def add_document(self, page_content: str, metadata: dict[str, Any] | None = None) -> None:
        """
        負責執行 LocalTaskVectorIndex 中的 add_document 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
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
        負責執行 LocalTaskVectorIndex 中的 add_documents 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            documents: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        changed = False
        for document in documents or []:
            if isinstance(document, dict):
                content = document.get("page_content") or document.get("content") or document.get("text") or ""
                metadata = document.get("metadata") or {}
            else:
                content = getattr(document, "page_content", "") or ""
                metadata = getattr(document, "metadata", {}) or {}
            task_id = _clean_text(metadata.get("task_id") or metadata.get("id"))
            if not task_id or not _clean_text(content):
                continue
            self.records[task_id] = TaskVectorRecord(
                task_id=task_id,
                text=_clean_text(content),
                metadata=dict(metadata),
                embedding=self.embed_text(content),
            )
            changed = True
        if changed:
            self._persist()

    def search_similar_tasks(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """
        負責執行 LocalTaskVectorIndex 中的 search_similar_tasks 流程，從記憶圖、向量索引或任務關聯中取回相關案例與策略提醒。
        
        Args:
            query: 目前要處理的任務、問題或查詢文字。
            k: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        query_vector = self.embed_text(query)
        if not query_vector:
            return []
        results = []
        for record in self.records.values():
            score = _cosine(query_vector, record.embedding)
            if score <= 0.0:
                continue
            results.append(
                {
                    "task_id": record.task_id,
                    "similarity": score,
                    "weight": score,
                    "question": record.text,
                    "metadata": dict(record.metadata),
                    "source": "local_vector_index",
                }
            )
        return sorted(results, key=lambda item: float(item.get("weight", 0.0)), reverse=True)[: max(1, int(k))]

    def similarity_search_with_score(self, query: str, k: int = 10) -> list[tuple[dict[str, Any], float]]:
        """
        負責執行 LocalTaskVectorIndex 中的 similarity_search_with_score 流程，依照 LocalTaskVectorIndex 的流程需求處理 similarity_search_with_score 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            query: 目前要處理的任務、問題或查詢文字。
            k: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[tuple[dict[str, Any], float]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        results = []
        for item in self.search_similar_tasks(query, k=k):
            doc = {
                "page_content": item.get("question", ""),
                "metadata": item.get("metadata", {}),
            }
            results.append((doc, 1.0 - float(item.get("weight", 0.0) or 0.0)))
        return results

    def embed_text(self, text: str) -> list[float]:
        """
        負責執行 LocalTaskVectorIndex 中的 embed_text 流程，依照 LocalTaskVectorIndex 的流程需求處理 embed_text 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[float]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        tokens = _tokenize(text)
        if not tokens:
            return []
        vector = [0.0] * self.dimensions
        for token in tokens:
            vector[_stable_bucket(token, self.dimensions)] += 1.0
        return vector

    def _load(self) -> None:
        """
        負責執行 LocalTaskVectorIndex 中的 _load 流程，依照 LocalTaskVectorIndex 的流程需求處理 _load 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.persist_path.exists():
            return
        try:
            data = json.loads(self.persist_path.read_text(encoding="utf-8"))
        except Exception:
            return
        for raw in data.get("records", []):
            try:
                record = TaskVectorRecord.from_dict(raw)
            except Exception:
                continue
            if record.task_id:
                self.records[record.task_id] = record

    def _persist(self) -> None:
        """
        負責執行 LocalTaskVectorIndex 中的 _persist 流程，依照 LocalTaskVectorIndex 的流程需求處理 _persist 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "namespace": self.namespace,
            "dimensions": self.dimensions,
            "records": [record.to_dict() for record in self.records.values()],
        }
        self.persist_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
