"""具備 TTL、token 限制與優先權衰減的短期工作記憶。"""

from datetime import datetime, timedelta
import heapq
from typing import Any, Dict, List

from ..base import BaseMemory, MemoryConfig, MemoryItem


class WorkingMemory(BaseMemory):
    """
    負責在 memory.types.working 中封裝 WorkingMemory，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        config: 控制此流程行為的設定資料。
        storage_backend: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        """
        負責執行 WorkingMemory 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            config: 控制此流程行為的設定資料。
            storage_backend: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        super().__init__(config, storage_backend)

        self.max_capacity = self.config.working_memory_capacity
        self.max_tokens = self.config.working_memory_tokens
        self.max_age_minutes = getattr(self.config, "working_memory_ttl_minutes", 120)
        self.current_tokens = 0
        self.session_start = datetime.now()

        self.memories: List[MemoryItem] = []
        self.memory_heap = []  # (-priority, timestamp, memory_id, memory_item)

    def add(self, memory_item: MemoryItem) -> str:
        """
        負責執行 WorkingMemory 中的 add 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            memory_item: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self._expire_old_memories()

        priority = self._calculate_priority(memory_item)
        heapq.heappush(
            self.memory_heap,
            (-priority, memory_item.timestamp, memory_item.id, memory_item),
        )
        self.memories.append(memory_item)
        self.current_tokens += len(memory_item.content.split())

        self._enforce_capacity_limits()
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, user_id: str = None, **kwargs) -> List[MemoryItem]:
        """
        負責執行 WorkingMemory 中的 retrieve 流程，從記憶圖、向量索引或任務關聯中取回相關案例與策略提醒。
        
        Args:
            query: 目前要處理的任務、問題或查詢文字。
            limit: 控制檢索、篩選或輸出數量的數值參數。
            user_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[MemoryItem]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self._expire_old_memories()
        if not self.memories:
            return []

        active_memories = [m for m in self.memories if not m.metadata.get("forgotten", False)]
        filtered_memories = active_memories
        if user_id:
            filtered_memories = [m for m in active_memories if m.user_id == user_id]

        if not filtered_memories:
            return []

        vector_scores = {}
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            documents = [query] + [m.content for m in filtered_memories]
            vectorizer = TfidfVectorizer(stop_words=None, lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(documents)
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()

            for idx, memory in enumerate(filtered_memories):
                vector_scores[memory.id] = similarities[idx]
        except Exception:
            vector_scores = {}

        query_lower = query.lower()
        scored_memories = []

        for memory in filtered_memories:
            content_lower = memory.content.lower()
            vector_score = vector_scores.get(memory.id, 0.0)

            keyword_score = 0.0
            if query_lower in content_lower:
                keyword_score = len(query_lower) / max(1, len(content_lower))
            else:
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                intersection = query_words.intersection(content_words)
                if intersection:
                    keyword_score = len(intersection) / max(1, len(query_words.union(content_words))) * 0.8

            if vector_score > 0:
                base_relevance = vector_score * 0.7 + keyword_score * 0.3
            else:
                base_relevance = keyword_score

            base_relevance *= self._calculate_time_decay(memory.timestamp)
            importance_weight = 0.8 + (memory.importance * 0.4)
            final_score = base_relevance * importance_weight

            if final_score > 0:
                scored_memories.append((final_score, memory))

        scored_memories.sort(key=lambda pair: pair[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]

    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        負責執行 WorkingMemory 中的 update 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            memory_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            content: 記憶系統提供的檢索結果、寫入資料或操作介面。
            importance: 記憶系統提供的檢索結果、寫入資料或操作介面。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        for memory in self.memories:
            if memory.id != memory_id:
                continue

            old_tokens = len(memory.content.split())

            if content is not None:
                memory.content = content
                new_tokens = len(content.split())
                self.current_tokens = self.current_tokens - old_tokens + new_tokens

            if importance is not None:
                memory.importance = importance

            if metadata is not None:
                memory.metadata.update(metadata)

            self._update_heap_priority(memory)
            return True

        return False

    def remove(self, memory_id: str) -> bool:
        """
        負責執行 WorkingMemory 中的 remove 流程，清除或移除指定資源、狀態或註冊資料，維持後續流程的一致性。
        
        Args:
            memory_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        for idx, memory in enumerate(self.memories):
            if memory.id != memory_id:
                continue

            removed_memory = self.memories.pop(idx)
            self._mark_deleted_in_heap(memory_id)
            self.current_tokens -= len(removed_memory.content.split())
            self.current_tokens = max(0, self.current_tokens)
            return True

        return False

    def has_memory(self, memory_id: str) -> bool:
        """
        負責執行 WorkingMemory 中的 has_memory 流程，檢查目前輸入、狀態或條件是否符合流程繼續執行的要求。
        
        Args:
            memory_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return any(memory.id == memory_id for memory in self.memories)

    def clear(self):
        """
        負責執行 WorkingMemory 中的 clear 流程，清除或移除指定資源、狀態或註冊資料，維持後續流程的一致性。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.memories.clear()
        self.memory_heap.clear()
        self.current_tokens = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        負責執行 WorkingMemory 中的 get_stats 流程，依照 WorkingMemory 的流程需求處理 get_stats 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self._expire_old_memories()
        active_memories = self.memories

        return {
            "count": len(active_memories),
            "forgotten_count": 0,
            "total_count": len(self.memories),
            "current_tokens": self.current_tokens,
            "max_capacity": self.max_capacity,
            "max_tokens": self.max_tokens,
            "max_age_minutes": self.max_age_minutes,
            "session_duration_minutes": (datetime.now() - self.session_start).total_seconds() / 60,
            "avg_importance": sum(m.importance for m in active_memories) / len(active_memories) if active_memories else 0.0,
            "capacity_usage": len(active_memories) / self.max_capacity if self.max_capacity > 0 else 0.0,
            "token_usage": self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0,
            "memory_type": "working",
        }

    def get_recent(self, limit: int = 10) -> List[MemoryItem]:
        """
        負責執行 WorkingMemory 中的 get_recent 流程，依照 WorkingMemory 的流程需求處理 get_recent 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            limit: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[MemoryItem]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        sorted_memories = sorted(self.memories, key=lambda item: item.timestamp, reverse=True)
        return sorted_memories[:limit]

    def get_important(self, limit: int = 10) -> List[MemoryItem]:
        """
        負責執行 WorkingMemory 中的 get_important 流程，依照 WorkingMemory 的流程需求處理 get_important 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            limit: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[MemoryItem]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        sorted_memories = sorted(self.memories, key=lambda item: item.importance, reverse=True)
        return sorted_memories[:limit]

    def get_all(self) -> List[MemoryItem]:
        """
        負責執行 WorkingMemory 中的 get_all 流程，依照 WorkingMemory 的流程需求處理 get_all 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[MemoryItem]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.memories.copy()

    def get_context_summary(self, max_length: int = 500) -> str:
        """
        負責執行 WorkingMemory 中的 get_context_summary 流程，依照 WorkingMemory 的流程需求處理 get_context_summary 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            max_length: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.memories:
            return "目前沒有可用的工作記憶。"

        sorted_memories = sorted(
            self.memories,
            key=lambda item: (item.importance, item.timestamp),
            reverse=True,
        )

        summary_parts = []
        current_length = 0

        for memory in sorted_memories:
            content = memory.content
            if current_length + len(content) <= max_length:
                summary_parts.append(content)
                current_length += len(content)
            else:
                remaining = max_length - current_length
                if remaining > 50:
                    summary_parts.append(content[:remaining] + "...")
                break

        return "工作記憶內容：\n" + "\n".join(summary_parts)

    def forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 1) -> int:
        """
        負責執行 WorkingMemory 中的 forget 流程，依照 WorkingMemory 的流程需求處理 forget 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            strategy: 記憶系統提供的檢索結果、寫入資料或操作介面。
            threshold: 控制檢索、篩選或輸出數量的數值參數。
            max_age_days: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 int。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        forgotten_count = 0
        current_time = datetime.now()
        to_remove = []

        cutoff_ttl = current_time - timedelta(minutes=self.max_age_minutes)
        for memory in self.memories:
            if memory.timestamp < cutoff_ttl:
                to_remove.append(memory.id)

        if strategy == "importance_based":
            for memory in self.memories:
                if memory.importance < threshold:
                    to_remove.append(memory.id)
        elif strategy == "time_based":
            cutoff_time = current_time - timedelta(hours=max_age_days * 24)
            for memory in self.memories:
                if memory.timestamp < cutoff_time:
                    to_remove.append(memory.id)
        elif strategy == "capacity_based" and len(self.memories) > self.max_capacity:
            sorted_memories = sorted(self.memories, key=lambda item: self._calculate_priority(item))
            excess_count = len(self.memories) - self.max_capacity
            for memory in sorted_memories[:excess_count]:
                to_remove.append(memory.id)

        for memory_id in to_remove:
            if self.remove(memory_id):
                forgotten_count += 1

        return forgotten_count

    def _calculate_priority(self, memory: MemoryItem) -> float:
        """
        負責執行 WorkingMemory 中的 _calculate_priority 流程，依照 WorkingMemory 的流程需求處理 _calculate_priority 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            memory: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 float。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return memory.importance * self._calculate_time_decay(memory.timestamp)

    def _calculate_time_decay(self, timestamp: datetime) -> float:
        """
        負責執行 WorkingMemory 中的 _calculate_time_decay 流程，依照 WorkingMemory 的流程需求處理 _calculate_time_decay 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            timestamp: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 float。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        time_diff = datetime.now() - timestamp
        hours_passed = time_diff.total_seconds() / 3600
        decay_factor = self.config.decay_factor ** (hours_passed / 6)
        return max(0.1, decay_factor)

    def _enforce_capacity_limits(self):
        """
        負責執行 WorkingMemory 中的 _enforce_capacity_limits 流程，依照 WorkingMemory 的流程需求處理 _enforce_capacity_limits 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        while len(self.memories) > self.max_capacity:
            self._remove_lowest_priority_memory()

        while self.current_tokens > self.max_tokens:
            self._remove_lowest_priority_memory()

    def _expire_old_memories(self):
        """
        負責執行 WorkingMemory 中的 _expire_old_memories 流程，依照 WorkingMemory 的流程需求處理 _expire_old_memories 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.memories:
            return

        cutoff_time = datetime.now() - timedelta(minutes=self.max_age_minutes)
        kept: List[MemoryItem] = []
        removed_token_sum = 0

        for memory in self.memories:
            if memory.timestamp >= cutoff_time:
                kept.append(memory)
            else:
                removed_token_sum += len(memory.content.split())

        if len(kept) == len(self.memories):
            return

        self.memories = kept
        self.current_tokens = max(0, self.current_tokens - removed_token_sum)
        self.memory_heap = []
        for memory in self.memories:
            priority = self._calculate_priority(memory)
            heapq.heappush(self.memory_heap, (-priority, memory.timestamp, memory.id, memory))

    def _remove_lowest_priority_memory(self):
        """
        負責執行 WorkingMemory 中的 _remove_lowest_priority_memory 流程，依照 WorkingMemory 的流程需求處理 _remove_lowest_priority_memory 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.memories:
            return

        lowest_priority = float("inf")
        lowest_memory = None

        for memory in self.memories:
            priority = self._calculate_priority(memory)
            if priority < lowest_priority:
                lowest_priority = priority
                lowest_memory = memory

        if lowest_memory:
            self.remove(lowest_memory.id)

    def _update_heap_priority(self, memory: MemoryItem):
        """
        負責執行 WorkingMemory 中的 _update_heap_priority 流程，依照 WorkingMemory 的流程需求處理 _update_heap_priority 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            memory: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.memory_heap = []
        for item in self.memories:
            priority = self._calculate_priority(item)
            heapq.heappush(self.memory_heap, (-priority, item.timestamp, item.id, item))

    def _mark_deleted_in_heap(self, memory_id: str):
        """
        負責執行 WorkingMemory 中的 _mark_deleted_in_heap 流程，依照 WorkingMemory 的流程需求處理 _mark_deleted_in_heap 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            memory_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        pass
