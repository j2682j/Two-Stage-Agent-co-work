"""用於保存會話、事件與經驗軌跡的情節記憶實作。"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import os
import math
import json
import logging

logger = logging.getLogger(__name__)

from ..base import BaseMemory, MemoryItem, MemoryConfig
from ..storage import SQLiteDocumentStore, QdrantVectorStore
from ..embedding import get_text_embedder, get_dimension

class Episode:
    """情節記憶中的單一事件紀錄。"""
    
    def __init__(
        self,
        episode_id: str,
        user_id: str,
        session_id: str,
        timestamp: datetime,
        content: str,
        context: Dict[str, Any],
        outcome: Optional[str] = None,
        importance: float = 0.5
    ):
        self.episode_id = episode_id
        self.user_id = user_id
        self.session_id = session_id
        self.timestamp = timestamp
        self.content = content
        self.context = context
        self.outcome = outcome
        self.importance = importance

class EpisodicMemory(BaseMemory):
    """用來保存事件序列、會話上下文與經驗摘要的記憶庫。"""
    
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        
        # 本地快取（記憶體）
        self.episodes: List[Episode] = []
        self.sessions: Dict[str, List[str]] = {}  # session_id -> episode_ids
        
        # 模式識別快取
        self.patterns_cache = {}
        self.last_pattern_analysis = None

        # 權威文檔儲存（SQLite）
        db_dir = self.config.storage_path if hasattr(self.config, 'storage_path') else "./memory_data"
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "memory.db")
        self.doc_store = SQLiteDocumentStore(db_path=db_path)

        # 統一嵌入模型（多語言，預設384維）
        self.embedder = get_text_embedder()

        # 向量儲存（Qdrant - 使用連線管理器避免重復連線）
        from ..storage.qdrant_store import QdrantConnectionManager
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.vector_store = QdrantConnectionManager.get_instance(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=os.getenv("QDRANT_COLLECTION", "hello_agents_vectors"),
            vector_size=get_dimension(getattr(self.embedder, 'dimension', 384)),
            distance=os.getenv("QDRANT_DISTANCE", "cosine")
        )
    
    def add(self, memory_item: MemoryItem) -> str:
        """添加情景記憶"""
        # 從元資料中提取情景資訊
        session_id = memory_item.metadata.get("session_id", "default_session")
        context = memory_item.metadata.get("context", {})
        outcome = memory_item.metadata.get("outcome")
        participants = memory_item.metadata.get("participants", [])
        tags = memory_item.metadata.get("tags", [])
        
        # 建立情景（記憶體快取）
        episode = Episode(
            episode_id=memory_item.id,
            user_id=memory_item.user_id,
            session_id=session_id,
            timestamp=memory_item.timestamp,
            content=memory_item.content,
            context=context,
            outcome=outcome,
            importance=memory_item.importance
        )
        self.episodes.append(episode)
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(episode.episode_id)

        # 1) 權威儲存（SQLite）
        ts_int = int(memory_item.timestamp.timestamp())
        self.doc_store.add_memory(
            memory_id=memory_item.id,
            user_id=memory_item.user_id,
            content=memory_item.content,
            memory_type="episodic",
            timestamp=ts_int,
            importance=memory_item.importance,
            properties={
                "session_id": session_id,
                "context": context,
                "outcome": outcome,
                "participants": participants,
                "tags": tags
            }
        )

        # 2) 向量索引（Qdrant）
        try:
            embedding = self.embedder.encode(memory_item.content)
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            self.vector_store.add_vectors(
                vectors=[embedding],
                metadata=[{
                    "memory_id": memory_item.id,
                    "user_id": memory_item.user_id,
                    "memory_type": "episodic",
                    "importance": memory_item.importance,
                    "session_id": session_id,
                    "content": memory_item.content
                }],
                ids=[memory_item.id]
            )
        except Exception:
            # 向量入庫失敗不影響權威儲存
            pass

        return memory_item.id
    
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """搜尋情景記憶（結構化過濾 + 語義向量搜尋）"""
        user_id = kwargs.get("user_id")
        session_id = kwargs.get("session_id")
        time_range: Optional[Tuple[datetime, datetime]] = kwargs.get("time_range")
        importance_threshold: Optional[float] = kwargs.get("importance_threshold")

        # 結構化過濾候選（來自權威庫）
        candidate_ids: Optional[set] = None
        if time_range is not None or importance_threshold is not None:
            start_ts = int(time_range[0].timestamp()) if time_range else None
            end_ts = int(time_range[1].timestamp()) if time_range else None
            docs = self.doc_store.search_memories(
                user_id=user_id,
                memory_type="episodic",
                start_time=start_ts,
                end_time=end_ts,
                importance_threshold=importance_threshold,
                limit=1000
            )
            candidate_ids = {d["memory_id"] for d in docs}

        # 向量搜尋（Qdrant）
        try:
            query_vec = self.embedder.encode(query)
            if hasattr(query_vec, "tolist"):
                query_vec = query_vec.tolist()
            where = {"memory_type": "episodic"}
            if user_id:
                where["user_id"] = user_id
            hits = self.vector_store.search_similar(
                query_vector=query_vec,
                limit=max(limit * 5, 20),
                where=where
            )
        except Exception:
            hits = []

        # 過濾與重排
        now_ts = int(datetime.now().timestamp())
        results: List[Tuple[float, MemoryItem]] = []
        seen = set()
        for hit in hits:
            meta = hit.get("metadata", {})
            mem_id = meta.get("memory_id")
            if not mem_id or mem_id in seen:
                continue
            
            # 檢查是否已遺忘
            episode = next((e for e in self.episodes if e.episode_id == mem_id), None)
            if episode and episode.context.get("forgotten", False):
                continue  # 跳過已遺忘的記憶
                
            if candidate_ids is not None and mem_id not in candidate_ids:
                continue
            if session_id and meta.get("session_id") != session_id:
                continue

            # 從權威庫讀取完整紀錄
            doc = self.doc_store.get_memory(mem_id)
            if not doc:
                continue

            # 計算綜合分數：向量0.6 + 近因0.2 + 重要性0.2
            vec_score = float(hit.get("score", 0.0))
            age_days = max(0.0, (now_ts - int(doc["timestamp"])) / 86400.0)
            recency_score = 1.0 / (1.0 + age_days)
            imp = float(doc.get("importance", 0.5))
            
            # 新評分算法：向量搜尋純基於相似度，重要性作為加權因子
            # 基礎相似度得分（不受重要性影響）
            base_relevance = vec_score * 0.8 + recency_score * 0.2
            
            # 重要性作為乘法加權因子，范圍 [0.8, 1.2]
            importance_weight = 0.8 + (imp * 0.4)
            
            # 最終得分：相似度 * 重要性權重
            combined = base_relevance * importance_weight

            item = MemoryItem(
                id=doc["memory_id"],
                content=doc["content"],
                memory_type=doc["memory_type"],
                user_id=doc["user_id"],
                timestamp=datetime.fromtimestamp(doc["timestamp"]),
                importance=doc.get("importance", 0.5),
                metadata={
                    **doc.get("properties", {}),
                    "relevance_score": combined,
                    "vector_score": vec_score,
                    "recency_score": recency_score
                }
            )
            results.append((combined, item))
            seen.add(mem_id)

        # 若向量搜尋無結果，回退到簡單關鍵詞匹配（記憶體快取）
        if not results:
            fallback = super()._generate_id  # 占位以避免未使用警告
            query_lower = query.lower()
            for ep in self._filter_episodes(user_id, session_id, time_range):
                if query_lower in ep.content.lower():
                    recency_score = 1.0 / (1.0 + max(0.0, (now_ts - int(ep.timestamp.timestamp())) / 86400.0))
                    # 回退匹配：新評分算法
                    keyword_score = 0.5  # 簡單關鍵詞匹配的基礎分數
                    base_relevance = keyword_score * 0.8 + recency_score * 0.2
                    importance_weight = 0.8 + (ep.importance * 0.4)
                    combined = base_relevance * importance_weight
                    item = MemoryItem(
                        id=ep.episode_id,
                        content=ep.content,
                        memory_type="episodic",
                        user_id=ep.user_id,
                        timestamp=ep.timestamp,
                        importance=ep.importance,
                        metadata={
                            "session_id": ep.session_id,
                            "context": ep.context,
                            "outcome": ep.outcome,
                            "relevance_score": combined
                        }
                    )
                    results.append((combined, item))

        results.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in results[:limit]]
    
    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """更新情景記憶（SQLite為權威，Qdrant按需重嵌入）"""
        updated = False
        for episode in self.episodes:
            if episode.episode_id == memory_id:
                if content is not None:
                    episode.content = content
                if importance is not None:
                    episode.importance = importance
                if metadata is not None:
                    episode.context.update(metadata.get("context", {}))
                    if "outcome" in metadata:
                        episode.outcome = metadata["outcome"]
                updated = True
                break

        # 更新SQLite
        doc_updated = self.doc_store.update_memory(
            memory_id=memory_id,
            content=content,
            importance=importance,
            properties=metadata
        )

        # 如內容變更，重嵌入並upsert到Qdrant
        if content is not None:
            try:
                embedding = self.embedder.encode(content)
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                # 取得更新後的紀錄以同步payload
                doc = self.doc_store.get_memory(memory_id)
                payload = {
                    "memory_id": memory_id,
                    "user_id": doc["user_id"] if doc else "",
                    "memory_type": "episodic",
                    "importance": (doc.get("importance") if doc else importance) or 0.5,
                    "session_id": (doc.get("properties", {}) or {}).get("session_id"),
                    "content": content
                }
                self.vector_store.add_vectors(
                    vectors=[embedding],
                    metadata=[payload],
                    ids=[memory_id]
                )
            except Exception:
                pass

        return updated or doc_updated
    
    def remove(self, memory_id: str) -> bool:
        """刪除情景記憶（SQLite + Qdrant）"""
        removed = False
        for i, episode in enumerate(self.episodes):
            if episode.episode_id == memory_id:
                removed_episode = self.episodes.pop(i)
                session_id = removed_episode.session_id
                if session_id in self.sessions:
                    self.sessions[session_id].remove(memory_id)
                    if not self.sessions[session_id]:
                        del self.sessions[session_id]
                removed = True
                break

        # 權威庫刪除
        doc_deleted = self.doc_store.delete_memory(memory_id)
        
        # 向量庫刪除
        try:
            self.vector_store.delete_memories([memory_id])
        except Exception:
            pass
        
        return removed or doc_deleted
    
    def has_memory(self, memory_id: str) -> bool:
        """檢查記憶是否存在"""
        return any(episode.episode_id == memory_id for episode in self.episodes)
    
    def clear(self):
        """清空所有情景記憶（僅清理episodic，不影響其他類型）"""
        # 記憶體快取
        self.episodes.clear()
        self.sessions.clear()
        self.patterns_cache.clear()

        # SQLite內的episodic全部刪除
        docs = self.doc_store.search_memories(memory_type="episodic", limit=10000)
        ids = [d["memory_id"] for d in docs]
        for mid in ids:
            self.doc_store.delete_memory(mid)

        # Qdrant按ID刪除對應向量
        try:
            if ids:
                self.vector_store.delete_memories(ids)
        except Exception:
            pass

    def forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 30) -> int:
        """情景記憶遺忘機制（硬刪除）"""
        forgotten_count = 0
        current_time = datetime.now()
        
        to_remove = []  # 收集要刪除的記憶ID
        
        for episode in self.episodes:
            should_forget = False
            
            if strategy == "importance_based":
                # 基於重要性遺忘
                if episode.importance < threshold:
                    should_forget = True
            elif strategy == "time_based":
                # 基於時間遺忘
                cutoff_time = current_time - timedelta(days=max_age_days)
                if episode.timestamp < cutoff_time:
                    should_forget = True
            elif strategy == "capacity_based":
                # 基於容量遺忘（保留最重要的）
                if len(self.episodes) > self.config.max_capacity:
                    sorted_episodes = sorted(self.episodes, key=lambda e: e.importance)
                    excess_count = len(self.episodes) - self.config.max_capacity
                    if episode in sorted_episodes[:excess_count]:
                        should_forget = True
            
            if should_forget:
                to_remove.append(episode.episode_id)
        
        # 執行硬刪除
        for episode_id in to_remove:
            if self.remove(episode_id):
                forgotten_count += 1
                logger.info("已硬刪除情節記憶：%s...（策略：%s）", episode_id[:8], strategy)
        
        return forgotten_count

    def get_all(self) -> List[MemoryItem]:
        """取得所有情景記憶（轉換為MemoryItem格式）"""
        memory_items = []
        for episode in self.episodes:
            memory_item = MemoryItem(
                id=episode.episode_id,
                content=episode.content,
                memory_type="episodic",
                user_id=episode.user_id,
                timestamp=episode.timestamp,
                importance=episode.importance,
                metadata=episode.metadata
            )
            memory_items.append(memory_item)
        return memory_items
    
    def get_stats(self) -> Dict[str, Any]:
        """取得情景記憶統計資訊（合併SQLite與Qdrant）"""
        # 硬刪除模式：所有episodes都是活躍的
        active_episodes = self.episodes
        
        db_stats = self.doc_store.get_database_stats()
        try:
            vs_stats = self.vector_store.get_collection_stats()
        except Exception:
            vs_stats = {"store_type": "qdrant"}
        return {
            "count": len(active_episodes),  # 活躍記憶數量
            "forgotten_count": 0,  # 硬刪除模式下已遺忘的記憶會被直接刪除
            "total_count": len(self.episodes),  # 總記憶數量
            "sessions_count": len(self.sessions),
            "avg_importance": sum(e.importance for e in active_episodes) / len(active_episodes) if active_episodes else 0.0,
            "time_span_days": self._calculate_time_span(),
            "memory_type": "episodic",
            "vector_store": vs_stats,
            "document_store": {k: v for k, v in db_stats.items() if k.endswith("_count") or k in ["store_type", "db_path"]}
        }
    
    def get_session_episodes(self, session_id: str) -> List[Episode]:
        """取得指定會話的所有情景"""
        if session_id not in self.sessions:
            return []
        
        episode_ids = self.sessions[session_id]
        return [e for e in self.episodes if e.episode_id in episode_ids]
    
    def find_patterns(self, user_id: str = None, min_frequency: int = 2) -> List[Dict[str, Any]]:
        """發現使用者行為模式"""
        # 檢查快取
        cache_key = f"{user_id}_{min_frequency}"
        if (cache_key in self.patterns_cache and 
            self.last_pattern_analysis and 
            (datetime.now() - self.last_pattern_analysis).hours < 1):
            return self.patterns_cache[cache_key]
        
        # 過濾情景
        episodes = [e for e in self.episodes if user_id is None or e.user_id == user_id]
        
        # 簡單的模式識別：基於內容關鍵詞
        keyword_patterns = {}
        context_patterns = {}
        
        for episode in episodes:
            # 提取關鍵詞
            words = episode.content.lower().split()
            for word in words:
                if len(word) > 3:  # 忽略短詞
                    keyword_patterns[word] = keyword_patterns.get(word, 0) + 1
            
            # 提取上下文模式
            for key, value in episode.context.items():
                pattern_key = f"{key}:{value}"
                context_patterns[pattern_key] = context_patterns.get(pattern_key, 0) + 1
        
        # 篩選頻繁模式
        patterns = []
        
        for keyword, frequency in keyword_patterns.items():
            if frequency >= min_frequency:
                patterns.append({
                    "type": "keyword",
                    "pattern": keyword,
                    "frequency": frequency,
                    "confidence": frequency / len(episodes)
                })
        
        for context_pattern, frequency in context_patterns.items():
            if frequency >= min_frequency:
                patterns.append({
                    "type": "context",
                    "pattern": context_pattern,
                    "frequency": frequency,
                    "confidence": frequency / len(episodes)
                })
        
        # 按頻率排序
        patterns.sort(key=lambda x: x["frequency"], reverse=True)
        
        # 快取結果
        self.patterns_cache[cache_key] = patterns
        self.last_pattern_analysis = datetime.now()
        
        return patterns
    
    def get_timeline(self, user_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """取得時間線視圖"""
        episodes = [e for e in self.episodes if user_id is None or e.user_id == user_id]
        episodes.sort(key=lambda x: x.timestamp, reverse=True)
        
        timeline = []
        for episode in episodes[:limit]:
            timeline.append({
                "episode_id": episode.episode_id,
                "timestamp": episode.timestamp.isoformat(),
                "content": episode.content[:100] + "..." if len(episode.content) > 100 else episode.content,
                "session_id": episode.session_id,
                "importance": episode.importance,
                "outcome": episode.outcome
            })
        
        return timeline
    
    def _filter_episodes(
        self,
        user_id: str = None,
        session_id: str = None,
        time_range: Tuple[datetime, datetime] = None
    ) -> List[Episode]:
        """過濾情景"""
        filtered = self.episodes
        
        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]
        
        if session_id:
            filtered = [e for e in filtered if e.session_id == session_id]
        
        if time_range:
            start_time, end_time = time_range
            filtered = [e for e in filtered if start_time <= e.timestamp <= end_time]
        
        return filtered
    
    def _calculate_time_span(self) -> float:
        """計算記憶時間跨度（天）"""
        if not self.episodes:
            return 0.0
        
        timestamps = [e.timestamp for e in self.episodes]
        min_time = min(timestamps)
        max_time = max(timestamps)
        
        return (max_time - min_time).days
    
    def _persist_episode(self, episode: Episode):
        """持久化情景到儲存後端"""
        if self.storage and hasattr(self.storage, 'add_memory'):
            self.storage.add_memory(
                memory_id=episode.episode_id,
                user_id=episode.user_id,
                content=episode.content,
                memory_type="episodic",
                timestamp=int(episode.timestamp.timestamp()),
                importance=episode.importance,
                properties={
                    "session_id": episode.session_id,
                    "context": episode.context,
                    "outcome": episode.outcome
                }
            )
    
    def _remove_from_storage(self, memory_id: str):
        """從儲存後端刪除"""
        if self.storage and hasattr(self.storage, 'delete_memory'):
            self.storage.delete_memory(memory_id)
