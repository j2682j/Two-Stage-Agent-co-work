"""
Qdrant向量資料庫儲存實現
使用專業的Qdrant向量資料庫替代ChromaDB
"""

import logging
import os
import uuid
import threading
from typing import Dict, List, Optional, Any, Union
import numpy as np
from datetime import datetime

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance, VectorParams, PointStruct, 
        Filter, FieldCondition, MatchValue, SearchRequest
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    models = None

logger = logging.getLogger(__name__)

class QdrantConnectionManager:
    """Qdrant連線管理器 - 防止重復連線和初始化"""
    _instances = {}  # key: (url, collection_name) -> QdrantVectorStore instance
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(
        cls, 
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "hello_agents_vectors",
        vector_size: int = 384,
        distance: str = "cosine",
        timeout: int = 30,
        **kwargs
    ) -> 'QdrantVectorStore':
        """取得或建立Qdrant實例（單例模式）"""
        # 建立唯一鍵
        key = (url or "local", collection_name)
        
        if key not in cls._instances:
            with cls._lock:
                # 雙重檢查鎖定
                if key not in cls._instances:
                    logger.debug(f"🔄 建立新的Qdrant連線: {collection_name}")
                    cls._instances[key] = QdrantVectorStore(
                        url=url,
                        api_key=api_key,
                        collection_name=collection_name,
                        vector_size=vector_size,
                        distance=distance,
                        timeout=timeout,
                        **kwargs
                    )
                else:
                    logger.debug(f"♻️ 復用現有Qdrant連線: {collection_name}")
        else:
            logger.debug(f"♻️ 復用現有Qdrant連線: {collection_name}")
            
        return cls._instances[key]

class QdrantVectorStore:
    """Qdrant向量資料庫儲存實現"""
    
    def __init__(
        self, 
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "hello_agents_vectors",
        vector_size: int = 384,
        distance: str = "cosine",
        timeout: int = 30,
        **kwargs
    ):
        """
        初始化Qdrant向量儲存 (支援云API)
        
        Args:
            url: Qdrant云服務URL (如果為None則使用本地)
            api_key: Qdrant云服務API密鑰
            collection_name: 集合名稱
            vector_size: 向量維度
            distance: 距離度量方式 (cosine, dot, euclidean)
            timeout: 連線逾時時間
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client未安裝。請執行: pip install qdrant-client>=1.6.0"
            )
        
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.timeout = timeout
        # HNSW/Query params via env
        try:
            self.hnsw_m = int(os.getenv("QDRANT_HNSW_M", "32"))
        except Exception:
            self.hnsw_m = 32
        try:
            self.hnsw_ef_construct = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "256"))
        except Exception:
            self.hnsw_ef_construct = 256
        try:
            self.search_ef = int(os.getenv("QDRANT_SEARCH_EF", "128"))
        except Exception:
            self.search_ef = 128
        self.search_exact = os.getenv("QDRANT_SEARCH_EXACT", "0") == "1"
        
        # 距離度量映射
        distance_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID,
        }
        self.distance = distance_map.get(distance.lower(), Distance.COSINE)
        
        # 初始化客戶端
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """初始化Qdrant客戶端和集合"""
        try:
            # 根據設定建立客戶端連線
            if self.url and self.api_key:
                # 使用云服務API
                self.client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=self.timeout
                )
                logger.info(f"[OK] 成功連線到Qdrant雲端服務: {self.url}")
            elif self.url:
                # 使用自定義URL（無API密鑰）
                self.client = QdrantClient(
                    url=self.url,
                    timeout=self.timeout
                )
                logger.info(f"[OK] 成功連線到Qdrant服務: {self.url}")
            else:
                # 使用本地服務（預設）
                self.client = QdrantClient(
                    host="localhost",
                    port=6333,
                    timeout=self.timeout
                )
                logger.info("[OK] 成功連線到本地Qdrant服務: localhost:6333")
            
            # 檢查連線
            collections = self.client.get_collections()
            
            # 建立或取得集合
            self._ensure_collection()
            
        except Exception as e:
            logger.error(f"[ERROR] Qdrant連線失敗: {e}")
            if not self.url:
                logger.info("[INFO] 本地連線失敗，可以考慮使用Qdrant云服務")
                logger.info("[INFO] 或啟動本地服務: docker run -p 6333:6333 qdrant/qdrant")
            else:
                logger.info("[INFO] 請檢查URL和API密鑰是否正確")
            raise
    
    def _ensure_collection(self):
        """確保集合存在，不存在則建立"""
        try:
            # 檢查集合是否存在
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # 建立新集合
                hnsw_cfg = None
                try:
                    hnsw_cfg = models.HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construct)
                except Exception:
                    hnsw_cfg = None
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance
                    ),
                    hnsw_config=hnsw_cfg
                )
                logger.info(f"[OK] 建立Qdrant集合: {self.collection_name}")
            else:
                logger.info(f"[OK] 使用現有Qdrant集合: {self.collection_name}")
                # 嘗試更新 HNSW 設定
                try:
                    self.client.update_collection(
                        collection_name=self.collection_name,
                        hnsw_config=models.HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construct)
                    )
                except Exception as ie:
                    logger.debug(f"跳過更新HNSW設定: {ie}")
            # 確保必要的payload索引
            self._ensure_payload_indexes()
                
        except Exception as e:
            logger.error(f"[ERROR] 集合初始化失敗: {e}")
            raise

    def _ensure_payload_indexes(self):
        """為常用過濾字段建立payload索引"""
        try:
            index_fields = [
                ("memory_type", models.PayloadSchemaType.KEYWORD),
                ("user_id", models.PayloadSchemaType.KEYWORD),
                ("memory_id", models.PayloadSchemaType.KEYWORD),
                ("timestamp", models.PayloadSchemaType.INTEGER),
                ("modality", models.PayloadSchemaType.KEYWORD),  # 感知記憶模態篩選
                ("source", models.PayloadSchemaType.KEYWORD),
                ("external", models.PayloadSchemaType.BOOL),
                ("namespace", models.PayloadSchemaType.KEYWORD),
                # RAG相關字段索引
                ("is_rag_data", models.PayloadSchemaType.BOOL),
                ("rag_namespace", models.PayloadSchemaType.KEYWORD),
                ("data_source", models.PayloadSchemaType.KEYWORD),
            ]
            for field_name, schema_type in index_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema_type,
                    )
                except Exception as ie:
                    # 索引已存在會報錯，忽略
                    logger.debug(f"索引 {field_name} 已存在或建立失敗: {ie}")
        except Exception as e:
            logger.debug(f"建立payload索引時出錯: {e}")
    
    def add_vectors(
        self, 
        vectors: List[List[float]], 
        metadata: List[Dict[str, Any]], 
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        添加向量到Qdrant
        
        Args:
            vectors: 向量列表
            metadata: 元資料列表
            ids: 可選的ID列表
        
        Returns:
            bool: 是否成功
        """
        try:
            if not vectors:
                logger.warning("[WARN] 向量列表為空")
                return False
                
            # 生成ID（如果未提供）
            if ids is None:
                ids = [f"vec_{i}_{int(datetime.now().timestamp() * 1000000)}" 
                       for i in range(len(vectors))]
            
            # 建構點資料
            logger.info(f"[Qdrant] add_vectors start: n_vectors={len(vectors)} n_meta={len(metadata)} collection={self.collection_name}")
            points = []
            for i, (vector, meta, point_id) in enumerate(zip(vectors, metadata, ids)):
                # 確保向量是正確的維度
                try:
                    vlen = len(vector)
                except Exception:
                    logger.error(f"[Qdrant] 非法向量類型: index={i} type={type(vector)} value={vector}")
                    continue
                if vlen != self.vector_size:
                    logger.warning(f"[WARN] 向量維度不匹配: 期望{self.vector_size}, 實際{len(vector)}")
                    continue
                    
                # 添加時間戳到元資料
                meta_with_timestamp = meta.copy()
                meta_with_timestamp["timestamp"] = int(datetime.now().timestamp())
                meta_with_timestamp["added_at"] = int(datetime.now().timestamp())
                if "external" in meta_with_timestamp and not isinstance(meta_with_timestamp.get("external"), bool):
                    # normalize to bool
                    val = meta_with_timestamp.get("external")
                    meta_with_timestamp["external"] = True if str(val).lower() in ("1", "true", "yes") else False
                # 確保點ID是Qdrant接受的類型（無符號整數或UUID字串）
                safe_id: Any
                if isinstance(point_id, int):
                    safe_id = point_id
                elif isinstance(point_id, str):
                    try:
                        uuid.UUID(point_id)
                        safe_id = point_id
                    except Exception:
                        safe_id = str(uuid.uuid4())
                else:
                    safe_id = str(uuid.uuid4())

                point = PointStruct(
                    id=safe_id,
                    vector=vector,
                    payload=meta_with_timestamp
                )
                points.append(point)
            
            if not points:
                logger.warning("[WARN] 沒有有效的向量點")
                return False
            
            # 批量插入
            logger.info(f"[Qdrant] upsert begin: points={len(points)}")
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            logger.info("[Qdrant] upsert done")
            
            logger.info(f"[OK] 成功添加 {len(points)} 個向量到Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] 添加向量失敗: {e}")
            return False
    
    def search_similar(
        self, 
        query_vector: List[float], 
        limit: int = 10, 
        score_threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜尋相似向量
        
        Args:
            query_vector: 查詢向量
            limit: 回傳結果數量限制
            score_threshold: 相似度閾值
            where: 過濾條件
        
        Returns:
            List[Dict]: 搜尋結果
        """
        try:
            if len(query_vector) != self.vector_size:
                logger.error(f"[ERROR] 查詢向量維度錯誤: 期望{self.vector_size}, 實際{len(query_vector)}")
                return []
            
            # 建構過濾器
            query_filter = None
            if where:
                conditions = []
                for key, value in where.items():
                    if isinstance(value, (str, int, float, bool)):
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value)
                            )
                        )
                
                if conditions:
                    query_filter = Filter(must=conditions)
            
            # 執行搜尋
            # 搜尋參數
            search_params = None
            try:
                search_params = models.SearchParams(hnsw_ef=self.search_ef, exact=self.search_exact)
            except Exception:
                search_params = None

            # 相容新舊 qdrant-client API
            # 1.16.0+ 使用 query_points(), <1.16.0 使用 search()
            try:
                # 嘗試新API (qdrant-client >= 1.16.0)
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params
                )
                search_result = response.points
            except AttributeError:
                # 回退到舊API (qdrant-client < 1.16.0)
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params
                )

            # 轉換結果格式
            results = []
            for hit in search_result:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "metadata": hit.payload or {}
                }
                results.append(result)

            logger.debug(f"[DEBUG] Qdrant搜尋回傳 {len(results)} 個結果")
            return results
            
        except Exception as e:
            logger.error(f"[ERROR] 向量搜尋失敗: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """
        刪除向量
        
        Args:
            ids: 要刪除的向量ID列表
        
        Returns:
            bool: 是否成功
        """
        try:
            if not ids:
                return True
                
            operation_info = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=ids
                ),
                wait=True
            )
            
            logger.info(f"[OK] 成功刪除 {len(ids)} 個向量")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] 刪除向量失敗: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        清空集合
        
        Returns:
            bool: 是否成功
        """
        try:
            # 刪除並重新建立集合
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()
            
            logger.info(f"[OK] 成功清空Qdrant集合: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] 清空集合失敗: {e}")
            return False
    
    def delete_memories(self, memory_ids: List[str]):
        """
        刪除指定記憶（通過payload中的 memory_id 過濾刪除）
        
        注意：由於寫入時可能將非UUID的點ID轉換為UUID，這裡不再依賴點ID，
        而是通過payload中的memory_id來匹配刪除，確保一致性。
        """
        try:
            if not memory_ids:
                return
            # 建構 should 過濾條件：memory_id 等於任一給定值
            conditions = [
                FieldCondition(key="memory_id", match=MatchValue(value=mid))
                for mid in memory_ids
            ]
            query_filter = Filter(should=conditions)
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=query_filter),
                wait=True,
            )
            logger.info(f"[OK] 成功按memory_id刪除 {len(memory_ids)} 個Qdrant向量")
        except Exception as e:
            logger.error(f"[ERROR] 刪除記憶失敗: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        取得集合資訊
        
        Returns:
            Dict: 集合資訊
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            info = {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "config": {
                    "vector_size": self.vector_size,
                    "distance": self.distance.value,
                }
            }
            
            return info
            
        except Exception as e:
            logger.error(f"[ERROR] 取得集合資訊失敗: {e}")
            return {}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        取得集合統計資訊（相容抽象介面）
        """
        info = self.get_collection_info()
        if not info:
            return {"store_type": "qdrant", "name": self.collection_name}
        info["store_type"] = "qdrant"
        return info
    
    def health_check(self) -> bool:
        """
        健康檢查
        
        Returns:
            bool: 服務是否健康
        """
        try:
            # 嘗試取得集合列表
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"[ERROR] Qdrant健康檢查失敗: {e}")
            return False
    
    def __del__(self):
        """析構函式，清理資源"""
        if hasattr(self, 'client') and self.client:
            try:
                self.client.close()
            except:
                pass
 

