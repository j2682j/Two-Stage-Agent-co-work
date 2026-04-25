"""
Semantic Memory:結合向量搜尋與知識圖譜的語義記憶實作
- 負責儲存抽象概念、規則、知識
- 需要建立實體和關系的圖譜結構，並與向量資料庫結合
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
import logging
import math
import numpy as np
import spacy

from ..base import BaseMemory, MemoryItem, MemoryConfig
from ..embedding import get_text_embedder, get_dimension


# 日誌設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Entity:
    """語義記憶中的實體節點。"""
    
    def __init__(
        self,
        entity_id: str,
        name: str,
        entity_type: str = "MISC",
        description: str = "",
        properties: Dict[str, Any] = None
    ):
        self.entity_id = entity_id
        self.name = name
        self.entity_type = entity_type  # PERSON, ORG, PRODUCT, SKILL, CONCEPT等
        self.description = description
        self.properties = properties or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.frequency = 1  # 出現頻率
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "properties": self.properties,
            "frequency": self.frequency
        }

class Relation:
    """語義記憶中的關聯邊。"""
    
    def __init__(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str,
        strength: float = 1.0,
        evidence: str = "",
        properties: Dict[str, Any] = None
    ):
        self.from_entity = from_entity
        self.to_entity = to_entity
        self.relation_type = relation_type
        self.strength = strength
        self.evidence = evidence  # 支援該關系的原文字
        self.properties = properties or {}
        self.created_at = datetime.now()
        self.frequency = 1  # 關系出現頻率
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "evidence": self.evidence,
            "properties": self.properties,
            "frequency": self.frequency
        }


class SemanticMemory(BaseMemory):
    """以向量搜尋搭配圖譜推理的混合式語義記憶。"""
    
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        
        # 嵌入模型（統一提供）
        self.embedding_model = None
        self._init_embedding_model()
        
        # 專業資料庫儲存
        self.vector_store = None
        self.graph_store = None
        self._init_databases()
        
        # 實體和關系快取 (用於快速訪問)
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        
        # 實體識別器
        self.nlp = None
        self._init_nlp()
        
        # 記憶儲存
        self.semantic_memories: List[MemoryItem] = []
        self.memory_embeddings: Dict[str, np.ndarray] = {}
        
        logger.info("語義記憶初始化完成，已接上 Qdrant 與 Neo4j")
    
    def _init_embedding_model(self):
        """初始化統一嵌入模型（由 embedding_provider 管理）。"""
        try:
            self.embedding_model = get_text_embedder()
            # 輕量健康檢查與日誌
            try:
                test_vec = self.embedding_model.encode("health_check")
                dim = getattr(self.embedding_model, "dimension", len(test_vec))
                logger.info("[OK] 嵌入模型就緒，維度：%s", dim)
            except Exception:
                logger.info("[OK] 嵌入模型就緒")
        except Exception as e:
            logger.error("[ERROR] 嵌入模型初始化失敗：%s", e)
            raise
    
    def _init_databases(self):
        """初始化專業資料庫儲存"""
        try:
            from core.database_config import get_database_config
            # 取得資料庫設定
            db_config = get_database_config()
            
            # 初始化Qdrant向量資料庫（使用連線管理器避免重復連線）
            from ..storage.qdrant_store import QdrantConnectionManager
            qdrant_config = db_config.get_qdrant_config() or {}
            qdrant_config["vector_size"] = get_dimension()
            self.vector_store = QdrantConnectionManager.get_instance(**qdrant_config)
            logger.info("[OK] Qdrant 向量資料庫初始化完成")
            
            # 初始化Neo4j圖形資料庫
            from ..storage.neo4j_store import Neo4jGraphStore
            neo4j_config = db_config.get_neo4j_config()
            self.graph_store = Neo4jGraphStore(**neo4j_config)
            logger.info("[OK] Neo4j 圖形資料庫初始化完成")
            
            # 驗證連線
            vector_health = self.vector_store.health_check()
            graph_health = self.graph_store.health_check()
            
            if not vector_health:
                logger.warning("[WARN] Qdrant 連線異常，部分功能可能受限")
            if not graph_health:
                logger.warning("[WARN] Neo4j 連線異常，圖搜尋功能可能受限")
            
            logger.info(
                "[INFO] 資料庫健康狀態：Qdrant=%s，Neo4j=%s",
                "[OK]" if vector_health else "[ERROR]",
                "[OK]" if graph_health else "[ERROR]",
            )
            
        except Exception as e:
            logger.error("[ERROR] 資料庫初始化失敗：%s", e)
            logger.info("[INFO] 請檢查資料庫設定與連線狀態")
            logger.info("[INFO] 可參考 DATABASE_SETUP_GUIDE.md 進行設定")
            raise
    
    def _init_nlp(self):
        """初始化NLP處理器 - 智慧多語言支援"""
        try:
            self.nlp_models = {}
            
            # 嘗試載入多語言模型
            models_to_try = [
                ("zh_core_web_sm", "中文"),
                ("en_core_web_sm", "英文")
            ]
            
            loaded_models = []
            for model_name, lang_name in models_to_try:
                try:
                    nlp = spacy.load(model_name)
                    self.nlp_models[model_name] = nlp
                    loaded_models.append(lang_name)
                    logger.info("[OK] 已載入 %s spaCy 模型：%s", lang_name, model_name)
                except OSError:
                    logger.warning("[WARN] %s spaCy 模型不可用：%s", lang_name, model_name)
            
            # 設定主要NLP處理器
            if "zh_core_web_sm" in self.nlp_models:
                self.nlp = self.nlp_models["zh_core_web_sm"]
                logger.info("[INFO] 目前以中文 spaCy 模型作為主要 NLP 管線")
            elif "en_core_web_sm" in self.nlp_models:
                self.nlp = self.nlp_models["en_core_web_sm"]
                logger.info("[INFO] 目前以英文 spaCy 模型作為主要 NLP 管線")
            else:
                self.nlp = None
                logger.warning("[WARN] 沒有可用的 spaCy 模型，實體擷取能力將受限")
            
            if loaded_models:
                logger.info("[INFO] 可用語言模型：%s", ", ".join(loaded_models))
                
        except ImportError:
            logger.warning("[WARN] spaCy 不可用，實體擷取能力將受限")
            self.nlp = None
            self.nlp_models = {}
    
    def add(self, memory_item: MemoryItem) -> str:
        """添加語義記憶"""
        try:
            # 1. 生成文字嵌入
            embedding = self.embedding_model.encode(memory_item.content)
            self.memory_embeddings[memory_item.id] = embedding
            
            # 2. 提取實體和關系
            entities = self._extract_entities(memory_item.content)
            relations = self._extract_relations(memory_item.content, entities)
            
            # 3. 儲存到Neo4j圖形資料庫
            for entity in entities:
                self._add_entity_to_graph(entity, memory_item)
            
            for relation in relations:
                self._add_relation_to_graph(relation, memory_item)
            
            # 4. 儲存到Qdrant向量資料庫
            metadata = {
                "memory_id": memory_item.id,
                "user_id": memory_item.user_id,
                "content": memory_item.content,
                "memory_type": memory_item.memory_type,
                "timestamp": int(memory_item.timestamp.timestamp()),
                "importance": memory_item.importance,
                "entities": [e.entity_id for e in entities],
                "entity_count": len(entities),
                "relation_count": len(relations)
            }
            
            success = self.vector_store.add_vectors(
                vectors=[embedding.tolist()],
                metadata=[metadata],
                ids=[memory_item.id]
            )
            
            if not success:
                logger.warning("[WARN] 向量資料庫寫入失敗，但圖譜資料已成功寫入")
            
            # 5. 添加實體資訊到元資料
            memory_item.metadata["entities"] = [e.entity_id for e in entities]
            memory_item.metadata["relations"] = [
                f"{r.from_entity}-{r.relation_type}-{r.to_entity}" for r in relations
            ]
            
            # 6. 儲存記憶
            self.semantic_memories.append(memory_item)
            
            logger.info("[OK] 已新增語義記憶：%s 個實體、%s 個關聯", len(entities), len(relations))
            return memory_item.id
        
        except Exception as e:
            logger.error("[ERROR] 新增語義記憶失敗：%s", e)
            raise
    
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """搜尋語義記憶"""
        try:
            user_id = kwargs.get("user_id")

            # 1. 向量搜尋
            vector_results = self._vector_search(query, limit * 2, user_id)
            
            # 2. 圖搜尋
            graph_results = self._graph_search(query, limit * 2, user_id)
            
            # 3. 混合排序
            combined_results = self._combine_and_rank_results(
                vector_results, graph_results, query, limit
            )

            # 3.1 計算概率（對 combined_score 做 softmax 歸一化）
            scores = [r.get("combined_score", r.get("vector_score", 0.0)) for r in combined_results]
            if scores:
                import math
                max_s = max(scores)
                exps = [math.exp(s - max_s) for s in scores]
                denom = sum(exps) or 1.0
                probs = [e / denom for e in exps]
            else:
                probs = []
            
            # 4. 過濾已遺忘記憶並轉換為MemoryItem
            result_memories = []
            for idx, result in enumerate(combined_results):
                memory_id = result.get("memory_id")
                
                # 檢查是否已遺忘
                memory = next((m for m in self.semantic_memories if m.id == memory_id), None)
                if memory and memory.metadata.get("forgotten", False):
                    continue  # 跳過已遺忘的記憶
                
                # 處理時間戳
                timestamp = result.get("timestamp")
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except ValueError:
                        timestamp = datetime.now()
                elif isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp)
                else:
                    timestamp = datetime.now()
                
                # 直接從結果資料建構MemoryItem（附帶分數與概率）
                memory_item = MemoryItem(
                    id=result["memory_id"],
                    content=result["content"],
                    memory_type="semantic",
                    user_id=result.get("user_id", "default"),
                    timestamp=timestamp,
                    importance=result.get("importance", 0.5),
                    metadata={
                        **result.get("metadata", {}),
                        "combined_score": result.get("combined_score", 0.0),
                        "vector_score": result.get("vector_score", 0.0),
                        "graph_score": result.get("graph_score", 0.0),
                        "probability": probs[idx] if idx < len(probs) else 0.0,
                    }
                )
                result_memories.append(memory_item)
            
            logger.info("[OK] 已取回 %s 筆相關語義記憶", len(result_memories))
            return result_memories[:limit]
                
        except Exception as e:
            logger.error("[ERROR] 檢索語義記憶失敗：%s", e)
            return []
    
    def _vector_search(self, query: str, limit: int, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Qdrant向量搜尋"""
        try:
            # 生成查詢向量
            query_embedding = self.embedding_model.encode(query)
            
            # 建構過濾條件
            where_filter = {"memory_type": "semantic"}
            if user_id:
                where_filter["user_id"] = user_id

            # Qdrant向量搜尋
            results = self.vector_store.search_similar(
                query_vector=query_embedding.tolist(),
                limit=limit,
                where=where_filter if where_filter else None
            )

            # 轉換結果格式以保持相容性
            formatted_results = []
            for result in results:
                formatted_result = {
                    "id": result["id"],
                    "score": result["score"],
                    **result["metadata"]  # 包含所有元資料
                }
                formatted_results.append(formatted_result)

            logger.debug(f"[DEBUG] Qdrant向量搜尋回傳 {len(formatted_results)} 個結果")
            return formatted_results
                
        except Exception as e:
            logger.error("[ERROR] Qdrant 向量搜尋失敗：%s", e)
            return []

    def _graph_search(self, query: str, limit: int, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Neo4j圖搜尋"""
        try:
            # 從查詢中提取實體
            query_entities = self._extract_entities(query)
            
            if not query_entities:
                # 如果沒有提取到實體，嘗試按名稱搜尋
                entities_by_name = self.graph_store.search_entities_by_name(
                    name_pattern=query, 
                    limit=10
                )
                if entities_by_name:
                    query_entities = [Entity(
                        entity_id=e["id"],
                        name=e["name"],
                        entity_type=e["type"]
                    ) for e in entities_by_name[:3]]
                else:
                    return []
            
            # 在Neo4j圖中查找相關實體和記憶
            related_memory_ids = set()
            
            for entity in query_entities:
                try:
                    # 查找相關實體
                    related_entities = self.graph_store.find_related_entities(
                        entity_id=entity.entity_id,
                        max_depth=2,
                        limit=20
                    )
                    
                    # 收集相關記憶ID
                    for rel_entity in related_entities:
                        if "memory_id" in rel_entity:
                            related_memory_ids.add(rel_entity["memory_id"])
                    
                    # 也添加直接匹配的實體記憶
                    entity_rels = self.graph_store.get_entity_relationships(entity.entity_id)
                    for rel in entity_rels:
                        rel_data = rel.get("relationship", {})
                        if "memory_id" in rel_data:
                            related_memory_ids.add(rel_data["memory_id"])
                            
                except Exception as e:
                    logger.debug(f"圖搜尋實體 {entity.entity_id} 失敗: {e}")
                    continue
            
            # 建構結果 - 從向量資料庫取得完整記憶資訊
            results = []
            for memory_id in list(related_memory_ids)[:limit * 2]:  # 取得更多候選
                try:
                    # 優先從本地快取取得記憶詳情，避免占位向量維度不一致問題
                    mem = self._find_memory_by_id(memory_id)
                    if not mem:
                        continue

                    if user_id and mem.user_id != user_id:
                        continue

                    metadata = {
                        "content": mem.content,
                        "user_id": mem.user_id,
                        "memory_type": mem.memory_type,
                        "importance": mem.importance,
                        "timestamp": int(mem.timestamp.timestamp()),
                        "entities": mem.metadata.get("entities", [])
                    }

                    # 計算圖相關性分數
                    graph_score = self._calculate_graph_relevance_neo4j(metadata, query_entities)

                    results.append({
                        "id": memory_id,
                        "memory_id": memory_id,
                        "content": metadata.get("content", ""),
                        "similarity": graph_score,
                        "user_id": metadata.get("user_id"),
                        "memory_type": metadata.get("memory_type"),
                        "importance": metadata.get("importance", 0.5),
                        "timestamp": metadata.get("timestamp"),
                        "entities": metadata.get("entities", [])
                    })

                except Exception as e:
                    logger.debug(f"取得記憶 {memory_id} 詳情失敗: {e}")
                    continue
            
            # 按圖相關性排序
            results.sort(key=lambda x: x["similarity"], reverse=True)
            logger.debug(f"🕸️ Neo4j圖搜尋回傳 {len(results)} 個結果")
            return results[:limit]
            
        except Exception as e:
            logger.error("[ERROR] Neo4j 圖搜尋失敗：%s", e)
            return []

    def _combine_and_rank_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """混合排序結果 - 僅基於向量與圖分數的簡單融合"""
        # 合併結果，按內容去重
        combined = {}
        content_seen = set()  # 用於內容去重
        
        # 添加向量結果
        for result in vector_results:
            memory_id = result["memory_id"]
            content = result.get("content", "")
            
            # 內容去重：檢查是否已經有相同或高度相似的內容
            content_hash = hash(content.strip())
            if content_hash in content_seen:
                logger.debug(f"[WARN] 跳過重復內容: {content[:30]}...")
                continue
            
            content_seen.add(content_hash)
            combined[memory_id] = {
                **result,
                "vector_score": result.get("score", 0.0), 
                "graph_score": 0.0,
                "content_hash": content_hash
            }
        
        # 添加圖結果
        for result in graph_results:
            memory_id = result["memory_id"]
            content = result.get("content", "")
            content_hash = hash(content.strip())
            
            if memory_id in combined:
                combined[memory_id]["graph_score"] = result.get("similarity", 0.0)
            elif content_hash not in content_seen:
                content_seen.add(content_hash)
                combined[memory_id] = {
                    **result,
                    "vector_score": 0.0,
                    "graph_score": result.get("similarity", 0.0),
                    "content_hash": content_hash
                }
        
        # 計算混合分數：相似度為主，重要性為輔助排序因子
        for memory_id, result in combined.items():
            vector_score = result["vector_score"]
            graph_score = result["graph_score"]
            importance = result.get("importance", 0.5)
            
            # 新評分算法：向量搜尋純基於相似度，重要性作為加權因子
            # 基礎相似度得分（不受重要性影響）
            base_relevance = vector_score * 0.7 + graph_score * 0.3
            
            # 重要性作為乘法加權因子，范圍 [0.8, 1.2]
            # importance in [0,1] -> weight in [0.8,1.2]
            importance_weight = 0.8 + (importance * 0.4)
            
            # 最終得分：相似度 * 重要性權重
            combined_score = base_relevance * importance_weight
            
            # 調試資訊：查看分數分解
            result["debug_info"] = {
                "base_relevance": base_relevance,
                "importance_weight": importance_weight,
                "combined_score": combined_score
            }

            result["combined_score"] = combined_score
        
        # 應用最小相關性閾值
        min_threshold = 0.1  # 最小相關性閾值
        filtered_results = [
            result for result in combined.values() 
            if result["combined_score"] >= min_threshold
        ]

        # 排序並回傳
        sorted_results = sorted(
            filtered_results,
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        # 調試資訊
        logger.debug(f"[DEBUG] 向量結果: {len(vector_results)}, 圖結果: {len(graph_results)}")
        logger.debug(f"[DEBUG] 去重后: {len(combined)}, 過濾后: {len(filtered_results)}")
        
        if logger.level <= logging.DEBUG:
            for i, result in enumerate(sorted_results[:3]):
                logger.debug(f"  結果{i+1}: 向量={result['vector_score']:.3f}, 圖={result['graph_score']:.3f}, 精確={result.get('exact_match_bonus', 0):.3f}, 關鍵詞={result.get('keyword_bonus', 0):.3f}, 公司={result.get('company_bonus', 0):.3f}, 實體={result.get('entity_type_bonus', 0):.3f}, 綜合={result['combined_score']:.3f}")
        
        return sorted_results[:limit]
    
    def _detect_language(self, text: str) -> str:
        """簡單的語言檢測"""
        # 統計中文字符比例（無正則，逐字符判斷范圍）
        chinese_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return "en"
        
        chinese_ratio = chinese_chars / total_chars
        return "zh" if chinese_ratio > 0.3 else "en"
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """智慧多語言實體提取"""
        entities = []
        
        # 檢測文字語言
        lang = self._detect_language(text)
        
        # 選擇合適的spaCy模型
        selected_nlp = None
        if lang == "zh" and "zh_core_web_sm" in self.nlp_models:
            selected_nlp = self.nlp_models["zh_core_web_sm"]
        elif lang == "en" and "en_core_web_sm" in self.nlp_models:
            selected_nlp = self.nlp_models["en_core_web_sm"]
        else:
            # 使用預設模型
            selected_nlp = self.nlp
        
        logger.debug(f"🌐 檢測語言: {lang}, 使用模型: {selected_nlp.meta['name'] if selected_nlp else 'None'}")
        
        # 使用spaCy進行實體識別和詞法分析
        if selected_nlp:
            try:
                doc = selected_nlp(text)
                logger.debug(f"[DEBUG] spaCy處理文字: '{text}' -> {len(doc.ents)} 個實體")
                
                # 儲存詞法分析結果，供Neo4j使用
                self._store_linguistic_analysis(doc, text)
                
                if not doc.ents:
                    # 如果沒有實體，紀錄詳細的詞元資訊
                    logger.debug("[DEBUG] 找不到實體，詞元分析:")
                    for token in doc[:5]:  # 只顯示前5個詞元
                        logger.debug(f"   '{token.text}' -> POS: {token.pos_}, TAG: {token.tag_}, ENT_IOB: {token.ent_iob_}")
                
                for ent in doc.ents:
                    entity = Entity(
                        entity_id=f"entity_{hash(ent.text)}",
                        name=ent.text,
                        entity_type=ent.label_,
                        description=f"從文字中識別的{ent.label_}實體"
                    )
                    entities.append(entity)
                    # 安全取得置信度資訊
                    confidence = "N/A"
                    try:
                        if hasattr(ent._, 'confidence'):
                            confidence = getattr(ent._, 'confidence', 'N/A')
                    except:
                        confidence = "N/A"
                    
                    logger.debug(f"🏷️ spaCy識別實體: '{ent.text}' -> {ent.label_} (置信度: {confidence})")
                
            except Exception as e:
                logger.warning("[WARN] spaCy 實體擷取失敗：%s", e)
                import traceback
                logger.debug(f"詳細錯誤: {traceback.format_exc()}")
        else:
            logger.warning("[WARN] 沒有可用的 spaCy 模型可供實體擷取")
        
        return entities
    
    def _store_linguistic_analysis(self, doc, text: str):
        """儲存spaCy詞法分析結果到Neo4j"""
        if not self.graph_store:
            return
            
        try:
            # 為每個詞元建立節點
            for token in doc:
                # 跳過標點符號和空格
                if token.is_punct or token.is_space:
                    continue
                    
                token_id = f"token_{hash(token.text + token.pos_)}"
                
                # 添加詞元節點到Neo4j
                self.graph_store.add_entity(
                    entity_id=token_id,
                    name=token.text,
                    entity_type="TOKEN",
                    properties={
                        "pos": token.pos_,        # 詞性（NOUN, VERB等）
                        "tag": token.tag_,        # 細粒度標簽
                        "lemma": token.lemma_,    # 詞元原形
                        "is_alpha": token.is_alpha,
                        "is_stop": token.is_stop,
                        "source_text": text[:50],  # 來源文字片段
                        "language": self._detect_language(text)
                    }
                )
                
                # 如果是名詞，可能是潛在的概念
                if token.pos_ in ["NOUN", "PROPN"]:
                    concept_id = f"concept_{hash(token.text)}"
                    self.graph_store.add_entity(
                        entity_id=concept_id,
                        name=token.text,
                        entity_type="CONCEPT",
                        properties={
                            "category": token.pos_,
                            "frequency": 1,  # 可以後續累計
                            "source_text": text[:50]
                        }
                    )
                    
                    # 建立詞元到概念的關系
                    self.graph_store.add_relationship(
                        from_entity_id=token_id,
                        to_entity_id=concept_id,
                        relationship_type="REPRESENTS",
                        properties={"confidence": 1.0}
                    )
            
            # 建立詞元之間的依存關系
            for token in doc:
                if token.is_punct or token.is_space or token.head == token:
                    continue
                    
                from_id = f"token_{hash(token.text + token.pos_)}"
                to_id = f"token_{hash(token.head.text + token.head.pos_)}"
                
                # Neo4j不允許關系類型包含冒號，需要清理
                relation_type = token.dep_.upper().replace(":", "_")
                
                self.graph_store.add_relationship(
                    from_entity_id=from_id,
                    to_entity_id=to_id,
                    relationship_type=relation_type,  # 清理後的依存關系類型
                    properties={
                        "dependency": token.dep_,  # 保留原始依存關系
                        "source_text": text[:50]
                    }
                )
            
            logger.debug(f"🔗 已將詞法分析結果儲存到Neo4j: {len([t for t in doc if not t.is_punct and not t.is_space])} 個詞元")
            
        except Exception as e:
            logger.warning("[WARN] 儲存詞法分析結果失敗：%s", e)
    
    def _extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """提取關系"""
        relations = []
        # 僅保留簡單共現關系，不做任何正則/關鍵詞匹配
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                relations.append(Relation(
                    from_entity=entity1.entity_id,
                    to_entity=entity2.entity_id,
                    relation_type="CO_OCCURS",
                    strength=0.5,
                    evidence=text[:100]
                ))
        return relations
    
    def _add_entity_to_graph(self, entity: Entity, memory_item: MemoryItem):
        """添加實體到Neo4j圖形資料庫"""
        try:
            # 準備實體屬性
            properties = {
                "name": entity.name,
                "description": entity.description,
                "frequency": entity.frequency,
                "memory_id": memory_item.id,
                "user_id": memory_item.user_id,
                "importance": memory_item.importance,
                **entity.properties
            }
            
            # 添加到Neo4j
            success = self.graph_store.add_entity(
                entity_id=entity.entity_id,
                name=entity.name,
                entity_type=entity.entity_type,
                properties=properties
            )
            
            if success:
                # 同時更新本地快取
                if entity.entity_id in self.entities:
                    self.entities[entity.entity_id].frequency += 1
                    self.entities[entity.entity_id].updated_at = datetime.now()
                else:
                    self.entities[entity.entity_id] = entity
                    
            return success
            
        except Exception as e:
            logger.error("[ERROR] 新增實體到圖形資料庫失敗：%s", e)
            return False
    
    def _add_relation_to_graph(self, relation: Relation, memory_item: MemoryItem):
        """添加關系到Neo4j圖形資料庫"""
        try:
            # 準備關系屬性
            properties = {
                "strength": relation.strength,
                "memory_id": memory_item.id,
                "user_id": memory_item.user_id,
                "importance": memory_item.importance,
                "evidence": relation.evidence
            }
            
            # 添加到Neo4j
            success = self.graph_store.add_relationship(
                from_entity_id=relation.from_entity,
                to_entity_id=relation.to_entity,
                relationship_type=relation.relation_type,
                properties=properties
            )
            
            if success:
                # 同時更新本地快取
                self.relations.append(relation)
                
            return success
            
        except Exception as e:
            logger.error("[ERROR] 新增關聯到圖形資料庫失敗：%s", e)
            return False
    
    def _calculate_graph_relevance_neo4j(self, memory_metadata: Dict[str, Any], query_entities: List[Entity]) -> float:
        """計算Neo4j圖相關性分數"""
        try:
            memory_entities = memory_metadata.get("entities", [])
            if not memory_entities or not query_entities:
                return 0.0
            
            # 實體匹配度
            query_entity_ids = {e.entity_id for e in query_entities}
            matching_entities = len(set(memory_entities).intersection(query_entity_ids))
            entity_score = matching_entities / len(query_entity_ids) if query_entity_ids else 0
            
            # 實體數量加權
            entity_count = memory_metadata.get("entity_count", 0)
            entity_density = min(entity_count / 10, 1.0)  # 歸一化到[0,1]
            
            # 關系數量加權
            relation_count = memory_metadata.get("relation_count", 0)
            relation_density = min(relation_count / 5, 1.0)  # 歸一化到[0,1]
            
            # 綜合分數
            relevance_score = (
                entity_score * 0.6 +           # 實體匹配權重60%
                entity_density * 0.2 +         # 實體密度權重20%
                relation_density * 0.2         # 關系密度權重20%
            )
            
            return min(relevance_score, 1.0)
            
        except Exception as e:
            logger.debug(f"計算圖相關性失敗: {e}")
            return 0.0

    def _add_or_update_entity(self, entity: Entity):
        """添加或更新實體"""
        if entity.entity_id in self.entities:
            # 更新現有實體
            existing = self.entities[entity.entity_id]
            existing.frequency += 1
            existing.updated_at = datetime.now()
        else:
            # 添加新實體
            self.entities[entity.entity_id] = entity
    
    def _add_or_update_relation(self, relation: Relation):
        """添加或更新關系"""
        # 檢查是否已存在相同關系
        existing_relation = None
        for r in self.relations:
            if (r.from_entity == relation.from_entity and
                r.to_entity == relation.to_entity and
                r.relation_type == relation.relation_type):
                existing_relation = r
                break
        
        if existing_relation:
            # 更新現有關系
            existing_relation.frequency += 1
            existing_relation.strength = min(1.0, existing_relation.strength + 0.1)
        else:
            # 添加新關系
            self.relations.append(relation)
    
    
    def _find_memory_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        """根據ID查找記憶"""
        logger.debug(f"[DEBUG] 查找記憶ID: {memory_id}, 目前記憶數: {len(self.semantic_memories)}")
        for memory in self.semantic_memories:
            if memory.id == memory_id:
                logger.debug(f"[OK] 找到記憶: {memory.content[:50]}...")
                return memory
        logger.debug(f"[ERROR] 找不到記憶ID: {memory_id}")
        return None
    
    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """更新語義記憶"""
        memory = self._find_memory_by_id(memory_id)
        if not memory:
            return False
        
        try:
            if content is not None:
                # 重新生成嵌入和提取實體
                embedding = self.embedding_model.encode(content)
                self.memory_embeddings[memory_id] = embedding
                
                # 清理舊的實體關系
                old_entities = memory.metadata.get("entities", [])
                self._cleanup_entities_and_relations(old_entities)
                
                # 提取新的實體和關系
                memory.content = content
                entities = self._extract_entities(content)
                relations = self._extract_relations(content, entities)
                
                # 更新知識圖譜
                for entity in entities:
                    self._add_or_update_entity(entity)
                for relation in relations:
                    self._add_or_update_relation(relation)
                
                # 更新元資料
                memory.metadata["entities"] = [e.entity_id for e in entities]
                memory.metadata["relations"] = [
                    f"{r.from_entity}-{r.relation_type}-{r.to_entity}" for r in relations
                ]
                
            if importance is not None:
                memory.importance = importance
            
            if metadata is not None:
                memory.metadata.update(metadata)
                
                return True
            
        except Exception as e:
            logger.error("[ERROR] 更新語義記憶失敗：%s", e)
        return False
    
    def remove(self, memory_id: str) -> bool:
        """刪除語義記憶"""
        memory = self._find_memory_by_id(memory_id)
        if not memory:
            return False
        
        try:
            # 刪除向量
            self.vector_store.delete_memories([memory_id])
            
            # 清理實體和關系
            entities = memory.metadata.get("entities", [])
            self._cleanup_entities_and_relations(entities)
            
            # 刪除記憶
            self.semantic_memories.remove(memory)
            if memory_id in self.memory_embeddings:
                del self.memory_embeddings[memory_id]
                
                return True
            
        except Exception as e:
            logger.error("[ERROR] 刪除語義記憶失敗：%s", e)
        return False
    
    def _cleanup_entities_and_relations(self, entity_ids: List[str]):
        """清理實體和關系"""
        # 這裡可以實現更智慧的清理邏輯
        # 例如，如果實體不再被任何記憶引用，則刪除它
        pass
    
    def has_memory(self, memory_id: str) -> bool:
        """檢查記憶是否存在"""
        return self._find_memory_by_id(memory_id) is not None
    
    def forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 30) -> int:
        """語義記憶遺忘機制（硬刪除）"""
        forgotten_count = 0
        current_time = datetime.now()
        
        to_remove = []  # 收集要刪除的記憶ID
        
        for memory in self.semantic_memories:
            should_forget = False
            
            if strategy == "importance_based":
                # 基於重要性遺忘
                if memory.importance < threshold:
                    should_forget = True
            elif strategy == "time_based":
                # 基於時間遺忘
                cutoff_time = current_time - timedelta(days=max_age_days)
                if memory.timestamp < cutoff_time:
                    should_forget = True
            elif strategy == "capacity_based":
                # 基於容量遺忘（保留最重要的）
                if len(self.semantic_memories) > self.config.max_capacity:
                    sorted_memories = sorted(self.semantic_memories, key=lambda m: m.importance)
                    excess_count = len(self.semantic_memories) - self.config.max_capacity
                    if memory in sorted_memories[:excess_count]:
                        should_forget = True
            
            if should_forget:
                to_remove.append(memory.id)
        
        # 執行硬刪除
        for memory_id in to_remove:
            if self.remove(memory_id):
                forgotten_count += 1
                logger.info("已硬刪除語義記憶：%s...（策略：%s）", memory_id[:8], strategy)
        
        return forgotten_count

    def clear(self):
        """清空所有語義記憶 - 包括專業資料庫"""
        try:
            # 清空Qdrant向量資料庫
            if self.vector_store:
                success = self.vector_store.clear_collection()
                if success:
                    logger.info("[OK] 已清空 Qdrant 向量資料庫")
                else:
                    logger.warning("[WARN] 清空 Qdrant 向量資料庫失敗")
            
            # 清空Neo4j圖形資料庫
            if self.graph_store:
                success = self.graph_store.clear_all()
                if success:
                    logger.info("[OK] 已清空 Neo4j 圖形資料庫")
                else:
                    logger.warning("[WARN] 清空 Neo4j 圖形資料庫失敗")
            
            # 清空本地快取
            self.semantic_memories.clear()
            self.memory_embeddings.clear()
            self.entities.clear()
            self.relations.clear()
            
            logger.info("[OK] 已完全清空語義記憶系統")
            
        except Exception as e:
            logger.error("[ERROR] 清空語義記憶失敗：%s", e)
            # 即使資料庫清空失敗，也要清空本地快取
        self.semantic_memories.clear()
        self.memory_embeddings.clear()
        self.entities.clear()
        self.relations.clear()

    def get_all(self) -> List[MemoryItem]:
        """取得所有語義記憶"""
        return self.semantic_memories.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """取得語義記憶統計資訊"""
        graph_stats = {}
        try:
            if self.graph_store:
                graph_stats = self.graph_store.get_stats() or {}
        except Exception:
            graph_stats = {}

        # 硬刪除模式：所有記憶都是活躍的
        active_memories = self.semantic_memories

        return {
            "count": len(active_memories),  # 活躍記憶數量
            "forgotten_count": 0,  # 硬刪除模式下已遺忘的記憶會被直接刪除
            "total_count": len(self.semantic_memories),  # 總記憶數量
            "entities_count": len(self.entities),
            "relations_count": len(self.relations),
            "graph_nodes": graph_stats.get("total_nodes", 0),
            "graph_edges": graph_stats.get("total_relationships", 0),
            "avg_importance": sum(m.importance for m in active_memories) / len(active_memories) if active_memories else 0.0,
            "memory_type": "enhanced_semantic"
        }
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """取得實體"""
        return self.entities.get(entity_id)
    
    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """搜尋實體"""
        query_lower = query.lower()
        scored_entities = []
        
        for entity in self.entities.values():
            score = 0.0
            
            # 名稱匹配
            if query_lower in entity.name.lower():
                score += 2.0
            
            # 類型匹配
            if query_lower in entity.entity_type.lower():
                score += 1.0
            
            # 描述匹配
            if query_lower in entity.description.lower():
                score += 0.5
            
            # 頻率權重
            score *= math.log(1 + entity.frequency)
            
            if score > 0:
                scored_entities.append((score, entity))
        
        scored_entities.sort(key=lambda x: x[0], reverse=True)
        return [entity for _, entity in scored_entities[:limit]]
    
    def get_related_entities(
        self,
        entity_id: str,
        relation_types: List[str] = None,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """取得相關實體 - 使用Neo4j圖形資料庫"""
        
        related = []
        
        try:
            # 使用Neo4j圖形資料庫查找相關實體
            if not self.graph_store:
                logger.warning("[WARN] Neo4j 圖形資料庫不可用")
                return []
            
            # 使用Neo4j查找相關實體
            related_entities = self.graph_store.find_related_entities(
                entity_id=entity_id,
                relationship_types=relation_types,
                max_depth=max_hops,
                limit=50
            )
            
            # 轉換格式以保持相容性
            for entity_data in related_entities:
                # 嘗試從本地快取取得實體對象
                entity_obj = self.entities.get(entity_data.get("id"))
                if not entity_obj:
                    # 如果本地快取沒有，建立臨時實體對象
                    entity_obj = Entity(
                        entity_id=entity_data.get("id", entity_id),
                        name=entity_data.get("name", ""),
                        entity_type=entity_data.get("type", "MISC")
                    )
                
                    related.append({
                    "entity": entity_obj,
                    "relation_type": entity_data.get("relationship_path", ["RELATED"])[-1] if entity_data.get("relationship_path") else "RELATED",
                    "strength": 1.0 / max(entity_data.get("distance", 1), 1),  # 距離越近強度越高
                    "distance": entity_data.get("distance", max_hops)
                })
            
            # 按距離和強度排序
            related.sort(key=lambda x: (x["distance"], -x["strength"]))
            
        except Exception as e:
            logger.error("[ERROR] 取得相關實體失敗：%s", e)
        
        return related
    
    def export_knowledge_graph(self) -> Dict[str, Any]:
        """匯出知識圖譜 - 從Neo4j取得統計資訊"""
        try:
            # 從Neo4j取得統計資訊
            stats = {}
            if self.graph_store:
                stats = self.graph_store.get_stats()
            
            return {
                "entities": {eid: entity.to_dict() for eid, entity in self.entities.items()},
                "relations": [relation.to_dict() for relation in self.relations],
                "graph_stats": {
                    "total_nodes": stats.get("total_nodes", 0),
                    "entity_nodes": stats.get("entity_nodes", 0),
                    "memory_nodes": stats.get("memory_nodes", 0),
                    "total_relationships": stats.get("total_relationships", 0),
                    "cached_entities": len(self.entities),
                    "cached_relations": len(self.relations)
                }
            }
        except Exception as e:
            logger.error("[ERROR] 匯出知識圖譜失敗：%s", e)
            return {
                "entities": {},
                "relations": [],
                "graph_stats": {"error": str(e)}
            }

