"""
Neo4j圖形資料庫儲存實現
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None

logger = logging.getLogger(__name__)

class Neo4jGraphStore:
    """
    負責在 memory.storage.neo4j_store 中封裝 Neo4jGraphStore，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        uri: 記憶系統提供的檢索結果、寫入資料或操作介面。
        username: 記憶系統提供的檢索結果、寫入資料或操作介面。
        password: 記憶系統提供的檢索結果、寫入資料或操作介面。
        database: 記憶系統提供的檢索結果、寫入資料或操作介面。
        max_connection_lifetime: 控制檢索、篩選或輸出數量的數值參數。
        max_connection_pool_size: 控制檢索、篩選或輸出數量的數值參數。
        connection_acquisition_timeout: 記憶系統提供的檢索結果、寫入資料或操作介面。
        **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j", 
        password: str = "hello-agents-password",
        database: str = "neo4j",
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50,
        connection_acquisition_timeout: int = 60,
        **kwargs
    ):
        """
        負責執行 Neo4jGraphStore 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            uri: 記憶系統提供的檢索結果、寫入資料或操作介面。
            username: 記憶系統提供的檢索結果、寫入資料或操作介面。
            password: 記憶系統提供的檢索結果、寫入資料或操作介面。
            database: 記憶系統提供的檢索結果、寫入資料或操作介面。
            max_connection_lifetime: 控制檢索、篩選或輸出數量的數值參數。
            max_connection_pool_size: 控制檢索、篩選或輸出數量的數值參數。
            connection_acquisition_timeout: 記憶系統提供的檢索結果、寫入資料或操作介面。
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j未安裝。請執行: pip install neo4j>=5.0.0"
            )
        
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        
        # 初始化驅動
        self.driver = None
        self._initialize_driver(
            max_connection_lifetime=max_connection_lifetime,
            max_connection_pool_size=max_connection_pool_size,
            connection_acquisition_timeout=connection_acquisition_timeout
        )
        
        # 建立索引
        self._create_indexes()
    
    def _initialize_driver(self, **config):
        """
        負責執行 Neo4jGraphStore 中的 _initialize_driver 流程，依照 Neo4jGraphStore 的流程需求處理 _initialize_driver 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            **config: 控制此流程行為的設定資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                **config
            )
            
            # 驗證連線
            self.driver.verify_connectivity()
            
            # 檢查是否是云服務
            if "neo4j.io" in self.uri or "aura" in self.uri.lower():
                logger.info(f"[OK] 成功連線到Neo4j云服務: {self.uri}")
            else:
                logger.info(f"[OK] 成功連線到Neo4j服務: {self.uri}")
                
        except AuthError as e:
            logger.error(f"[ERROR] Neo4j認證失敗: {e}")
            logger.info("[INFO] 請檢查使用者名和密碼是否正確")
            raise
        except ServiceUnavailable as e:
            logger.error(f"[ERROR] Neo4j服務不可用: {e}")
            if "localhost" in self.uri:
                logger.info("[INFO] 本地連線失敗，可以考慮使用Neo4j Aura云服務")
                logger.info("[INFO] 或啟動本地服務: docker run -p 7474:7474 -p 7687:7687 neo4j:5.14")
            else:
                logger.info("[INFO] 請檢查URL和網路連線")
            raise
        except Exception as e:
            logger.error(f"[ERROR] Neo4j連線失敗: {e}")
            raise
    
    def _create_indexes(self):
        """
        負責執行 Neo4jGraphStore 中的 _create_indexes 流程，依照 Neo4jGraphStore 的流程需求處理 _create_indexes 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        indexes = [
            # 實體索引
            "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            
            # 記憶索引
            "CREATE INDEX memory_id_index IF NOT EXISTS FOR (m:Memory) ON (m.id)",
            "CREATE INDEX memory_type_index IF NOT EXISTS FOR (m:Memory) ON (m.memory_type)",
            "CREATE INDEX memory_timestamp_index IF NOT EXISTS FOR (m:Memory) ON (m.timestamp)",
        ]
        
        with self.driver.session(database=self.database) as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                except Exception as e:
                    logger.debug(f"索引建立跳過 (可能已存在): {e}")
        
        logger.info("[OK] Neo4j索引建立完成")
    
    def add_entity(self, entity_id: str, name: str, entity_type: str, properties: Dict[str, Any] = None) -> bool:
        """
        負責執行 Neo4jGraphStore 中的 add_entity 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            entity_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            name: 記憶系統提供的檢索結果、寫入資料或操作介面。
            entity_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            properties: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            props = properties or {}
            props.update({
                "id": entity_id,
                "name": name,
                "type": entity_type,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
            
            query = """
            MERGE (e:Entity {id: $entity_id})
            SET e += $properties
            RETURN e
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id, properties=props)
                record = result.single()
                
                if record:
                    logger.debug(f"[OK] 添加實體: {name} ({entity_type})")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] 添加實體失敗: {e}")
            return False
    
    def add_relationship(
        self, 
        from_entity_id: str, 
        to_entity_id: str, 
        relationship_type: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        """
        負責執行 Neo4jGraphStore 中的 add_relationship 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            from_entity_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            to_entity_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            relationship_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            properties: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            props = properties or {}
            props.update({
                "type": relationship_type,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
            
            query = f"""
            MATCH (from:Entity {{id: $from_id}})
            MATCH (to:Entity {{id: $to_id}})
            MERGE (from)-[r:{relationship_type}]->(to)
            SET r += $properties
            RETURN r
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    from_id=from_entity_id,
                    to_id=to_entity_id,
                    properties=props
                )
                record = result.single()
                
                if record:
                    logger.debug(f"[OK] 添加關系: {from_entity_id} -{relationship_type}-> {to_entity_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] 添加關系失敗: {e}")
            return False
    
    def find_related_entities(
        self, 
        entity_id: str, 
        relationship_types: List[str] = None,
        max_depth: int = 2,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        負責執行 Neo4jGraphStore 中的 find_related_entities 流程，從記憶圖、向量索引或任務關聯中取回相關案例與策略提醒。
        
        Args:
            entity_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            relationship_types: 記憶系統提供的檢索結果、寫入資料或操作介面。
            max_depth: 控制檢索、篩選或輸出數量的數值參數。
            limit: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            # 建構關系類型過濾
            rel_filter = ""
            if relationship_types:
                rel_types = "|".join(relationship_types)
                rel_filter = f":{rel_types}"
            
            query = f"""
            MATCH path = (start:Entity {{id: $entity_id}})-[r{rel_filter}*1..{max_depth}]-(related:Entity)
            WHERE start.id <> related.id
            RETURN DISTINCT related, 
                   length(path) as distance,
                   [rel in relationships(path) | type(rel)] as relationship_path
            ORDER BY distance, related.name
            LIMIT $limit
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id, limit=limit)
                
                entities = []
                for record in result:
                    entity_data = dict(record["related"])
                    entity_data["distance"] = record["distance"]
                    entity_data["relationship_path"] = record["relationship_path"]
                    entities.append(entity_data)
                
                logger.debug(f"[DEBUG] 找到 {len(entities)} 個相關實體")
                return entities
                
        except Exception as e:
            logger.error(f"[ERROR] 查找相關實體失敗: {e}")
            return []
    
    def search_entities_by_name(self, name_pattern: str, entity_types: List[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        負責執行 Neo4jGraphStore 中的 search_entities_by_name 流程，從記憶圖、向量索引或任務關聯中取回相關案例與策略提醒。
        
        Args:
            name_pattern: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            entity_types: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            limit: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            # 建構類型過濾
            type_filter = ""
            params = {"pattern": f".*{name_pattern}.*", "limit": limit}
            
            if entity_types:
                type_filter = "AND e.type IN $types"
                params["types"] = entity_types
            
            query = f"""
            MATCH (e:Entity)
            WHERE e.name =~ $pattern {type_filter}
            RETURN e
            ORDER BY e.name
            LIMIT $limit
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, **params)
                
                entities = []
                for record in result:
                    entity_data = dict(record["e"])
                    entities.append(entity_data)
                
                logger.debug(f"[DEBUG] 按名稱搜尋到 {len(entities)} 個實體")
                return entities
                
        except Exception as e:
            logger.error(f"[ERROR] 按名稱搜尋實體失敗: {e}")
            return []
    
    def get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        負責執行 Neo4jGraphStore 中的 get_entity_relationships 流程，依照 Neo4jGraphStore 的流程需求處理 get_entity_relationships 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            entity_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            query = """
            MATCH (e:Entity {id: $entity_id})-[r]-(other:Entity)
            RETURN r, other, 
                   CASE WHEN startNode(r).id = $entity_id THEN 'outgoing' ELSE 'incoming' END as direction
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id)
                
                relationships = []
                for record in result:
                    rel_data = dict(record["r"])
                    other_data = dict(record["other"])
                    
                    relationship = {
                        "relationship": rel_data,
                        "other_entity": other_data,
                        "direction": record["direction"]
                    }
                    relationships.append(relationship)
                
                return relationships
                
        except Exception as e:
            logger.error(f"[ERROR] 取得實體關系失敗: {e}")
            return []
    
    def delete_entity(self, entity_id: str) -> bool:
        """
        負責執行 Neo4jGraphStore 中的 delete_entity 流程，依照 Neo4jGraphStore 的流程需求處理 delete_entity 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            entity_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            query = """
            MATCH (e:Entity {id: $entity_id})
            DETACH DELETE e
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id)
                summary = result.consume()
                
                deleted_count = summary.counters.nodes_deleted
                logger.info(f"[OK] 刪除實體: {entity_id} (刪除 {deleted_count} 個節點)")
                return deleted_count > 0
                
        except Exception as e:
            logger.error(f"[ERROR] 刪除實體失敗: {e}")
            return False
    
    def clear_all(self) -> bool:
        """
        負責執行 Neo4jGraphStore 中的 clear_all 流程，清除或移除指定資源、狀態或註冊資料，維持後續流程的一致性。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            query = "MATCH (n) DETACH DELETE n"
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                summary = result.consume()
                
                deleted_nodes = summary.counters.nodes_deleted
                deleted_relationships = summary.counters.relationships_deleted
                
                logger.info(f"[OK] 清空Neo4j資料庫: 刪除 {deleted_nodes} 個節點, {deleted_relationships} 個關系")
                return True
                
        except Exception as e:
            logger.error(f"[ERROR] 清空資料庫失敗: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        負責執行 Neo4jGraphStore 中的 get_stats 流程，依照 Neo4jGraphStore 的流程需求處理 get_stats 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            queries = {
                "total_nodes": "MATCH (n) RETURN count(n) as count",
                "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
                "entity_nodes": "MATCH (n:Entity) RETURN count(n) as count",
                "memory_nodes": "MATCH (n:Memory) RETURN count(n) as count",
            }
            
            stats = {}
            with self.driver.session(database=self.database) as session:
                for key, query in queries.items():
                    result = session.run(query)
                    record = result.single()
                    stats[key] = record["count"] if record else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"[ERROR] 取得統計資訊失敗: {e}")
            return {}
    
    def health_check(self) -> bool:
        """
        負責執行 Neo4jGraphStore 中的 health_check 流程，依照 Neo4jGraphStore 的流程需求處理 health_check 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as health")
                record = result.single()
                return record["health"] == 1
        except Exception as e:
            logger.error(f"[ERROR] Neo4j健康檢查失敗: {e}")
            return False
    
    def __del__(self):
        """
        負責執行 Neo4jGraphStore 中的 __del__ 流程，依照 Neo4jGraphStore 的流程需求處理 __del__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.close()
            except:
                pass

