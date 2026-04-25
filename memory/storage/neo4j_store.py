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
    """Neo4j圖形資料庫儲存實現"""
    
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
        初始化Neo4j圖儲存 (支援云API)
        
        Args:
            uri: Neo4j連線URI (本地: bolt://localhost:7687, 云: neo4j+s://xxx.databases.neo4j.io)
            username: 使用者名
            password: 密碼
            database: 資料庫名稱
            max_connection_lifetime: 最大連線生命周期(秒)
            max_connection_pool_size: 最大連線池大小
            connection_acquisition_timeout: 連線取得逾時(秒)
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
        """初始化Neo4j驅動"""
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
        """建立必要的索引以提高查詢性能"""
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
        添加實體節點
        
        Args:
            entity_id: 實體ID
            name: 實體名稱
            entity_type: 實體類型
            properties: 附加屬性
        
        Returns:
            bool: 是否成功
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
        添加實體間關系
        
        Args:
            from_entity_id: 源實體ID
            to_entity_id: 目標實體ID  
            relationship_type: 關系類型
            properties: 關系屬性
        
        Returns:
            bool: 是否成功
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
        查找相關實體
        
        Args:
            entity_id: 起始實體ID
            relationship_types: 關系類型過濾
            max_depth: 最大搜尋深度
            limit: 結果限制
        
        Returns:
            List[Dict]: 相關實體列表
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
        按名稱搜尋實體
        
        Args:
            name_pattern: 名稱模式 (支援部分匹配)
            entity_types: 實體類型過濾
            limit: 結果限制
        
        Returns:
            List[Dict]: 匹配的實體列表
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
        取得實體的所有關系
        
        Args:
            entity_id: 實體ID
        
        Returns:
            List[Dict]: 關系列表
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
        刪除實體及其所有關系
        
        Args:
            entity_id: 實體ID
        
        Returns:
            bool: 是否成功
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
        清空所有資料
        
        Returns:
            bool: 是否成功
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
        取得圖形資料庫統計資訊
        
        Returns:
            Dict: 統計資訊
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
        健康檢查
        
        Returns:
            bool: 服務是否健康
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
        """析構函式，清理資源"""
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.close()
            except:
                pass

