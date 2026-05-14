"""文檔儲存實現

支援多種文檔資料庫後端：
- SQLite: 輕量級關系型資料庫
- PostgreSQL: 企業級關系型資料庫（可擴展）
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import sqlite3
import json
import os
import threading


class DocumentStore(ABC):
    """
    負責在 memory.storage.document_store 中封裝 DocumentStore，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    @abstractmethod
    def add_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: float,
        properties: Dict[str, Any] = None
    ) -> str:
        """
        負責執行 DocumentStore 中的 add_memory 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            memory_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            user_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            content: 記憶系統提供的檢索結果、寫入資料或操作介面。
            memory_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            timestamp: 記憶系統提供的檢索結果、寫入資料或操作介面。
            importance: 記憶系統提供的檢索結果、寫入資料或操作介面。
            properties: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        pass
    
    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        負責執行 DocumentStore 中的 get_memory 流程，依照 DocumentStore 的流程需求處理 get_memory 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            memory_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Optional[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        pass
    
    @abstractmethod
    def search_memories(
        self,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        importance_threshold: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        負責執行 DocumentStore 中的 search_memories 流程，從記憶圖、向量索引或任務關聯中取回相關案例與策略提醒。
        
        Args:
            user_id: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            memory_type: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            start_time: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            end_time: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            importance_threshold: 控制檢索、篩選或輸出數量的數值參數。
            limit: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        pass
    
    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        properties: Dict[str, Any] = None
    ) -> bool:
        """
        負責執行 DocumentStore 中的 update_memory 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            memory_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            content: 記憶系統提供的檢索結果、寫入資料或操作介面。
            importance: 記憶系統提供的檢索結果、寫入資料或操作介面。
            properties: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        pass
    
    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """
        負責執行 DocumentStore 中的 delete_memory 流程，依照 DocumentStore 的流程需求處理 delete_memory 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            memory_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        pass
    
    @abstractmethod
    def get_database_stats(self) -> Dict[str, Any]:
        """
        負責執行 DocumentStore 中的 get_database_stats 流程，依照 DocumentStore 的流程需求處理 get_database_stats 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        pass
    
    @abstractmethod
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        負責執行 DocumentStore 中的 add_document 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            content: 記憶系統提供的檢索結果、寫入資料或操作介面。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        負責執行 DocumentStore 中的 get_document 流程，依照 DocumentStore 的流程需求處理 get_document 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            document_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Optional[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        pass

class SQLiteDocumentStore(DocumentStore):
    """
    負責在 memory.storage.document_store 中封裝 SQLiteDocumentStore，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        db_path: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    _instances = {}  # 儲存已建立的實例
    _initialized_dbs = set()  # 儲存已初始化的資料庫路徑
    
    def __new__(cls, db_path: str = "./memory.db"):
        """
        負責執行 SQLiteDocumentStore 中的 __new__ 流程，依照 SQLiteDocumentStore 的流程需求處理 __new__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            db_path: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        abs_path = os.path.abspath(db_path)
        if abs_path not in cls._instances:
            instance = super(SQLiteDocumentStore, cls).__new__(cls)
            cls._instances[abs_path] = instance
        return cls._instances[abs_path]
    
    def __init__(self, db_path: str = "./memory.db"):
        # 避免重復初始化
        """
        負責執行 SQLiteDocumentStore 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            db_path: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if hasattr(self, '_initialized'):
            return
            
        self.db_path = db_path
        self.local = threading.local()
        
        # 確保目錄存在
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        # 初始化資料庫（只初始化一次）
        abs_path = os.path.abspath(db_path)
        if abs_path not in self._initialized_dbs:
            self._init_database()
            self._initialized_dbs.add(abs_path)
            print(f"[OK] SQLite 文檔儲存初始化完成: {db_path}")
        
        self._initialized = True
    
    def _get_connection(self):
        """
        負責執行 SQLiteDocumentStore 中的 _get_connection 流程，依照 SQLiteDocumentStore 的流程需求處理 _get_connection 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not hasattr(self.local, 'connection'):
            self.local.connection = sqlite3.connect(self.db_path)
            self.local.connection.row_factory = sqlite3.Row  # 使結果可以按列名訪問
        return self.local.connection
    
    def _init_database(self):
        """
        負責執行 SQLiteDocumentStore 中的 _init_database 流程，依照 SQLiteDocumentStore 的流程需求處理 _init_database 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 建立使用者表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 建立記憶表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                importance REAL NOT NULL,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # 建立概念表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 建立記憶-概念關聯表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_concepts (
                memory_id TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                relevance_score REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (memory_id, concept_id),
                FOREIGN KEY (memory_id) REFERENCES memories (id) ON DELETE CASCADE,
                FOREIGN KEY (concept_id) REFERENCES concepts (id) ON DELETE CASCADE
            )
        """)
        
        # 建立概念關系表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concept_relationships (
                from_concept_id TEXT NOT NULL,
                to_concept_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (from_concept_id, to_concept_id, relationship_type),
                FOREIGN KEY (from_concept_id) REFERENCES concepts (id) ON DELETE CASCADE,
                FOREIGN KEY (to_concept_id) REFERENCES concepts (id) ON DELETE CASCADE
            )
        """)
        
        # 建立索引
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories (importance)",
            "CREATE INDEX IF NOT EXISTS idx_memory_concepts_memory ON memory_concepts (memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_concepts_concept ON memory_concepts (concept_id)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        print("[OK] SQLite 資料庫表和索引建立完成")
    
    def add_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: float,
        properties: Dict[str, Any] = None
    ) -> str:
        """
        負責執行 SQLiteDocumentStore 中的 add_memory 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            memory_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            user_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            content: 記憶系統提供的檢索結果、寫入資料或操作介面。
            memory_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            timestamp: 記憶系統提供的檢索結果、寫入資料或操作介面。
            importance: 記憶系統提供的檢索結果、寫入資料或操作介面。
            properties: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 確保使用者存在
        cursor.execute("INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)", (user_id, user_id))
        
        # 插入記憶
        cursor.execute("""
            INSERT OR REPLACE INTO memories 
            (id, user_id, content, memory_type, timestamp, importance, properties, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            memory_id,
            user_id,
            content,
            memory_type,
            timestamp,
            importance,
            json.dumps(properties) if properties else None
        ))
        
        conn.commit()
        return memory_id
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        負責執行 SQLiteDocumentStore 中的 get_memory 流程，依照 SQLiteDocumentStore 的流程需求處理 get_memory 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            memory_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Optional[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, user_id, content, memory_type, timestamp, importance, properties, created_at
            FROM memories
            WHERE id = ?
        """, (memory_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            "memory_id": row["id"],
            "user_id": row["user_id"],
            "content": row["content"],
            "memory_type": row["memory_type"],
            "timestamp": row["timestamp"],
            "importance": row["importance"],
            "properties": json.loads(row["properties"]) if row["properties"] else {},
            "created_at": row["created_at"]
        }
    
    def search_memories(
        self,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        importance_threshold: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        負責執行 SQLiteDocumentStore 中的 search_memories 流程，從記憶圖、向量索引或任務關聯中取回相關案例與策略提醒。
        
        Args:
            user_id: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            memory_type: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            start_time: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            end_time: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            importance_threshold: 控制檢索、篩選或輸出數量的數值參數。
            limit: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 建構查詢條件
        where_conditions = []
        params = []
        
        if user_id:
            where_conditions.append("user_id = ?")
            params.append(user_id)
        
        if memory_type:
            where_conditions.append("memory_type = ?")
            params.append(memory_type)
        
        if start_time:
            where_conditions.append("timestamp >= ?")
            params.append(start_time)
        
        if end_time:
            where_conditions.append("timestamp <= ?")
            params.append(end_time)
        
        if importance_threshold:
            where_conditions.append("importance >= ?")
            params.append(importance_threshold)
        
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        cursor.execute(f"""
            SELECT id, user_id, content, memory_type, timestamp, importance, properties, created_at
            FROM memories
            {where_clause}
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        """, params + [limit])
        
        memories = []
        for row in cursor.fetchall():
            memories.append({
                "memory_id": row["id"],
                "user_id": row["user_id"],
                "content": row["content"],
                "memory_type": row["memory_type"],
                "timestamp": row["timestamp"],
                "importance": row["importance"],
                "properties": json.loads(row["properties"]) if row["properties"] else {},
                "created_at": row["created_at"]
            })
        
        return memories
    
    def update_memory(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        properties: Dict[str, Any] = None
    ) -> bool:
        """
        負責執行 SQLiteDocumentStore 中的 update_memory 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            memory_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            content: 記憶系統提供的檢索結果、寫入資料或操作介面。
            importance: 記憶系統提供的檢索結果、寫入資料或操作介面。
            properties: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 建構更新字段
        update_fields = []
        params = []
        
        if content is not None:
            update_fields.append("content = ?")
            params.append(content)
        
        if importance is not None:
            update_fields.append("importance = ?")
            params.append(importance)
        
        if properties is not None:
            update_fields.append("properties = ?")
            params.append(json.dumps(properties))
        
        if not update_fields:
            return False
        
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        params.append(memory_id)
        
        cursor.execute(f"""
            UPDATE memories
            SET {', '.join(update_fields)}
            WHERE id = ?
        """, params)
        
        conn.commit()
        return cursor.rowcount > 0
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        負責執行 SQLiteDocumentStore 中的 delete_memory 流程，依照 SQLiteDocumentStore 的流程需求處理 delete_memory 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            memory_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        return deleted_count > 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        負責執行 SQLiteDocumentStore 中的 get_database_stats 流程，依照 SQLiteDocumentStore 的流程需求處理 get_database_stats 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # 統計各表的紀錄數
        tables = ["users", "memories", "concepts", "memory_concepts", "concept_relationships"]
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            stats[f"{table}_count"] = cursor.fetchone()["count"]
        
        # 統計記憶類型分布
        cursor.execute("""
            SELECT memory_type, COUNT(*) as count
            FROM memories
            GROUP BY memory_type
        """)
        memory_types = {}
        for row in cursor.fetchall():
            memory_types[row["memory_type"]] = row["count"]
        stats["memory_types"] = memory_types
        
        # 統計使用者分布
        cursor.execute("""
            SELECT user_id, COUNT(*) as count
            FROM memories
            GROUP BY user_id
            ORDER BY count DESC
            LIMIT 10
        """)
        top_users = {}
        for row in cursor.fetchall():
            top_users[row["user_id"]] = row["count"]
        stats["top_users"] = top_users
        
        stats["store_type"] = "sqlite"
        stats["db_path"] = self.db_path
        
        return stats
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        負責執行 SQLiteDocumentStore 中的 add_document 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            content: 記憶系統提供的檢索結果、寫入資料或操作介面。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        import uuid
        import time
        
        doc_id = str(uuid.uuid4())
        user_id = metadata.get("user_id", "system") if metadata else "system"
        
        return self.add_memory(
            memory_id=doc_id,
            user_id=user_id,
            content=content,
            memory_type="document",
            timestamp=int(time.time()),
            importance=0.5,
            properties=metadata or {}
        )
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        負責執行 SQLiteDocumentStore 中的 get_document 流程，依照 SQLiteDocumentStore 的流程需求處理 get_document 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            document_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Optional[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.get_memory(document_id)

    def close(self):
        """
        負責執行 SQLiteDocumentStore 中的 close 流程，清除或移除指定資源、狀態或註冊資料，維持後續流程的一致性。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if hasattr(self.local, 'connection'):
            self.local.connection.close()
            delattr(self.local, 'connection')
            print("[OK] SQLite 連線已關閉")

