"""
資料庫設定管理
支援Qdrant向量資料庫和Neo4j圖形資料庫的設定
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

# Load environment variables early so DB configs pick them up
load_dotenv()


class QdrantConfig(BaseModel):
    """
    負責在 core.database_config 中封裝 QdrantConfig，封裝儲存後端操作，處理資料寫入、查詢與連線管理。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    # 連線設定
    url: Optional[str] = Field(
        default=None,
        description="Qdrant服務URL (云服務或自定義URL)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API密鑰 (云服務需要)"
    )
    
    # 集合設定
    collection_name: str = Field(
        default="hello_agents_vectors",
        description="向量集合名稱"
    )
    vector_size: int = Field(
        default=384,
        description="向量維度"
    )
    distance: str = Field(
        default="cosine",
        description="距離度量方式 (cosine, dot, euclidean)"
    )
    
    # 連線設定
    timeout: int = Field(
        default=30,
        description="連線逾時時間(秒)"
    )
    
    @classmethod
    def from_env(cls) -> "QdrantConfig":
        """
        負責執行 QdrantConfig 中的 from_env 流程，依照 QdrantConfig 的流程需求處理 from_env 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 'QdrantConfig'。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return cls(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION", "hello_agents_vectors"),
            vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "384")),
            distance=os.getenv("QDRANT_DISTANCE", "cosine"),
            timeout=int(os.getenv("QDRANT_TIMEOUT", "30"))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        負責執行 QdrantConfig 中的 to_dict 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.model_dump(exclude_none=True)


class Neo4jConfig(BaseModel):
    """
    負責在 core.database_config 中封裝 Neo4jConfig，封裝儲存後端操作，處理資料寫入、查詢與連線管理。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    # 連線設定
    uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j連線URI"
    )
    username: str = Field(
        default="neo4j",
        description="使用者名"
    )
    password: str = Field(
        default="hello-agents-password",
        description="密碼"
    )
    database: str = Field(
        default="neo4j",
        description="資料庫名稱"
    )
    
    # 連線池設定
    max_connection_lifetime: int = Field(
        default=3600,
        description="最大連線生命周期(秒)"
    )
    max_connection_pool_size: int = Field(
        default=50,
        description="最大連線池大小"
    )
    connection_acquisition_timeout: int = Field(
        default=60,
        description="連線取得逾時(秒)"
    )
    
    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """
        負責執行 Neo4jConfig 中的 from_env 流程，依照 Neo4jConfig 的流程需求處理 from_env 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 'Neo4jConfig'。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "hello-agents-password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            max_connection_lifetime=int(os.getenv("NEO4J_MAX_CONNECTION_LIFETIME", "3600")),
            max_connection_pool_size=int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "50")),
            connection_acquisition_timeout=int(os.getenv("NEO4J_CONNECTION_TIMEOUT", "60"))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        負責執行 Neo4jConfig 中的 to_dict 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.model_dump()


class DatabaseConfig(BaseModel):
    """
    負責在 core.database_config 中封裝 DatabaseConfig，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    qdrant: QdrantConfig = Field(
        default_factory=QdrantConfig,
        description="Qdrant向量資料庫設定"
    )
    neo4j: Neo4jConfig = Field(
        default_factory=Neo4jConfig,
        description="Neo4j圖形資料庫設定"
    )
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """
        負責執行 DatabaseConfig 中的 from_env 流程，依照 DatabaseConfig 的流程需求處理 from_env 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 'DatabaseConfig'。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return cls(
            qdrant=QdrantConfig.from_env(),
            neo4j=Neo4jConfig.from_env()
        )
    
    def get_qdrant_config(self) -> Dict[str, Any]:
        """
        負責執行 DatabaseConfig 中的 get_qdrant_config 流程，依照 DatabaseConfig 的流程需求處理 get_qdrant_config 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.qdrant.to_dict()
    
    def get_neo4j_config(self) -> Dict[str, Any]:
        """
        負責執行 DatabaseConfig 中的 get_neo4j_config 流程，依照 DatabaseConfig 的流程需求處理 get_neo4j_config 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.neo4j.to_dict()
    
    def validate_connections(self) -> Dict[str, bool]:
        """
        負責執行 DatabaseConfig 中的 validate_connections 流程，檢查目前輸入、狀態或條件是否符合流程繼續執行的要求。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, bool]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        results = {}
        
        # 驗證Qdrant設定
        try:
            from ..memory.storage.qdrant_store import QdrantVectorStore
            qdrant_store = QdrantVectorStore(**self.get_qdrant_config())
            results["qdrant"] = qdrant_store.health_check()
            logger.info(f"✅ Qdrant連線驗證: {'成功' if results['qdrant'] else '失敗'}")
        except Exception as e:
            results["qdrant"] = False
            logger.error(f"❌ Qdrant連線驗證失敗: {e}")
        
        # 驗證Neo4j設定
        try:
            from ..memory.storage.neo4j_store import Neo4jGraphStore
            neo4j_store = Neo4jGraphStore(**self.get_neo4j_config())
            results["neo4j"] = neo4j_store.health_check()
            logger.info(f"✅ Neo4j連線驗證: {'成功' if results['neo4j'] else '失敗'}")
        except Exception as e:
            results["neo4j"] = False
            logger.error(f"❌ Neo4j連線驗證失敗: {e}")
        
        return results


# 全局設定實例
db_config = DatabaseConfig.from_env()


def get_database_config() -> DatabaseConfig:
    """
    負責執行 core.database_config 中的 get_database_config 流程，依照 core.database_config 的流程需求處理 get_database_config 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 DatabaseConfig。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return db_config


def update_database_config(**kwargs) -> None:
    """
    負責執行 core.database_config 中的 update_database_config 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
    
    Args:
        **kwargs: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 None。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    global db_config
    
    if "qdrant" in kwargs:
        db_config.qdrant = QdrantConfig(**kwargs["qdrant"])
    
    if "neo4j" in kwargs:
        db_config.neo4j = Neo4jConfig(**kwargs["neo4j"])
    
    logger.info("✅ 資料庫設定已更新")
 