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
    """Qdrant向量資料庫設定"""
    
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
        """從環境變數建立設定"""
        return cls(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION", "hello_agents_vectors"),
            vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "384")),
            distance=os.getenv("QDRANT_DISTANCE", "cosine"),
            timeout=int(os.getenv("QDRANT_TIMEOUT", "30"))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return self.model_dump(exclude_none=True)


class Neo4jConfig(BaseModel):
    """Neo4j圖形資料庫設定"""
    
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
        """從環境變數建立設定"""
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
        """轉換為字典"""
        return self.model_dump()


class DatabaseConfig(BaseModel):
    """資料庫設定管理器"""
    
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
        """從環境變數建立設定"""
        return cls(
            qdrant=QdrantConfig.from_env(),
            neo4j=Neo4jConfig.from_env()
        )
    
    def get_qdrant_config(self) -> Dict[str, Any]:
        """取得Qdrant設定字典"""
        return self.qdrant.to_dict()
    
    def get_neo4j_config(self) -> Dict[str, Any]:
        """取得Neo4j設定字典"""
        return self.neo4j.to_dict()
    
    def validate_connections(self) -> Dict[str, bool]:
        """驗證資料庫連線設定"""
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
    """取得資料庫設定"""
    return db_config


def update_database_config(**kwargs) -> None:
    """更新資料庫設定"""
    global db_config
    
    if "qdrant" in kwargs:
        db_config.qdrant = QdrantConfig(**kwargs["qdrant"])
    
    if "neo4j" in kwargs:
        db_config.neo4j = Neo4jConfig(**kwargs["neo4j"])
    
    logger.info("✅ 資料庫設定已更新")
 