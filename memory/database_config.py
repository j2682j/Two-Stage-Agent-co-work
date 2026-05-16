"""memory.database_config 模組。

提供此模組相關的資料結構、流程輔助或整合邏輯。
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
    """QdrantConfig 類別。
    
    封裝此類別負責的狀態、設定與操作流程。
    """
    
    # ???閮剖?
    url: Optional[str] = Field(
        default=None,
        description="Qdrant??URL (鈭????芸?蝢垃RL)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API撖 (鈭???閬?"
    )
    
    # ??閮剖?
    collection_name: str = Field(
        default="hello_agents_vectors",
        description="?????迂"
    )
    vector_size: int = Field(
        default=384,
        description="??蝬剖漲"
    )
    distance: str = Field(
        default="cosine",
        description="頝摨阡??孵? (cosine, dot, euclidean)"
    )
    
    # ???閮剖?
    timeout: int = Field(
        default=30,
        description="????暹???(蝘?"
    )
    
    @classmethod
    def from_env(cls) -> "QdrantConfig":
        """處理 from_env 流程並回傳結果。
        
        回傳:
            此函式的處理結果。
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
        """處理 to_dict 流程並回傳結果。
        
        回傳:
            此函式的處理結果。
        """
        return self.model_dump(exclude_none=True)


class Neo4jConfig(BaseModel):
    """Neo4jConfig 類別。
    
    封裝此類別負責的狀態、設定與操作流程。
    """
    
    # ???閮剖?
    uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j???URI"
    )
    username: str = Field(
        default="neo4j",
        description="雿輻??"
    )
    password: str = Field(
        default="hello-agents-password",
        description="撖Ⅳ"
    )
    database: str = Field(
        default="neo4j",
        description="Neo4j database name"
    )
    
    # ???瘙身摰?
    max_connection_lifetime: int = Field(
        default=3600,
        description="?憭折????冽?(蝘?"
    )
    max_connection_pool_size: int = Field(
        default=50,
        description="Maximum Neo4j connection pool size"
    )
    connection_acquisition_timeout: int = Field(
        default=60,
        description="??????暹?(蝘?"
    )
    
    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """處理 from_env 流程並回傳結果。
        
        回傳:
            此函式的處理結果。
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
        """處理 to_dict 流程並回傳結果。
        
        回傳:
            此函式的處理結果。
        """
        return self.model_dump()


class DatabaseConfig(BaseModel):
    """DatabaseConfig 類別。
    
    封裝此類別負責的狀態、設定與操作流程。
    """
    
    qdrant: QdrantConfig = Field(
        default_factory=QdrantConfig,
        description="Qdrant database configuration"
    )
    neo4j: Neo4jConfig = Field(
        default_factory=Neo4jConfig,
        description="Neo4j database configuration"
    )
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """處理 from_env 流程並回傳結果。
        
        回傳:
            此函式的處理結果。
        """
        return cls(
            qdrant=QdrantConfig.from_env(),
            neo4j=Neo4jConfig.from_env()
        )
    
    def get_qdrant_config(self) -> Dict[str, Any]:
        """取得 get_qdrant_config 對應的資料。
        
        回傳:
            此函式的處理結果。
        """
        return self.qdrant.to_dict()
    
    def get_neo4j_config(self) -> Dict[str, Any]:
        """取得 get_neo4j_config 對應的資料。
        
        回傳:
            此函式的處理結果。
        """
        return self.neo4j.to_dict()
    
    def validate_connections(self) -> Dict[str, bool]:
        """驗證 validate_connections 的狀態或輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        results = {}
        
        # 撽?Qdrant閮剖?
        try:
            from ..memory.storage.qdrant_store import QdrantVectorStore
            qdrant_store = QdrantVectorStore(**self.get_qdrant_config())
            results["qdrant"] = qdrant_store.health_check()
            logger.info(f"??Qdrant???撽?: {'??' if results['qdrant'] else '憭望?'}")
        except Exception as e:
            results["qdrant"] = False
            logger.error(f"??Qdrant???撽?憭望?: {e}")
        
        # 撽?Neo4j閮剖?
        try:
            from ..memory.storage.neo4j_store import Neo4jGraphStore
            neo4j_store = Neo4jGraphStore(**self.get_neo4j_config())
            results["neo4j"] = neo4j_store.health_check()
            logger.info(f"??Neo4j???撽?: {'??' if results['neo4j'] else '憭望?'}")
        except Exception as e:
            results["neo4j"] = False
            logger.error(f"??Neo4j???撽?憭望?: {e}")
        
        return results


# ?典?閮剖?撖虫?
db_config = DatabaseConfig.from_env()


def get_database_config() -> DatabaseConfig:
    """取得 get_database_config 對應的資料。
    
    回傳:
        此函式的處理結果。
    """
    return db_config


def update_database_config(**kwargs) -> None:
    """更新 update_database_config 相關資料。
    
    參數:
        **kwargs: 此流程需要使用的輸入資料。
    
    回傳:
        此函式的處理結果。
    """
    global db_config
    
    if "qdrant" in kwargs:
        db_config.qdrant = QdrantConfig(**kwargs["qdrant"])
    
    if "neo4j" in kwargs:
        db_config.neo4j = Neo4jConfig(**kwargs["neo4j"])
    
    logger.info("??鞈?摨怨身摰歇?湔")
 
