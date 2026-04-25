"""???????"""


import os
from typing import Optional, Dict, Any
from pydantic import BaseModel

class Config(BaseModel):
    """HelloAgents設定類別"""
    
    # LLM設定
    default_model: str = "gpt-3.5-turbo"
    default_provider: str = "openai"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    # 系統設定
    debug: bool = False
    log_level: str = "INFO"
    
    # 其他設定
    max_history_length: int = 100
    
    @classmethod
    def from_env(cls) -> "Config":
        """從環境變數建立設定"""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return self.dict()
