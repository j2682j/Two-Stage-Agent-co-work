"""?????"""


from abc import ABC, abstractmethod
from typing import Optional
from .message import Message
from .llm import HelloAgentsLLM
from .config import Config

class Agent(ABC):
    """Agent基類"""
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []
    
    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """執行Agent"""
        pass
    
    def add_message(self, message: Message):
        """添加消息到歷史紀錄"""
        self._history.append(message)
    
    def clear_history(self):
        """清空歷史紀錄"""
        self._history.clear()
    
    def get_history(self) -> list[Message]:
        """取得歷史紀錄"""
        return self._history.copy()
    
    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
    
    def __repr__(self) -> str:
        return self.__str__()
