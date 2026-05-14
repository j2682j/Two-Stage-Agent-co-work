"""
建立 SLM 實例並提供一致的呼叫介面。
"""

import os
from typing import Iterator, Optional

from openai import OpenAI

from .exceptions import AgentsException


MODEL_ID_MAP = {
    "nemotron-mini:4b": os.getenv("Nemotron_MODEL_ID"),
    "phi3:3.8b": os.getenv("Phi_MODEL_ID"),
    "qwen3:4b": os.getenv("Qwen_MODEL_ID"),
    "gemma3:4b": os.getenv("Gemma_MODEL_ID"),
    "minicpm3_4b:latest": os.getenv("Minicpm_MODEL_ID"),
    "gpt-oss:20b": os.getenv("GPT_OSS_MODEL_ID"),
}


def estimate_text_tokens(text: str) -> int:
    """
    負責執行 network.slm_agent 中的 estimate_text_tokens 流程，依照 network.slm_agent 的流程需求處理 estimate_text_tokens 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        text: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 int。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    normalized = str(text or "")
    if not normalized:
        return 0
    # Lightweight fallback for local OpenAI-compatible servers that omit usage.
    return max(1, int(len(normalized) / 4))


def estimate_chat_tokens(messages: list[dict[str, str]]) -> int:
    """
    負責執行 network.slm_agent 中的 estimate_chat_tokens 流程，依照 network.slm_agent 的流程需求處理 estimate_chat_tokens 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        messages: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 int。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    total = 0
    for message in messages or []:
        total += estimate_text_tokens(str(message.get("role", "")))
        total += estimate_text_tokens(str(message.get("content", "")))
        total += 4
    return total


class SLM_4b_Agent:
    """
    負責在 network.slm_agent 中封裝 SLM_4b_Agent，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        api_key: 此流程需要使用的輸入資料。
        base_url: 此流程需要使用的輸入資料。
        temperature: 此流程需要使用的輸入資料。
        max_tokens: 控制檢索、篩選或輸出數量的數值參數。
        timeout: 此流程需要使用的輸入資料。
        model_name: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        **kwargs: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        """
        負責執行 SLM_4b_Agent 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            api_key: 此流程需要使用的輸入資料。
            base_url: 此流程需要使用的輸入資料。
            temperature: 此流程需要使用的輸入資料。
            max_tokens: 控制檢索、篩選或輸出數量的數值參數。
            timeout: 此流程需要使用的輸入資料。
            model_name: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        model_env_key_map = {
            "nemotron-mini:4b": "Nemotron_MODEL_ID",
            "phi3:3.8b": "Phi_MODEL_ID",
            "qwen3:4b": "Qwen_MODEL_ID",
            "gemma3:4b": "Gemma_MODEL_ID",
            "minicpm3_4b:latest": "Minicpm_MODEL_ID",
            "gpt-oss:20b": "GPT_OSS_MODEL_ID",
        }

        env_key = model_env_key_map.get(model_name)
        self.model = os.getenv(env_key) if env_key else None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or int(os.getenv("OLLAMA_TIMEOUT", "60"))
        self.kwargs = kwargs

        self.api_key = api_key or os.getenv("OLLAMA_API_KEY")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL")
        self._client = self._create_client()

    def _create_client(self) -> OpenAI:
        """
        負責執行 SLM_4b_Agent 中的 _create_client 流程，依照 SLM_4b_Agent 的流程需求處理 _create_client 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 OpenAI。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def think(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """
        負責執行 SLM_4b_Agent 中的 think 流程，依照 SLM_4b_Agent 的流程需求處理 think 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            messages: 此流程需要使用的輸入資料。
            temperature: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Iterator[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        print(f"正在呼叫 {self.model} 模型...")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
            print("大型語言模型回應成功:")
            return response
        except Exception as e:
            print(f"呼叫 LLM API 發生錯誤: {e}")
            raise AgentsException(f"SLM 呼叫失敗: {str(e)}")

    def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        負責執行 SLM_4b_Agent 中的 invoke 流程，呼叫模型、工具或外部服務並整理回傳結果。
        
        Args:
            messages: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]},
            )
            return response.choices[0].message.content
        except Exception as e:
            raise AgentsException(f"SLM 呼叫失敗: {str(e)}")

    def invoke_with_usage(self, messages: list[dict[str, str]], **kwargs) -> tuple[str, int, int]:
        """
        負責執行 SLM_4b_Agent 中的 invoke_with_usage 流程，呼叫模型、工具或外部服務並整理回傳結果。
        
        Args:
            messages: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 tuple[str, int, int]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]},
            )
            content = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens if response.usage and hasattr(response.usage, "prompt_tokens") else 0
            completion_tokens = response.usage.completion_tokens if response.usage and hasattr(response.usage, "completion_tokens") else 0
            if prompt_tokens <= 0:
                prompt_tokens = estimate_chat_tokens(messages)
            if completion_tokens <= 0:
                completion_tokens = estimate_text_tokens(content)
            return content, prompt_tokens, completion_tokens
        except Exception as e:
            raise AgentsException(f"SLM 呼叫失敗: {str(e)}")

    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        """
        負責執行 SLM_4b_Agent 中的 stream_invoke 流程，依照 SLM_4b_Agent 的流程需求處理 stream_invoke 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            messages: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Iterator[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        temperature = kwargs.get("temperature")
        yield from self.think(messages, temperature)
