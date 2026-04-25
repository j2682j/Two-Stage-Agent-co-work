"""????????????"""


import os
from typing import Literal, Optional, Iterator
from openai import OpenAI

from .exceptions import HelloAgentsException

# 支援的LLM提供商
SUPPORTED_PROVIDERS = Literal[
    "openai",
    "deepseek",
    "qwen",
    "modelscope",
    "kimi",
    "zhipu",
    "ollama",
    "vllm",
    "local",
    "auto",
    "custom",
]

class HelloAgentsLLM:
    """
    為HelloAgents定制的LLM客戶端。
    它用於呼叫任何相容OpenAI介面的服務，並預設使用流式回應。

    設計理念：
    - 參數優先，環境變數兜底
    - 流式回應為預設，提供更好的使用者體驗
    - 支援多種LLM提供商
    - 統一的呼叫介面
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[SUPPORTED_PROVIDERS] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ):
        """
        初始化客戶端。優先使用傳入參數，如果未提供，則從環境變數載入。
        支援自動檢測provider或使用統一的LLM_*環境變數設定。

        Args:
            model: 模型名稱，如果未提供則從環境變數LLM_MODEL_ID讀取
            api_key: API密鑰，如果未提供則從環境變數讀取
            base_url: 服務地址，如果未提供則從環境變數LLM_BASE_URL讀取
            provider: LLM提供商，如果未提供則自動檢測
            temperature: 溫度參數
            max_tokens: 最大token 數
            timeout: 逾時時間，從環境變數LLM_TIMEOUT讀取，預設60秒
        """
        # 優先使用傳入參數，如果未提供，則從環境變數載入
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        self.kwargs = kwargs

        # 自動檢測provider或使用指定的provider
        requested_provider = (provider or "").lower() if provider else None
        self.provider = provider or self._auto_detect_provider(api_key, base_url)

        if requested_provider == "custom":
            self.provider = "custom"
            self.api_key = api_key or os.getenv("LLM_API_KEY")
            self.base_url = base_url or os.getenv("LLM_BASE_URL")
        else:
            # 根據provider確定API密鑰和base_url
            self.api_key, self.base_url = self._resolve_credentials(api_key, base_url)

        # 驗證必要參數
        if not self.model:
            self.model = self._get_default_model()
        if not all([self.api_key, self.base_url]):
            raise HelloAgentsException("API密鑰和服務地址必須被提供或在.env檔案中定義。")

        # 建立OpenAI客戶端
        self._client = self._create_client()

    def _auto_detect_provider(self, api_key: Optional[str], base_url: Optional[str]) -> str:
        """
        自動檢測LLM提供商

        檢測邏輯：
        1. 優先檢查特定提供商的環境變數
        2. 根據API密鑰格式判斷
        3. 根據base_url判斷
        4. 預設回傳通用設定
        """
        # 1. 檢查特定提供商的環境變數
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("DEEPSEEK_API_KEY"):
            return "deepseek"
        if os.getenv("DASHSCOPE_API_KEY"):
            return "qwen"
        if os.getenv("MODELSCOPE_API_KEY"):
            return "modelscope"
        if os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY"):
            return "kimi"
        if os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY"):
            return "zhipu"
        if os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_HOST"):
            return "ollama"
        if os.getenv("VLLM_API_KEY") or os.getenv("VLLM_HOST"):
            return "vllm"

        # 2. 根據API密鑰格式判斷
        actual_api_key = api_key or os.getenv("LLM_API_KEY")
        if actual_api_key:
            actual_key_lower = actual_api_key.lower()
            if actual_api_key.startswith("ms-"):
                return "modelscope"
            elif actual_key_lower == "ollama":
                return "ollama"
            elif actual_key_lower == "vllm":
                return "vllm"
            elif actual_key_lower == "local":
                return "local"
            elif actual_api_key.startswith("sk-") and len(actual_api_key) > 50:
                # 可能是OpenAI、DeepSeek或Kimi，需要進一步判斷
                pass
            elif actual_api_key.endswith(".") or "." in actual_api_key[-20:]:
                # 智譜AI的API密鑰格式通常包含點號
                return "zhipu"

        # 3. 根據base_url判斷
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")
        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            if "api.openai.com" in base_url_lower:
                return "openai"
            elif "api.deepseek.com" in base_url_lower:
                return "deepseek"
            elif "dashscope.aliyuncs.com" in base_url_lower:
                return "qwen"
            elif "api-inference.modelscope.cn" in base_url_lower:
                return "modelscope"
            elif "api.moonshot.cn" in base_url_lower:
                return "kimi"
            elif "open.bigmodel.cn" in base_url_lower:
                return "zhipu"
            elif "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                # 本地部署檢測 - 優先檢查特定服務
                if ":11434" in base_url_lower or "ollama" in base_url_lower:
                    return "ollama"
                elif ":8000" in base_url_lower and "vllm" in base_url_lower:
                    return "vllm"
                elif ":8080" in base_url_lower or ":7860" in base_url_lower:
                    return "local"
                else:
                    # 根據API密鑰進一步判斷
                    if actual_api_key and actual_api_key.lower() == "ollama":
                        return "ollama"
                    elif actual_api_key and actual_api_key.lower() == "vllm":
                        return "vllm"
                    else:
                        return "local"
            elif any(port in base_url_lower for port in [":8080", ":7860", ":5000"]):
                # 常見的本地部署端口
                return "local"

        # 4. 預設回傳auto，使用通用設定
        return "auto"

    def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple[str, str]:
        """根據provider解析API密鑰和base_url"""
        if self.provider == "openai":
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "deepseek":
            resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
            return resolved_api_key, resolved_base_url

        elif self.provider == "qwen":
            resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "modelscope":
            resolved_api_key = api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api-inference.modelscope.cn/v1/"
            return resolved_api_key, resolved_base_url

        elif self.provider == "kimi":
            resolved_api_key = api_key or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.moonshot.cn/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "zhipu":
            resolved_api_key = api_key or os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4"
            return resolved_api_key, resolved_base_url

        elif self.provider == "ollama":
            resolved_api_key = api_key or os.getenv("OLLAMA_API_KEY") or os.getenv("LLM_API_KEY") or "ollama"
            resolved_base_url = base_url or os.getenv("OLLAMA_HOST") or os.getenv("LLM_BASE_URL") or "http://localhost:11434/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "vllm":
            resolved_api_key = api_key or os.getenv("VLLM_API_KEY") or os.getenv("LLM_API_KEY") or "vllm"
            resolved_base_url = base_url or os.getenv("VLLM_HOST") or os.getenv("LLM_BASE_URL") or "http://localhost:8000/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "local":
            resolved_api_key = api_key or os.getenv("LLM_API_KEY") or "local"
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "http://localhost:8000/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "custom":
            resolved_api_key = api_key or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL")
            return resolved_api_key, resolved_base_url

        else:
            # auto或其他情況：使用通用設定，支援任何OpenAI相容的服務
            resolved_api_key = api_key or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL")
            return resolved_api_key, resolved_base_url

    def _create_client(self) -> OpenAI:
        """建立OpenAI客戶端"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    def _get_default_model(self) -> str:
        """取得預設模型"""
        if self.provider == "openai":
            return "gpt-3.5-turbo"
        elif self.provider == "deepseek":
            return "deepseek-chat"
        elif self.provider == "qwen":
            return "qwen-plus"
        elif self.provider == "modelscope":
            return "Qwen/Qwen2.5-72B-Instruct"
        elif self.provider == "kimi":
            return "moonshot-v1-8k"
        elif self.provider == "zhipu":
            return "glm-4"
        elif self.provider == "ollama":
            return "llama3.2"  # Ollama常用模型
        elif self.provider == "vllm":
            return "meta-llama/Llama-2-7b-chat-hf"  # vLLM常用模型
        elif self.provider == "local":
            return "local-model"  # 本地模型占位符
        elif self.provider == "custom":
            return self.model or "gpt-3.5-turbo"
        else:
            # auto或其他情況：根據base_url智慧推斷預設模型
            base_url = os.getenv("LLM_BASE_URL", "")
            base_url_lower = base_url.lower()
            if "modelscope" in base_url_lower:
                return "Qwen/Qwen2.5-72B-Instruct"
            elif "deepseek" in base_url_lower:
                return "deepseek-chat"
            elif "dashscope" in base_url_lower:
                return "qwen-plus"
            elif "moonshot" in base_url_lower:
                return "moonshot-v1-8k"
            elif "bigmodel" in base_url_lower:
                return "glm-4"
            elif "ollama" in base_url_lower or ":11434" in base_url_lower:
                return "llama3.2"
            elif ":8000" in base_url_lower or "vllm" in base_url_lower:
                return "meta-llama/Llama-2-7b-chat-hf"
            elif "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                return "local-model"
            else:
                return "gpt-3.5-turbo"

    def think(self, messages: list[dict[str, str]], temperature: Optional[float] = None) -> Iterator[str]:
        """
        呼叫大語言模型進行思考，並回傳流式回應。
        這是主要的呼叫方法，預設使用流式回應以獲得更好的使用者體驗。

        Args:
            messages: 消息列表
            temperature: 溫度參數，如果未提供則使用初始化時的值

        Yields:
            str: 流式回應的文字片段
        """
        print(f"🧠 正在呼叫 {self.model} 模型...")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            # 處理流式回應
            print("✅ 大語言模型回應成功:")
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
                    yield content
            print()  # 在流式輸出結束后換行

        except Exception as e:
            print(f"❌ 呼叫LLM API時發生錯誤: {e}")
            raise HelloAgentsException(f"LLM呼叫失敗: {str(e)}")

    def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        非流式呼叫LLM，回傳完整回應。
        適用於不需要流式輸出的場景。
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
            )
            return response.choices[0].message.content
        except Exception as e:
            raise HelloAgentsException(f"LLM呼叫失敗: {str(e)}")

    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        """
        流式呼叫LLM的別名方法，與think方法功能相同。
        保持向後相容性。
        """
        temperature = kwargs.get('temperature')
        yield from self.think(messages, temperature)
