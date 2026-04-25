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


class SLM_4b_Agent:
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
            return content, prompt_tokens, completion_tokens
        except Exception as e:
            raise AgentsException(f"SLM 呼叫失敗: {str(e)}")

    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        temperature = kwargs.get("temperature")
        yield from self.think(messages, temperature)
