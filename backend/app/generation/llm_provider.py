import json
import httpx
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List, Optional
from backend.app.config import settings
from backend.app.core.errors import LLMProviderError
from backend.app.core.logging import logger

class LLMProvider(ABC):
    @abstractmethod
    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Stream tokens asynchronously from the LLM."""
        pass

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate a complete text response synchronously/asynchronously."""
        pass

    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """Test provider availability and latency."""
        pass

class OllamaProvider(LLMProvider):
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL

    async def test_connection(self) -> Dict[str, Any]:
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    models_data = res.json()
                    available_models = [m.get("name") for m in models_data.get("models", [])]
                    model_found = any(self.model in m for m in available_models)
                    return {
                        "status": "connected",
                        "provider": "ollama",
                        "model": self.model,
                        "model_present": model_found,
                        "available_models": available_models,
                        "message": f"Ollama online. Model '{self.model}' {'ready' if model_found else 'not pulled yet (run `ollama pull ' + self.model + '`)'}."
                    }
                else:
                    return {
                        "status": "error",
                        "provider": "ollama",
                        "message": f"Ollama returned HTTP status {res.status_code}"
                    }
        except httpx.ConnectError:
            return {
                "status": "offline",
                "provider": "ollama",
                "message": f"Cannot connect to Ollama at {self.base_url}. Ensure Ollama is running."
            }
        except Exception as e:
            return {
                "status": "error",
                "provider": "ollama",
                "message": f"Error contacting Ollama: {str(e)}"
            }

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": kwargs.get("model", self.model),
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", 0.2),
                "num_ctx": kwargs.get("num_ctx", 4096 if not settings.LOW_MEMORY_MODE else 2048)
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        raise LLMProviderError(f"Ollama error {response.status_code}: {err_text.decode('utf-8', errors='ignore')}")
                    
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
        except httpx.RequestError as e:
            logger.error("Ollama connection error during streaming: %s", str(e))
            raise LLMProviderError(f"Could not stream response from Ollama: {str(e)}")

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": kwargs.get("model", self.model),
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.2),
                "num_ctx": kwargs.get("num_ctx", 4096 if not settings.LOW_MEMORY_MODE else 2048)
            }
        }
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                res = await client.post(url, json=payload)
                if res.status_code != 200:
                    raise LLMProviderError(f"Ollama returned {res.status_code}: {res.text}")
                data = res.json()
                return data.get("response", "")
        except httpx.RequestError as e:
            raise LLMProviderError(f"Ollama request failed: {str(e)}")

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY or ""
        self.model = model or settings.OPENAI_MODEL

    async def test_connection(self) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "unconfigured",
                "provider": "openai",
                "message": "OpenAI API key is not set. Provide OPENAI_API_KEY in Settings."
            }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                if res.status_code == 200:
                    return {
                        "status": "connected",
                        "provider": "openai",
                        "model": self.model,
                        "message": f"OpenAI API connected successfully with model '{self.model}'."
                    }
                else:
                    return {
                        "status": "error",
                        "provider": "openai",
                        "message": f"OpenAI returned HTTP {res.status_code}: {res.text}"
                    }
        except Exception as e:
            return {
                "status": "error",
                "provider": "openai",
                "message": f"Failed to reach OpenAI: {str(e)}"
            }

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        if not self.api_key:
            raise LLMProviderError("OpenAI API key is missing.")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
            "stream": True
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        raise LLMProviderError(f"OpenAI error {response.status_code}: {err_text.decode('utf-8', errors='ignore')}")
                    
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise LLMProviderError(f"OpenAI streaming error: {str(e)}")

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        if not self.api_key:
            raise LLMProviderError("OpenAI API key is missing.")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2)
        }
        
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                res = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload
                )
                if res.status_code != 200:
                    raise LLMProviderError(f"OpenAI returned {res.status_code}: {res.text}")
                data = res.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            raise LLMProviderError(f"OpenAI completion error: {str(e)}")

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY or ""
        self.model = model or settings.GEMINI_MODEL

    async def test_connection(self) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "unconfigured",
                "provider": "gemini",
                "message": "Gemini API key is not set. Provide GEMINI_API_KEY in Settings."
            }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}?key={self.api_key}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    return {
                        "status": "connected",
                        "provider": "gemini",
                        "model": self.model,
                        "message": f"Gemini connected successfully with model '{self.model}'."
                    }
                else:
                    return {
                        "status": "error",
                        "provider": "gemini",
                        "message": f"Gemini returned HTTP {res.status_code}: {res.text}"
                    }
        except Exception as e:
            return {
                "status": "error",
                "provider": "gemini",
                "message": f"Gemini connection error: {str(e)}"
            }

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        if not self.api_key:
            raise LLMProviderError("Gemini API key is missing.")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"
        
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"[System Instructions]:\n{system_prompt}\n\n[User Query]:\n{prompt}"}]})
        else:
            contents.append({"role": "user", "parts": [{"text": prompt}]})
            
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.2)
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        raise LLMProviderError(f"Gemini error {response.status_code}: {err_text.decode('utf-8', errors='ignore')}")
                    
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        try:
                            data = json.loads(data_str)
                            candidates = data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for part in parts:
                                    if part.get("thought", False):
                                        continue
                                    text = part.get("text", "")
                                    if text:
                                        yield text
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise LLMProviderError(f"Gemini streaming error: {str(e)}")

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        if not self.api_key:
            raise LLMProviderError("Gemini API key is missing.")
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"[System Instructions]:\n{system_prompt}\n\n[User Query]:\n{prompt}"}]})
        else:
            contents.append({"role": "user", "parts": [{"text": prompt}]})
            
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.2)
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                res = await client.post(url, json=payload)
                if res.status_code != 200:
                    raise LLMProviderError(f"Gemini returned {res.status_code}: {res.text}")
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    return "".join(part.get("text", "") for part in parts if not part.get("thought", False))
                return ""
        except Exception as e:
            raise LLMProviderError(f"Gemini completion error: {str(e)}")

def get_llm_provider(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None
) -> LLMProvider:
    name = (provider_name or settings.DEFAULT_LLM_PROVIDER).lower()
    if name == "ollama":
        return OllamaProvider(base_url=base_url, model=model)
    elif name == "openai":
        return OpenAIProvider(api_key=api_key, model=model)
    elif name == "gemini":
        return GeminiProvider(api_key=api_key, model=model)
    else:
        logger.warning(f"Unknown LLM provider '{provider_name}', falling back to Ollama.")
        return OllamaProvider(base_url=base_url, model=model)
