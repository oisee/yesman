"""
LLM Provider implementations for YesMan
Supports OpenAI, Ollama, Anthropic, and other providers
"""

import requests
import json
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    error: Optional[str] = None
    provider: str = ""
    model: str = ""
    cached: bool = False


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                pass
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        if not self.client:
            return LLMResponse("", error="OpenAI client not available", provider=self.name)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 10),
                timeout=kwargs.get("timeout", 10.0)
            )
            
            content = response.choices[0].message.content.strip()
            return LLMResponse(content, provider=self.name, model=self.model)
            
        except Exception as e:
            return LLMResponse("", error=str(e), provider=self.name)
    
    def is_available(self) -> bool:
        return self.client is not None
    
    @property
    def name(self) -> str:
        return "openai"


class OllamaProvider(LLMProvider):
    """Ollama local provider"""
    
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip('/')
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.1),
                        "num_predict": kwargs.get("max_tokens", 10),
                    }
                },
                timeout=kwargs.get("timeout", 30.0)
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "").strip()
                return LLMResponse(content, provider=self.name, model=self.model)
            else:
                return LLMResponse("", error=f"HTTP {response.status_code}", provider=self.name)
                
        except Exception as e:
            return LLMResponse("", error=str(e), provider=self.name)
    
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except:
            return False
    
    @property
    def name(self) -> str:
        return "ollama"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""
    
    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                pass
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        if not self.client:
            return LLMResponse("", error="Anthropic client not available", provider=self.name)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 10),
                temperature=kwargs.get("temperature", 0.1),
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text.strip()
            return LLMResponse(content, provider=self.name, model=self.model)
            
        except Exception as e:
            return LLMResponse("", error=str(e), provider=self.name)
    
    def is_available(self) -> bool:
        return self.client is not None
    
    @property
    def name(self) -> str:
        return "anthropic"


class GroqProvider(LLMProvider):
    """Groq API provider (fast inference)"""
    
    def __init__(self, model: str = "llama3-8b-8192", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                pass
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        if not self.client:
            return LLMResponse("", error="Groq client not available", provider=self.name)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 10),
            )
            
            content = response.choices[0].message.content.strip()
            return LLMResponse(content, provider=self.name, model=self.model)
            
        except Exception as e:
            return LLMResponse("", error=str(e), provider=self.name)
    
    def is_available(self) -> bool:
        return self.client is not None
    
    @property
    def name(self) -> str:
        return "groq"


# Model recommendations by provider
RECOMMENDED_MODELS = {
    "openai": {
        "fast": "gpt-3.5-turbo",
        "balanced": "gpt-4o-mini", 
        "best": "gpt-4o"
    },
    "ollama": {
        "fast": "llama3.2:1b",      # Very fast, good for this task
        "balanced": "llama3.2:3b",  # Best balance of speed/quality
        "best": "llama3.1:8b"       # Higher quality
    },
    "anthropic": {
        "fast": "claude-3-haiku-20240307",
        "balanced": "claude-3-haiku-20240307",
        "best": "claude-3-5-sonnet-20241022"
    },
    "groq": {
        "fast": "llama-3.1-8b-instant",
        "balanced": "llama3-8b-8192",
        "best": "llama-3.1-70b-versatile"
    }
}


def create_provider(provider_name: str, model: str = None, **kwargs) -> LLMProvider:
    """Factory function to create LLM providers"""
    
    provider_name = provider_name.lower()
    
    # Use recommended model if none specified
    if not model and provider_name in RECOMMENDED_MODELS:
        model = RECOMMENDED_MODELS[provider_name]["balanced"]
    
    if provider_name == "openai":
        return OpenAIProvider(model=model or "gpt-3.5-turbo", **kwargs)
    elif provider_name == "ollama":
        return OllamaProvider(model=model or "llama3.2:3b", **kwargs)
    elif provider_name == "anthropic":
        return AnthropicProvider(model=model or "claude-3-haiku-20240307", **kwargs)
    elif provider_name == "groq":
        return GroqProvider(model=model or "llama3-8b-8192", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def get_available_providers() -> Dict[str, bool]:
    """Check which providers are available"""
    providers = {}
    
    for name in ["openai", "ollama", "anthropic", "groq"]:
        try:
            provider = create_provider(name)
            providers[name] = provider.is_available()
        except:
            providers[name] = False
    
    return providers


def auto_select_provider() -> Optional[LLMProvider]:
    """Automatically select the best available provider"""
    
    # Priority order: local (ollama) > fast cloud (groq) > others
    priority = ["ollama", "groq", "openai", "anthropic"]
    
    for provider_name in priority:
        try:
            provider = create_provider(provider_name)
            if provider.is_available():
                return provider
        except:
            continue
    
    return None