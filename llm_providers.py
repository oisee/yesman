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
    
    def __init__(self, model: str = "llama3.2:3b", base_url: str = None):
        self.model = model
        # Check OLLAMA_HOST environment variable
        if base_url is None:
            base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.base_url = base_url.rstrip('/')
        self.last_error = None
    
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
                self.last_error = None
                return LLMResponse(content, provider=self.name, model=self.model)
            elif response.status_code == 404:
                error_detail = f"Model '{self.model}' not found. Try: ollama pull {self.model}"
                self.last_error = error_detail
                return LLMResponse("", error=error_detail, provider=self.name)
            else:
                error_detail = f"HTTP {response.status_code}: {response.text[:100]}"
                self.last_error = error_detail
                return LLMResponse("", error=error_detail, provider=self.name)
                
        except requests.exceptions.ConnectionError:
            error_detail = f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
            self.last_error = error_detail
            return LLMResponse("", error=error_detail, provider=self.name)
        except requests.exceptions.Timeout:
            error_detail = "Request timeout. Ollama might be busy."
            self.last_error = error_detail
            return LLMResponse("", error=error_detail, provider=self.name)
        except Exception as e:
            error_detail = f"Unexpected error: {str(e)}"
            self.last_error = error_detail
            return LLMResponse("", error=error_detail, provider=self.name)
    
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                # Check if the specified model is available
                tags = response.json().get("models", [])
                model_names = [model.get("name", "") for model in tags]
                # Check for exact match or partial match (e.g., "llama3.2:3b" in "llama3.2:3b-instruct")
                model_available = any(self.model in name or name.startswith(self.model) for name in model_names)
                if not model_available:
                    self.last_error = f"Model '{self.model}' not found in Ollama. Available: {', '.join(model_names[:3])}"
                return model_available
            else:
                self.last_error = f"Ollama API returned {response.status_code}"
                return False
        except requests.exceptions.ConnectionError:
            self.last_error = f"Cannot connect to Ollama at {self.base_url}"
            return False
        except Exception as e:
            self.last_error = f"Error checking Ollama: {str(e)}"
            return False
    
    def get_diagnostic_info(self) -> str:
        """Get diagnostic information for troubleshooting"""
        info = []
        info.append(f"Ollama URL: {self.base_url}")
        info.append(f"Model: {self.model}")
        
        try:
            # Test connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                tags = response.json().get("models", [])
                model_names = [model.get("name", "") for model in tags]
                info.append(f"✅ Connected to Ollama")
                info.append(f"Available models: {', '.join(model_names)}")
                
                if not any(self.model in name for name in model_names):
                    info.append(f"❌ Model '{self.model}' not found")
                    info.append(f"💡 Try: ollama pull {self.model}")
            else:
                info.append(f"❌ HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            info.append(f"❌ Cannot connect to {self.base_url}")
            info.append(f"💡 Is Ollama running? Try: ollama serve")
        except Exception as e:
            info.append(f"❌ Error: {e}")
            
        if self.last_error:
            info.append(f"Last error: {self.last_error}")
            
        return "\n".join(info)
    
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
        "fast": "phi4-mini:latest",      # Very fast, small model
        "balanced": "phi4:latest",       # Good balance for this task
        "best": "mistral-small:latest"   # Higher quality, still fast
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
        return OllamaProvider(model=model or "phi4:latest", **kwargs)
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
            if provider_name == "ollama":
                # Try different models for Ollama in order of preference
                for model in ["phi4-mini:latest", "phi4:latest", "mistral-small:latest", 
                             "qwen3:4b", "qwen3:latest", "deepseek-coder:latest"]:
                    try:
                        provider = create_provider(provider_name, model=model)
                        if provider.is_available():
                            return provider
                    except:
                        continue
            else:
                provider = create_provider(provider_name)
                if provider.is_available():
                    return provider
        except:
            continue
    
    return None