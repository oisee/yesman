#!/usr/bin/env python3
"""
YesMan - Smart terminal wrapper with multi-provider LLM automation
A single-file solution for automating interactive CLI applications
"""

import os
import pty
import select
import sys
import re
import threading
import queue
import time
import argparse
import hashlib
import json
import termios
import tty
import requests
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass

# TUI imports
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table


# ============================================================================
# LLM Provider System
# ============================================================================

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
    
    def __init__(self, model: str = "phi4:latest", base_url: str = None):
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
                # Check for exact match or partial match
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


# ============================================================================
# Main Application
# ============================================================================

class Mode(Enum):
    AUTO = "AUTO"
    MANUAL = "MANUAL"


class YesMan:
    def __init__(self, command: list[str], provider_name: str = "auto", 
                 model: str = None, mode: Mode = Mode.AUTO, 
                 pause_seconds: int = 3, cache_file: str = ".yesman_cache.json"):
        self.command = command
        self.mode = mode
        self.pause_seconds = pause_seconds
        self.cache_file = cache_file
        
        # Initialize LLM provider
        if provider_name == "auto":
            self.llm_provider = auto_select_provider()
            if not self.llm_provider:
                print("❌ No LLM providers available. Please install and configure one:")
                print("  • Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
                print("  • OpenAI: export OPENAI_API_KEY=your_key")
                print("  • Anthropic: export ANTHROPIC_API_KEY=your_key") 
                print("  • Groq: export GROQ_API_KEY=your_key")
                sys.exit(1)
        else:
            try:
                self.llm_provider = create_provider(provider_name, model=model)
                if not self.llm_provider.is_available():
                    print(f"❌ Provider '{provider_name}' is not available")
                    if hasattr(self.llm_provider, 'last_error') and self.llm_provider.last_error:
                        print(f"💡 {self.llm_provider.last_error}")
                    sys.exit(1)
            except Exception as e:
                print(f"❌ Error creating provider '{provider_name}': {e}")
                sys.exit(1)
        
        # Terminal state
        self.output_lines = []
        self.max_lines = 1000
        self.running = True
        self.pty_master = None
        self.pty_pid = None
        self.output_queue = queue.Queue()
        self.input_queue = queue.Queue()
        
        # Control state
        self.is_paused = False
        self.countdown = 0
        self.countdown_active = False
        self.llm_suggestion = ""
        self.detected_prompt = ""
        self.last_llm_check = 0
        self.show_help = False
        self.show_providers = False
        self.show_diagnostics = False
        
        # Cache for LLM responses
        self.response_cache = self.load_cache()
        
        # Status and initial diagnostics
        if hasattr(self.llm_provider, 'last_error') and self.llm_provider.last_error:
            self.status_text = f"⚠️ {self.llm_provider.name}: {self.llm_provider.last_error}"
        else:
            self.status_text = f"Using {self.llm_provider.name} provider"
        
        # Question detection patterns
        self.question_patterns = [
            r'\?\s*$',
            r'\[y/n\]',
            r'\[Y/n\]',
            r'\(yes/no\)',
            r'(?i)press enter',
            r'(?i)continue\?',
            r'(?i)proceed\?',
            r'(?i)confirm',
            r'(?i)choice:',
            r'(?i)select.*:',
        ]
        
        # Save original terminal settings
        self.old_settings = None
        
    def load_cache(self) -> Dict[str, str]:
        """Load response cache from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
        
    def save_cache(self):
        """Save response cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.response_cache, f, indent=2)
        except:
            pass
            
    def setup_terminal(self):
        """Set terminal to raw mode for key capture"""
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except:
            pass
            
    def restore_terminal(self):
        """Restore original terminal settings"""
        if self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except:
                pass
                
    def start_pty(self):
        """Fork a PTY and run the command"""
        self.pty_pid, self.pty_master = pty.fork()
        
        if self.pty_pid == 0:  # Child process
            os.execvp(self.command[0], self.command)
            
    def read_pty_output(self):
        """Background thread to read PTY output"""
        while self.running:
            try:
                r, _, _ = select.select([self.pty_master], [], [], 0.1)
                if r:
                    data = os.read(self.pty_master, 4096)
                    if data:
                        text = data.decode('utf-8', errors='replace')
                        self.output_queue.put(text)
                    else:
                        break  # EOF
            except OSError:
                break
                
    def write_pty_input(self):
        """Background thread to write input to PTY"""
        while self.running:
            try:
                input_data = self.input_queue.get(timeout=0.1)
                if input_data:
                    os.write(self.pty_master, input_data.encode())
            except queue.Empty:
                continue
            except OSError:
                break
                
    def read_keyboard(self):
        """Background thread to read keyboard input"""
        while self.running:
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if r:
                    key = sys.stdin.read(1)
                    if key:
                        self.handle_key(key)
            except:
                pass
                
    def handle_key(self, key: str):
        """Process keyboard input"""
        if key == ' ' and self.countdown > 0:
            self.is_paused = True
            self.countdown = 0
            self.status_text = "Paused - Press Enter to accept, Esc for manual control"
        elif key == '\n' and self.is_paused and self.llm_suggestion:
            self.send_input(self.llm_suggestion)
            self.is_paused = False
            self.status_text = f"Sent: {self.llm_suggestion}"
            self.llm_suggestion = ""
            self.detected_prompt = ""
        elif key == '\x1b':  # ESC key - always available for manual intervention
            if self.is_paused:
                self.is_paused = False
            self.mode = Mode.MANUAL
            self.countdown = 0
            self.llm_suggestion = ""
            self.detected_prompt = ""
            self.status_text = "🔧 Manual mode - automation disabled. Type directly to the program."
        elif key == 'a':  # 'a' key for auto mode
            self.mode = Mode.AUTO
            self.status_text = "🤖 Auto mode - automation enabled"
        elif key == 'm':  # 'm' key for manual mode
            self.mode = Mode.MANUAL
            self.countdown = 0
            self.llm_suggestion = ""
            self.detected_prompt = ""
            self.status_text = "🔧 Manual mode - automation disabled. Type directly to the program."
        elif key == '?':
            self.show_help = not self.show_help
        elif key == 'p':
            self.show_providers = not self.show_providers
        elif key == 'd':  # 'd' for diagnostics
            self.show_diagnostics = not getattr(self, 'show_diagnostics', False)
        elif key == 'r' and self.mode == Mode.AUTO:  # 'r' to retry LLM on error
            if "LLM Error" in self.status_text and self.detect_question():
                self.last_llm_check = 0  # Reset to allow immediate retry
                self.status_text = "🔄 Retrying LLM..."
        elif self.is_paused and len(key) == 1 and key.isprintable():
            self.send_input(key)
            self.is_paused = False
            self.status_text = f"Manual input: {key}"
        elif self.mode == Mode.MANUAL and len(key) == 1 and key.isprintable():
            # In manual mode, pass through all printable characters
            self.send_input(key)
            self.status_text = f"Manual input: {key}"
        elif self.mode == Mode.MANUAL and key == '\n':
            # Handle Enter key in manual mode
            self.send_input("ENTER")
            self.status_text = "Manual input: ENTER"
            
    def process_output(self, data: str):
        """Process output data into lines"""
        lines = data.split('\n')
        
        for i, line in enumerate(lines):
            if i == 0 and self.output_lines:
                self.output_lines[-1] += line
            else:
                if line or i > 0:
                    self.output_lines.append(line)
                    
        if len(self.output_lines) > self.max_lines:
            self.output_lines = self.output_lines[-self.max_lines:]
            
    def detect_question(self) -> bool:
        """Check if recent output contains a question/prompt"""
        if not self.output_lines:
            return False
            
        recent_lines = self.output_lines[-5:]
        combined = '\n'.join(recent_lines).lower()
        
        for pattern in self.question_patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                return True
        return False
        
    def get_context_hash(self, context: str) -> str:
        """Get hash of context for caching"""
        cleaned = re.sub(r'\s+', ' ', context.strip())
        # Include provider name in hash for provider-specific caching
        cache_key = f"{self.llm_provider.name}:{cleaned}"
        return hashlib.md5(cache_key.encode()).hexdigest()
        
    def ask_llm(self, context: str) -> Optional[str]:
        """Ask LLM what to respond to the prompt, with caching"""
        if not self.llm_provider:
            return None
            
        # Check cache first
        context_hash = self.get_context_hash(context)
        if context_hash in self.response_cache:
            self.status_text = f"Using cached response (cache: {len(self.response_cache)})"
            return self.response_cache[context_hash]
            
        # Optimized prompt for terminal automation
        prompt = f"""Terminal automation task: A program is waiting for input.
Output: {context}

Respond with ONLY the exact input needed (y/n/1/2/ENTER/etc). Examples:
- "Continue? [Y/n]" → y
- "Press ENTER" → ENTER  
- "Choose (1/2/3)" → 1
- Uncertain → ENTER

Response:"""

        try:
            response = self.llm_provider.generate(
                prompt, 
                temperature=0.1, 
                max_tokens=10,
                timeout=10.0
            )
            
            if response.error:
                self.status_text = f"LLM Error: {response.error[:30]}..."
                return None
            
            result = response.content
            
            # Cache the response
            self.response_cache[context_hash] = result
            self.save_cache()
            
            return result
            
        except Exception as e:
            self.status_text = f"LLM Error: {str(e)[:30]}..."
            return None
            
    def send_input(self, text: str):
        """Send input to the PTY"""
        if text.upper() == "ENTER":
            self.input_queue.put("\n")
        else:
            self.input_queue.put(text + "\n")
            
    def countdown_thread(self):
        """Handle countdown in background"""
        while self.countdown > 0 and not self.is_paused:
            time.sleep(1)
            self.countdown -= 1
            if self.countdown == 0 and self.llm_suggestion and not self.is_paused:
                self.send_input(self.llm_suggestion)
                self.status_text = f"Auto-sent: {self.llm_suggestion}"
                self.llm_suggestion = ""
                self.detected_prompt = ""
                
    def create_layout(self) -> Layout:
        """Create the TUI layout"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="terminal", ratio=1),
            Layout(name="status", size=3)
        )
        return layout
        
    def get_visible_output(self, height: int) -> str:
        """Get the visible portion of output"""
        if not self.output_lines:
            return ""
        visible_lines = self.output_lines[-height:] if height > 0 else []
        return '\n'.join(visible_lines)
        
    def create_provider_table(self) -> Table:
        """Create a table showing available providers"""
        table = Table(title="🤖 Available LLM Providers")
        table.add_column("Provider", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Recommended Models")
        
        providers = get_available_providers()
        
        for name, available in providers.items():
            status = "✅ Available" if available else "❌ Not Available"
            models = ", ".join(RECOMMENDED_MODELS.get(name, {}).values())
            table.add_row(name.title(), status, models)
        
        # Highlight current provider
        if hasattr(self, 'llm_provider') and self.llm_provider:
            table.add_row()
            table.add_row(f"Current: {self.llm_provider.name}", "🟢 Active", 
                         getattr(self.llm_provider, 'model', 'Unknown'), style="bold yellow")
        
        return table
        
    def run(self):
        """Main TUI loop"""
        console = Console()
        layout = self.create_layout()
        
        self.setup_terminal()
        
        try:
            self.start_pty()
            
            output_thread = threading.Thread(target=self.read_pty_output, daemon=True)
            input_thread = threading.Thread(target=self.write_pty_input, daemon=True)
            keyboard_thread = threading.Thread(target=self.read_keyboard, daemon=True)
            
            output_thread.start()
            input_thread.start()
            keyboard_thread.start()
            
            with Live(layout, console=console, screen=True, refresh_per_second=10) as live:
                while self.running:
                    try:
                        # Process output
                        try:
                            output = self.output_queue.get_nowait()
                            self.process_output(output)
                            
                            # Check for questions
                            current_time = time.time()
                            if (self.mode == Mode.AUTO and 
                                self.detect_question() and 
                                current_time - self.last_llm_check > 2):
                                
                                self.last_llm_check = current_time
                                
                                context_lines = self.output_lines[-20:] if len(self.output_lines) > 20 else self.output_lines
                                context = '\n'.join(context_lines)
                                
                                prompt_lines = self.output_lines[-3:] if len(self.output_lines) > 3 else self.output_lines
                                self.detected_prompt = '\n'.join(prompt_lines).strip()
                                
                                response = self.ask_llm(context)
                                if response:
                                    self.llm_suggestion = response
                                    
                                    if self.pause_seconds > 0:
                                        self.countdown = self.pause_seconds
                                        self.countdown_active = True
                                        countdown_thread = threading.Thread(target=self.countdown_thread, daemon=True)
                                        countdown_thread.start()
                                    else:
                                        self.send_input(response)
                                        self.status_text = f"Auto-sent: {response}"
                                        
                        except queue.Empty:
                            pass
                        
                        # Update UI
                        provider_info = f"{self.llm_provider.name}"
                        if hasattr(self.llm_provider, 'model'):
                            provider_info += f":{self.llm_provider.model}"
                        
                        mode_str = f"[{self.mode.value}]"
                        layout["header"].update(
                            Panel(f"{mode_str} YesMan ({provider_info}) - Command: {' '.join(self.command)}", 
                                  style="bold blue")
                        )
                        
                        base_height = 10
                        if self.is_paused:
                            base_height += 8
                        if self.show_help:
                            base_height += 10
                        if self.show_providers:
                            base_height += 15
                        if self.show_diagnostics:
                            base_height += 12
                            
                        terminal_height = console.height - base_height
                        if terminal_height < 5:
                            terminal_height = 5
                        
                        visible_output = self.get_visible_output(terminal_height)
                        terminal_content = Text(visible_output)
                        
                        # Add pause menu
                        if self.is_paused and self.llm_suggestion:
                            terminal_content.append("\n\n")
                            terminal_content.append("─" * 50 + "\n", style="yellow")
                            terminal_content.append(f"⏸  Paused - Detected: ", style="yellow bold")
                            terminal_content.append(f"{self.detected_prompt[-60:]}\n")
                            terminal_content.append(f"🤖 {self.llm_provider.name} suggests: '{self.llm_suggestion}'\n\n", style="green")
                            terminal_content.append("Enter: Accept  |  Esc: Manual mode  |  Any key: Send that key\n", style="dim")
                            terminal_content.append("─" * 50, style="yellow")
                        
                        # Add help
                        if self.show_help:
                            terminal_content.append("\n\n")
                            terminal_content.append("─" * 50 + "\n", style="dim")
                            terminal_content.append("🎮 Controls:\n", style="bold")
                            terminal_content.append("  Space - Pause automation during countdown\n")
                            terminal_content.append("  Enter - Accept LLM suggestion (when paused)\n")
                            terminal_content.append("  Esc   - Switch to manual mode (ALWAYS available)\n")
                            terminal_content.append("  a     - Switch to auto mode\n")
                            terminal_content.append("  m     - Switch to manual mode\n")
                            terminal_content.append("  r     - Retry LLM on error\n")
                            terminal_content.append("  ?     - Toggle this help\n")
                            terminal_content.append("  p     - Show provider info\n")
                            terminal_content.append("  d     - Show diagnostics\n")
                            terminal_content.append("\n📝 In manual mode: type directly to send input\n", style="yellow")
                            terminal_content.append("─" * 50, style="dim")
                        
                        # Add provider info
                        if self.show_providers:
                            terminal_content.append("\n\n")
                            provider_table = self.create_provider_table()
                            terminal_content.append(str(provider_table))
                        
                        # Add diagnostics
                        if self.show_diagnostics and hasattr(self.llm_provider, 'get_diagnostic_info'):
                            terminal_content.append("\n\n")
                            terminal_content.append("─" * 50 + "\n", style="cyan")
                            terminal_content.append("🔍 Provider Diagnostics:\n", style="bold cyan")
                            diag_info = self.llm_provider.get_diagnostic_info()
                            terminal_content.append(diag_info + "\n")
                            terminal_content.append("─" * 50, style="cyan")
                        
                        layout["terminal"].update(
                            Panel(terminal_content, title="Terminal Output", border_style="green")
                        )
                        
                        # Status
                        status = self.status_text
                        if self.countdown > 0 and not self.is_paused:
                            status = f"Auto-responding in {self.countdown}s... Press Space to pause"
                        
                        # Add helpful hints for error states
                        if "LLM Error" in status:
                            status += " | Press 'r' to retry, 'Esc' for manual mode"
                        elif self.mode == Mode.MANUAL:
                            status += " | Press 'a' to re-enable automation"
                        
                        cache_info = f" | Cache: {len(self.response_cache)} responses"
                        
                        # Color status based on state
                        border_color = "blue"
                        if "Error" in status:
                            border_color = "red"
                        elif self.mode == Mode.MANUAL:
                            border_color = "yellow"
                        elif self.countdown > 0:
                            border_color = "green"
                            
                        layout["status"].update(
                            Panel(status + cache_info, border_style=border_color)
                        )
                        
                        # Check process status
                        if self.pty_pid:
                            pid, status = os.waitpid(self.pty_pid, os.WNOHANG)
                            if pid != 0:
                                self.status_text = f"Process exited with status: {status >> 8}"
                                self.running = False
                                time.sleep(1)
                                
                    except KeyboardInterrupt:
                        self.running = False
                    except Exception as e:
                        self.status_text = f"Error: {str(e)[:50]}..."
                        
        finally:
            self.restore_terminal()
            if self.pty_master:
                try:
                    os.close(self.pty_master)
                except:
                    pass


def main():
    parser = argparse.ArgumentParser(
        description="YesMan - Smart terminal wrapper with multi-provider LLM automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🤖 Supported Providers:
  ollama     - Local inference (recommended for speed/privacy)
  openai     - OpenAI GPT models (requires API key)
  anthropic  - Claude models (requires API key)  
  groq       - Fast cloud inference (requires API key)

📋 Examples:
  %(prog)s ./install.sh                           # Auto-detect provider
  %(prog)s --provider ollama ./setup.sh          # Use Ollama
  %(prog)s --provider openai --model gpt-4o ./configure
  %(prog)s --provider ollama --model phi4-mini:latest --auto npm install
  
🔧 Setup:
  Ollama:    curl -fsSL https://ollama.ai/install.sh | sh && ollama pull phi4
  OpenAI:    export OPENAI_API_KEY=your_key
  Anthropic: export ANTHROPIC_API_KEY=your_key
  Groq:      export GROQ_API_KEY=your_key
        """
    )
    
    parser.add_argument('command', nargs='+', help='Command to run')
    parser.add_argument('--provider', choices=['auto', 'ollama', 'openai', 'anthropic', 'groq'], 
                       default='auto', help='LLM provider to use')
    parser.add_argument('--model', help='Specific model to use')
    parser.add_argument('--auto', action='store_true', help='Full auto mode (no pause)')
    parser.add_argument('--manual', action='store_true', help='Manual mode (no automation)')
    parser.add_argument('--pause', type=int, default=3, help='Pause duration in seconds')
    parser.add_argument('--cache-file', default='.yesman_cache.json', help='Cache file for responses')
    parser.add_argument('--list-providers', action='store_true', help='List available providers and exit')
    parser.add_argument('--setup-ollama', action='store_true', help='Setup Ollama with recommended model')
    
    args = parser.parse_args()
    
    if args.list_providers:
        print("🤖 Available LLM Providers:")
        providers = get_available_providers()
        for name, available in providers.items():
            status = "✅" if available else "❌"
            models = list(RECOMMENDED_MODELS.get(name, {}).values())
            print(f"  {status} {name:12} - Models: {', '.join(models)}")
        return
    
    if args.setup_ollama:
        print("🚀 Setting up Ollama...")
        try:
            provider = create_provider("ollama")
            print(f"Diagnostic info:\n{provider.get_diagnostic_info()}")
            
            if not provider.is_available():
                print("\n💡 Recommended setup:")
                print("1. ollama pull phi4-mini:latest    # Fastest model")
                print("2. ollama pull phi4:latest         # Balanced model") 
                print("3. ollama pull mistral-small:latest # High quality")
        except Exception as e:
            print(f"Error: {e}")
        return
    
    # Determine mode
    if args.manual:
        mode = Mode.MANUAL
        pause = 0
    elif args.auto:
        mode = Mode.AUTO
        pause = 0
    else:
        mode = Mode.AUTO
        pause = args.pause
    
    wrapper = YesMan(
        command=args.command,
        provider_name=args.provider,
        model=args.model,
        mode=mode,
        pause_seconds=pause,
        cache_file=args.cache_file
    )
    
    try:
        wrapper.run()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()