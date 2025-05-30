#!/usr/bin/env python3
"""
Smart TUI wrapper that monitors CLI output and uses LLM to handle prompts
Version 4: Multi-provider support (OpenAI, Ollama, Anthropic, Groq)
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
from datetime import datetime
from typing import Optional, Tuple, Dict
from enum import Enum

# TUI imports
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table

# LLM providers
from llm_providers import (
    create_provider, get_available_providers, auto_select_provider,
    RECOMMENDED_MODELS, LLMProvider, LLMResponse
)


class Mode(Enum):
    AUTO = "AUTO"
    MANUAL = "MANUAL"


class SmartTerminalWrapper:
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
        
        # Cache for LLM responses
        self.response_cache = self.load_cache()
        
        # Status
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
        elif key == '\x1b' and self.is_paused:  # ESC key
            self.mode = Mode.MANUAL
            self.is_paused = False
            self.status_text = "Manual mode - automation disabled"
        elif key == '?':
            self.show_help = not self.show_help
        elif key == 'p':
            self.show_providers = not self.show_providers
        elif self.is_paused and len(key) == 1 and key.isprintable():
            self.send_input(key)
            self.is_paused = False
            self.status_text = f"Manual input: {key}"
            
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
                            terminal_content.append("  Space - Pause automation\n")
                            terminal_content.append("  Enter - Accept LLM suggestion\n")
                            terminal_content.append("  Esc   - Switch to manual mode\n") 
                            terminal_content.append("  ?     - Toggle this help\n")
                            terminal_content.append("  p     - Show provider info\n")
                            terminal_content.append("─" * 50, style="dim")
                        
                        # Add provider info
                        if self.show_providers:
                            terminal_content.append("\n\n")
                            provider_table = self.create_provider_table()
                            terminal_content.append(str(provider_table))
                        
                        layout["terminal"].update(
                            Panel(terminal_content, title="Terminal Output", border_style="green")
                        )
                        
                        # Status
                        status = self.status_text
                        if self.countdown > 0 and not self.is_paused:
                            status = f"Auto-responding in {self.countdown}s... Press Space to pause"
                        
                        cache_info = f" | Cache: {len(self.response_cache)} responses"
                        layout["status"].update(
                            Panel(status + cache_info, border_style="blue")
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
  %(prog)s --provider ollama --model llama3.2:1b --auto npm install
  
🔧 Setup:
  Ollama:    curl -fsSL https://ollama.ai/install.sh | sh && ollama pull llama3.2
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
    
    args = parser.parse_args()
    
    if args.list_providers:
        print("🤖 Available LLM Providers:")
        providers = get_available_providers()
        for name, available in providers.items():
            status = "✅" if available else "❌"
            models = list(RECOMMENDED_MODELS.get(name, {}).values())
            print(f"  {status} {name:12} - Models: {', '.join(models)}")
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
    
    wrapper = SmartTerminalWrapper(
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