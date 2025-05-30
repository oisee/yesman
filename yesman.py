#!/usr/bin/env python3
"""
Smart TUI wrapper that monitors CLI output and uses LLM to handle prompts
Version 3: Fixed keyboard handling for terminal environments
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
from rich.align import Align

# For LLM integration
from openai import OpenAI


class Mode(Enum):
    AUTO = "AUTO"
    MANUAL = "MANUAL"


class SmartTerminalWrapper:
    def __init__(self, command: list[str], mode: Mode = Mode.AUTO, 
                 pause_seconds: int = 3, cache_file: str = ".yesman_cache.json"):
        self.command = command
        self.mode = mode
        self.pause_seconds = pause_seconds
        self.cache_file = cache_file
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm_client = OpenAI(api_key=api_key) if api_key else None
        
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
        
        # Cache for LLM responses
        self.response_cache = self.load_cache()
        
        # Status
        self.status_text = "Starting..."
        
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
            # Execute the command directly - it can be ANY executable
            os.execvp(self.command[0], self.command)
            
    def read_pty_output(self):
        """Background thread to read PTY output"""
        while self.running:
            try:
                r, _, _ = select.select([self.pty_master], [], [], 0.1)
                if r:
                    data = os.read(self.pty_master, 4096)
                    if data:
                        # Decode and clean the output
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
                # Check if input is available
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
        elif self.is_paused and len(key) == 1 and key.isprintable():
            # Manual character input during pause
            self.send_input(key)
            self.is_paused = False
            self.status_text = f"Manual input: {key}"
            
    def process_output(self, data: str):
        """Process output data into lines"""
        lines = data.split('\n')
        
        for i, line in enumerate(lines):
            if i == 0 and self.output_lines:
                # Append to last line if data didn't start with newline
                self.output_lines[-1] += line
            else:
                # Add as new line
                if line or i > 0:  # Keep empty lines except first
                    self.output_lines.append(line)
                    
        # Keep buffer size manageable
        if len(self.output_lines) > self.max_lines:
            self.output_lines = self.output_lines[-self.max_lines:]
            
    def detect_question(self) -> bool:
        """Check if recent output contains a question/prompt"""
        if not self.output_lines:
            return False
            
        # Get last 5 lines
        recent_lines = self.output_lines[-5:]
        combined = '\n'.join(recent_lines).lower()
        
        for pattern in self.question_patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                return True
        return False
        
    def get_context_hash(self, context: str) -> str:
        """Get hash of context for caching"""
        # Clean context for better cache hits
        cleaned = re.sub(r'\s+', ' ', context.strip())
        return hashlib.md5(cleaned.encode()).hexdigest()
        
    def ask_llm(self, context: str) -> Optional[str]:
        """Ask LLM what to respond to the prompt, with caching"""
        if not self.llm_client:
            return None
            
        # Check cache first
        context_hash = self.get_context_hash(context)
        if context_hash in self.response_cache:
            self.status_text = f"Using cached response (cache size: {len(self.response_cache)})"
            return self.response_cache[context_hash]
            
        try:
            prompt = f"""You are watching a terminal application output. 
The following is currently shown:

{context}

The application appears to be waiting for user input. 
Analyze the output and determine what input should be provided.
Respond with ONLY the exact input to send (e.g., 'y', 'n', '1', 'yes', or just 'ENTER' for pressing enter).
If unsure, respond with 'ENTER'."""

            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10,
                timeout=10.0
            )
            
            result = response.choices[0].message.content.strip()
            
            # Cache the response
            self.response_cache[context_hash] = result
            self.save_cache()
            
            return result
        except Exception as e:
            self.status_text = f"LLM Error: {str(e)[:50]}..."
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
                # Auto-send after countdown
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
            
        # Get last N lines that fit in view
        visible_lines = self.output_lines[-height:] if height > 0 else []
        return '\n'.join(visible_lines)
        
    def run(self):
        """Main TUI loop"""
        console = Console()
        layout = self.create_layout()
        
        # Setup terminal
        self.setup_terminal()
        
        try:
            # Start PTY
            self.start_pty()
            
            # Start background threads
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
                                
                                # Get context
                                context_lines = self.output_lines[-20:] if len(self.output_lines) > 20 else self.output_lines
                                context = '\n'.join(context_lines)
                                
                                # Get last few lines for prompt display
                                prompt_lines = self.output_lines[-5:] if len(self.output_lines) > 5 else self.output_lines
                                self.detected_prompt = '\n'.join(prompt_lines).strip()
                                
                                # Ask LLM
                                response = self.ask_llm(context)
                                if response:
                                    self.llm_suggestion = response
                                    
                                    if self.pause_seconds > 0:
                                        # Start countdown
                                        self.countdown = self.pause_seconds
                                        self.countdown_active = True
                                        countdown_thread = threading.Thread(target=self.countdown_thread, daemon=True)
                                        countdown_thread.start()
                                    else:
                                        # Immediate mode
                                        self.send_input(response)
                                        self.status_text = f"Auto-sent: {response}"
                                        
                        except queue.Empty:
                            pass
                        
                        # Update UI
                        mode_str = f"[{self.mode.value}]"
                        layout["header"].update(
                            Panel(f"{mode_str} YesMan - Command: {' '.join(self.command)}", 
                                  style="bold blue")
                        )
                        
                        # Calculate terminal height
                        base_height = 10  # Header + status + borders
                        if self.is_paused:
                            base_height += 8
                        if self.show_help:
                            base_height += 8
                            
                        terminal_height = console.height - base_height
                        if terminal_height < 5:
                            terminal_height = 5
                        
                        # Terminal output
                        visible_output = self.get_visible_output(terminal_height)
                        
                        # Build terminal panel content
                        terminal_content = Text(visible_output)
                        
                        # Add pause menu if needed
                        if self.is_paused and self.llm_suggestion:
                            terminal_content.append("\n\n")
                            terminal_content.append("─" * 40 + "\n", style="yellow")
                            terminal_content.append(f"⏸  Paused - Detected: ", style="yellow bold")
                            terminal_content.append(f"{self.detected_prompt[-50:]}\n")
                            terminal_content.append(f"💡 LLM suggests: '{self.llm_suggestion}'\n\n", style="green")
                            terminal_content.append("Enter: Accept  |  Esc: Manual mode  |  Any key: Send that key\n", style="dim")
                            terminal_content.append("─" * 40, style="yellow")
                        
                        # Add help if needed
                        if self.show_help:
                            terminal_content.append("\n\n")
                            terminal_content.append("─" * 40 + "\n", style="dim")
                            terminal_content.append("Help:\n", style="bold")
                            terminal_content.append("  Space - Pause automation\n")
                            terminal_content.append("  Enter - Accept LLM suggestion\n")
                            terminal_content.append("  Esc   - Switch to manual mode\n")
                            terminal_content.append("  ?     - Toggle this help\n")
                            terminal_content.append("─" * 40, style="dim")
                        
                        layout["terminal"].update(
                            Panel(terminal_content, title="Terminal Output", border_style="green")
                        )
                        
                        # Status
                        status = self.status_text
                        if self.countdown > 0 and not self.is_paused:
                            status = f"Auto-responding in {self.countdown}s... Press Space to pause"
                        
                        # Show cache stats
                        cache_info = f" | Cache: {len(self.response_cache)} responses"
                        layout["status"].update(
                            Panel(status + cache_info, border_style="blue")
                        )
                        
                        # Check if process is still running
                        if self.pty_pid:
                            pid, status = os.waitpid(self.pty_pid, os.WNOHANG)
                            if pid != 0:
                                self.status_text = f"Process exited with status: {status >> 8}"
                                self.running = False
                                time.sleep(1)  # Give time to see the exit status
                                
                    except KeyboardInterrupt:
                        self.running = False
                    except Exception as e:
                        self.status_text = f"Error: {str(e)[:50]}..."
                        
        finally:
            # Cleanup
            self.restore_terminal()
            if self.pty_master:
                try:
                    os.close(self.pty_master)
                except:
                    pass


def main():
    parser = argparse.ArgumentParser(
        description="YesMan - Smart terminal wrapper with LLM automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./install.sh                    # Shell script
  %(prog)s apt install htop                # System command  
  %(prog)s python script.py                # Python script
  %(prog)s node app.js                     # Node.js app
  %(prog)s ./binary-program                # Any executable
  
  %(prog)s --auto npm install              # No pause, immediate responses
  %(prog)s --manual apt upgrade            # Manual control only
  %(prog)s --pause 5 ./configure           # 5 second pause before auto-response
        """
    )
    
    parser.add_argument('command', nargs='+', help='Command to run (can be ANY program)')
    parser.add_argument('--auto', action='store_true', 
                       help='Full auto mode (no pause)')
    parser.add_argument('--manual', action='store_true',
                       help='Manual mode (no automation)')
    parser.add_argument('--pause', type=int, default=3,
                       help='Pause duration in seconds (default: 3)')
    parser.add_argument('--cache-file', default='.yesman_cache.json',
                       help='Cache file for LLM responses')
    
    args = parser.parse_args()
    
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
        mode=mode,
        pause_seconds=pause,
        cache_file=args.cache_file
    )
    
    try:
        wrapper.run()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()