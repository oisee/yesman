#!/usr/bin/env python3
"""
Smart TUI wrapper that monitors CLI output and uses LLM to handle prompts
"""

import os
import pty
import select
import sys
import re
import threading
import queue
from datetime import datetime
from typing import Optional, Tuple
import subprocess

# TUI imports
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.align import Align

# For LLM integration
from openai import OpenAI
import json

class SmartTerminalWrapper:
    def __init__(self, command: list[str], llm_api_key: str = None):
        self.command = command
        self.llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=self.llm_api_key) if self.llm_api_key else None
        self.output_buffer = []
        self.max_buffer_lines = 1000
        self.running = True
        self.pty_master = None
        self.pty_pid = None
        self.output_queue = queue.Queue()
        self.input_queue = queue.Queue()
        
        # Question detection patterns
        self.question_patterns = [
            r'\?[\s]*$',
            r'\[y/n\]',
            r'\[Y/n\]',
            r'\(yes/no\)',
            r'press enter',
            r'continue\?',
            r'proceed\?',
            r'confirm',
            r'choice:',
            r'select.*:',
        ]
        
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
                    data = os.read(self.pty_master, 4096).decode('utf-8', errors='replace')
                    if data:
                        self.output_queue.put(data)
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
                
    def detect_question(self, recent_output: str) -> bool:
        """Check if recent output contains a question/prompt"""
        last_lines = recent_output.strip().split('\n')[-5:]  # Check last 5 lines
        combined = '\n'.join(last_lines).lower()
        
        for pattern in self.question_patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                return True
        return False
        
    def ask_llm(self, context: str) -> Optional[str]:
        """Ask LLM what to respond to the prompt"""
        if not self.openai_client:
            return None
            
        try:
            prompt = f"""You are watching a terminal application output. 
The following is currently shown:

{context}

The application appears to be waiting for user input. 
Analyze the output and determine what input should be provided.
Respond with ONLY the exact input to send (e.g., 'y', 'n', '1', 'yes', or just 'ENTER' for pressing enter).
If unsure, respond with 'ENTER'."""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return None
            
    def send_input(self, text: str):
        """Send input to the PTY"""
        if text.upper() == "ENTER":
            self.input_queue.put("\n")
        else:
            self.input_queue.put(text + "\n")
            
    def create_layout(self) -> Layout:
        """Create the TUI layout"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="terminal", ratio=1),
            Layout(name="status", size=3)
        )
        return layout
        
    def run(self):
        """Main TUI loop"""
        console = Console()
        layout = self.create_layout()
        
        # Start PTY
        self.start_pty()
        
        # Start background threads
        output_thread = threading.Thread(target=self.read_pty_output, daemon=True)
        input_thread = threading.Thread(target=self.write_pty_input, daemon=True)
        output_thread.start()
        input_thread.start()
        
        terminal_content = []
        status_text = "Monitoring output..."
        last_llm_check = 0
        
        with Live(layout, console=console, screen=True, refresh_per_second=10) as live:
            while self.running:
                try:
                    # Process output
                    while not self.output_queue.empty():
                        chunk = self.output_queue.get_nowait()
                        terminal_content.append(chunk)
                        
                        # Keep buffer reasonable
                        if len(terminal_content) > self.max_buffer_lines:
                            terminal_content = terminal_content[-self.max_buffer_lines:]
                    
                    # Get recent output
                    recent_output = ''.join(terminal_content[-50:])  # Last 50 chunks
                    
                    # Check for questions
                    if self.detect_question(recent_output):
                        current_time = datetime.now().timestamp()
                        if current_time - last_llm_check > 2:  # Debounce (2 seconds)
                            status_text = "Question detected! Asking LLM..."
                            layout["status"].update(Panel(status_text, title="Status", border_style="yellow"))
                            live.refresh()
                            
                            # Get last 20 lines for context
                            lines = recent_output.strip().split('\n')
                            context = '\n'.join(lines[-20:])
                            
                            response = self.ask_llm(context)
                            if response:
                                status_text = f"LLM suggested: '{response}' - Sending..."
                                layout["status"].update(Panel(status_text, title="Status", border_style="green"))
                                live.refresh()
                                self.send_input(response)
                                last_llm_check = current_time
                            else:
                                status_text = "LLM couldn't determine response"
                                
                    # Update UI
                    layout["header"].update(
                        Panel(f"Smart Terminal Wrapper - Command: {' '.join(self.command)}", 
                              title="YesMan TUI", border_style="blue")
                    )
                    
                    # Terminal output
                    terminal_text = Text(''.join(terminal_content[-100:]))  # Show last 100 chunks
                    layout["terminal"].update(
                        Panel(terminal_text, title="Terminal Output", border_style="green")
                    )
                    
                    layout["status"].update(
                        Panel(status_text, title="Status", border_style="blue")
                    )
                    
                    # Check if process is still running
                    if self.pty_pid:
                        pid, status = os.waitpid(self.pty_pid, os.WNOHANG)
                        if pid != 0:
                            status_text = f"Process exited with status: {status}"
                            self.running = False
                            
                except KeyboardInterrupt:
                    self.running = False
                except Exception as e:
                    status_text = f"Error: {e}"
                    
        # Cleanup
        if self.pty_master:
            os.close(self.pty_master)


def main():
    if len(sys.argv) < 2:
        print("Usage: python tui_wrapper.py <command> [args...]")
        print("Example: python tui_wrapper.py apt install htop")
        sys.exit(1)
        
    command = sys.argv[1:]
    wrapper = SmartTerminalWrapper(command)
    wrapper.run()


if __name__ == "__main__":
    main()