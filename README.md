# YesMan - Smart Terminal Wrapper

A TUI wrapper that monitors CLI application output and automatically handles interactive prompts using LLM.

## Features

- **Terminal-in-terminal display** using PTY (pseudo-terminal)
- **Real-time output monitoring** with pattern detection
- **LLM integration** to intelligently answer prompts
- **Rich TUI interface** showing output and status

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

Run any interactive command:
```bash
python tui_wrapper.py python test_interactive_app.py
```

Or wrap real commands:
```bash
python tui_wrapper.py apt install htop
python tui_wrapper.py npm install
```

## How it works

1. Spawns the command in a PTY subprocess
2. Monitors output for question patterns (?, [y/n], "press enter", etc.)
3. When detected, sends recent output context to LLM
4. LLM suggests appropriate response
5. Automatically sends the response to the application

## Testing

Run the included test app:
```bash
chmod +x test_interactive_app.py
python tui_wrapper.py ./test_interactive_app.py
```

The test app will ask various types of questions to demonstrate the wrapper's capabilities.