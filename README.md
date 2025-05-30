# YesMan 🤖

**Smart terminal wrapper with multi-provider LLM automation**

YesMan is a TUI (Terminal User Interface) application that monitors CLI program output and automatically handles interactive prompts using Large Language Models. It shows your program's output in a terminal-in-terminal view while intelligently responding to questions and prompts.

## 🤔 Why Do You Need This?

**YesMan enables fully automated AI pair programming with visual oversight.**

When using AI coding tools like Codex CLI or Claude Code, you constantly face interrupting prompts:

- `"Apply this change? [Y/n]"`
- `"Create new file? [Y/n]"`  
- `"Run this command? [Y/n]"`
- `"Continue with refactoring? [Y/n]"`
- `"Install dependencies? [Y/n]"`

**The Problem**: These safety prompts interrupt the AI's flow, requiring manual intervention every few seconds and defeating the purpose of automation.

**The Solution**: YesMan acts as an intelligent supervisor that:

### ✅ **Auto-Approves AI Suggestions**
- Let Claude Code refactor entire codebases while you watch
- Enable true "autopilot" coding sessions where AI completes full features
- Remove friction from AI-driven development workflows

### 👀 **Provides Full Visual Control** 
- See exactly what the AI is doing in real-time
- Pause or intervene at any moment with a single keypress
- Terminal-in-terminal view shows all AI actions transparently

### 🚀 **Enables Autonomous Development**
- Run overnight AI-driven development sessions
- Automate repetitive coding tasks with AI assistance
- Build proof-of-concepts with minimal human intervention

### 🛡️ **Maintains Safety**
- Smart context-aware defaults for AI tool prompts
- Manual override capability always available
- Pause countdown before executing any action

### 💡 **Perfect For AI Coding Workflows**
- **Feature Development**: Let AI implement entire features while you supervise
- **Codebase Refactoring**: Automated large-scale code improvements  
- **Dependency Management**: AI handles package installations and updates
- **Testing & CI**: Automated test writing and pipeline setup
- **Documentation**: AI generates docs while handling all confirmations
- **Code Reviews**: AI applies suggested changes across multiple files

![YesMan Demo](https://img.shields.io/badge/Status-Working-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- 🖥️ **Terminal-in-terminal display** using PTY (pseudo-terminal)
- 🔍 **Real-time output monitoring** with intelligent pattern detection
- 🤖 **Multi-provider LLM support** (Ollama, OpenAI, Anthropic, Groq)
- 🎮 **Interactive controls** with pause, manual mode, and help
- ⚡ **Response caching** for faster repeated interactions
- 🎯 **Smart prompt detection** for various question formats
- 🔧 **Manual intervention** available at any time
- 📊 **Rich TUI interface** with status, diagnostics, and provider info

## 🚀 Quick Start

### Installation

```bash
# Clone or download yesman.py (single file solution)
chmod +x yesman.py

# Install dependencies
pip install rich requests
```

### Setup LLM Provider

**Option 1: Ollama (Recommended - Local & Fast)**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a fast model
ollama pull qwen3:4b

# YesMan will auto-detect Ollama
./yesman.py python your_script.py
```

**Option 2: OpenAI**
```bash
export OPENAI_API_KEY="your-api-key"
./yesman.py --provider openai python your_script.py
```

**Option 3: Other Providers**
```bash
# Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key"
./yesman.py --provider anthropic python your_script.py

# Groq (fast cloud inference)
export GROQ_API_KEY="your-api-key"
./yesman.py --provider groq python your_script.py
```

## 📖 Usage

### Basic Usage
```bash
# Auto-detect best provider
./yesman.py python install_script.py

# Use specific provider and model
./yesman.py --provider ollama --model qwen3:4b ./configure.sh

# Manual mode (no automation)
./yesman.py --manual npm install

# Full auto mode (no pause)
./yesman.py --auto apt update
```

### Interactive Controls

- **Space** - Pause automation during countdown
- **Enter** - Accept LLM suggestion (when paused)
- **Esc** - Switch to manual mode (always available)
- **a** - Enable auto mode
- **m** - Switch to manual mode
- **r** - Retry LLM on error
- **?** - Toggle help
- **p** - Show provider info
- **d** - Show diagnostics

### AI Coding Examples
```bash
# Let Claude Code refactor entire codebase autonomously
./yesman.py claude-code "refactor this React app to use TypeScript"

# Automated feature development with oversight
./yesman.py codex "implement user authentication system"

# AI-driven dependency management
./yesman.py claude-code "update all packages and fix breaking changes"

# Autonomous testing and CI setup
./yesman.py --auto claude-code "add comprehensive tests for all components"

# Full application scaffolding
./yesman.py codex "create a complete CRUD API with database setup"
```

## 🔧 How It Works

1. **PTY Subprocess**: Spawns your command in a pseudo-terminal
2. **Pattern Detection**: Monitors output for questions using regex patterns
3. **LLM Analysis**: Sends context to LLM when prompts are detected
4. **Smart Response**: LLM suggests appropriate responses (y/n/enter/etc.)
5. **Auto-execution**: Sends response after countdown (with pause option)

### Supported Prompt Patterns

- `[Y/n]`, `[y/N]` - Yes/no questions
- `(yes/no)` - Confirmation prompts  
- `Press ENTER` - Continue prompts
- `? ` - General questions
- `Continue?`, `Proceed?` - Confirmation
- `Choose 1/2/3` - Selection menus

## 🤖 LLM Providers

| Provider | Models | Setup | Speed | Cost |
|----------|---------|--------|-------|------|
| **Ollama** | qwen3:4b, phi4-mini, devstral | Local install | ⚡ Fast | 🆓 Free |
| **OpenAI** | gpt-3.5-turbo, gpt-4o-mini, gpt-4o | API key | 🔥 Very Fast | 💰 Paid |
| **Anthropic** | claude-3-haiku, claude-3.5-sonnet | API key | 🚀 Fast | 💰 Paid |
| **Groq** | llama-3.1-8b-instant, llama3-70b | API key | ⚡ Very Fast | 🆓 Free tier |

### Model Recommendations

- **Speed**: qwen3:4b (Ollama), gpt-3.5-turbo (OpenAI)
- **Balance**: phi4-mini (Ollama), gpt-4o-mini (OpenAI) 
- **Quality**: devstral (Ollama), gpt-4o (OpenAI)

## 📋 Command Line Options

```bash
./yesman.py [OPTIONS] COMMAND [ARGS...]

Options:
  --provider {auto,ollama,openai,anthropic,groq}  LLM provider
  --model MODEL                                   Specific model to use
  --auto                                          Full auto mode (no pause)
  --manual                                        Manual mode (no automation)
  --pause SECONDS                                 Pause duration (default: 3)
  --cache-file FILE                               Response cache file
  --list-providers                                Show available providers
  --setup-ollama                                  Setup Ollama with models
```

## 🛠️ Advanced Configuration

### Environment Variables
```bash
# Ollama configuration
export OLLAMA_HOST="http://localhost:11434"  # Custom Ollama server

# API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."  
export GROQ_API_KEY="gsk_..."
```

### Response Caching
YesMan caches LLM responses in `.yesman_cache.json` to speed up repeated interactions with the same prompts.

### Remote Ollama
```bash
# Use remote Ollama instance
export OLLAMA_HOST="http://192.168.1.100:11434"
./yesman.py python script.py
```

## 🧪 Testing

Test YesMan with the included interactive test application:

```bash
# Basic test
./yesman.py python test_interactive_app.py

# Test with specific provider
./yesman.py --provider ollama --model qwen3:4b python test_interactive_app.py
```

The test app simulates various prompt types to validate automation capabilities.

## 🔍 Troubleshooting

### Check Available Providers
```bash
./yesman.py --list-providers echo test
```

### Ollama Issues
```bash
# Check Ollama status
./yesman.py --setup-ollama

# Pull missing models
ollama pull qwen3:4b

# Check running models
ollama ps
```

### Debug Mode
```bash
# Enable diagnostics view
# Press 'd' during execution to see provider diagnostics
```

## 🤝 Contributing

YesMan is a single-file solution for easy deployment and modification. Feel free to:

- Add new LLM providers
- Improve prompt detection patterns
- Enhance the TUI interface
- Add new automation features

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Rich](https://github.com/Textualize/rich) for beautiful TUI
- LLM integration supports multiple providers
- PTY handling for true terminal emulation