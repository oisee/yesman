# YesMan - Go Implementation

A smart terminal wrapper written in Go that monitors CLI application output and automatically handles interactive prompts using LLM.

## Features

- **PTY-based terminal emulation** using creack/pty
- **Beautiful TUI** with Charm's BubbleTea framework
- **Real-time pattern detection** for interactive prompts
- **OpenAI integration** for intelligent responses
- **Simple control system** with just 4 shortcuts
- **Configurable pause behavior** before auto-responses

## Prerequisites

- Go 1.21 or higher
- OpenAI API key

## Installation

```bash
# Install dependencies
make deps

# Build the binaries
make build
```

## Usage

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

Run any interactive command:
```bash
./yesman <command> [args...]

# Examples:
./yesman apt install htop
./yesman npm install
./yesman ./test_app
```

## Testing

Run the included test application:
```bash
make test
```

## Building from source

```bash
# Get dependencies
go mod download

# Build
go build -o yesman main.go

# Or use make
make build
```

## Architecture

- **main.go**: Core application with BubbleTea model
- **PTY handling**: Uses creack/pty for pseudo-terminal
- **Concurrent design**: Separate goroutines for PTY I/O
- **Pattern matching**: Regex-based prompt detection
- **LLM integration**: sashabaranov/go-openai client

## Key Components

1. **Model**: BubbleTea application state
2. **PTY wrapper**: Spawns and manages subprocess
3. **Output buffer**: Circular buffer for terminal output
4. **Question detector**: Regex patterns for prompts
5. **LLM handler**: Sends context and gets responses

## Controls

**Just 4 shortcuts to remember:**
- `Space` - Pause automation (stop and wait for manual input)
- `Enter` - Accept LLM suggestion and continue
- `Esc` - Switch to manual mode (disable automation)
- `?` - Show/hide help

## Operating Modes

### Default Mode (3 second pause)
```bash
./yesman ./install.sh
```
When a prompt is detected, shows countdown: "Auto-responding in 3s... Press Space to pause"

### Auto Mode (no pause)
```bash
./yesman --auto npm install
```
Immediately responds to all prompts without pause

### Manual Mode
```bash
./yesman --manual apt upgrade
```
No automation - you control everything

### Custom Pause
```bash
./yesman --pause 5 ./configure
```
Wait 5 seconds before each auto-response

## Pause Menu

When you press Space during countdown:
```
┌─ Paused ──────────────────┐
│ "Continue? [y/N]"         │
│ LLM suggests: 'y'         │
│                           │
│ Enter: Accept             │
│ Esc: Manual control       │
│ Any key: Manual input     │
└───────────────────────────┘
```