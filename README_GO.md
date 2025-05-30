# YesMan - Go Implementation

A smart terminal wrapper written in Go that monitors CLI application output and automatically handles interactive prompts using LLM.

## Features

- **PTY-based terminal emulation** using creack/pty
- **Beautiful TUI** with Charm's BubbleTea framework
- **Real-time pattern detection** for interactive prompts
- **OpenAI integration** for intelligent responses
- **Concurrent processing** with Go routines

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

- `q` or `Ctrl+C`: Quit the application