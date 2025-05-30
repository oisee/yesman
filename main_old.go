package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/creack/pty"
	"github.com/sashabaranov/go-openai"
)

const (
	maxBufferLines = 1000
	contextLines   = 20
	defaultPause   = 3 * time.Second
)

// Operating modes
type Mode int

const (
	ModeAuto Mode = iota
	ModeManual
	ModePaused
)

var (
	// Styles
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("39")).
			BorderStyle(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("39")).
			Padding(0, 1)

	outputStyle = lipgloss.NewStyle().
			BorderStyle(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("42")).
			Padding(1)

	statusStyle = lipgloss.NewStyle().
			BorderStyle(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("214")).
			Padding(0, 1)

	// Question patterns
	questionPatterns = []*regexp.Regexp{
		regexp.MustCompile(`\?\s*$`),
		regexp.MustCompile(`\[y/n\]`),
		regexp.MustCompile(`\[Y/n\]`),
		regexp.MustCompile(`\(yes/no\)`),
		regexp.MustCompile(`(?i)press enter`),
		regexp.MustCompile(`(?i)continue\?`),
		regexp.MustCompile(`(?i)proceed\?`),
		regexp.MustCompile(`(?i)confirm`),
		regexp.MustCompile(`(?i)choice:`),
		regexp.MustCompile(`(?i)select.*:`),
	}
)

type model struct {
	command         []string
	outputBuffer    []string
	statusText      string
	width           int
	height          int
	ptyFile         *os.File
	cmd             *exec.Cmd
	outputChan      chan string
	llmClient       *openai.Client
	lastLLMCheck    time.Time
	mu              sync.Mutex
	quitting        bool
	mode            Mode
	pauseDuration   time.Duration
	isPaused        bool
	countdown       int
	llmSuggestion   string
	detectedPrompt  string
	showHelp        bool
}

type outputMsg string
type statusMsg string
type tickMsg time.Time
type countdownMsg int
type llmResponseMsg struct {
	suggestion string
	prompt     string
}

func initialModel(command []string, mode Mode, pauseDuration time.Duration) model {
	apiKey := os.Getenv("OPENAI_API_KEY")
	var client *openai.Client
	if apiKey != "" {
		client = openai.NewClient(apiKey)
	}

	return model{
		command:       command,
		outputBuffer:  []string{},
		statusText:    "Starting...",
		outputChan:    make(chan string, 100),
		llmClient:     client,
		lastLLMCheck:  time.Now(),
		mode:          mode,
		pauseDuration: pauseDuration,
		width:         80,  // Default width
		height:        24,  // Default height
	}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(
		m.startPTY(),
		tickCmd(),
	)
}

func tickCmd() tea.Cmd {
	return tea.Tick(100*time.Millisecond, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
}

func (m *model) startPTY() tea.Cmd {
	return func() tea.Msg {
		cmd := exec.Command(m.command[0], m.command[1:]...)
		
		ptyFile, err := pty.Start(cmd)
		if err != nil {
			return statusMsg(fmt.Sprintf("Error starting PTY: %v", err))
		}
		
		m.cmd = cmd
		m.ptyFile = ptyFile
		
		// Start reading PTY output
		go m.readPTYOutput()
		
		return statusMsg("Process started")
	}
}

func (m *model) readPTYOutput() {
	buf := make([]byte, 4096)
	for {
		n, err := m.ptyFile.Read(buf)
		if err != nil {
			if err != io.EOF {
				m.outputChan <- fmt.Sprintf("\nError reading PTY: %v", err)
			}
			break
		}
		if n > 0 {
			// Clean the output - remove null bytes and control sequences
			output := string(buf[:n])
			output = strings.ReplaceAll(output, "\x00", "")
			// Remove ANSI escape sequences that might corrupt display
			// Keep newlines and printable characters
			cleaned := ""
			inEscape := false
			for _, ch := range output {
				if ch == '\x1b' {
					inEscape = true
				} else if inEscape && ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')) {
					inEscape = false
				} else if !inEscape {
					cleaned += string(ch)
				}
			}
			if cleaned != "" {
				m.outputChan <- cleaned
			}
		}
	}
	close(m.outputChan)
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			m.quitting = true
			if m.cmd != nil && m.cmd.Process != nil {
				m.cmd.Process.Kill()
			}
			return m, tea.Quit
		case "q":
			if !m.isPaused {
				m.quitting = true
				if m.cmd != nil && m.cmd.Process != nil {
					m.cmd.Process.Kill()
				}
				return m, tea.Quit
			}
		case " ":
			if m.countdown > 0 {
				m.isPaused = true
				m.countdown = 0
				m.statusText = "Paused - Press Enter to accept, Esc for manual control"
			}
		case "enter":
			if m.isPaused && m.llmSuggestion != "" {
				suggestion := m.llmSuggestion
				m.sendInput(suggestion)
				m.isPaused = false
				m.llmSuggestion = ""
				m.detectedPrompt = ""
				m.statusText = fmt.Sprintf("Sent: %s", suggestion)
			}
		case "esc":
			if m.isPaused {
				m.mode = ModeManual
				m.isPaused = false
				m.statusText = "Manual mode - automation disabled"
			}
		case "?":
			m.showHelp = !m.showHelp
		default:
			if m.isPaused && len(msg.String()) == 1 {
				// Any single key press sends that input
				m.sendInput(msg.String())
				m.isPaused = false
				m.statusText = fmt.Sprintf("Manual input: %s", msg.String())
			}
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

	case tickMsg:
		// Check for new output
		select {
		case output, ok := <-m.outputChan:
			if ok {
				m.mu.Lock()
				m.outputBuffer = append(m.outputBuffer, output)
				if len(m.outputBuffer) > maxBufferLines {
					m.outputBuffer = m.outputBuffer[len(m.outputBuffer)-maxBufferLines:]
				}
				m.mu.Unlock()
				
				// Check for questions
				if m.mode != ModeManual && m.detectQuestion() && time.Since(m.lastLLMCheck) > 2*time.Second {
					m.lastLLMCheck = time.Now()
					if m.mode == ModeAuto && m.pauseDuration > 0 {
						// Start countdown
						m.countdown = int(m.pauseDuration.Seconds())
						return m, tea.Batch(
							m.handleQuestion(),
							m.countdownCmd(),
						)
					} else {
						// No pause, immediate response
						return m, m.handleQuestion()
					}
				}
			} else {
				m.statusText = "Process completed"
				return m, tea.Quit
			}
		default:
			// No new output
		}
		
		return m, tickCmd()

	case statusMsg:
		m.statusText = string(msg)
	
	case countdownMsg:
		if m.countdown > 0 && !m.isPaused {
			m.countdown = int(msg)
			if m.countdown > 0 {
				m.statusText = fmt.Sprintf("Auto-responding in %ds... Press Space to pause", m.countdown)
				return m, m.countdownCmd()
			} else if m.llmSuggestion != "" {
				// Countdown finished, auto-send
				m.sendInput(m.llmSuggestion)
				m.statusText = fmt.Sprintf("Auto-sent: %s", m.llmSuggestion)
				m.llmSuggestion = ""
				m.detectedPrompt = ""
			}
		}
	
	case llmResponseMsg:
		m.llmSuggestion = msg.suggestion
		m.detectedPrompt = msg.prompt
		if m.countdown == 0 && m.mode == ModeAuto && m.pauseDuration == 0 {
			// Immediate mode
			m.sendInput(m.llmSuggestion)
			m.statusText = fmt.Sprintf("Auto-sent: %s", m.llmSuggestion)
			m.llmSuggestion = ""
			m.detectedPrompt = ""
		}
	}

	return m, nil
}

func (m *model) detectQuestion() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	// Get recent output
	recentOutput := strings.Join(m.outputBuffer, "")
	lines := strings.Split(recentOutput, "\n")
	
	// Check last 5 lines
	start := len(lines) - 5
	if start < 0 {
		start = 0
	}
	
	lastLines := strings.Join(lines[start:], "\n")
	lowerLines := strings.ToLower(lastLines)
	
	for _, pattern := range questionPatterns {
		if pattern.MatchString(lowerLines) {
			return true
		}
	}
	
	return false
}

func (m *model) countdownCmd() tea.Cmd {
	return tea.Tick(time.Second, func(t time.Time) tea.Msg {
		return countdownMsg(m.countdown - 1)
	})
}

func (m *model) handleQuestion() tea.Cmd {
	return func() tea.Msg {
		if m.llmClient == nil {
			return statusMsg("No OpenAI API key set")
		}
		
		m.mu.Lock()
		recentContext := m.getRecentContext()
		m.mu.Unlock()
		
		// Extract last few lines for prompt detection
		lines := strings.Split(recentContext, "\n")
		lastFew := lines
		if len(lines) > 5 {
			lastFew = lines[len(lines)-5:]
		}
		promptText := strings.Join(lastFew, "\n")
		
		prompt := fmt.Sprintf(`You are watching a terminal application output. 
The following is currently shown:

%s

The application appears to be waiting for user input. 
Analyze the output and determine what input should be provided.
Respond with ONLY the exact input to send (e.g., 'y', 'n', '1', 'yes', or just 'ENTER' for pressing enter).
If unsure, respond with 'ENTER'.`, recentContext)
		
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		
		resp, err := m.llmClient.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
			Model:       openai.GPT3Dot5Turbo,
			Messages:    []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: prompt}},
			Temperature: 0.1,
			MaxTokens:   10,
		})
		
		if err != nil {
			return statusMsg(fmt.Sprintf("LLM error: %v", err))
		}
		
		if len(resp.Choices) > 0 {
			response := strings.TrimSpace(resp.Choices[0].Message.Content)
			return llmResponseMsg{
				suggestion: response,
				prompt:     promptText,
			}
		}
		
		return statusMsg("No LLM response")
	}
}

func (m *model) getRecentContext() string {
	// Get last N lines
	allOutput := strings.Join(m.outputBuffer, "")
	lines := strings.Split(allOutput, "\n")
	
	start := len(lines) - contextLines
	if start < 0 {
		start = 0
	}
	
	return strings.Join(lines[start:], "\n")
}

func (m model) sendInput(input string) {
	if m.ptyFile == nil {
		return
	}
	
	if strings.ToUpper(input) == "ENTER" {
		m.ptyFile.Write([]byte("\n"))
	} else {
		m.ptyFile.Write([]byte(input + "\n"))
	}
}

func (m model) View() string {
	if m.quitting {
		return ""
	}
	
	// Mode indicator
	modeStr := "AUTO"
	if m.mode == ModeManual {
		modeStr = "MANUAL"
	}
	
	// Title with mode
	title := titleStyle.Render(fmt.Sprintf("[%s] YesMan - Command: %s", modeStr, strings.Join(m.command, " ")))
	
	// Output
	m.mu.Lock()
	outputText := strings.Join(m.outputBuffer, "")
	m.mu.Unlock()
	
	// Calculate view height
	baseHeight := 10
	if m.showHelp {
		baseHeight = 20
	}
	if m.isPaused {
		baseHeight = 15
	}
	
	viewHeight := m.height - baseHeight
	if viewHeight < 5 {
		viewHeight = 5
	}
	
	// Clean and prepare output
	outputText = strings.TrimSpace(outputText)
	lines := strings.Split(outputText, "\n")
	
	// Get last N lines that fit in view
	start := len(lines) - viewHeight
	if start < 0 {
		start = 0
	}
	
	visibleLines := lines[start:]
	// Trim each line to prevent wrapping issues
	maxLineWidth := m.width - 8
	if maxLineWidth < 10 {
		maxLineWidth = 10
	}
	for i, line := range visibleLines {
		if len(line) > maxLineWidth {
			visibleLines[i] = line[:maxLineWidth]
		}
	}
	
	visibleOutput := strings.Join(visibleLines, "\n")
	output := outputStyle.Width(m.width - 4).Height(viewHeight).Render(visibleOutput)
	
	// Build components
	components := []string{title, output}
	
	// Pause menu
	if m.isPaused && m.llmSuggestion != "" {
		pauseBox := lipgloss.NewStyle().
			BorderStyle(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("226")).
			Padding(1).
			Width(m.width - 4).
			Render(fmt.Sprintf("┌─ Paused ──────────────────┐\n│ %s\n│ LLM suggests: '%s'\n│\n│ Enter: Accept\n│ Esc: Manual control\n│ Any key: Manual input\n└───────────────────────────┘",
				m.detectedPrompt, m.llmSuggestion))
		components = append(components, pauseBox)
	}
	
	// Help
	if m.showHelp {
		help := lipgloss.NewStyle().
			BorderStyle(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("241")).
			Padding(1).
			Width(m.width - 4).
			Render("Controls:\n  Space - Pause automation\n  Enter - Accept LLM suggestion\n  Esc   - Manual mode\n  ?     - Toggle help\n  Ctrl+C - Quit")
		components = append(components, help)
	}
	
	// Status
	statusText := m.statusText
	if m.countdown > 0 && !m.isPaused {
		statusText = fmt.Sprintf("Auto-responding in %ds... Press Space to pause", m.countdown)
	}
	
	status := statusStyle.Width(m.width - 4).Render(statusText)
	components = append(components, status)
	
	return lipgloss.JoinVertical(lipgloss.Left, components...)
}

func main() {
	var (
		autoMode = flag.Bool("auto", false, "Full auto mode (no pause)")
		manualMode = flag.Bool("manual", false, "Manual mode (no automation)")
		pauseSec = flag.Int("pause", 3, "Pause duration in seconds before auto-response")
	)
	
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options] <command> [args...]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s ./install.sh                    # Default mode (3s pause)\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --auto npm install              # No pause\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --manual apt upgrade            # Manual control only\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --pause 5 ./configure           # 5 second pause\n", os.Args[0])
	}
	
	flag.Parse()
	
	if flag.NArg() < 1 {
		flag.Usage()
		os.Exit(1)
	}
	
	// Determine mode
	mode := ModeAuto
	pauseDuration := time.Duration(*pauseSec) * time.Second
	
	if *manualMode {
		mode = ModeManual
		pauseDuration = 0
	} else if *autoMode {
		pauseDuration = 0
	}
	
	command := flag.Args()
	
	p := tea.NewProgram(initialModel(command, mode, pauseDuration), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
}