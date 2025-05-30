package main

import (
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
	command      []string
	outputBuffer []string
	statusText   string
	width        int
	height       int
	ptyFile      *os.File
	cmd          *exec.Cmd
	outputChan   chan string
	llmClient    *openai.Client
	lastLLMCheck time.Time
	mu           sync.Mutex
	quitting     bool
}

type outputMsg string
type statusMsg string
type tickMsg time.Time

func initialModel(command []string) model {
	apiKey := os.Getenv("OPENAI_API_KEY")
	var client *openai.Client
	if apiKey != "" {
		client = openai.NewClient(apiKey)
	}

	return model{
		command:      command,
		outputBuffer: []string{},
		statusText:   "Starting...",
		outputChan:   make(chan string, 100),
		llmClient:    client,
		lastLLMCheck: time.Now(),
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
			m.outputChan <- string(buf[:n])
		}
	}
	close(m.outputChan)
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			m.quitting = true
			if m.cmd != nil && m.cmd.Process != nil {
				m.cmd.Process.Kill()
			}
			return m, tea.Quit
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
				if m.detectQuestion() && time.Since(m.lastLLMCheck) > 2*time.Second {
					m.lastLLMCheck = time.Now()
					return m, m.handleQuestion()
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

func (m *model) handleQuestion() tea.Cmd {
	return func() tea.Msg {
		if m.llmClient == nil {
			return statusMsg("No OpenAI API key set")
		}
		
		m.mu.Lock()
		context := m.getRecentContext()
		m.mu.Unlock()
		
		prompt := fmt.Sprintf(`You are watching a terminal application output. 
The following is currently shown:

%s

The application appears to be waiting for user input. 
Analyze the output and determine what input should be provided.
Respond with ONLY the exact input to send (e.g., 'y', 'n', '1', 'yes', or just 'ENTER' for pressing enter).
If unsure, respond with 'ENTER'.`, context)
		
		resp, err := m.llmClient.CreateChatCompletion(nil, openai.ChatCompletionRequest{
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
			m.sendInput(response)
			return statusMsg(fmt.Sprintf("Sent: %s", response))
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

func (m *model) sendInput(input string) {
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
	
	// Title
	title := titleStyle.Render(fmt.Sprintf("YesMan TUI - Command: %s", strings.Join(m.command, " ")))
	
	// Output
	m.mu.Lock()
	outputText := strings.Join(m.outputBuffer, "")
	m.mu.Unlock()
	
	// Get last lines that fit in view
	lines := strings.Split(outputText, "\n")
	viewHeight := m.height - 10 // Account for borders and status
	if viewHeight < 5 {
		viewHeight = 5
	}
	
	start := len(lines) - viewHeight
	if start < 0 {
		start = 0
	}
	
	visibleOutput := strings.Join(lines[start:], "\n")
	output := outputStyle.Width(m.width - 4).Height(viewHeight).Render(visibleOutput)
	
	// Status
	status := statusStyle.Width(m.width - 4).Render(fmt.Sprintf("Status: %s", m.statusText))
	
	return lipgloss.JoinVertical(lipgloss.Left, title, output, status)
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: yesman <command> [args...]")
		fmt.Println("Example: yesman apt install htop")
		os.Exit(1)
	}
	
	p := tea.NewProgram(initialModel(os.Args[1:]), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
}