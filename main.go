package main

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"

	openai "github.com/sashabaranov/go-openai"
)

type Message struct {
	Content string
	Role    string
	CauseBy string
}

type Memory struct {
	mu      sync.Mutex
	history []Message
}

func (m *Memory) Add(msg Message) {
	m.mu.Lock()
	defer m.mu.Unlock()
	//使用切片存储历史消息
	m.history = append(m.history, msg)
}

func (m *Memory) GetRecent() []Message {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.history) == 0 {
		return nil
	}
	return m.history[len(m.history)-1:]
}


type Action interface {
	Run(ctx context.Context, input string) (string, error)
	Name() string
}


type SimpleWriteCode struct {
	llmClient *openai.Client
}

// Name returns the name identifier for the SimpleWriteCode agent type.
func (a *SimpleWriteCode) Name() string { return "SimpleWriteCode" }

func (a *SimpleWriteCode) Run(ctx context.Context, instruction string) (string, error) {
	prompt := fmt.Sprintf("Write a python function that can %s.\nReturn ```python\nyour_code_here``` with NO other texts.", instruction)
	
	// Azure OpenAI 调用配置
	req := openai.ChatCompletionRequest{
		Model: "gpt-4", // 使用部署名称而非模型ID
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			},
		},
	}
	
	resp, err := a.llmClient.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("Azure OpenAI API error: %w", err)
	}
	
	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == "" {
		return "", errors.New("no response from Azure OpenAI")
	}
	
	return parseCode(resp.Choices[0].Message.Content), nil
}

type SimpleWriteTest struct {
	llmClient *openai.Client
}

func (a *SimpleWriteTest) Name() string { return "SimpleWriteTest" }

func (a *SimpleWriteTest) Run(ctx context.Context, contextData string) (string, error) {
	prompt := fmt.Sprintf("Context: %s\nWrite 3 unit tests using pytest for the given function, assuming you have imported it.\nReturn ```python\nyour_code_here``` with NO other texts.", contextData)
	
	req := openai.ChatCompletionRequest{
		Model: "gpt-4", // 使用部署名称
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			},
		},
	}
	
	resp, err := a.llmClient.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("Azure OpenAI API error: %w", err)
	}
	
	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == "" {
		return "", errors.New("no response from Azure OpenAI")
	}
	
	return parseCode(resp.Choices[0].Message.Content), nil
}

type SimpleWriteReview struct {
	llmClient *openai.Client
}

func (a *SimpleWriteReview) Name() string { return "SimpleWriteReview" }

func (a *SimpleWriteReview) Run(ctx context.Context, contextData string) (string, error) {
	prompt := fmt.Sprintf("Context: %s\nReview the test cases and provide one critical comment:", contextData)
	
	req := openai.ChatCompletionRequest{
		Model: "gpt-4", // 使用部署名称
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			},
		},
	}
	
	resp, err := a.llmClient.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("Azure OpenAI API error: %w", err)
	}
	
	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == "" {
		return "", errors.New("no response from Azure OpenAI")
	}
	
	return resp.Choices[0].Message.Content, nil
}

func parseCode(rsp string) string {
	re := regexp.MustCompile("")
	matches := re.FindStringSubmatch(rsp)
	if len(matches) > 1 {
		return strings.TrimSpace(matches[1])
	}
	
	// 尝试匹配不带语言标签的代码块
	re = regexp.MustCompile("")
	matches = re.FindStringSubmatch(rsp)
	if len(matches) > 1 {
		return strings.TrimSpace(matches[1])
	}
	
	return rsp
}


type Role struct {
	Name      string
	Profile   string
	Actions   []Action
	WatchList []string
	Memory    *Memory
}

func (r *Role) Act(ctx context.Context) (Message, error) {
	contextData := ""
	for _, msg := range r.Memory.GetRecent() {
		contextData += fmt.Sprintf("[%s]: %s\n", msg.Role, msg.Content)
	}

	for _, action := range r.Actions {
		output, err := action.Run(ctx, contextData)
		if err != nil {
			return Message{}, fmt.Errorf("%s action failed: %w", action.Name(), err)
		}

		msg := Message{
			Content: output,
			Role:    r.Profile,
			CauseBy: action.Name(),
		}

		r.Memory.Add(msg)
		return msg, nil
	}

	return Message{}, errors.New("no suitable action found")
}


type Team struct {
	Roles       []*Role
	ProjectIdea string
}

func (t *Team) RunProject(ctx context.Context) {
	userReq := Message{
		Content: t.ProjectIdea,
		Role:    "User",
		CauseBy: "UserRequirement",
	}

	for _, role := range t.Roles {
		role.Memory.Add(userReq)
	}

	var wg sync.WaitGroup
	results := make(chan Message, len(t.Roles))

	for _, role := range t.Roles {
		wg.Add(1)
		go func(r *Role) {
			defer wg.Done()
			msg, err := r.Act(ctx)
			if err != nil {
				fmt.Printf("%s error: %v\n", r.Profile, err)
				return
			}
			results <- msg
		}(role)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	for msg := range results {
		fmt.Printf("=== [%s] OUTPUT ===\n%s\n\n", msg.Role, msg.Content)
		
		for _, role := range t.Roles {
			for _, watchType := range role.WatchList {
				if watchType == msg.CauseBy {
					role.Memory.Add(msg)
				}
			}
		}
	}
}

func main() {
	apiKey := "" // Azure API密钥
	azureEndpoint := "https://azure-openai-wus3.openai.azure.com/" // Azure终结点
	
	// 创建Azure OpenAI客户端配置
	config := openai.DefaultAzureConfig(apiKey, azureEndpoint)
	config.AzureModelMapperFunc = func(model string) string {
		// 将模型名称映射到Azure部署名称
		return "gpt-4" // 使用您在Azure门户中创建的部署名称
	}
	
	llmClient := openai.NewClientWithConfig(config)

	// 创建角色
	coder := &Role{
		Name:    "Alice",
		Profile: "SimpleCoder",
		Actions: []Action{
			&SimpleWriteCode{llmClient: llmClient},
		},
		WatchList: []string{"UserRequirement"},
		Memory:    &Memory{},
	}

	tester := &Role{
		Name:    "Bob",
		Profile: "SimpleTester",
		Actions: []Action{
			&SimpleWriteTest{llmClient: llmClient},
		},
		WatchList: []string{"SimpleWriteCode"},
		Memory:    &Memory{},
	}

	reviewer := &Role{
		Name:    "Charlie",
		Profile: "SimpleReviewer",
		Actions: []Action{
			&SimpleWriteReview{llmClient: llmClient},
		},
		WatchList: []string{"SimpleWriteTest"},
		Memory:    &Memory{},
	}

	// 创建团队并运行项目
	team := Team{
		Roles: []*Role{ tester,coder, reviewer},
		ProjectIdea: "write a function that calculates the product of a list",
	}

	team.RunProject(context.Background())
}