package siftrank

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"

	"github.com/meganerd/siftrank/pkg/siftrank/eval"
	"github.com/openai/openai-go"
)

// ProviderType identifies the LLM provider implementation
type ProviderType string

const (
	ProviderTypeOpenAI     ProviderType = "openai"
	ProviderTypeOpenRouter ProviderType = "openrouter"
	ProviderTypeAnthropic  ProviderType = "anthropic"
	ProviderTypeGoogle     ProviderType = "google"
	ProviderTypeOllama     ProviderType = "ollama"
)

// ProviderConfig contains common configuration for all providers
type ProviderConfig struct {
	// Provider type (required)
	Type ProviderType

	// Authentication (required for most providers)
	APIKey string `json:"-"` // Used for Bearer auth (OpenAI, OpenRouter, Ollama)

	// Model configuration
	Model    string // Model identifier (required)
	BaseURL  string // Custom base URL (optional, for vLLM, OpenRouter, Ollama)
	Encoding string // Tokenizer encoding (optional, defaults per provider)

	// Advanced options
	Effort string       // Reasoning effort for o1/o3 models (optional)
	Logger *slog.Logger // Logger instance (optional, creates default if nil)

	// Model comparison (optional)
	CompareModels string // Comma-separated list of models to compare (format: "provider:model,provider:model")
}

// NewProvider creates an LLMProvider instance based on the configuration
func NewProvider(cfg ProviderConfig) (LLMProvider, error) {
	// Validate required fields
	if cfg.Type == "" {
		return nil, fmt.Errorf("provider type is required")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("model is required")
	}

	// Create default logger if not provided
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}

	// Route to provider-specific constructor
	switch cfg.Type {
	case ProviderTypeOpenAI, ProviderTypeOpenRouter:
		return newOpenAICompatibleProvider(cfg, logger)
	case ProviderTypeAnthropic:
		return newAnthropicProvider(cfg, logger)
	case ProviderTypeGoogle:
		return nil, fmt.Errorf("google provider not yet implemented")
	case ProviderTypeOllama:
		return newOllamaProvider(cfg, logger)
	default:
		return nil, fmt.Errorf("unknown provider type: %s", cfg.Type)
	}
}

// newOpenAICompatibleProvider creates a provider for OpenAI and OpenRouter
func newOpenAICompatibleProvider(cfg ProviderConfig, logger *slog.Logger) (LLMProvider, error) {
	// OpenAI and OpenRouter both use Bearer token authentication
	// and the OpenAI SDK client format
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("%s provider requires an API key", cfg.Type)
	}

	// Default encoding for OpenAI
	encoding := cfg.Encoding
	if encoding == "" {
		encoding = DefaultEncoding
	}

	return NewOpenAIProvider(OpenAIConfig{
		Auth:     NewBearerAuth(cfg.APIKey),
		Model:    openai.ChatModel(cfg.Model),
		BaseURL:  cfg.BaseURL,
		Encoding: encoding,
		Effort:   cfg.Effort,
		Logger:   logger,
	})
}

// newAnthropicProvider creates a provider for Anthropic
func newAnthropicProvider(cfg ProviderConfig, logger *slog.Logger) (LLMProvider, error) {
	// Anthropic uses x-api-key header authentication
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("anthropic provider requires an API key")
	}

	// Default encoding for Anthropic (cl100k_base works well for Claude)
	encoding := cfg.Encoding
	if encoding == "" {
		encoding = DefaultEncoding
	}

	return NewAnthropicProvider(AnthropicConfig{
		Auth:     NewHeaderAuth("x-api-key", cfg.APIKey),
		Model:    cfg.Model,
		BaseURL:  cfg.BaseURL,
		Encoding: encoding,
		Logger:   logger,
	})
}

// newOllamaProvider creates a provider for Ollama
func newOllamaProvider(cfg ProviderConfig, logger *slog.Logger) (LLMProvider, error) {
	// Ollama uses optional authentication
	// If no API key provided, use NoAuth strategy
	var auth AuthStrategy
	if cfg.APIKey != "" {
		auth = NewBearerAuth(cfg.APIKey)
	} else {
		auth = NewNoAuth()
	}

	// Ollama requires a base URL (typically http://localhost:11434)
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("ollama provider requires a base URL")
	}

	// Default encoding for Ollama (uses tiktoken for estimation)
	encoding := cfg.Encoding
	if encoding == "" {
		encoding = DefaultEncoding
	}

	return NewOpenAIProvider(OpenAIConfig{
		Auth:     auth,
		Model:    openai.ChatModel(cfg.Model),
		BaseURL:  cfg.BaseURL,
		Encoding: encoding,
		Effort:   cfg.Effort,
		Logger:   logger,
	})
}

// llmProviderAdapter adapts siftrank.LLMProvider to eval.LLMProvider
type llmProviderAdapter struct {
	provider LLMProvider
}

func (a *llmProviderAdapter) Complete(ctx context.Context, prompt string, opts eval.CompletionOptionsInterface) (string, error) {
	// Convert eval.CompletionOptionsInterface to *CompletionOptions
	var siftOpts *CompletionOptions
	if opts != nil {
		// Create a CompletionOptions to pass to underlying provider
		siftOpts = &CompletionOptions{}
	}

	// Call underlying provider
	result, err := a.provider.Complete(ctx, prompt, siftOpts)

	// Populate usage back to interface if needed (done by provider)
	return result, err
}

// completionOptionsAdapter adapts *CompletionOptions to eval.CompletionOptionsInterface
type completionOptionsAdapter struct {
	opts *CompletionOptions
}

func (c *completionOptionsAdapter) GetUsage() (int, int) {
	if c.opts == nil {
		return 0, 0
	}
	return c.opts.Usage.InputTokens, c.opts.Usage.OutputTokens
}

// roundRobinSelector implements eval.ProviderSelector with round-robin model selection
type roundRobinSelector struct {
	mu        sync.Mutex
	providers map[string]eval.LLMProvider
	sequence  []string // Model IDs in rotation order
	index     int
}

func (rr *roundRobinSelector) SelectProvider(ctx context.Context) (eval.LLMProvider, string, error) {
	if len(rr.sequence) == 0 {
		return nil, "", fmt.Errorf("no models configured for comparison")
	}

	rr.mu.Lock()
	modelID := rr.sequence[rr.index%len(rr.sequence)]
	rr.index++
	rr.mu.Unlock()

	provider, ok := rr.providers[modelID]
	if !ok {
		return nil, "", fmt.Errorf("provider not found for model: %s", modelID)
	}

	return provider, modelID, nil
}

// evalProviderWrapper wraps eval.EvalProvider to implement siftrank.LLMProvider
type evalProviderWrapper struct {
	evalProvider *eval.EvalProvider
}

func (w *evalProviderWrapper) Complete(ctx context.Context, prompt string, opts *CompletionOptions) (string, error) {
	// Adapt opts to CompletionOptionsInterface
	var optsAdapter eval.CompletionOptionsInterface
	if opts != nil {
		optsAdapter = &completionOptionsAdapter{opts: opts}
	}

	// Call eval provider
	result, err := w.evalProvider.Complete(ctx, prompt, optsAdapter)

	// Usage is populated by underlying provider directly in opts
	return result, err
}

// NewEvalProvider creates an EvalProvider that compares multiple models
// The compareModels string should be in format: "provider:model,provider:model"
// Example: "openai:gpt-4o-mini,ollama:qwen2.5-coder:32b"
func NewEvalProvider(compareModels string, logger *slog.Logger) (LLMProvider, *eval.MetricsCollector, error) {
	if compareModels == "" {
		return nil, nil, fmt.Errorf("compareModels is empty")
	}

	// Parse the compare models string
	modelSpecs := strings.Split(compareModels, ",")
	if len(modelSpecs) == 0 {
		return nil, nil, fmt.Errorf("no models specified in compareModels")
	}

	// Create providers for each model
	providers := make(map[string]eval.LLMProvider)
	sequence := make([]string, 0, len(modelSpecs))

	for _, spec := range modelSpecs {
		spec = strings.TrimSpace(spec)
		if spec == "" {
			continue
		}

		// Parse spec: "provider:model" or "provider:model:variant"
		parts := strings.SplitN(spec, ":", 2)
		if len(parts) < 2 {
			return nil, nil, fmt.Errorf("invalid model spec format: %s (expected provider:model)", spec)
		}

		providerType := ProviderType(parts[0])
		modelID := parts[1]

		// Construct full model identifier (used as key)
		fullModelID := spec

		// Determine API key based on provider
		apiKey := ""
		baseURL := ""

		switch providerType {
		case ProviderTypeOpenAI:
			apiKey = os.Getenv("OPENAI_API_KEY")
		case ProviderTypeOpenRouter:
			apiKey = os.Getenv("OPENROUTER_API_KEY")
		case ProviderTypeOllama:
			// Ollama typically doesn't need API key
			baseURL = os.Getenv("OLLAMA_BASE_URL")
			if baseURL == "" {
				baseURL = "http://localhost:11434"
			}
		case ProviderTypeAnthropic:
			apiKey = os.Getenv("ANTHROPIC_API_KEY")
		case ProviderTypeGoogle:
			return nil, nil, fmt.Errorf("google provider not yet implemented")
		default:
			return nil, nil, fmt.Errorf("unknown provider type: %s", providerType)
		}

		// Create provider config
		cfg := ProviderConfig{
			Type:     providerType,
			APIKey:   apiKey,
			Model:    modelID,
			BaseURL:  baseURL,
			Encoding: DefaultEncoding,
			Logger:   logger,
		}

		// Create provider
		provider, err := NewProvider(cfg)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create provider for %s: %w", fullModelID, err)
		}

		// Wrap provider to adapt to eval.LLMProvider interface
		adaptedProvider := &llmProviderAdapter{provider: provider}
		providers[fullModelID] = adaptedProvider
		sequence = append(sequence, fullModelID)
	}

	if len(providers) == 0 {
		return nil, nil, fmt.Errorf("no valid providers created from compareModels")
	}

	// Create round-robin selector
	selector := &roundRobinSelector{
		providers: providers,
		sequence:  sequence,
		index:     0,
	}

	// Create metrics collector
	collector := eval.NewMetricsCollector()

	// Create EvalProvider
	evalProvider := eval.NewEvalProvider(selector, collector)

	// Wrap to implement siftrank.LLMProvider
	wrapper := &evalProviderWrapper{evalProvider: evalProvider}

	return wrapper, collector, nil
}
