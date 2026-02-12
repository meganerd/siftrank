package siftrank

import (
	"fmt"
	"log/slog"

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
	APIKey string // Used for Bearer auth (OpenAI, OpenRouter, Ollama)

	// Model configuration
	Model    string // Model identifier (required)
	BaseURL  string // Custom base URL (optional, for vLLM, OpenRouter, Ollama)
	Encoding string // Tokenizer encoding (optional, defaults per provider)

	// Advanced options
	Effort string       // Reasoning effort for o1/o3 models (optional)
	Logger *slog.Logger // Logger instance (optional, creates default if nil)
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
		return nil, fmt.Errorf("anthropic provider not yet implemented")
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
