package siftrank

import (
	"testing"

	"github.com/openai/openai-go"
)

func TestNewProvider_OpenAI(t *testing.T) {
	cfg := ProviderConfig{
		Type:     ProviderTypeOpenAI,
		APIKey:   "test-key",
		Model:    string(openai.ChatModelGPT4oMini),
		Encoding: "o200k_base",
	}

	provider, err := NewProvider(cfg)
	if err != nil {
		t.Fatalf("NewProvider failed: %v", err)
	}

	if provider == nil {
		t.Fatal("Expected non-nil provider")
	}

	// Verify it implements LLMProvider
	_, ok := provider.(LLMProvider)
	if !ok {
		t.Fatal("Provider does not implement LLMProvider interface")
	}

	// Verify it implements TokenEstimator (OpenAI provider should)
	_, ok = provider.(TokenEstimator)
	if !ok {
		t.Fatal("OpenAI provider should implement TokenEstimator interface")
	}
}

func TestNewProvider_OpenRouter(t *testing.T) {
	cfg := ProviderConfig{
		Type:     ProviderTypeOpenRouter,
		APIKey:   "test-key",
		Model:    "openai/gpt-4o-mini",
		BaseURL:  "https://openrouter.ai/api/v1",
		Encoding: "o200k_base",
	}

	provider, err := NewProvider(cfg)
	if err != nil {
		t.Fatalf("NewProvider failed: %v", err)
	}

	if provider == nil {
		t.Fatal("Expected non-nil provider")
	}
}

func TestNewProvider_Ollama_NoAuth(t *testing.T) {
	cfg := ProviderConfig{
		Type:     ProviderTypeOllama,
		Model:    "llama3.1:8b",
		BaseURL:  "http://localhost:11434",
		Encoding: "o200k_base",
	}

	provider, err := NewProvider(cfg)
	if err != nil {
		t.Fatalf("NewProvider failed: %v", err)
	}

	if provider == nil {
		t.Fatal("Expected non-nil provider")
	}
}

func TestNewProvider_Ollama_WithAuth(t *testing.T) {
	cfg := ProviderConfig{
		Type:     ProviderTypeOllama,
		APIKey:   "test-key",
		Model:    "llama3.1:8b",
		BaseURL:  "http://localhost:11434",
		Encoding: "o200k_base",
	}

	provider, err := NewProvider(cfg)
	if err != nil {
		t.Fatalf("NewProvider failed: %v", err)
	}

	if provider == nil {
		t.Fatal("Expected non-nil provider")
	}
}

func TestNewProvider_MissingType(t *testing.T) {
	cfg := ProviderConfig{
		APIKey:   "test-key",
		Model:    "gpt-4o-mini",
		Encoding: "o200k_base",
	}

	_, err := NewProvider(cfg)
	if err == nil {
		t.Fatal("Expected error for missing provider type")
	}
}

func TestNewProvider_MissingModel(t *testing.T) {
	cfg := ProviderConfig{
		Type:     ProviderTypeOpenAI,
		APIKey:   "test-key",
		Encoding: "o200k_base",
	}

	_, err := NewProvider(cfg)
	if err == nil {
		t.Fatal("Expected error for missing model")
	}
}

func TestNewProvider_MissingAPIKey(t *testing.T) {
	cfg := ProviderConfig{
		Type:     ProviderTypeOpenAI,
		Model:    string(openai.ChatModelGPT4oMini),
		Encoding: "o200k_base",
	}

	_, err := NewProvider(cfg)
	if err == nil {
		t.Fatal("Expected error for missing API key")
	}
}

func TestNewProvider_OllamaMissingBaseURL(t *testing.T) {
	cfg := ProviderConfig{
		Type:     ProviderTypeOllama,
		Model:    "llama3.1:8b",
		Encoding: "o200k_base",
	}

	_, err := NewProvider(cfg)
	if err == nil {
		t.Fatal("Expected error for Ollama without base URL")
	}
}

func TestNewProvider_UnimplementedProvider(t *testing.T) {
	cfg := ProviderConfig{
		Type:     ProviderTypeAnthropic,
		APIKey:   "test-key",
		Model:    "claude-3-opus",
		Encoding: "o200k_base",
	}

	_, err := NewProvider(cfg)
	if err == nil {
		t.Fatal("Expected error for unimplemented Anthropic provider")
	}
}
