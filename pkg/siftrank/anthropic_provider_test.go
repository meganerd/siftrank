package siftrank

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestAnthropicProviderCreation verifies that the provider can be created
// with proper configuration
func TestAnthropicProviderCreation(t *testing.T) {
	cfg := AnthropicConfig{
		Auth:     NewHeaderAuth("x-api-key", "test-key"),
		Model:    "claude-3-5-sonnet-20241022",
		Encoding: "cl100k_base",
		Logger:   slog.Default(),
	}

	provider, err := NewAnthropicProvider(cfg)
	if err != nil {
		t.Fatalf("NewAnthropicProvider failed: %v", err)
	}

	if provider == nil {
		t.Fatal("Expected non-nil provider")
	}
}

// TestAnthropicProviderImplementsLLMProvider verifies interface compliance
func TestAnthropicProviderImplementsLLMProvider(t *testing.T) {
	cfg := AnthropicConfig{
		Auth:     NewHeaderAuth("x-api-key", "test-key"),
		Model:    "claude-3-5-sonnet-20241022",
		Encoding: "cl100k_base",
		Logger:   slog.Default(),
	}

	provider, err := NewAnthropicProvider(cfg)
	if err != nil {
		t.Fatalf("NewAnthropicProvider failed: %v", err)
	}

	// Verify it implements LLMProvider
	var _ LLMProvider = provider
}

// TestAnthropicProviderComplete tests the Complete method with a mock server
func TestAnthropicProviderComplete(t *testing.T) {
	// Create a mock server that returns Anthropic-style responses
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request method and path
		if r.Method != "POST" {
			t.Errorf("Expected POST request, got %s", r.Method)
		}
		if !strings.HasSuffix(r.URL.Path, "/messages") {
			t.Errorf("Expected path ending with /messages, got %s", r.URL.Path)
		}

		// Verify headers
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("Expected x-api-key header to be 'test-key', got '%s'", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("Expected Content-Type header to be 'application/json', got '%s'", r.Header.Get("Content-Type"))
		}

		// Read and validate request body
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("Failed to read request body: %v", err)
		}

		var reqBody map[string]interface{}
		if err := json.Unmarshal(body, &reqBody); err != nil {
			t.Errorf("Failed to parse request body: %v", err)
		}

		// Verify model is present
		if reqBody["model"] != "claude-3-5-sonnet-20241022" {
			t.Errorf("Expected model 'claude-3-5-sonnet-20241022', got %v", reqBody["model"])
		}

		// Return mock response
		response := map[string]interface{}{
			"id":    "msg_01XFDUDYJgAACzvnptvVoYEL",
			"type":  "message",
			"role":  "assistant",
			"model": "claude-3-5-sonnet-20241022",
			"content": []map[string]interface{}{
				{
					"type": "text",
					"text": "Hello! I'm Claude.",
				},
			},
			"stop_reason": "end_turn",
			"usage": map[string]interface{}{
				"input_tokens":  10,
				"output_tokens": 5,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	cfg := AnthropicConfig{
		Auth:     NewHeaderAuth("x-api-key", "test-key"),
		Model:    "claude-3-5-sonnet-20241022",
		BaseURL:  server.URL,
		Encoding: "cl100k_base",
		Logger:   slog.Default(),
	}

	provider, err := NewAnthropicProvider(cfg)
	if err != nil {
		t.Fatalf("NewAnthropicProvider failed: %v", err)
	}

	opts := &CompletionOptions{}
	result, err := provider.Complete(context.Background(), "Hello!", opts)
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}

	if result != "Hello! I'm Claude." {
		t.Errorf("Expected 'Hello! I'm Claude.', got '%s'", result)
	}

	// Verify usage was populated
	if opts.Usage.InputTokens != 10 {
		t.Errorf("Expected 10 input tokens, got %d", opts.Usage.InputTokens)
	}
	if opts.Usage.OutputTokens != 5 {
		t.Errorf("Expected 5 output tokens, got %d", opts.Usage.OutputTokens)
	}
	if opts.ModelUsed != "claude-3-5-sonnet-20241022" {
		t.Errorf("Expected model 'claude-3-5-sonnet-20241022', got '%s'", opts.ModelUsed)
	}
	if opts.FinishReason != "end_turn" {
		t.Errorf("Expected finish reason 'end_turn', got '%s'", opts.FinishReason)
	}
}

// TestAnthropicProviderCompleteWithOptions tests temperature and max tokens
func TestAnthropicProviderCompleteWithOptions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var reqBody map[string]interface{}
		json.Unmarshal(body, &reqBody)

		// Verify temperature and max_tokens were set
		if temp, ok := reqBody["temperature"].(float64); !ok || temp != 0.5 {
			t.Errorf("Expected temperature 0.5, got %v", reqBody["temperature"])
		}
		if maxTokens, ok := reqBody["max_tokens"].(float64); !ok || maxTokens != 100 {
			t.Errorf("Expected max_tokens 100, got %v", reqBody["max_tokens"])
		}

		response := map[string]interface{}{
			"id":          "msg_123",
			"type":        "message",
			"role":        "assistant",
			"model":       "claude-3-5-sonnet-20241022",
			"content":     []map[string]interface{}{{"type": "text", "text": "Response"}},
			"stop_reason": "end_turn",
			"usage":       map[string]interface{}{"input_tokens": 5, "output_tokens": 3},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	cfg := AnthropicConfig{
		Auth:     NewHeaderAuth("x-api-key", "test-key"),
		Model:    "claude-3-5-sonnet-20241022",
		BaseURL:  server.URL,
		Encoding: "cl100k_base",
		Logger:   slog.Default(),
	}

	provider, err := NewAnthropicProvider(cfg)
	if err != nil {
		t.Fatalf("NewAnthropicProvider failed: %v", err)
	}

	temp := 0.5
	maxTokens := 100
	opts := &CompletionOptions{
		Temperature: &temp,
		MaxTokens:   &maxTokens,
	}

	_, err = provider.Complete(context.Background(), "Hello!", opts)
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
}

// TestAnthropicProviderRateLimitRetry tests rate limit handling
func TestAnthropicProviderRateLimitRetry(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			// First call: rate limited
			w.Header().Set("retry-after", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"type": "error",
				"error": map[string]interface{}{
					"type":    "rate_limit_error",
					"message": "Rate limited",
				},
			})
			return
		}

		// Second call: success
		response := map[string]interface{}{
			"id":          "msg_123",
			"type":        "message",
			"role":        "assistant",
			"model":       "claude-3-5-sonnet-20241022",
			"content":     []map[string]interface{}{{"type": "text", "text": "Success after retry"}},
			"stop_reason": "end_turn",
			"usage":       map[string]interface{}{"input_tokens": 5, "output_tokens": 3},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	cfg := AnthropicConfig{
		Auth:     NewHeaderAuth("x-api-key", "test-key"),
		Model:    "claude-3-5-sonnet-20241022",
		BaseURL:  server.URL,
		Encoding: "cl100k_base",
		Logger:   slog.Default(),
	}

	provider, err := NewAnthropicProvider(cfg)
	if err != nil {
		t.Fatalf("NewAnthropicProvider failed: %v", err)
	}

	result, err := provider.Complete(context.Background(), "Hello!", nil)
	if err != nil {
		t.Fatalf("Complete failed after retry: %v", err)
	}

	if result != "Success after retry" {
		t.Errorf("Expected 'Success after retry', got '%s'", result)
	}

	if callCount != 2 {
		t.Errorf("Expected 2 calls (1 rate limited + 1 success), got %d", callCount)
	}
}

// TestAnthropicProviderServerErrorRetry tests 5xx error retry behavior
func TestAnthropicProviderServerErrorRetry(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount <= 2 {
			// First two calls: server error
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"type": "error",
				"error": map[string]interface{}{
					"type":    "api_error",
					"message": "Internal server error",
				},
			})
			return
		}

		// Third call: success
		response := map[string]interface{}{
			"id":          "msg_123",
			"type":        "message",
			"role":        "assistant",
			"model":       "claude-3-5-sonnet-20241022",
			"content":     []map[string]interface{}{{"type": "text", "text": "Success"}},
			"stop_reason": "end_turn",
			"usage":       map[string]interface{}{"input_tokens": 5, "output_tokens": 3},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	cfg := AnthropicConfig{
		Auth:     NewHeaderAuth("x-api-key", "test-key"),
		Model:    "claude-3-5-sonnet-20241022",
		BaseURL:  server.URL,
		Encoding: "cl100k_base",
		Logger:   slog.Default(),
	}

	provider, err := NewAnthropicProvider(cfg)
	if err != nil {
		t.Fatalf("NewAnthropicProvider failed: %v", err)
	}

	result, err := provider.Complete(context.Background(), "Hello!", nil)
	if err != nil {
		t.Fatalf("Complete failed after retries: %v", err)
	}

	if result != "Success" {
		t.Errorf("Expected 'Success', got '%s'", result)
	}
}

// TestAnthropicProviderUnrecoverableError tests client error handling
func TestAnthropicProviderUnrecoverableError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"type": "error",
			"error": map[string]interface{}{
				"type":    "authentication_error",
				"message": "Invalid API key",
			},
		})
	}))
	defer server.Close()

	cfg := AnthropicConfig{
		Auth:     NewHeaderAuth("x-api-key", "invalid-key"),
		Model:    "claude-3-5-sonnet-20241022",
		BaseURL:  server.URL,
		Encoding: "cl100k_base",
		Logger:   slog.Default(),
	}

	provider, err := NewAnthropicProvider(cfg)
	if err != nil {
		t.Fatalf("NewAnthropicProvider failed: %v", err)
	}

	_, err = provider.Complete(context.Background(), "Hello!", nil)
	if err == nil {
		t.Fatal("Expected error for invalid API key")
	}

	if !strings.Contains(err.Error(), "unrecoverable") {
		t.Errorf("Expected unrecoverable error, got: %v", err)
	}
}

// TestAnthropicProviderContextCancellation tests context handling
func TestAnthropicProviderContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Slow response to allow cancellation
		time.Sleep(5 * time.Second)
		response := map[string]interface{}{
			"id":          "msg_123",
			"type":        "message",
			"role":        "assistant",
			"model":       "claude-3-5-sonnet-20241022",
			"content":     []map[string]interface{}{{"type": "text", "text": "Response"}},
			"stop_reason": "end_turn",
			"usage":       map[string]interface{}{"input_tokens": 5, "output_tokens": 3},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	cfg := AnthropicConfig{
		Auth:     NewHeaderAuth("x-api-key", "test-key"),
		Model:    "claude-3-5-sonnet-20241022",
		BaseURL:  server.URL,
		Encoding: "cl100k_base",
		Logger:   slog.Default(),
	}

	provider, err := NewAnthropicProvider(cfg)
	if err != nil {
		t.Fatalf("NewAnthropicProvider failed: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, err = provider.Complete(ctx, "Hello!", nil)
	if err == nil {
		t.Fatal("Expected error due to context timeout")
	}

	// Should be context error
	if ctx.Err() == nil {
		t.Error("Expected context to be cancelled/timed out")
	}
}

// TestAnthropicProviderConcurrentCalls tests thread safety
func TestAnthropicProviderConcurrentCalls(t *testing.T) {
	var mu sync.Mutex
	callCount := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		callCount++
		currentCall := callCount
		mu.Unlock()

		response := map[string]interface{}{
			"id":          "msg_123",
			"type":        "message",
			"role":        "assistant",
			"model":       "claude-3-5-sonnet-20241022",
			"content":     []map[string]interface{}{{"type": "text", "text": "Response " + string(rune('A'-1+currentCall))}},
			"stop_reason": "end_turn",
			"usage":       map[string]interface{}{"input_tokens": 5, "output_tokens": 3},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	cfg := AnthropicConfig{
		Auth:     NewHeaderAuth("x-api-key", "test-key"),
		Model:    "claude-3-5-sonnet-20241022",
		BaseURL:  server.URL,
		Encoding: "cl100k_base",
		Logger:   slog.Default(),
	}

	provider, err := NewAnthropicProvider(cfg)
	if err != nil {
		t.Fatalf("NewAnthropicProvider failed: %v", err)
	}

	const numGoroutines = 10
	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := provider.Complete(context.Background(), "Hello!", nil)
			if err != nil {
				errors <- err
			}
		}()
	}

	wg.Wait()
	close(errors)

	for err := range errors {
		t.Errorf("Concurrent call failed: %v", err)
	}

	mu.Lock()
	if callCount != numGoroutines {
		t.Errorf("Expected %d calls, got %d", numGoroutines, callCount)
	}
	mu.Unlock()
}

// TestAnthropicProviderMultipleTextBlocks tests handling multiple content blocks
func TestAnthropicProviderMultipleTextBlocks(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := map[string]interface{}{
			"id":    "msg_123",
			"type":  "message",
			"role":  "assistant",
			"model": "claude-3-5-sonnet-20241022",
			"content": []map[string]interface{}{
				{"type": "text", "text": "First part. "},
				{"type": "text", "text": "Second part."},
			},
			"stop_reason": "end_turn",
			"usage":       map[string]interface{}{"input_tokens": 5, "output_tokens": 8},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	cfg := AnthropicConfig{
		Auth:     NewHeaderAuth("x-api-key", "test-key"),
		Model:    "claude-3-5-sonnet-20241022",
		BaseURL:  server.URL,
		Encoding: "cl100k_base",
		Logger:   slog.Default(),
	}

	provider, err := NewAnthropicProvider(cfg)
	if err != nil {
		t.Fatalf("NewAnthropicProvider failed: %v", err)
	}

	result, err := provider.Complete(context.Background(), "Hello!", nil)
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}

	if result != "First part. Second part." {
		t.Errorf("Expected concatenated text 'First part. Second part.', got '%s'", result)
	}
}

// TestAnthropicProviderEstimateTokens tests token estimation
func TestAnthropicProviderEstimateTokens(t *testing.T) {
	cfg := AnthropicConfig{
		Auth:     NewHeaderAuth("x-api-key", "test-key"),
		Model:    "claude-3-5-sonnet-20241022",
		Encoding: "cl100k_base",
		Logger:   slog.Default(),
	}

	provider, err := NewAnthropicProvider(cfg)
	if err != nil {
		t.Fatalf("NewAnthropicProvider failed: %v", err)
	}

	// Verify it implements TokenEstimator by casting to LLMProvider first
	var llmProvider LLMProvider = provider
	estimator, ok := llmProvider.(TokenEstimator)
	if !ok {
		t.Fatal("AnthropicProvider should implement TokenEstimator")
	}

	text := "Hello, how are you doing today?"
	tokens := estimator.EstimateTokens(text)

	// Should return a reasonable number of tokens
	if tokens <= 0 {
		t.Errorf("Expected positive token count, got %d", tokens)
	}

	// Roughly 4 chars per token for English text
	expectedApprox := len(text) / 4
	if tokens < expectedApprox/2 || tokens > expectedApprox*3 {
		t.Errorf("Token count %d seems unreasonable for text length %d", tokens, len(text))
	}
}

// TestAnthropicProviderNilOptions tests Complete with nil options
func TestAnthropicProviderNilOptions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := map[string]interface{}{
			"id":          "msg_123",
			"type":        "message",
			"role":        "assistant",
			"model":       "claude-3-5-sonnet-20241022",
			"content":     []map[string]interface{}{{"type": "text", "text": "Response"}},
			"stop_reason": "end_turn",
			"usage":       map[string]interface{}{"input_tokens": 5, "output_tokens": 3},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	cfg := AnthropicConfig{
		Auth:     NewHeaderAuth("x-api-key", "test-key"),
		Model:    "claude-3-5-sonnet-20241022",
		BaseURL:  server.URL,
		Encoding: "cl100k_base",
		Logger:   slog.Default(),
	}

	provider, err := NewAnthropicProvider(cfg)
	if err != nil {
		t.Fatalf("NewAnthropicProvider failed: %v", err)
	}

	// Pass nil options - should not panic
	result, err := provider.Complete(context.Background(), "Hello!", nil)
	if err != nil {
		t.Fatalf("Complete with nil options failed: %v", err)
	}

	if result != "Response" {
		t.Errorf("Expected 'Response', got '%s'", result)
	}
}

// TestAnthropicFactoryIntegration tests that the factory creates Anthropic providers
func TestAnthropicFactoryIntegration(t *testing.T) {
	cfg := ProviderConfig{
		Type:     ProviderTypeAnthropic,
		APIKey:   "test-key",
		Model:    "claude-3-5-sonnet-20241022",
		Encoding: "cl100k_base",
	}

	provider, err := NewProvider(cfg)
	if err != nil {
		t.Fatalf("NewProvider for Anthropic failed: %v", err)
	}

	if provider == nil {
		t.Fatal("Expected non-nil provider from factory")
	}

	// Verify it implements LLMProvider
	var _ LLMProvider = provider
}
