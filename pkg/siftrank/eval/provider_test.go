package eval

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

// mockCompletionOptions implements CompletionOptionsInterface for testing
type mockCompletionOptions struct {
	inputTokens  int
	outputTokens int
}

func (m *mockCompletionOptions) GetUsage() (int, int) {
	return m.inputTokens, m.outputTokens
}

// mockProvider is a test LLMProvider implementation
type mockProvider struct {
	modelID      string
	response     string
	err          error
	latency      time.Duration
	inputTokens  int
	outputTokens int
}

func (m *mockProvider) Complete(ctx context.Context, prompt string, opts CompletionOptionsInterface) (string, error) {
	// Simulate latency
	if m.latency > 0 {
		time.Sleep(m.latency)
	}

	if m.err != nil {
		return "", m.err
	}

	return m.response, nil
}

// mockSelector is a test ProviderSelector implementation
type mockSelector struct {
	mu        sync.Mutex
	providers map[string]LLMProvider
	sequence  []string // Models to rotate through
	index     int
}

func (ms *mockSelector) SelectProvider(ctx context.Context) (LLMProvider, string, error) {
	if len(ms.sequence) == 0 {
		return nil, "", errors.New("no models configured")
	}

	ms.mu.Lock()
	modelID := ms.sequence[ms.index%len(ms.sequence)]
	ms.index++
	ms.mu.Unlock()

	provider, ok := ms.providers[modelID]
	if !ok {
		return nil, "", errors.New("provider not found")
	}

	return provider, modelID, nil
}

func TestEvalProvider_Complete_Success(t *testing.T) {
	collector := NewMetricsCollector()

	mock := &mockProvider{
		modelID:      "openai:gpt-4o-mini",
		response:     "test response",
		inputTokens:  50,
		outputTokens: 25,
		latency:      10 * time.Millisecond,
	}

	selector := &mockSelector{
		providers: map[string]LLMProvider{
			"openai:gpt-4o-mini": mock,
		},
		sequence: []string{"openai:gpt-4o-mini"},
	}

	evalProvider := NewEvalProvider(selector, collector)

	ctx := context.Background()
	opts := &mockCompletionOptions{
		inputTokens:  50,
		outputTokens: 25,
	}

	response, err := evalProvider.Complete(ctx, "test prompt", opts)

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if response != "test response" {
		t.Errorf("Expected 'test response', got '%s'", response)
	}

	// Verify metrics were recorded
	metrics := collector.GetMetrics()
	if len(metrics) != 1 {
		t.Fatalf("Expected 1 metric, got %d", len(metrics))
	}

	m := metrics[0]
	if m.ModelID != "openai:gpt-4o-mini" {
		t.Errorf("Expected ModelID 'openai:gpt-4o-mini', got '%s'", m.ModelID)
	}

	if !m.Success {
		t.Error("Expected Success=true")
	}

	if m.InputTokens != 50 {
		t.Errorf("Expected InputTokens=50, got %d", m.InputTokens)
	}

	if m.OutputTokens != 25 {
		t.Errorf("Expected OutputTokens=25, got %d", m.OutputTokens)
	}

	// Latency should be >= 10ms (the mock sleep time)
	if m.LatencyMs < 10 {
		t.Errorf("Expected LatencyMs >= 10, got %d", m.LatencyMs)
	}
}

func TestEvalProvider_Complete_Error(t *testing.T) {
	collector := NewMetricsCollector()

	mock := &mockProvider{
		modelID: "openai:gpt-4o-mini",
		err:     errors.New("rate limit exceeded"),
	}

	selector := &mockSelector{
		providers: map[string]LLMProvider{
			"openai:gpt-4o-mini": mock,
		},
		sequence: []string{"openai:gpt-4o-mini"},
	}

	evalProvider := NewEvalProvider(selector, collector)

	ctx := context.Background()
	opts := &mockCompletionOptions{}

	_, err := evalProvider.Complete(ctx, "test prompt", opts)

	if err == nil {
		t.Fatal("Expected error, got nil")
	}

	// Verify metrics were recorded for failed call
	metrics := collector.GetMetrics()
	if len(metrics) != 1 {
		t.Fatalf("Expected 1 metric, got %d", len(metrics))
	}

	m := metrics[0]
	if m.Success {
		t.Error("Expected Success=false for error case")
	}

	if m.ErrorType != "rate limit exceeded" {
		t.Errorf("Expected ErrorType='rate limit exceeded', got '%s'", m.ErrorType)
	}
}

func TestEvalProvider_ModelRotation(t *testing.T) {
	collector := NewMetricsCollector()

	mock1 := &mockProvider{
		modelID:      "openai:gpt-4o-mini",
		response:     "response1",
		inputTokens:  50,
		outputTokens: 25,
	}

	mock2 := &mockProvider{
		modelID:      "ollama:qwen2.5-coder:32b",
		response:     "response2",
		inputTokens:  60,
		outputTokens: 30,
	}

	selector := &mockSelector{
		providers: map[string]LLMProvider{
			"openai:gpt-4o-mini":       mock1,
			"ollama:qwen2.5-coder:32b": mock2,
		},
		sequence: []string{"openai:gpt-4o-mini", "ollama:qwen2.5-coder:32b"},
	}

	evalProvider := NewEvalProvider(selector, collector)

	ctx := context.Background()

	// First call should use mock1
	response1, err := evalProvider.Complete(ctx, "prompt1", &mockCompletionOptions{})
	if err != nil {
		t.Fatalf("Call 1 error: %v", err)
	}
	if response1 != "response1" {
		t.Errorf("Expected 'response1', got '%s'", response1)
	}

	// Second call should use mock2
	response2, err := evalProvider.Complete(ctx, "prompt2", &mockCompletionOptions{})
	if err != nil {
		t.Fatalf("Call 2 error: %v", err)
	}
	if response2 != "response2" {
		t.Errorf("Expected 'response2', got '%s'", response2)
	}

	// Third call should rotate back to mock1
	response3, err := evalProvider.Complete(ctx, "prompt3", &mockCompletionOptions{})
	if err != nil {
		t.Fatalf("Call 3 error: %v", err)
	}
	if response3 != "response1" {
		t.Errorf("Expected 'response1', got '%s'", response3)
	}

	// Verify metrics recorded for both models
	metrics := collector.GetMetrics()
	if len(metrics) != 3 {
		t.Fatalf("Expected 3 metrics, got %d", len(metrics))
	}

	// Check model IDs
	if metrics[0].ModelID != "openai:gpt-4o-mini" {
		t.Errorf("Metric 0: expected 'openai:gpt-4o-mini', got '%s'", metrics[0].ModelID)
	}
	if metrics[1].ModelID != "ollama:qwen2.5-coder:32b" {
		t.Errorf("Metric 1: expected 'ollama:qwen2.5-coder:32b', got '%s'", metrics[1].ModelID)
	}
	if metrics[2].ModelID != "openai:gpt-4o-mini" {
		t.Errorf("Metric 2: expected 'openai:gpt-4o-mini', got '%s'", metrics[2].ModelID)
	}
}

func TestEvalProvider_GetCollector(t *testing.T) {
	collector := NewMetricsCollector()

	mock := &mockProvider{
		modelID: "openai:gpt-4o-mini",
	}

	selector := &mockSelector{
		providers: map[string]LLMProvider{
			"openai:gpt-4o-mini": mock,
		},
		sequence: []string{"openai:gpt-4o-mini"},
	}

	evalProvider := NewEvalProvider(selector, collector)

	retrievedCollector := evalProvider.GetCollector()

	if retrievedCollector != collector {
		t.Error("GetCollector returned different collector than provided")
	}
}

func TestEvalProvider_ConcurrentCalls(t *testing.T) {
	collector := NewMetricsCollector()

	mock := &mockProvider{
		modelID:      "openai:gpt-4o-mini",
		response:     "response",
		inputTokens:  50,
		outputTokens: 25,
		latency:      5 * time.Millisecond,
	}

	selector := &mockSelector{
		providers: map[string]LLMProvider{
			"openai:gpt-4o-mini": mock,
		},
		sequence: []string{"openai:gpt-4o-mini"},
	}

	evalProvider := NewEvalProvider(selector, collector)

	// Make 10 concurrent calls
	numCalls := 10
	done := make(chan bool, numCalls)
	errors := make(chan error, numCalls)

	ctx := context.Background()

	for i := 0; i < numCalls; i++ {
		go func() {
			_, err := evalProvider.Complete(ctx, "test", &mockCompletionOptions{})
			if err != nil {
				errors <- err
			}
			done <- true
		}()
	}

	// Wait for all calls to complete
	for i := 0; i < numCalls; i++ {
		<-done
	}

	close(errors)
	for err := range errors {
		t.Errorf("Concurrent call error: %v", err)
	}

	// Verify all metrics were recorded
	metrics := collector.GetMetrics()
	if len(metrics) != numCalls {
		t.Errorf("Expected %d metrics, got %d", numCalls, len(metrics))
	}
}

func TestEvalProvider_NilOptions(t *testing.T) {
	collector := NewMetricsCollector()

	mock := &mockProvider{
		modelID:  "openai:gpt-4o-mini",
		response: "response",
	}

	selector := &mockSelector{
		providers: map[string]LLMProvider{
			"openai:gpt-4o-mini": mock,
		},
		sequence: []string{"openai:gpt-4o-mini"},
	}

	evalProvider := NewEvalProvider(selector, collector)

	ctx := context.Background()

	// Call with nil options should not panic
	_, err := evalProvider.Complete(ctx, "test", nil)

	if err != nil {
		t.Fatalf("Expected no error with nil options, got %v", err)
	}

	// Metrics should still be recorded
	metrics := collector.GetMetrics()
	if len(metrics) != 1 {
		t.Fatalf("Expected 1 metric, got %d", len(metrics))
	}
}

// BenchmarkEvalProvider_Overhead measures the overhead of metrics collection
func BenchmarkEvalProvider_Overhead(b *testing.B) {
	collector := NewMetricsCollector()

	mock := &mockProvider{
		modelID:      "openai:gpt-4o-mini",
		response:     "response",
		inputTokens:  50,
		outputTokens: 25,
	}

	selector := &mockSelector{
		providers: map[string]LLMProvider{
			"openai:gpt-4o-mini": mock,
		},
		sequence: []string{"openai:gpt-4o-mini"},
	}

	evalProvider := NewEvalProvider(selector, collector)

	ctx := context.Background()
	opts := &mockCompletionOptions{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := evalProvider.Complete(ctx, "test", opts)
		if err != nil {
			b.Fatal(err)
		}
	}
}
