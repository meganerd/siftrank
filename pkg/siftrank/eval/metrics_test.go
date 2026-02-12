package eval

import (
	"context"
	"sync"
	"testing"
	"time"
)

func TestCallMetrics_Basic(t *testing.T) {
	metrics := CallMetrics{
		ModelID:       "openai:gpt-4o-mini",
		LatencyMs:     150,
		InputTokens:   100,
		OutputTokens:  50,
		PromptTokens:  100,
		Success:       true,
		ErrorType:     "",
		Timestamp:     time.Now(),
	}

	if metrics.ModelID != "openai:gpt-4o-mini" {
		t.Errorf("Expected ModelID openai:gpt-4o-mini, got %s", metrics.ModelID)
	}

	if metrics.LatencyMs != 150 {
		t.Errorf("Expected LatencyMs 150, got %d", metrics.LatencyMs)
	}

	if !metrics.Success {
		t.Error("Expected Success true, got false")
	}
}

func TestMetricsCollector_RecordCall(t *testing.T) {
	collector := NewMetricsCollector()

	metrics := CallMetrics{
		ModelID:      "openai:gpt-4o-mini",
		LatencyMs:    100,
		InputTokens:  50,
		OutputTokens: 25,
		Success:      true,
		Timestamp:    time.Now(),
	}

	collector.RecordCall(metrics)

	allMetrics := collector.GetMetrics()
	if len(allMetrics) != 1 {
		t.Fatalf("Expected 1 metric, got %d", len(allMetrics))
	}

	if allMetrics[0].ModelID != "openai:gpt-4o-mini" {
		t.Errorf("Expected ModelID openai:gpt-4o-mini, got %s", allMetrics[0].ModelID)
	}
}

func TestMetricsCollector_ConcurrentAccess(t *testing.T) {
	collector := NewMetricsCollector()
	var wg sync.WaitGroup

	// Simulate 100 concurrent calls
	numCalls := 100
	wg.Add(numCalls)

	for i := 0; i < numCalls; i++ {
		go func(id int) {
			defer wg.Done()
			metrics := CallMetrics{
				ModelID:      "openai:gpt-4o-mini",
				LatencyMs:    int64(100 + id),
				InputTokens:  50,
				OutputTokens: 25,
				Success:      true,
				Timestamp:    time.Now(),
			}
			collector.RecordCall(metrics)
		}(i)
	}

	wg.Wait()

	allMetrics := collector.GetMetrics()
	if len(allMetrics) != numCalls {
		t.Errorf("Expected %d metrics, got %d", numCalls, len(allMetrics))
	}
}

func TestMetricsCollector_Reset(t *testing.T) {
	collector := NewMetricsCollector()

	// Add some metrics
	for i := 0; i < 5; i++ {
		metrics := CallMetrics{
			ModelID:   "openai:gpt-4o-mini",
			LatencyMs: int64(100 + i),
			Success:   true,
			Timestamp: time.Now(),
		}
		collector.RecordCall(metrics)
	}

	if len(collector.GetMetrics()) != 5 {
		t.Fatalf("Expected 5 metrics before reset, got %d", len(collector.GetMetrics()))
	}

	collector.Reset()

	if len(collector.GetMetrics()) != 0 {
		t.Errorf("Expected 0 metrics after reset, got %d", len(collector.GetMetrics()))
	}
}

func TestMetricsCollector_GetMetricsByModel(t *testing.T) {
	collector := NewMetricsCollector()

	// Add metrics for different models
	models := []string{"openai:gpt-4o-mini", "ollama:qwen2.5-coder:32b"}
	for i := 0; i < 10; i++ {
		modelID := models[i%2]
		metrics := CallMetrics{
			ModelID:   modelID,
			LatencyMs: int64(100 + i),
			Success:   true,
			Timestamp: time.Now(),
		}
		collector.RecordCall(metrics)
	}

	gptMetrics := collector.GetMetricsByModel("openai:gpt-4o-mini")
	if len(gptMetrics) != 5 {
		t.Errorf("Expected 5 metrics for gpt-4o-mini, got %d", len(gptMetrics))
	}

	ollamaMetrics := collector.GetMetricsByModel("ollama:qwen2.5-coder:32b")
	if len(ollamaMetrics) != 5 {
		t.Errorf("Expected 5 metrics for ollama, got %d", len(ollamaMetrics))
	}

	unknownMetrics := collector.GetMetricsByModel("unknown:model")
	if len(unknownMetrics) != 0 {
		t.Errorf("Expected 0 metrics for unknown model, got %d", len(unknownMetrics))
	}
}

func TestMetricsCollector_ErrorTracking(t *testing.T) {
	collector := NewMetricsCollector()

	// Record successful call
	collector.RecordCall(CallMetrics{
		ModelID:   "openai:gpt-4o-mini",
		LatencyMs: 100,
		Success:   true,
		Timestamp: time.Now(),
	})

	// Record failed call
	collector.RecordCall(CallMetrics{
		ModelID:   "openai:gpt-4o-mini",
		LatencyMs: 50,
		Success:   false,
		ErrorType: "rate_limit",
		Timestamp: time.Now(),
	})

	allMetrics := collector.GetMetrics()
	if len(allMetrics) != 2 {
		t.Fatalf("Expected 2 metrics, got %d", len(allMetrics))
	}

	// Verify error was recorded
	var foundError bool
	for _, m := range allMetrics {
		if !m.Success && m.ErrorType == "rate_limit" {
			foundError = true
			break
		}
	}

	if !foundError {
		t.Error("Expected to find error metric with rate_limit error type")
	}
}

func TestMetricsCollector_TimestampOrdering(t *testing.T) {
	collector := NewMetricsCollector()

	now := time.Now()

	// Record metrics with different timestamps
	for i := 0; i < 5; i++ {
		metrics := CallMetrics{
			ModelID:   "openai:gpt-4o-mini",
			LatencyMs: int64(100 + i),
			Success:   true,
			Timestamp: now.Add(time.Duration(i) * time.Second),
		}
		collector.RecordCall(metrics)
	}

	allMetrics := collector.GetMetrics()

	// Verify timestamps are in order
	for i := 1; i < len(allMetrics); i++ {
		if allMetrics[i].Timestamp.Before(allMetrics[i-1].Timestamp) {
			t.Error("Metrics are not in chronological order")
		}
	}
}

// BenchmarkMetricsCollector_RecordCall tests that recording metrics has <1ms overhead
func BenchmarkMetricsCollector_RecordCall(b *testing.B) {
	collector := NewMetricsCollector()
	ctx := context.Background()

	metrics := CallMetrics{
		ModelID:      "openai:gpt-4o-mini",
		LatencyMs:    100,
		InputTokens:  50,
		OutputTokens: 25,
		Success:      true,
		Timestamp:    time.Now(),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		collector.RecordCall(metrics)
	}

	_ = ctx // Use ctx to avoid unused variable error
}

// BenchmarkMetricsCollector_ConcurrentRecording tests concurrent recording overhead
func BenchmarkMetricsCollector_ConcurrentRecording(b *testing.B) {
	collector := NewMetricsCollector()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			metrics := CallMetrics{
				ModelID:      "openai:gpt-4o-mini",
				LatencyMs:    100,
				InputTokens:  50,
				OutputTokens: 25,
				Success:      true,
				Timestamp:    time.Now(),
			}
			collector.RecordCall(metrics)
		}
	})
}
