package eval

import (
	"sync"
	"time"
)

// CallMetrics captures performance data for a single LLM API call
type CallMetrics struct {
	// Model identification
	ModelID string // Format: "provider:model" (e.g., "openai:gpt-4o-mini")

	// Performance metrics
	LatencyMs int64 // End-to-end latency in milliseconds

	// Token consumption
	InputTokens   int // Prompt tokens
	OutputTokens  int // Completion tokens
	PromptTokens  int // Alternative naming for input tokens (some providers use this)

	// Success/failure tracking
	Success   bool   // True if call completed successfully
	ErrorType string // Error category if Success=false (e.g., "rate_limit", "timeout")

	// Timing
	Timestamp time.Time // When the call was made
}

// MetricsCollector provides thread-safe collection of CallMetrics
// Zero value is ready to use (call NewMetricsCollector for clarity)
type MetricsCollector struct {
	mu      sync.RWMutex
	metrics []CallMetrics
}

// NewMetricsCollector creates a new MetricsCollector
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		metrics: make([]CallMetrics, 0),
	}
}

// RecordCall adds a CallMetrics entry to the collector
// Safe for concurrent use from multiple goroutines
func (mc *MetricsCollector) RecordCall(m CallMetrics) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.metrics = append(mc.metrics, m)
}

// GetMetrics returns a copy of all recorded metrics
// Safe for concurrent use from multiple goroutines
func (mc *MetricsCollector) GetMetrics() []CallMetrics {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	// Return a copy to prevent external mutation
	result := make([]CallMetrics, len(mc.metrics))
	copy(result, mc.metrics)
	return result
}

// GetMetricsByModel returns all metrics for a specific model ID
// Safe for concurrent use from multiple goroutines
func (mc *MetricsCollector) GetMetricsByModel(modelID string) []CallMetrics {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	result := make([]CallMetrics, 0)
	for _, m := range mc.metrics {
		if m.ModelID == modelID {
			result = append(result, m)
		}
	}
	return result
}

// Reset clears all recorded metrics
// Safe for concurrent use from multiple goroutines
func (mc *MetricsCollector) Reset() {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.metrics = make([]CallMetrics, 0)
}
