package eval

import (
	"math"
	"testing"
	"time"
)

func TestPercentile_EmptySlice(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for empty slice, got none")
		}
	}()

	var empty []int64
	_ = percentile(empty, 50)
}

func TestPercentile_SingleValue(t *testing.T) {
	values := []int64{100}
	result := percentile(values, 50)

	if result != 100 {
		t.Errorf("Expected 100, got %d", result)
	}
}

func TestPercentile_P50(t *testing.T) {
	values := []int64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
	result := percentile(values, 50)

	// P50 should be around the median (55)
	if result < 50 || result > 60 {
		t.Errorf("Expected P50 around 55, got %d", result)
	}
}

func TestPercentile_P95(t *testing.T) {
	values := []int64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
	result := percentile(values, 95)

	// P95 should be near the high end
	if result < 90 {
		t.Errorf("Expected P95 >= 90, got %d", result)
	}
}

func TestPercentile_P99(t *testing.T) {
	values := []int64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
	result := percentile(values, 99)

	// P99 should be at or near the maximum
	if result < 95 {
		t.Errorf("Expected P99 >= 95, got %d", result)
	}
}

func TestPercentile_UnsortedInput(t *testing.T) {
	values := []int64{100, 10, 50, 30, 90, 20, 70, 40, 80, 60}
	result := percentile(values, 50)

	// Should handle unsorted input correctly
	if result < 50 || result > 60 {
		t.Errorf("Expected P50 around 55, got %d", result)
	}
}

func TestPercentile_DuplicateValues(t *testing.T) {
	values := []int64{50, 50, 50, 50, 50}
	result := percentile(values, 50)

	if result != 50 {
		t.Errorf("Expected 50 for all duplicates, got %d", result)
	}
}

func TestModelStats_Calculation(t *testing.T) {
	stats := ModelStats{
		ModelID:     "openai:gpt-4o-mini",
		CallCount:   10,
		SuccessRate: 0.95,
		AvgLatency:  150,
		P50Latency:  145,
		P95Latency:  200,
		P99Latency:  250,
		TotalTokens: 1500,
		ErrorCount:  1,
	}

	if stats.ModelID != "openai:gpt-4o-mini" {
		t.Errorf("Expected ModelID openai:gpt-4o-mini, got %s", stats.ModelID)
	}

	if stats.CallCount != 10 {
		t.Errorf("Expected CallCount 10, got %d", stats.CallCount)
	}

	expectedErrorRate := 1.0 - 0.95
	if math.Abs(stats.SuccessRate-0.95) > 0.01 {
		t.Errorf("Expected SuccessRate 0.95, got %f", stats.SuccessRate)
	}

	_ = expectedErrorRate // Use variable
}

func TestSessionAggregator_AggregateMetrics_Empty(t *testing.T) {
	aggregator := NewSessionAggregator()
	var empty []CallMetrics

	result := aggregator.AggregateMetrics(empty)

	if result.ModelID != "" {
		t.Error("Expected empty ModelID for empty input")
	}

	if result.CallCount != 0 {
		t.Errorf("Expected CallCount 0, got %d", result.CallCount)
	}
}

func TestSessionAggregator_AggregateMetrics_SingleModel(t *testing.T) {
	aggregator := NewSessionAggregator()

	now := time.Now()
	metrics := []CallMetrics{
		{
			ModelID:      "openai:gpt-4o-mini",
			LatencyMs:    100,
			InputTokens:  50,
			OutputTokens: 25,
			Success:      true,
			Timestamp:    now,
		},
		{
			ModelID:      "openai:gpt-4o-mini",
			LatencyMs:    150,
			InputTokens:  60,
			OutputTokens: 30,
			Success:      true,
			Timestamp:    now.Add(1 * time.Second),
		},
		{
			ModelID:      "openai:gpt-4o-mini",
			LatencyMs:    200,
			InputTokens:  70,
			OutputTokens: 35,
			Success:      false,
			ErrorType:    "rate_limit",
			Timestamp:    now.Add(2 * time.Second),
		},
	}

	result := aggregator.AggregateMetrics(metrics)

	if result.ModelID != "openai:gpt-4o-mini" {
		t.Errorf("Expected ModelID openai:gpt-4o-mini, got %s", result.ModelID)
	}

	if result.CallCount != 3 {
		t.Errorf("Expected CallCount 3, got %d", result.CallCount)
	}

	if result.ErrorCount != 1 {
		t.Errorf("Expected ErrorCount 1, got %d", result.ErrorCount)
	}

	expectedSuccessRate := 2.0 / 3.0
	if math.Abs(result.SuccessRate-expectedSuccessRate) > 0.01 {
		t.Errorf("Expected SuccessRate %.2f, got %.2f", expectedSuccessRate, result.SuccessRate)
	}

	expectedAvgLatency := int64((100 + 150 + 200) / 3)
	if result.AvgLatency != expectedAvgLatency {
		t.Errorf("Expected AvgLatency %d, got %d", expectedAvgLatency, result.AvgLatency)
	}

	expectedTotalTokens := (50 + 25) + (60 + 30) + (70 + 35)
	if result.TotalTokens != expectedTotalTokens {
		t.Errorf("Expected TotalTokens %d, got %d", expectedTotalTokens, result.TotalTokens)
	}

	// Check percentiles are calculated
	if result.P50Latency == 0 {
		t.Error("Expected P50Latency to be non-zero")
	}

	if result.P95Latency == 0 {
		t.Error("Expected P95Latency to be non-zero")
	}

	if result.P99Latency == 0 {
		t.Error("Expected P99Latency to be non-zero")
	}
}

func TestSessionAggregator_AggregateByModel(t *testing.T) {
	aggregator := NewSessionAggregator()

	now := time.Now()
	metrics := []CallMetrics{
		{
			ModelID:      "openai:gpt-4o-mini",
			LatencyMs:    100,
			InputTokens:  50,
			OutputTokens: 25,
			Success:      true,
			Timestamp:    now,
		},
		{
			ModelID:      "ollama:qwen2.5-coder:32b",
			LatencyMs:    200,
			InputTokens:  60,
			OutputTokens: 30,
			Success:      true,
			Timestamp:    now.Add(1 * time.Second),
		},
		{
			ModelID:      "openai:gpt-4o-mini",
			LatencyMs:    150,
			InputTokens:  70,
			OutputTokens: 35,
			Success:      true,
			Timestamp:    now.Add(2 * time.Second),
		},
	}

	results := aggregator.AggregateByModel(metrics)

	if len(results) != 2 {
		t.Fatalf("Expected 2 model stats, got %d", len(results))
	}

	// Find stats for each model
	var gptStats, ollamaStats *ModelStats
	for i := range results {
		if results[i].ModelID == "openai:gpt-4o-mini" {
			gptStats = &results[i]
		} else if results[i].ModelID == "ollama:qwen2.5-coder:32b" {
			ollamaStats = &results[i]
		}
	}

	if gptStats == nil {
		t.Fatal("Expected to find stats for openai:gpt-4o-mini")
	}

	if ollamaStats == nil {
		t.Fatal("Expected to find stats for ollama:qwen2.5-coder:32b")
	}

	if gptStats.CallCount != 2 {
		t.Errorf("Expected gpt CallCount 2, got %d", gptStats.CallCount)
	}

	if ollamaStats.CallCount != 1 {
		t.Errorf("Expected ollama CallCount 1, got %d", ollamaStats.CallCount)
	}
}

func TestSessionAggregator_AllSuccessful(t *testing.T) {
	aggregator := NewSessionAggregator()

	now := time.Now()
	metrics := []CallMetrics{
		{
			ModelID:      "openai:gpt-4o-mini",
			LatencyMs:    100,
			InputTokens:  50,
			OutputTokens: 25,
			Success:      true,
			Timestamp:    now,
		},
		{
			ModelID:      "openai:gpt-4o-mini",
			LatencyMs:    150,
			InputTokens:  60,
			OutputTokens: 30,
			Success:      true,
			Timestamp:    now.Add(1 * time.Second),
		},
	}

	result := aggregator.AggregateMetrics(metrics)

	if result.ErrorCount != 0 {
		t.Errorf("Expected ErrorCount 0, got %d", result.ErrorCount)
	}

	if result.SuccessRate != 1.0 {
		t.Errorf("Expected SuccessRate 1.0, got %.2f", result.SuccessRate)
	}
}

func TestSessionAggregator_AllFailed(t *testing.T) {
	aggregator := NewSessionAggregator()

	now := time.Now()
	metrics := []CallMetrics{
		{
			ModelID:   "openai:gpt-4o-mini",
			LatencyMs: 50,
			Success:   false,
			ErrorType: "timeout",
			Timestamp: now,
		},
		{
			ModelID:   "openai:gpt-4o-mini",
			LatencyMs: 75,
			Success:   false,
			ErrorType: "rate_limit",
			Timestamp: now.Add(1 * time.Second),
		},
	}

	result := aggregator.AggregateMetrics(metrics)

	if result.ErrorCount != 2 {
		t.Errorf("Expected ErrorCount 2, got %d", result.ErrorCount)
	}

	if result.SuccessRate != 0.0 {
		t.Errorf("Expected SuccessRate 0.0, got %.2f", result.SuccessRate)
	}

	if result.CallCount != 2 {
		t.Errorf("Expected CallCount 2, got %d", result.CallCount)
	}
}

func TestSessionAggregator_LatencyPercentiles(t *testing.T) {
	aggregator := NewSessionAggregator()

	now := time.Now()
	// Create metrics with known latencies: 100, 200, 300, ..., 1000
	metrics := make([]CallMetrics, 10)
	for i := 0; i < 10; i++ {
		metrics[i] = CallMetrics{
			ModelID:      "openai:gpt-4o-mini",
			LatencyMs:    int64((i + 1) * 100),
			InputTokens:  50,
			OutputTokens: 25,
			Success:      true,
			Timestamp:    now.Add(time.Duration(i) * time.Second),
		}
	}

	result := aggregator.AggregateMetrics(metrics)

	// P50 should be around 500-600 (median of 100-1000)
	if result.P50Latency < 400 || result.P50Latency > 700 {
		t.Errorf("Expected P50Latency around 500-600, got %d", result.P50Latency)
	}

	// P95 should be near the high end
	if result.P95Latency < 900 {
		t.Errorf("Expected P95Latency >= 900, got %d", result.P95Latency)
	}

	// P99 should be at or near max (1000)
	if result.P99Latency < 950 {
		t.Errorf("Expected P99Latency >= 950, got %d", result.P99Latency)
	}

	// Verify ordering: P50 < P95 < P99
	if !(result.P50Latency <= result.P95Latency && result.P95Latency <= result.P99Latency) {
		t.Errorf("Percentiles not in order: P50=%d, P95=%d, P99=%d",
			result.P50Latency, result.P95Latency, result.P99Latency)
	}
}
