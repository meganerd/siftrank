package eval

import (
	"sort"
)

// ModelStats contains aggregated performance statistics for a single model
type ModelStats struct {
	ModelID     string  // Format: "provider:model"
	CallCount   int     // Total number of calls
	SuccessRate float64 // Ratio of successful calls (0.0-1.0)
	ErrorCount  int     // Number of failed calls

	// Latency statistics (milliseconds)
	AvgLatency int64 // Mean latency
	P50Latency int64 // Median latency
	P95Latency int64 // 95th percentile latency
	P99Latency int64 // 99th percentile latency

	// Token consumption
	TotalTokens int // Sum of all input + output tokens
}

// SessionAggregator aggregates CallMetrics into ModelStats
type SessionAggregator struct {
	// Currently stateless, but struct allows for future extensions
	// (e.g., session-level caching, custom percentile algorithms)
}

// NewSessionAggregator creates a new SessionAggregator
func NewSessionAggregator() *SessionAggregator {
	return &SessionAggregator{}
}

// AggregateMetrics aggregates a slice of CallMetrics for a single model into ModelStats
func (sa *SessionAggregator) AggregateMetrics(metrics []CallMetrics) ModelStats {
	if len(metrics) == 0 {
		return ModelStats{}
	}

	// Assume all metrics are for the same model (caller's responsibility to filter)
	modelID := metrics[0].ModelID

	var (
		callCount    = len(metrics)
		successCount = 0
		errorCount   = 0
		totalLatency int64
		totalTokens  = 0
		latencies    = make([]int64, 0, len(metrics))
	)

	for _, m := range metrics {
		if m.Success {
			successCount++
		} else {
			errorCount++
		}

		totalLatency += m.LatencyMs
		latencies = append(latencies, m.LatencyMs)

		// Handle both InputTokens/OutputTokens and PromptTokens naming
		inputTokens := m.InputTokens
		if inputTokens == 0 && m.PromptTokens > 0 {
			inputTokens = m.PromptTokens
		}
		totalTokens += inputTokens + m.OutputTokens
	}

	successRate := float64(successCount) / float64(callCount)
	avgLatency := totalLatency / int64(callCount)

	return ModelStats{
		ModelID:     modelID,
		CallCount:   callCount,
		SuccessRate: successRate,
		ErrorCount:  errorCount,
		AvgLatency:  avgLatency,
		P50Latency:  percentile(latencies, 50),
		P95Latency:  percentile(latencies, 95),
		P99Latency:  percentile(latencies, 99),
		TotalTokens: totalTokens,
	}
}

// AggregateByModel aggregates metrics grouped by ModelID
func (sa *SessionAggregator) AggregateByModel(metrics []CallMetrics) []ModelStats {
	// Group metrics by ModelID
	grouped := make(map[string][]CallMetrics)
	for _, m := range metrics {
		grouped[m.ModelID] = append(grouped[m.ModelID], m)
	}

	// Aggregate each group
	results := make([]ModelStats, 0, len(grouped))
	for modelID, modelMetrics := range grouped {
		stats := sa.AggregateMetrics(modelMetrics)
		// Ensure ModelID is set even if AggregateMetrics doesn't set it
		stats.ModelID = modelID
		results = append(results, stats)
	}

	// Sort by ModelID for deterministic output
	sort.Slice(results, func(i, j int) bool {
		return results[i].ModelID < results[j].ModelID
	})

	return results
}

// percentile calculates the p-th percentile of a slice of int64 values
// p should be in the range [0, 100]
// Panics if values is empty
func percentile(values []int64, p int) int64 {
	if len(values) == 0 {
		panic("percentile: empty slice")
	}

	// Make a copy to avoid mutating the input
	sorted := make([]int64, len(values))
	copy(sorted, values)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i] < sorted[j]
	})

	if len(sorted) == 1 {
		return sorted[0]
	}

	// Linear interpolation method (R-7, Excel PERCENTILE.INC)
	// rank = p/100 * (N-1) + 1 (1-indexed)
	// Convert to 0-indexed: rank = p/100 * (N-1)
	rank := float64(p) / 100.0 * float64(len(sorted)-1)
	lower := int(rank)
	upper := lower + 1

	if upper >= len(sorted) {
		return sorted[len(sorted)-1]
	}

	// Linear interpolation between sorted[lower] and sorted[upper]
	fraction := rank - float64(lower)
	result := float64(sorted[lower]) + fraction*float64(sorted[upper]-sorted[lower])

	return int64(result)
}
