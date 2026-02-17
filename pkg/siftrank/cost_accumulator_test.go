package siftrank

import (
	"math"
	"sync"
	"testing"
)

func TestCostAccumulator_KnownModel(t *testing.T) {
	registry := DefaultPricingRegistry()
	acc := NewCostAccumulator("gpt-4o-mini", registry)

	acc.Add(Usage{InputTokens: 1000, OutputTokens: 500})

	summary := acc.Summary()
	if summary.Model != "gpt-4o-mini" {
		t.Errorf("Expected model gpt-4o-mini, got %s", summary.Model)
	}
	if summary.TotalUsage.InputTokens != 1000 {
		t.Errorf("Expected 1000 input tokens, got %d", summary.TotalUsage.InputTokens)
	}
	if summary.TotalUsage.OutputTokens != 500 {
		t.Errorf("Expected 500 output tokens, got %d", summary.TotalUsage.OutputTokens)
	}

	// gpt-4o-mini: $0.15/1M input, $0.60/1M output
	// 1000 * 0.00000015 = 0.00015
	// 500 * 0.0000006 = 0.0003
	// Total = 0.00045
	expected := 0.00045
	if math.Abs(summary.TotalCost-expected) > 1e-10 {
		t.Errorf("Expected cost $%.10f, got $%.10f", expected, summary.TotalCost)
	}
}

func TestCostAccumulator_UnknownModel(t *testing.T) {
	registry := DefaultPricingRegistry()
	acc := NewCostAccumulator("unknown-model-xyz", registry)

	acc.Add(Usage{InputTokens: 10000, OutputTokens: 5000})

	summary := acc.Summary()
	if summary.TotalCost != 0 {
		t.Errorf("Expected zero cost for unknown model, got $%.6f", summary.TotalCost)
	}
	// Usage should still be tracked
	if summary.TotalUsage.InputTokens != 10000 {
		t.Errorf("Expected 10000 input tokens, got %d", summary.TotalUsage.InputTokens)
	}
}

func TestCostAccumulator_NilRegistry(t *testing.T) {
	acc := NewCostAccumulator("gpt-4o", nil)

	acc.Add(Usage{InputTokens: 1000, OutputTokens: 500})

	summary := acc.Summary()
	if summary.TotalCost != 0 {
		t.Errorf("Expected zero cost with nil registry, got $%.6f", summary.TotalCost)
	}
	if summary.TotalUsage.InputTokens != 1000 {
		t.Errorf("Expected 1000 input tokens, got %d", summary.TotalUsage.InputTokens)
	}
}

func TestCostAccumulator_ZeroUsage(t *testing.T) {
	registry := DefaultPricingRegistry()
	acc := NewCostAccumulator("gpt-4o", registry)

	summary := acc.Summary()
	if summary.TotalCost != 0 {
		t.Errorf("Expected zero cost before any adds, got $%.6f", summary.TotalCost)
	}
	if summary.TotalUsage.InputTokens != 0 || summary.TotalUsage.OutputTokens != 0 {
		t.Error("Expected zero usage before any adds")
	}
}

func TestCostAccumulator_MultipleAdds(t *testing.T) {
	registry := DefaultPricingRegistry()
	acc := NewCostAccumulator("gpt-4o-mini", registry)

	// Simulate 5 API calls
	for i := 0; i < 5; i++ {
		acc.Add(Usage{InputTokens: 10000, OutputTokens: 2000})
	}

	summary := acc.Summary()

	if summary.TotalUsage.InputTokens != 50000 {
		t.Errorf("Expected 50000 input tokens, got %d", summary.TotalUsage.InputTokens)
	}
	if summary.TotalUsage.OutputTokens != 10000 {
		t.Errorf("Expected 10000 output tokens, got %d", summary.TotalUsage.OutputTokens)
	}

	// 50000 * $0.15/1M = $0.0075
	// 10000 * $0.60/1M = $0.006
	// Total = $0.0135
	expected := 0.0135
	if math.Abs(summary.TotalCost-expected) > 1e-10 {
		t.Errorf("Expected cost $%.10f, got $%.10f", expected, summary.TotalCost)
	}
}

func TestCostAccumulator_ReasoningTokens(t *testing.T) {
	registry := DefaultPricingRegistry()
	acc := NewCostAccumulator("gpt-4o-mini", registry)

	acc.Add(Usage{
		InputTokens:     1000,
		OutputTokens:    500,
		ReasoningTokens: 200,
	})

	summary := acc.Summary()
	if summary.TotalUsage.ReasoningTokens != 200 {
		t.Errorf("Expected 200 reasoning tokens, got %d", summary.TotalUsage.ReasoningTokens)
	}
	// Reasoning tokens are tracked but not priced (no standard pricing)
	// Cost should be based on input + output only
	expected := float64(1000)*perMillion(0.15) + float64(500)*perMillion(0.60)
	if math.Abs(summary.TotalCost-expected) > 1e-10 {
		t.Errorf("Expected cost $%.10f, got $%.10f", expected, summary.TotalCost)
	}
}

func TestCostAccumulator_Concurrent(t *testing.T) {
	registry := DefaultPricingRegistry()
	acc := NewCostAccumulator("gpt-4o-mini", registry)

	numGoroutines := 100
	usagePerCall := Usage{InputTokens: 1000, OutputTokens: 500}

	var wg sync.WaitGroup
	wg.Add(numGoroutines)
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			acc.Add(usagePerCall)
		}()
	}
	wg.Wait()

	summary := acc.Summary()

	expectedInput := 1000 * numGoroutines
	expectedOutput := 500 * numGoroutines
	if summary.TotalUsage.InputTokens != expectedInput {
		t.Errorf("Expected %d input tokens, got %d", expectedInput, summary.TotalUsage.InputTokens)
	}
	if summary.TotalUsage.OutputTokens != expectedOutput {
		t.Errorf("Expected %d output tokens, got %d", expectedOutput, summary.TotalUsage.OutputTokens)
	}

	// 100 * (1000 * $0.15/1M + 500 * $0.60/1M) = 100 * $0.00045 = $0.045
	expectedCost := 0.045
	if math.Abs(summary.TotalCost-expectedCost) > 1e-10 {
		t.Errorf("Expected cost $%.10f, got $%.10f", expectedCost, summary.TotalCost)
	}
}

func TestCostAccumulator_OllamaZeroCost(t *testing.T) {
	registry := DefaultPricingRegistry()
	acc := NewCostAccumulator("llama3.1:8b", registry)

	acc.Add(Usage{InputTokens: 100000, OutputTokens: 50000})

	summary := acc.Summary()
	if summary.TotalCost != 0 {
		t.Errorf("Expected zero cost for Ollama model, got $%.6f", summary.TotalCost)
	}
	if summary.TotalUsage.InputTokens != 100000 {
		t.Errorf("Expected 100000 input tokens tracked, got %d", summary.TotalUsage.InputTokens)
	}
}

func TestCostAccumulator_SummaryIsSnapshot(t *testing.T) {
	registry := DefaultPricingRegistry()
	acc := NewCostAccumulator("gpt-4o-mini", registry)

	acc.Add(Usage{InputTokens: 1000, OutputTokens: 500})
	snap1 := acc.Summary()

	acc.Add(Usage{InputTokens: 2000, OutputTokens: 1000})
	snap2 := acc.Summary()

	// snap1 should not have changed
	if snap1.TotalUsage.InputTokens != 1000 {
		t.Errorf("Snapshot 1 should still show 1000 input tokens, got %d", snap1.TotalUsage.InputTokens)
	}
	if snap2.TotalUsage.InputTokens != 3000 {
		t.Errorf("Snapshot 2 should show 3000 input tokens, got %d", snap2.TotalUsage.InputTokens)
	}
}
