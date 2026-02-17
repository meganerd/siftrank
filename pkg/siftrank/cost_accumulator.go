package siftrank

import "sync"

// CostSummary contains the accumulated cost and usage for a model.
type CostSummary struct {
	TotalCost  float64 // Total cost in USD
	TotalUsage Usage   // Accumulated token usage
	Model      string  // Model identifier
}

// CostAccumulator tracks token usage and cost across multiple API calls.
// It is safe for concurrent use by multiple goroutines.
type CostAccumulator struct {
	mu         sync.Mutex
	totalCost  float64
	totalUsage Usage
	pricing    ModelPricing
	model      string
}

// NewCostAccumulator creates a CostAccumulator for the given model.
// It looks up pricing from the registry. If the model is not found,
// the accumulator still tracks usage but reports zero cost.
func NewCostAccumulator(model string, registry *PricingRegistry) *CostAccumulator {
	var pricing ModelPricing
	if registry != nil {
		pricing, _ = registry.Get(model)
	}
	return &CostAccumulator{
		pricing: pricing,
		model:   model,
	}
}

// Add records token usage from a single API call and accumulates cost.
func (c *CostAccumulator) Add(usage Usage) {
	cost := CalculateCost(usage, c.pricing)

	c.mu.Lock()
	c.totalUsage.Add(usage)
	c.totalCost += cost
	c.mu.Unlock()
}

// Summary returns a snapshot of the accumulated cost and usage.
func (c *CostAccumulator) Summary() CostSummary {
	c.mu.Lock()
	summary := CostSummary{
		TotalCost:  c.totalCost,
		TotalUsage: c.totalUsage,
		Model:      c.model,
	}
	c.mu.Unlock()
	return summary
}
