package siftrank

import (
	"fmt"
	"sort"
)

// Recommendation represents a model suggestion with cost analysis.
type Recommendation struct {
	ModelID       string       // Recommended model identifier
	Provider      ProviderType // Provider serving this model
	EstimatedCost float64      // What this model would cost for the same workload
	CostSavings   float64      // Savings vs baseline (positive = cheaper, negative = more expensive)
	Rationale     string       // Human-readable explanation
}

// RecommendOptions configures the recommendation engine.
type RecommendOptions struct {
	ExcludeProviders []ProviderType // Providers to exclude from recommendations
}

// Recommend scores all catalog models against a workload defined by input/output
// token counts, comparing each to the baseline model that was actually used.
// Returns recommendations sorted by estimated cost (cheapest first).
//
// If the baseline model is not in the catalog, savings are reported as zero
// and the rationale notes that the baseline cost is unknown.
func Recommend(inputTokens, outputTokens int, baselineModel string, catalog *ModelCatalog, opts *RecommendOptions) []Recommendation {
	if catalog == nil {
		return nil
	}

	// Compute baseline cost
	var baselineCost float64
	var baselineKnown bool
	if info, ok := catalog.Get(baselineModel); ok {
		baselineCost = estimateCost(inputTokens, outputTokens, info.Pricing)
		baselineKnown = true
	}

	// Build exclusion set
	excluded := make(map[ProviderType]bool)
	if opts != nil {
		for _, p := range opts.ExcludeProviders {
			excluded[p] = true
		}
	}

	var recs []Recommendation
	for _, info := range catalog.All() {
		if excluded[info.Provider] {
			continue
		}

		cost := estimateCost(inputTokens, outputTokens, info.Pricing)

		var savings float64
		var rationale string
		if baselineKnown {
			savings = baselineCost - cost
			rationale = buildRationale(info, cost, baselineModel, baselineCost, savings)
		} else {
			rationale = fmt.Sprintf("%s (%s): estimated $%.6f for this workload (baseline %q not in catalog)",
				info.ID, info.Provider, cost, baselineModel)
		}

		recs = append(recs, Recommendation{
			ModelID:       info.ID,
			Provider:      info.Provider,
			EstimatedCost: cost,
			CostSavings:   savings,
			Rationale:     rationale,
		})
	}

	sort.Slice(recs, func(i, j int) bool {
		return recs[i].EstimatedCost < recs[j].EstimatedCost
	})

	return recs
}

func estimateCost(inputTokens, outputTokens int, pricing ModelPricing) float64 {
	return float64(inputTokens)*pricing.InputPricePerToken +
		float64(outputTokens)*pricing.OutputPricePerToken
}

func buildRationale(info ModelInfo, cost float64, baselineModel string, baselineCost, savings float64) string {
	if savings > 0 {
		pct := (savings / baselineCost) * 100
		return fmt.Sprintf("%s (%s): $%.6f — saves $%.6f (%.0f%% cheaper than %s)",
			info.ID, info.Provider, cost, savings, pct, baselineModel)
	} else if savings < 0 {
		pct := (-savings / baselineCost) * 100
		return fmt.Sprintf("%s (%s): $%.6f — costs $%.6f more (%.0f%% more than %s)",
			info.ID, info.Provider, cost, -savings, pct, baselineModel)
	}
	return fmt.Sprintf("%s (%s): $%.6f — same cost as %s",
		info.ID, info.Provider, cost, baselineModel)
}
