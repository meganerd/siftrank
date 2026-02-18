package siftrank

import (
	"math"
	"strings"
	"testing"
)

func testCatalog() *ModelCatalog {
	c := NewModelCatalog()
	c.Add(ModelInfo{
		ID: "cheap-model", Provider: ProviderTypeOpenAI, ContextWindow: 128000,
		Pricing: ModelPricing{InputPricePerToken: perMillion(0.10), OutputPricePerToken: perMillion(0.40)},
	})
	c.Add(ModelInfo{
		ID: "mid-model", Provider: ProviderTypeAnthropic, ContextWindow: 200000,
		Pricing: ModelPricing{InputPricePerToken: perMillion(3.00), OutputPricePerToken: perMillion(15.00)},
	})
	c.Add(ModelInfo{
		ID: "expensive-model", Provider: ProviderTypeGoogle, ContextWindow: 1048576,
		Pricing: ModelPricing{InputPricePerToken: perMillion(10.00), OutputPricePerToken: perMillion(30.00)},
	})
	c.Add(ModelInfo{
		ID: "free-model", Provider: ProviderTypeOllama, ContextWindow: 131072,
		Pricing: ModelPricing{InputPricePerToken: 0, OutputPricePerToken: 0},
	})
	return c
}

func TestRecommend_SortedByCostAscending(t *testing.T) {
	recs := Recommend(100000, 50000, "mid-model", testCatalog(), nil)

	if len(recs) != 4 {
		t.Fatalf("expected 4 recommendations, got %d", len(recs))
	}

	for i := 1; i < len(recs); i++ {
		if recs[i].EstimatedCost < recs[i-1].EstimatedCost {
			t.Errorf("recommendations not sorted: [%d]=%f < [%d]=%f",
				i, recs[i].EstimatedCost, i-1, recs[i-1].EstimatedCost)
		}
	}

	// Free model should be first
	if recs[0].ModelID != "free-model" {
		t.Errorf("expected free-model first, got %q", recs[0].ModelID)
	}
}

func TestRecommend_CostSavings(t *testing.T) {
	recs := Recommend(100000, 50000, "mid-model", testCatalog(), nil)

	// Find the cheap-model recommendation
	var cheapRec *Recommendation
	for i := range recs {
		if recs[i].ModelID == "cheap-model" {
			cheapRec = &recs[i]
			break
		}
	}
	if cheapRec == nil {
		t.Fatal("cheap-model not found in recommendations")
	}

	// Savings should be positive (cheaper than mid-model)
	if cheapRec.CostSavings <= 0 {
		t.Errorf("expected positive savings for cheap-model vs mid-model, got %f", cheapRec.CostSavings)
	}

	// Find the expensive-model recommendation
	var expRec *Recommendation
	for i := range recs {
		if recs[i].ModelID == "expensive-model" {
			expRec = &recs[i]
			break
		}
	}
	if expRec == nil {
		t.Fatal("expensive-model not found in recommendations")
	}

	// Savings should be negative (more expensive than mid-model)
	if expRec.CostSavings >= 0 {
		t.Errorf("expected negative savings for expensive-model vs mid-model, got %f", expRec.CostSavings)
	}
}

func TestRecommend_ProviderExclusion(t *testing.T) {
	opts := &RecommendOptions{
		ExcludeProviders: []ProviderType{ProviderTypeOllama, ProviderTypeGoogle},
	}
	recs := Recommend(100000, 50000, "mid-model", testCatalog(), opts)

	if len(recs) != 2 {
		t.Fatalf("expected 2 recommendations after excluding 2 providers, got %d", len(recs))
	}

	for _, r := range recs {
		if r.Provider == ProviderTypeOllama || r.Provider == ProviderTypeGoogle {
			t.Errorf("excluded provider %q still in results", r.Provider)
		}
	}
}

func TestRecommend_ZeroCostModel(t *testing.T) {
	recs := Recommend(100000, 50000, "mid-model", testCatalog(), nil)

	if recs[0].ModelID != "free-model" {
		t.Fatalf("expected free-model as cheapest, got %q", recs[0].ModelID)
	}
	if recs[0].EstimatedCost != 0 {
		t.Errorf("expected $0 estimated cost for free-model, got %f", recs[0].EstimatedCost)
	}
	if recs[0].CostSavings <= 0 {
		t.Errorf("expected positive savings for free-model, got %f", recs[0].CostSavings)
	}
}

func TestRecommend_UnknownBaseline(t *testing.T) {
	recs := Recommend(100000, 50000, "unknown-model-xyz", testCatalog(), nil)

	if len(recs) != 4 {
		t.Fatalf("expected 4 recommendations, got %d", len(recs))
	}

	// All savings should be zero when baseline is unknown
	for _, r := range recs {
		if r.CostSavings != 0 {
			t.Errorf("expected zero savings for unknown baseline, %q has %f", r.ModelID, r.CostSavings)
		}
		if !strings.Contains(r.Rationale, "not in catalog") {
			t.Errorf("expected 'not in catalog' in rationale for %q, got %q", r.ModelID, r.Rationale)
		}
	}
}

func TestRecommend_NilCatalog(t *testing.T) {
	recs := Recommend(100000, 50000, "some-model", nil, nil)
	if recs != nil {
		t.Errorf("expected nil for nil catalog, got %v", recs)
	}
}

func TestRecommend_EmptyCatalog(t *testing.T) {
	recs := Recommend(100000, 50000, "some-model", NewModelCatalog(), nil)
	if len(recs) != 0 {
		t.Errorf("expected 0 recommendations for empty catalog, got %d", len(recs))
	}
}

func TestRecommend_CostCalculationAccuracy(t *testing.T) {
	c := NewModelCatalog()
	c.Add(ModelInfo{
		ID: "test-model", Provider: ProviderTypeOpenAI, ContextWindow: 128000,
		Pricing: ModelPricing{InputPricePerToken: perMillion(2.00), OutputPricePerToken: perMillion(8.00)},
	})

	recs := Recommend(1000000, 500000, "test-model", c, nil)
	if len(recs) != 1 {
		t.Fatalf("expected 1 recommendation, got %d", len(recs))
	}

	// 1M input * $2/1M + 500K output * $8/1M = $2.00 + $4.00 = $6.00
	expected := 6.0
	if math.Abs(recs[0].EstimatedCost-expected) > 0.0001 {
		t.Errorf("estimated cost = %f, want %f", recs[0].EstimatedCost, expected)
	}
}

func TestRecommend_RationaleContent(t *testing.T) {
	recs := Recommend(100000, 50000, "mid-model", testCatalog(), nil)

	for _, r := range recs {
		if r.Rationale == "" {
			t.Errorf("empty rationale for %q", r.ModelID)
		}
		// All rationales should mention the model name
		if !strings.Contains(r.Rationale, r.ModelID) {
			t.Errorf("rationale for %q doesn't contain model ID: %q", r.ModelID, r.Rationale)
		}
	}
}
