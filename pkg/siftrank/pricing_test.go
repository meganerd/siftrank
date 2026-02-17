package siftrank

import (
	"math"
	"testing"
)

func TestModelPricing_Fields(t *testing.T) {
	p := ModelPricing{
		InputPricePerToken:  perMillion(0.15),
		OutputPricePerToken: perMillion(0.60),
	}

	if p.InputPricePerToken <= 0 {
		t.Error("Expected positive InputPricePerToken")
	}
	if p.OutputPricePerToken <= 0 {
		t.Error("Expected positive OutputPricePerToken")
	}
	if p.OutputPricePerToken <= p.InputPricePerToken {
		t.Error("Expected output price > input price for typical models")
	}
}

func TestCalculateCost_Basic(t *testing.T) {
	pricing := ModelPricing{
		InputPricePerToken:  perMillion(0.15),  // $0.15/1M tokens
		OutputPricePerToken: perMillion(0.60),   // $0.60/1M tokens
	}

	usage := Usage{
		InputTokens:  1_000_000,
		OutputTokens: 1_000_000,
	}

	cost := CalculateCost(usage, pricing)

	// 1M input tokens at $0.15/1M = $0.15
	// 1M output tokens at $0.60/1M = $0.60
	// Total = $0.75
	expected := 0.75
	if math.Abs(cost-expected) > 0.0001 {
		t.Errorf("Expected cost $%.4f, got $%.4f", expected, cost)
	}
}

func TestCalculateCost_ZeroUsage(t *testing.T) {
	pricing := ModelPricing{
		InputPricePerToken:  perMillion(2.50),
		OutputPricePerToken: perMillion(10.00),
	}

	usage := Usage{}
	cost := CalculateCost(usage, pricing)

	if cost != 0 {
		t.Errorf("Expected zero cost for zero usage, got $%.6f", cost)
	}
}

func TestCalculateCost_ZeroPricing(t *testing.T) {
	pricing := ModelPricing{
		InputPricePerToken:  0,
		OutputPricePerToken: 0,
	}

	usage := Usage{
		InputTokens:  500_000,
		OutputTokens: 100_000,
	}

	cost := CalculateCost(usage, pricing)
	if cost != 0 {
		t.Errorf("Expected zero cost for free model, got $%.6f", cost)
	}
}

func TestCalculateCost_RealisticGPT4oMini(t *testing.T) {
	pricing := ModelPricing{
		InputPricePerToken:  perMillion(0.15),
		OutputPricePerToken: perMillion(0.60),
	}

	// Typical siftrank run: ~50k input, ~10k output
	usage := Usage{
		InputTokens:  50_000,
		OutputTokens: 10_000,
	}

	cost := CalculateCost(usage, pricing)

	// 50k * $0.15/1M = $0.0075
	// 10k * $0.60/1M = $0.006
	// Total = $0.0135
	expected := 0.0135
	if math.Abs(cost-expected) > 0.0001 {
		t.Errorf("Expected cost ~$%.4f, got $%.4f", expected, cost)
	}
}

func TestPricingRegistry_SetAndGet(t *testing.T) {
	r := NewPricingRegistry()

	pricing := ModelPricing{
		InputPricePerToken:  perMillion(1.00),
		OutputPricePerToken: perMillion(3.00),
	}

	r.Set("test-model", pricing)

	got, ok := r.Get("test-model")
	if !ok {
		t.Fatal("Expected to find test-model in registry")
	}
	if got.InputPricePerToken != pricing.InputPricePerToken {
		t.Errorf("InputPricePerToken mismatch: want %v, got %v",
			pricing.InputPricePerToken, got.InputPricePerToken)
	}
	if got.OutputPricePerToken != pricing.OutputPricePerToken {
		t.Errorf("OutputPricePerToken mismatch: want %v, got %v",
			pricing.OutputPricePerToken, got.OutputPricePerToken)
	}
}

func TestPricingRegistry_GetUnknown(t *testing.T) {
	r := NewPricingRegistry()

	_, ok := r.Get("nonexistent-model")
	if ok {
		t.Error("Expected false for unknown model")
	}
}

func TestPricingRegistry_Overwrite(t *testing.T) {
	r := NewPricingRegistry()

	r.Set("model-a", ModelPricing{
		InputPricePerToken:  perMillion(1.00),
		OutputPricePerToken: perMillion(3.00),
	})

	// Overwrite with new pricing
	r.Set("model-a", ModelPricing{
		InputPricePerToken:  perMillion(0.50),
		OutputPricePerToken: perMillion(1.50),
	})

	got, ok := r.Get("model-a")
	if !ok {
		t.Fatal("Expected to find model-a after overwrite")
	}
	if math.Abs(got.InputPricePerToken-perMillion(0.50)) > 1e-15 {
		t.Errorf("Expected updated input price, got %v", got.InputPricePerToken)
	}
}

func TestPricingRegistry_Models(t *testing.T) {
	r := NewPricingRegistry()
	r.Set("a", ModelPricing{})
	r.Set("b", ModelPricing{})
	r.Set("c", ModelPricing{})

	models := r.Models()
	if len(models) != 3 {
		t.Errorf("Expected 3 models, got %d", len(models))
	}
}

func TestDefaultPricingRegistry_HasOpenAIModels(t *testing.T) {
	r := DefaultPricingRegistry()

	for _, model := range []string{"gpt-4o", "gpt-4o-mini"} {
		if _, ok := r.Get(model); !ok {
			t.Errorf("Default registry missing OpenAI model: %s", model)
		}
	}
}

func TestDefaultPricingRegistry_HasAnthropicModels(t *testing.T) {
	r := DefaultPricingRegistry()

	for _, model := range []string{"claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20241022"} {
		if _, ok := r.Get(model); !ok {
			t.Errorf("Default registry missing Anthropic model: %s", model)
		}
	}
}

func TestDefaultPricingRegistry_HasGoogleModels(t *testing.T) {
	r := DefaultPricingRegistry()

	for _, model := range []string{"gemini-2.5-flash", "gemini-2.5-pro"} {
		if _, ok := r.Get(model); !ok {
			t.Errorf("Default registry missing Google model: %s", model)
		}
	}
}

func TestDefaultPricingRegistry_HasOllamaModels(t *testing.T) {
	r := DefaultPricingRegistry()

	p, ok := r.Get("llama3.1:8b")
	if !ok {
		t.Fatal("Default registry missing Ollama model: llama3.1:8b")
	}
	if p.InputPricePerToken != 0 || p.OutputPricePerToken != 0 {
		t.Error("Ollama models should have zero pricing")
	}
}

func TestDefaultPricingRegistry_HasOpenRouterModels(t *testing.T) {
	r := DefaultPricingRegistry()

	if _, ok := r.Get("openai/gpt-4o-mini"); !ok {
		t.Error("Default registry missing OpenRouter model: openai/gpt-4o-mini")
	}
}

func TestDefaultPricingRegistry_TotalModelCount(t *testing.T) {
	r := DefaultPricingRegistry()

	models := r.Models()
	if len(models) < 10 {
		t.Errorf("Expected at least 10 models in default registry, got %d", len(models))
	}
}

func TestPerMillion_Conversion(t *testing.T) {
	// $1.00 per million tokens = $0.000001 per token
	result := perMillion(1.00)
	expected := 0.000001
	if math.Abs(result-expected) > 1e-15 {
		t.Errorf("perMillion(1.00) = %v, want %v", result, expected)
	}
}
