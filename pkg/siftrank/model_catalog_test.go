package siftrank

import (
	"testing"
)

func TestDefaultModelCatalog_HasAllProviders(t *testing.T) {
	c := DefaultModelCatalog()
	providers := []ProviderType{
		ProviderTypeOpenAI,
		ProviderTypeAnthropic,
		ProviderTypeGoogle,
		ProviderTypeOllama,
		ProviderTypeOpenRouter,
	}
	for _, pt := range providers {
		models := c.ByProvider(pt)
		if len(models) == 0 {
			t.Errorf("no models found for provider %q", pt)
		}
	}
}

func TestDefaultModelCatalog_TotalModelCount(t *testing.T) {
	c := DefaultModelCatalog()
	all := c.All()
	if len(all) != 17 {
		t.Errorf("expected 17 models in default catalog, got %d", len(all))
	}
}

func TestModelCatalog_Get(t *testing.T) {
	c := DefaultModelCatalog()

	info, ok := c.Get("gpt-4o-mini")
	if !ok {
		t.Fatal("expected to find gpt-4o-mini")
	}
	if info.Provider != ProviderTypeOpenAI {
		t.Errorf("Provider = %q, want %q", info.Provider, ProviderTypeOpenAI)
	}
	if info.ContextWindow != 128000 {
		t.Errorf("ContextWindow = %d, want 128000", info.ContextWindow)
	}
	if info.Pricing.InputPricePerToken != perMillion(0.15) {
		t.Errorf("InputPricePerToken = %v, want %v", info.Pricing.InputPricePerToken, perMillion(0.15))
	}
}

func TestModelCatalog_GetUnknown(t *testing.T) {
	c := DefaultModelCatalog()
	_, ok := c.Get("nonexistent-model")
	if ok {
		t.Error("expected ok=false for unknown model")
	}
}

func TestModelCatalog_ByProvider(t *testing.T) {
	c := DefaultModelCatalog()

	openai := c.ByProvider(ProviderTypeOpenAI)
	if len(openai) != 6 {
		t.Errorf("OpenAI models = %d, want 6", len(openai))
	}
	for _, m := range openai {
		if m.Provider != ProviderTypeOpenAI {
			t.Errorf("model %q has provider %q, want %q", m.ID, m.Provider, ProviderTypeOpenAI)
		}
	}

	anthropic := c.ByProvider(ProviderTypeAnthropic)
	if len(anthropic) != 3 {
		t.Errorf("Anthropic models = %d, want 3", len(anthropic))
	}

	ollama := c.ByProvider(ProviderTypeOllama)
	if len(ollama) != 3 {
		t.Errorf("Ollama models = %d, want 3", len(ollama))
	}
}

func TestModelCatalog_ByProviderEmpty(t *testing.T) {
	c := NewModelCatalog()
	result := c.ByProvider(ProviderTypeOpenAI)
	if len(result) != 0 {
		t.Errorf("expected empty result for empty catalog, got %d", len(result))
	}
}

func TestModelCatalog_ContextWindows(t *testing.T) {
	c := DefaultModelCatalog()

	tests := []struct {
		model  string
		window int
	}{
		{"gpt-4o", 128000},
		{"gpt-4.1", 1047576},
		{"claude-sonnet-4-5-20250929", 200000},
		{"gemini-2.5-pro", 1048576},
		{"llama3.1:8b", 131072},
		{"qwen2.5-coder:32b", 32768},
	}

	for _, tc := range tests {
		info, ok := c.Get(tc.model)
		if !ok {
			t.Errorf("model %q not found", tc.model)
			continue
		}
		if info.ContextWindow != tc.window {
			t.Errorf("%s ContextWindow = %d, want %d", tc.model, info.ContextWindow, tc.window)
		}
	}
}

func TestModelCatalog_PricingRegistryConsistency(t *testing.T) {
	c := DefaultModelCatalog()
	r := c.PricingRegistry()

	// Every model in catalog should be in the derived pricing registry
	for _, info := range c.All() {
		pricing, ok := r.Get(info.ID)
		if !ok {
			t.Errorf("model %q in catalog but not in derived PricingRegistry", info.ID)
			continue
		}
		if pricing.InputPricePerToken != info.Pricing.InputPricePerToken {
			t.Errorf("%s input pricing mismatch: registry=%v catalog=%v",
				info.ID, pricing.InputPricePerToken, info.Pricing.InputPricePerToken)
		}
		if pricing.OutputPricePerToken != info.Pricing.OutputPricePerToken {
			t.Errorf("%s output pricing mismatch: registry=%v catalog=%v",
				info.ID, pricing.OutputPricePerToken, info.Pricing.OutputPricePerToken)
		}
	}

	// Model counts should match
	if len(r.Models()) != len(c.All()) {
		t.Errorf("PricingRegistry has %d models, catalog has %d", len(r.Models()), len(c.All()))
	}
}

func TestModelCatalog_OllamaZeroCost(t *testing.T) {
	c := DefaultModelCatalog()
	for _, m := range c.ByProvider(ProviderTypeOllama) {
		if m.Pricing.InputPricePerToken != 0 || m.Pricing.OutputPricePerToken != 0 {
			t.Errorf("Ollama model %q should have zero cost, got input=%v output=%v",
				m.ID, m.Pricing.InputPricePerToken, m.Pricing.OutputPricePerToken)
		}
	}
}
