package siftrank

// ModelInfo describes a model's capabilities and pricing.
type ModelInfo struct {
	ID            string       // Model identifier (e.g., "gpt-4o-mini")
	Provider      ProviderType // Provider that serves this model
	ContextWindow int          // Maximum context window in tokens
	Pricing       ModelPricing // Per-token pricing
}

// ModelCatalog is a registry of known models and their metadata.
type ModelCatalog struct {
	models map[string]ModelInfo
}

// NewModelCatalog creates an empty ModelCatalog.
func NewModelCatalog() *ModelCatalog {
	return &ModelCatalog{
		models: make(map[string]ModelInfo),
	}
}

// Add registers a model in the catalog.
func (c *ModelCatalog) Add(info ModelInfo) {
	c.models[info.ID] = info
}

// Get returns metadata for a model. Returns ok=false if not found.
func (c *ModelCatalog) Get(id string) (ModelInfo, bool) {
	info, ok := c.models[id]
	return info, ok
}

// ByProvider returns all models for a given provider type.
func (c *ModelCatalog) ByProvider(pt ProviderType) []ModelInfo {
	var result []ModelInfo
	for _, info := range c.models {
		if info.Provider == pt {
			result = append(result, info)
		}
	}
	return result
}

// All returns all models in the catalog.
func (c *ModelCatalog) All() []ModelInfo {
	result := make([]ModelInfo, 0, len(c.models))
	for _, info := range c.models {
		result = append(result, info)
	}
	return result
}

// PricingRegistry derives a PricingRegistry from the catalog's pricing data.
func (c *ModelCatalog) PricingRegistry() *PricingRegistry {
	r := NewPricingRegistry()
	for id, info := range c.models {
		r.Set(id, info.Pricing)
	}
	return r
}

// DefaultModelCatalog returns a catalog pre-populated with known models
// across all supported providers. This is the single source of truth
// for model metadata including pricing.
//
// Prices sourced from provider pricing pages (as of Feb 2026).
// Ollama models are local inference and have zero cost.
func DefaultModelCatalog() *ModelCatalog {
	c := NewModelCatalog()

	// --- OpenAI ---
	c.Add(ModelInfo{
		ID: "gpt-4o", Provider: ProviderTypeOpenAI, ContextWindow: 128000,
		Pricing: ModelPricing{InputPricePerToken: perMillion(2.50), OutputPricePerToken: perMillion(10.00)},
	})
	c.Add(ModelInfo{
		ID: "gpt-4o-mini", Provider: ProviderTypeOpenAI, ContextWindow: 128000,
		Pricing: ModelPricing{InputPricePerToken: perMillion(0.15), OutputPricePerToken: perMillion(0.60)},
	})
	c.Add(ModelInfo{
		ID: "gpt-4.1", Provider: ProviderTypeOpenAI, ContextWindow: 1047576,
		Pricing: ModelPricing{InputPricePerToken: perMillion(2.00), OutputPricePerToken: perMillion(8.00)},
	})
	c.Add(ModelInfo{
		ID: "gpt-4.1-mini", Provider: ProviderTypeOpenAI, ContextWindow: 1047576,
		Pricing: ModelPricing{InputPricePerToken: perMillion(0.40), OutputPricePerToken: perMillion(1.60)},
	})
	c.Add(ModelInfo{
		ID: "gpt-4.1-nano", Provider: ProviderTypeOpenAI, ContextWindow: 1047576,
		Pricing: ModelPricing{InputPricePerToken: perMillion(0.10), OutputPricePerToken: perMillion(0.40)},
	})
	c.Add(ModelInfo{
		ID: "o3-mini", Provider: ProviderTypeOpenAI, ContextWindow: 200000,
		Pricing: ModelPricing{InputPricePerToken: perMillion(1.10), OutputPricePerToken: perMillion(4.40)},
	})

	// --- Anthropic ---
	c.Add(ModelInfo{
		ID: "claude-sonnet-4-5-20250929", Provider: ProviderTypeAnthropic, ContextWindow: 200000,
		Pricing: ModelPricing{InputPricePerToken: perMillion(3.00), OutputPricePerToken: perMillion(15.00)},
	})
	c.Add(ModelInfo{
		ID: "claude-haiku-3-5-20241022", Provider: ProviderTypeAnthropic, ContextWindow: 200000,
		Pricing: ModelPricing{InputPricePerToken: perMillion(0.80), OutputPricePerToken: perMillion(4.00)},
	})
	c.Add(ModelInfo{
		ID: "claude-3-5-sonnet-20241022", Provider: ProviderTypeAnthropic, ContextWindow: 200000,
		Pricing: ModelPricing{InputPricePerToken: perMillion(3.00), OutputPricePerToken: perMillion(15.00)},
	})

	// --- Google ---
	c.Add(ModelInfo{
		ID: "gemini-2.5-flash", Provider: ProviderTypeGoogle, ContextWindow: 1048576,
		Pricing: ModelPricing{InputPricePerToken: perMillion(0.15), OutputPricePerToken: perMillion(0.60)},
	})
	c.Add(ModelInfo{
		ID: "gemini-2.5-pro", Provider: ProviderTypeGoogle, ContextWindow: 1048576,
		Pricing: ModelPricing{InputPricePerToken: perMillion(1.25), OutputPricePerToken: perMillion(10.00)},
	})
	c.Add(ModelInfo{
		ID: "gemini-2.0-flash", Provider: ProviderTypeGoogle, ContextWindow: 1048576,
		Pricing: ModelPricing{InputPricePerToken: perMillion(0.10), OutputPricePerToken: perMillion(0.40)},
	})

	// --- Ollama (local inference, zero cost) ---
	c.Add(ModelInfo{
		ID: "llama3.1:8b", Provider: ProviderTypeOllama, ContextWindow: 131072,
		Pricing: ModelPricing{InputPricePerToken: 0, OutputPricePerToken: 0},
	})
	c.Add(ModelInfo{
		ID: "llama3.1:70b", Provider: ProviderTypeOllama, ContextWindow: 131072,
		Pricing: ModelPricing{InputPricePerToken: 0, OutputPricePerToken: 0},
	})
	c.Add(ModelInfo{
		ID: "qwen2.5-coder:32b", Provider: ProviderTypeOllama, ContextWindow: 32768,
		Pricing: ModelPricing{InputPricePerToken: 0, OutputPricePerToken: 0},
	})

	// --- OpenRouter (uses underlying model pricing, common models) ---
	c.Add(ModelInfo{
		ID: "openai/gpt-4o-mini", Provider: ProviderTypeOpenRouter, ContextWindow: 128000,
		Pricing: ModelPricing{InputPricePerToken: perMillion(0.15), OutputPricePerToken: perMillion(0.60)},
	})
	c.Add(ModelInfo{
		ID: "anthropic/claude-sonnet-4-5-20250929", Provider: ProviderTypeOpenRouter, ContextWindow: 200000,
		Pricing: ModelPricing{InputPricePerToken: perMillion(3.00), OutputPricePerToken: perMillion(15.00)},
	})

	return c
}
