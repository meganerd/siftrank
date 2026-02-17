package siftrank

// ModelPricing holds per-token pricing for a specific model.
// Prices are in USD per token (e.g., $0.15/1M input tokens = 0.00000015 per token).
type ModelPricing struct {
	InputPricePerToken  float64 // USD per input token
	OutputPricePerToken float64 // USD per output token
}

// CalculateCost computes the total cost in USD from token usage and pricing.
func CalculateCost(usage Usage, pricing ModelPricing) float64 {
	inputCost := float64(usage.InputTokens) * pricing.InputPricePerToken
	outputCost := float64(usage.OutputTokens) * pricing.OutputPricePerToken
	return inputCost + outputCost
}

// PricingRegistry maps model identifiers to their pricing data.
type PricingRegistry struct {
	models map[string]ModelPricing
}

// NewPricingRegistry creates an empty PricingRegistry.
func NewPricingRegistry() *PricingRegistry {
	return &PricingRegistry{
		models: make(map[string]ModelPricing),
	}
}

// Set adds or updates pricing for a model.
func (r *PricingRegistry) Set(model string, pricing ModelPricing) {
	r.models[model] = pricing
}

// Get returns pricing for a model. Returns zero pricing and false if not found.
func (r *PricingRegistry) Get(model string) (ModelPricing, bool) {
	p, ok := r.models[model]
	return p, ok
}

// Models returns all registered model names.
func (r *PricingRegistry) Models() []string {
	names := make([]string, 0, len(r.models))
	for name := range r.models {
		names = append(names, name)
	}
	return names
}

// helper to convert $/1M tokens to $/token
func perMillion(price float64) float64 {
	return price / 1_000_000
}

// DefaultPricingRegistry returns a registry pre-populated with current pricing
// for popular models across all supported providers.
//
// Prices sourced from provider pricing pages (as of Feb 2026).
// Ollama models are local inference and have zero cost.
func DefaultPricingRegistry() *PricingRegistry {
	r := NewPricingRegistry()

	// --- OpenAI ---
	r.Set("gpt-4o", ModelPricing{
		InputPricePerToken:  perMillion(2.50),
		OutputPricePerToken: perMillion(10.00),
	})
	r.Set("gpt-4o-mini", ModelPricing{
		InputPricePerToken:  perMillion(0.15),
		OutputPricePerToken: perMillion(0.60),
	})
	r.Set("gpt-4.1", ModelPricing{
		InputPricePerToken:  perMillion(2.00),
		OutputPricePerToken: perMillion(8.00),
	})
	r.Set("gpt-4.1-mini", ModelPricing{
		InputPricePerToken:  perMillion(0.40),
		OutputPricePerToken: perMillion(1.60),
	})
	r.Set("gpt-4.1-nano", ModelPricing{
		InputPricePerToken:  perMillion(0.10),
		OutputPricePerToken: perMillion(0.40),
	})
	r.Set("o3-mini", ModelPricing{
		InputPricePerToken:  perMillion(1.10),
		OutputPricePerToken: perMillion(4.40),
	})

	// --- Anthropic ---
	r.Set("claude-sonnet-4-5-20250929", ModelPricing{
		InputPricePerToken:  perMillion(3.00),
		OutputPricePerToken: perMillion(15.00),
	})
	r.Set("claude-haiku-3-5-20241022", ModelPricing{
		InputPricePerToken:  perMillion(0.80),
		OutputPricePerToken: perMillion(4.00),
	})
	r.Set("claude-3-5-sonnet-20241022", ModelPricing{
		InputPricePerToken:  perMillion(3.00),
		OutputPricePerToken: perMillion(15.00),
	})

	// --- Google ---
	r.Set("gemini-2.5-flash", ModelPricing{
		InputPricePerToken:  perMillion(0.15),
		OutputPricePerToken: perMillion(0.60),
	})
	r.Set("gemini-2.5-pro", ModelPricing{
		InputPricePerToken:  perMillion(1.25),
		OutputPricePerToken: perMillion(10.00),
	})
	r.Set("gemini-2.0-flash", ModelPricing{
		InputPricePerToken:  perMillion(0.10),
		OutputPricePerToken: perMillion(0.40),
	})

	// --- Ollama (local inference, zero cost) ---
	r.Set("llama3.1:8b", ModelPricing{
		InputPricePerToken:  0,
		OutputPricePerToken: 0,
	})
	r.Set("llama3.1:70b", ModelPricing{
		InputPricePerToken:  0,
		OutputPricePerToken: 0,
	})
	r.Set("qwen2.5-coder:32b", ModelPricing{
		InputPricePerToken:  0,
		OutputPricePerToken: 0,
	})

	// --- OpenRouter (uses underlying model pricing, common models) ---
	r.Set("openai/gpt-4o-mini", ModelPricing{
		InputPricePerToken:  perMillion(0.15),
		OutputPricePerToken: perMillion(0.60),
	})
	r.Set("anthropic/claude-sonnet-4-5-20250929", ModelPricing{
		InputPricePerToken:  perMillion(3.00),
		OutputPricePerToken: perMillion(15.00),
	})

	return r
}
