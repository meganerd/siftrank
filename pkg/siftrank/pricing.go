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
// It derives pricing from DefaultModelCatalog, which is the single source of truth.
func DefaultPricingRegistry() *PricingRegistry {
	return DefaultModelCatalog().PricingRegistry()
}
