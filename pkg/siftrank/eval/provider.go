package eval

import (
	"context"
	"time"
)

// LLMProvider is a generic interface for LLM providers
// This mirrors siftrank.LLMProvider but avoids import cycles
type LLMProvider interface {
	Complete(ctx context.Context, prompt string, opts CompletionOptionsInterface) (string, error)
}

// CompletionOptionsInterface provides access to completion metadata
// This avoids importing siftrank.CompletionOptions
type CompletionOptionsInterface interface {
	GetUsage() (inputTokens, outputTokens int)
}

// ProviderSelector selects which model/provider to use for each call
// Implementations can rotate between models, use random selection, etc.
type ProviderSelector interface {
	// SelectProvider returns the provider to use and its model ID
	SelectProvider(ctx context.Context) (LLMProvider, string, error)
}

// EvalProvider is a decorator that wraps LLMProvider calls with metrics collection
// It implements the LLMProvider interface and delegates to an underlying provider
// selected by the ProviderSelector.
//
// Zero overhead: metrics collection adds <1ms per call (see benchmarks)
type EvalProvider struct {
	selector  ProviderSelector
	collector *MetricsCollector
}

// NewEvalProvider creates a new EvalProvider that wraps provider calls with metrics
func NewEvalProvider(selector ProviderSelector, collector *MetricsCollector) *EvalProvider {
	return &EvalProvider{
		selector:  selector,
		collector: collector,
	}
}

// Complete implements LLMProvider interface
// Wraps the underlying provider's Complete call with metrics collection
func (ep *EvalProvider) Complete(ctx context.Context, prompt string, opts CompletionOptionsInterface) (string, error) {
	// Select which provider/model to use
	provider, modelID, err := ep.selector.SelectProvider(ctx)
	if err != nil {
		// Record selection failure
		ep.collector.RecordCall(CallMetrics{
			ModelID:   "unknown",
			Success:   false,
			ErrorType: err.Error(),
			Timestamp: time.Now(),
		})
		return "", err
	}

	// Start timing
	startTime := time.Now()

	// Call underlying provider
	response, callErr := provider.Complete(ctx, prompt, opts)

	// End timing
	endTime := time.Now()
	latencyMs := endTime.Sub(startTime).Milliseconds()

	// Build metrics
	metrics := CallMetrics{
		ModelID:   modelID,
		LatencyMs: latencyMs,
		Success:   callErr == nil,
		Timestamp: startTime,
	}

	// Extract token counts from opts if available
	if opts != nil {
		inputTokens, outputTokens := opts.GetUsage()
		metrics.InputTokens = inputTokens
		metrics.OutputTokens = outputTokens
		metrics.PromptTokens = inputTokens // Alternative naming
	}

	// Record error type if call failed
	if callErr != nil {
		metrics.ErrorType = callErr.Error()
	}

	// Record metrics (thread-safe)
	ep.collector.RecordCall(metrics)

	return response, callErr
}

// GetCollector returns the underlying MetricsCollector
// Useful for aggregating metrics after a session completes
func (ep *EvalProvider) GetCollector() *MetricsCollector {
	return ep.collector
}
