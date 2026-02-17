package siftrank

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/pkoukk/tiktoken-go"
	"google.golang.org/genai"
)

// GoogleProvider implements LLMProvider using Google Gemini API
type GoogleProvider struct {
	client   *genai.Client
	model    string
	logger   *slog.Logger
	encoding *tiktoken.Tiktoken
}

// GoogleConfig configures the Google provider
type GoogleConfig struct {
	APIKey   string       // API key for Gemini API
	Model    string       // Model identifier (e.g., "gemini-2.5-flash")
	Encoding string       // Tokenizer encoding
	Logger   *slog.Logger
}

// NewGoogleProvider creates a new Google Gemini provider
func NewGoogleProvider(cfg GoogleConfig) (*GoogleProvider, error) {
	// Create encoding for token estimation
	encoding, err := tiktoken.GetEncoding(cfg.Encoding)
	if err != nil {
		return nil, fmt.Errorf("failed to get tiktoken encoding: %w", err)
	}

	// Create Google GenAI client
	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  cfg.APIKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Google GenAI client: %w", err)
	}

	return &GoogleProvider{
		client:   client,
		model:    cfg.Model,
		logger:   cfg.Logger,
		encoding: encoding,
	}, nil
}

// Complete implements LLMProvider.Complete
// Handles network-level retries only. Returns raw response without validation.
func (p *GoogleProvider) Complete(ctx context.Context, prompt string, opts *CompletionOptions) (string, error) {
	backoff := time.Second
	maxBackoff := 30 * time.Second

	// Create default options if nil
	if opts == nil {
		opts = &CompletionOptions{}
	}

	var totalUsage Usage

	for {
		// Check if context cancelled
		if ctx.Err() != nil {
			return "", ctx.Err()
		}

		// Create timeout context for this attempt
		timeoutCtx, cancel := context.WithTimeout(ctx, 30*time.Second)

		// Build config
		var config *genai.GenerateContentConfig
		if opts.Temperature != nil || opts.MaxTokens != nil {
			config = &genai.GenerateContentConfig{}
			if opts.Temperature != nil {
				temp := float32(*opts.Temperature)
				config.Temperature = &temp
			}
			if opts.MaxTokens != nil {
				config.MaxOutputTokens = int32(*opts.MaxTokens)
			}
		}

		// Build content
		contents := []*genai.Content{
			{
				Parts: []*genai.Part{
					{Text: prompt},
				},
				Role: "user",
			},
		}

		// Make API call
		response, err := p.client.Models.GenerateContent(timeoutCtx, p.model, contents, config)
		cancel()

		if err == nil && response != nil {
			// Extract text from response
			text, textErr := extractGoogleText(response)
			if textErr != nil {
				return "", textErr
			}

			// Populate usage from response metadata
			if response.UsageMetadata != nil {
				callUsage := Usage{
					InputTokens:  int(response.UsageMetadata.PromptTokenCount),
					OutputTokens: int(response.UsageMetadata.CandidatesTokenCount),
				}
				totalUsage.Add(callUsage)
			}

			// Populate output fields in opts
			opts.Usage = totalUsage
			opts.ModelUsed = p.model

			// Extract finish reason from first candidate
			if len(response.Candidates) > 0 && response.Candidates[0].FinishReason != "" {
				opts.FinishReason = string(response.Candidates[0].FinishReason)
			}

			p.logger.Debug("Google call successful",
				"input_tokens", totalUsage.InputTokens,
				"output_tokens", totalUsage.OutputTokens,
				"model", opts.ModelUsed)

			return text, nil
		}

		// Check if context cancelled
		if ctx.Err() != nil {
			return "", ctx.Err()
		}

		// Handle timeout
		if err == context.DeadlineExceeded {
			p.logger.Debug("Request timeout, retrying", "backoff", backoff)
			time.Sleep(backoff)
			backoff = minDuration(backoff*2, maxBackoff)
			continue
		}

		// Classify error for retry decisions
		errStr := ""
		if err != nil {
			errStr = err.Error()
		}

		// Rate limit errors (429)
		if strings.Contains(errStr, "429") || strings.Contains(errStr, "RESOURCE_EXHAUSTED") {
			p.logger.Debug("Rate limit exceeded, retrying", "backoff", backoff)
			time.Sleep(backoff)
			backoff = minDuration(backoff*2, maxBackoff)
			continue
		}

		// Server errors (5xx) - retry
		if strings.Contains(errStr, "500") || strings.Contains(errStr, "502") ||
			strings.Contains(errStr, "503") || strings.Contains(errStr, "INTERNAL") ||
			strings.Contains(errStr, "UNAVAILABLE") {
			p.logger.Debug("Server error, retrying", "error", err, "backoff", backoff)
			time.Sleep(backoff)
			backoff = minDuration(backoff*2, maxBackoff)
			continue
		}

		// Client errors (bad auth, invalid model, etc.) are unrecoverable
		if strings.Contains(errStr, "400") || strings.Contains(errStr, "401") ||
			strings.Contains(errStr, "403") || strings.Contains(errStr, "404") ||
			strings.Contains(errStr, "INVALID_ARGUMENT") || strings.Contains(errStr, "PERMISSION_DENIED") {
			p.logger.Error("Unrecoverable client error", "error", err)
			return "", fmt.Errorf("unrecoverable error: %w", err)
		}

		// Other errors - retry with backoff
		p.logger.Debug("Request failed, retrying", "error", err, "backoff", backoff)
		time.Sleep(backoff)
		backoff = minDuration(backoff*2, maxBackoff)
	}
}

// extractGoogleText extracts the text content from a Gemini response
func extractGoogleText(response *genai.GenerateContentResponse) (string, error) {
	if len(response.Candidates) == 0 {
		return "", fmt.Errorf("no candidates in response")
	}

	candidate := response.Candidates[0]
	if candidate.Content == nil || len(candidate.Content.Parts) == 0 {
		return "", fmt.Errorf("no content in response candidate")
	}

	var builder strings.Builder
	for _, part := range candidate.Content.Parts {
		if part.Text != "" {
			builder.WriteString(part.Text)
		}
	}

	return builder.String(), nil
}

// EstimateTokens implements TokenEstimator.EstimateTokens
func (p *GoogleProvider) EstimateTokens(text string) int {
	return len(p.encoding.Encode(text, nil, nil))
}
