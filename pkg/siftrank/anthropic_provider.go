package siftrank

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/pkoukk/tiktoken-go"
)

// anthropicCustomTransport captures response headers and body for rate limit handling
type anthropicCustomTransport struct {
	mu         sync.Mutex
	Transport  http.RoundTripper
	Headers    http.Header
	StatusCode int
	Body       []byte
}

func (t *anthropicCustomTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.Transport.RoundTrip(req)
	if err != nil {
		return nil, err
	}

	t.mu.Lock()
	t.Headers = resp.Header
	t.StatusCode = resp.StatusCode

	t.Body, err = io.ReadAll(resp.Body)
	// Create a copy of the body to use after unlocking
	bodyCopy := make([]byte, len(t.Body))
	copy(bodyCopy, t.Body)
	t.mu.Unlock()

	if err != nil {
		return nil, err
	}

	resp.Body = io.NopCloser(bytes.NewBuffer(bodyCopy))

	return resp, nil
}

// anthropicAuthTransport applies authentication strategy to HTTP requests
type anthropicAuthTransport struct {
	Transport http.RoundTripper
	Auth      AuthStrategy
}

func (t *anthropicAuthTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Apply authentication before forwarding to next transport
	t.Auth.ApplyAuth(req)
	return t.Transport.RoundTrip(req)
}

// AnthropicProvider implements LLMProvider using Anthropic API
type AnthropicProvider struct {
	client    *anthropic.Client
	model     anthropic.Model
	logger    *slog.Logger
	encoding  *tiktoken.Tiktoken
	transport *anthropicCustomTransport
}

// AnthropicConfig configures the Anthropic provider
type AnthropicConfig struct {
	Auth     AuthStrategy // Authentication strategy (HeaderAuth for Anthropic with x-api-key)
	Model    string       // Model identifier (e.g., "claude-3-5-sonnet-20241022")
	BaseURL  string       // Optional: for custom endpoints
	Encoding string       // Tokenizer encoding
	Logger   *slog.Logger
}

// NewAnthropicProvider creates a new Anthropic provider
func NewAnthropicProvider(cfg AnthropicConfig) (*AnthropicProvider, error) {
	// Create encoding
	encoding, err := tiktoken.GetEncoding(cfg.Encoding)
	if err != nil {
		return nil, fmt.Errorf("failed to get tiktoken encoding: %w", err)
	}

	// Create transport chain: auth -> custom (rate limit handling) -> default
	customTransport := &anthropicCustomTransport{Transport: http.DefaultTransport}
	authTransport := &anthropicAuthTransport{
		Transport: customTransport,
		Auth:      cfg.Auth,
	}
	httpClient := &http.Client{Transport: authTransport}

	// Create Anthropic client options
	clientOptions := []option.RequestOption{
		option.WithHTTPClient(httpClient),
		option.WithMaxRetries(0), // We handle retries ourselves
	}

	if cfg.BaseURL != "" {
		baseURL := cfg.BaseURL
		if !strings.HasSuffix(baseURL, "/") {
			baseURL += "/"
		}
		clientOptions = append(clientOptions, option.WithBaseURL(baseURL))
	}

	client := anthropic.NewClient(clientOptions...)

	return &AnthropicProvider{
		client:    &client,
		model:     anthropic.Model(cfg.Model),
		logger:    cfg.Logger,
		encoding:  encoding,
		transport: customTransport,
	}, nil
}

// Complete implements LLMProvider.Complete
// Handles network-level retries only. Returns raw response without validation.
func (p *AnthropicProvider) Complete(ctx context.Context, prompt string, opts *CompletionOptions) (string, error) {
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
		timeoutCtx, cancel := context.WithTimeout(ctx, 15*time.Second)

		// Build request parameters
		params := anthropic.MessageNewParams{
			Model: p.model,
			Messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
			},
			MaxTokens: 4096, // Default max tokens for Anthropic
		}

		// Add temperature if provided
		if opts.Temperature != nil {
			params.Temperature = anthropic.Float(*opts.Temperature)
		}

		// Add max tokens if provided
		if opts.MaxTokens != nil {
			params.MaxTokens = int64(*opts.MaxTokens)
		}

		// Make API call
		message, err := p.client.Messages.New(timeoutCtx, params)
		cancel() // Cancel immediately after API call to avoid resource leak

		if err == nil {
			// Success! Populate usage and metadata
			callUsage := Usage{
				InputTokens:  int(message.Usage.InputTokens),
				OutputTokens: int(message.Usage.OutputTokens),
			}

			totalUsage.Add(callUsage)

			// Populate output fields in opts
			opts.Usage = totalUsage
			opts.ModelUsed = string(message.Model)
			opts.FinishReason = string(message.StopReason)
			opts.RequestID = message.ID

			// Extract text content from response
			// Anthropic returns an array of content blocks, we concatenate all text blocks
			var contentBuilder strings.Builder
			for _, block := range message.Content {
				switch b := block.AsAny().(type) {
				case anthropic.TextBlock:
					contentBuilder.WriteString(b.Text)
				}
			}
			content := contentBuilder.String()

			p.logger.Debug("Anthropic call successful",
				"input_tokens", callUsage.InputTokens,
				"output_tokens", callUsage.OutputTokens,
				"model", opts.ModelUsed)

			// Return raw content - no validation
			return content, nil
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

		// Get status code under lock for error handling
		p.transport.mu.Lock()
		statusCode := p.transport.StatusCode
		p.transport.mu.Unlock()

		// Handle rate limits (429)
		if statusCode == http.StatusTooManyRequests {
			p.handleRateLimit(&backoff, maxBackoff)
			continue
		}

		// Handle server errors (5xx) - retry
		if statusCode >= 500 && statusCode < 600 {
			p.logger.Debug("Server error, retrying",
				"status", statusCode,
				"backoff", backoff)
			time.Sleep(backoff)
			backoff = minDuration(backoff*2, maxBackoff)
			continue
		}

		// Client errors (4xx except 429) are unrecoverable
		if statusCode >= 400 && statusCode < 500 {
			p.logger.Error("Unrecoverable client error",
				"status", statusCode,
				"error", err)
			return "", fmt.Errorf("unrecoverable error (status %d): %w",
				statusCode, err)
		}

		// Other errors - retry with backoff
		p.logger.Debug("Request failed, retrying", "error", err, "backoff", backoff)
		time.Sleep(backoff)
		backoff = minDuration(backoff*2, maxBackoff)
	}
}

// handleRateLimit handles rate limit errors with intelligent backoff
func (p *AnthropicProvider) handleRateLimit(backoff *time.Duration, maxBackoff time.Duration) {
	// Get headers and body under lock
	p.transport.mu.Lock()
	headers := p.transport.Headers
	body := p.transport.Body
	p.transport.mu.Unlock()

	// Log rate limit headers (Anthropic uses different header names)
	for key, values := range headers {
		if strings.Contains(strings.ToLower(key), "rate") ||
			strings.Contains(strings.ToLower(key), "retry") ||
			strings.Contains(strings.ToLower(key), "limit") {
			for _, value := range values {
				p.logger.Debug("Rate limit header", "key", key, "value", value)
			}
		}
	}

	if body != nil {
		p.logger.Debug("Rate limit response body", "body", string(body))
	}

	// Extract suggested wait time from retry-after header
	retryAfterStr := headers.Get("retry-after")
	if retryAfterStr == "" {
		retryAfterStr = headers.Get("Retry-After")
	}

	var retryAfter time.Duration
	if retryAfterStr != "" {
		// Try parsing as seconds first
		if seconds, err := strconv.Atoi(retryAfterStr); err == nil {
			retryAfter = time.Duration(seconds) * time.Second
		} else {
			// Try parsing as duration
			retryAfter, _ = time.ParseDuration(retryAfterStr)
		}
	}

	p.logger.Debug("Rate limit exceeded",
		"retry_after", retryAfter)

	// Use suggested wait time if available, otherwise exponential backoff
	if retryAfter > 0 {
		p.logger.Debug("Waiting for rate limit reset", "duration", retryAfter)
		time.Sleep(retryAfter)
	} else {
		p.logger.Debug("Waiting with exponential backoff", "duration", *backoff)
		time.Sleep(*backoff)
		*backoff = minDuration(*backoff*2, maxBackoff)
	}
}

// EstimateTokens implements TokenEstimator.EstimateTokens
func (p *AnthropicProvider) EstimateTokens(text string) int {
	return len(p.encoding.Encode(text, nil, nil))
}
