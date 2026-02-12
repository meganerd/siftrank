package siftrank

import "net/http"

// AuthStrategy defines how a provider authenticates HTTP requests.
// Different LLM providers use different authentication methods:
//   - OpenAI, OpenRouter: Bearer token in Authorization header
//   - Anthropic: Custom X-API-Key header
//   - Google: API key in query parameters (not yet supported via headers)
//   - Ollama: Optional authentication (NoAuth when not configured)
type AuthStrategy interface {
	// ApplyAuth adds authentication headers to an HTTP request.
	// This method is safe for concurrent use.
	ApplyAuth(req *http.Request)
}

// BearerAuth implements AuthStrategy for Bearer token authentication.
// Used by: OpenAI, OpenRouter
type BearerAuth struct {
	Token string
}

// ApplyAuth adds the Bearer token to the Authorization header.
func (b *BearerAuth) ApplyAuth(req *http.Request) {
	req.Header.Set("Authorization", "Bearer "+b.Token)
}

// NewBearerAuth creates a Bearer token auth strategy.
// This is the most common authentication method for LLM APIs.
func NewBearerAuth(token string) *BearerAuth {
	return &BearerAuth{Token: token}
}

// HeaderAuth implements AuthStrategy for custom header authentication.
// Used by: Anthropic (X-API-Key)
type HeaderAuth struct {
	HeaderName  string
	HeaderValue string
}

// ApplyAuth adds the custom header to the request.
func (h *HeaderAuth) ApplyAuth(req *http.Request) {
	req.Header.Set(h.HeaderName, h.HeaderValue)
}

// NewHeaderAuth creates a custom header auth strategy.
// Example: NewHeaderAuth("X-API-Key", "sk-ant-...")
func NewHeaderAuth(name, value string) *HeaderAuth {
	return &HeaderAuth{
		HeaderName:  name,
		HeaderValue: value,
	}
}

// NoAuth implements AuthStrategy for providers that don't require authentication.
// Used by: Ollama (when running locally without auth), custom endpoints
type NoAuth struct{}

// ApplyAuth is a no-op since no authentication is required.
func (n *NoAuth) ApplyAuth(req *http.Request) {
	// No authentication headers to add
}

// NewNoAuth creates a no-auth strategy.
// Use this for Ollama instances without authentication configured.
func NewNoAuth() *NoAuth {
	return &NoAuth{}
}
