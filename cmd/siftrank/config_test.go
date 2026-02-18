package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/meganerd/siftrank/pkg/siftrank"
)

func TestLoadConfig_ValidFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	content := `provider: anthropic
model: claude-sonnet-4-5-20250929
api_key: sk-global-key
base_url: https://custom.api.example.com
api_keys:
  openai: sk-openai-key
  anthropic: sk-anthropic-key
`
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}

	cfg := loadConfig(path)

	if cfg.Provider != "anthropic" {
		t.Errorf("Provider = %q, want %q", cfg.Provider, "anthropic")
	}
	if cfg.Model != "claude-sonnet-4-5-20250929" {
		t.Errorf("Model = %q, want %q", cfg.Model, "claude-sonnet-4-5-20250929")
	}
	if cfg.APIKey != "sk-global-key" {
		t.Errorf("APIKey = %q, want %q", cfg.APIKey, "sk-global-key")
	}
	if cfg.BaseURL != "https://custom.api.example.com" {
		t.Errorf("BaseURL = %q, want %q", cfg.BaseURL, "https://custom.api.example.com")
	}
	if cfg.APIKeys["openai"] != "sk-openai-key" {
		t.Errorf("APIKeys[openai] = %q, want %q", cfg.APIKeys["openai"], "sk-openai-key")
	}
	if cfg.APIKeys["anthropic"] != "sk-anthropic-key" {
		t.Errorf("APIKeys[anthropic] = %q, want %q", cfg.APIKeys["anthropic"], "sk-anthropic-key")
	}
}

func TestLoadConfig_MissingFile(t *testing.T) {
	cfg := loadConfig("/nonexistent/path/config.yaml")

	if cfg.Provider != "" {
		t.Errorf("expected empty Provider for missing file, got %q", cfg.Provider)
	}
	if cfg.Model != "" {
		t.Errorf("expected empty Model for missing file, got %q", cfg.Model)
	}
}

func TestLoadConfig_MalformedYAML(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte("{{invalid yaml"), 0600); err != nil {
		t.Fatal(err)
	}

	// Should not panic, returns empty config
	cfg := loadConfig(path)
	if cfg.Provider != "" {
		t.Errorf("expected empty Provider for malformed YAML, got %q", cfg.Provider)
	}
}

func TestLoadConfig_EmptyPath(t *testing.T) {
	cfg := loadConfig("")
	if cfg.Provider != "" {
		t.Errorf("expected empty config for empty path, got %q", cfg.Provider)
	}
}

func TestLoadConfig_PartialConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	content := `model: gpt-4o
`
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}

	cfg := loadConfig(path)

	if cfg.Model != "gpt-4o" {
		t.Errorf("Model = %q, want %q", cfg.Model, "gpt-4o")
	}
	if cfg.Provider != "" {
		t.Errorf("expected empty Provider for partial config, got %q", cfg.Provider)
	}
}

func TestResolveAPIKeyFromConfig(t *testing.T) {
	cfg := Config{
		APIKeys: map[string]string{
			"openai":    "sk-from-config-openai",
			"anthropic": "sk-from-config-anthropic",
		},
	}

	if got := resolveAPIKeyFromConfig(cfg, "openai"); got != "sk-from-config-openai" {
		t.Errorf("resolveAPIKeyFromConfig(openai) = %q, want %q", got, "sk-from-config-openai")
	}
	if got := resolveAPIKeyFromConfig(cfg, "anthropic"); got != "sk-from-config-anthropic" {
		t.Errorf("resolveAPIKeyFromConfig(anthropic) = %q, want %q", got, "sk-from-config-anthropic")
	}
	if got := resolveAPIKeyFromConfig(cfg, "google"); got != "" {
		t.Errorf("resolveAPIKeyFromConfig(google) = %q, want empty", got)
	}
}

func TestResolveAPIKeyFromConfig_NilMap(t *testing.T) {
	cfg := Config{}
	if got := resolveAPIKeyFromConfig(cfg, "openai"); got != "" {
		t.Errorf("expected empty for nil api_keys map, got %q", got)
	}
}

func TestResolveAPIKey_EnvOverridesConfig(t *testing.T) {
	// Set up config with a key
	oldConfig := appConfig
	appConfig = Config{
		APIKeys: map[string]string{
			"openai": "sk-from-config",
		},
	}
	defer func() { appConfig = oldConfig }()

	// Set env var — should win over config
	t.Setenv("OPENAI_API_KEY", "sk-from-env")

	key := resolveAPIKey(siftrank.ProviderTypeOpenAI)
	if key != "sk-from-env" {
		t.Errorf("expected env var to win over config, got %q", key)
	}
}

func TestResolveAPIKey_ConfigFallback(t *testing.T) {
	// Set up config with a key, no env var
	oldConfig := appConfig
	appConfig = Config{
		APIKeys: map[string]string{
			"openai": "sk-from-config",
		},
	}
	defer func() { appConfig = oldConfig }()

	// Clear env var
	t.Setenv("OPENAI_API_KEY", "")

	key := resolveAPIKey(siftrank.ProviderTypeOpenAI)
	if key != "sk-from-config" {
		t.Errorf("expected config fallback, got %q", key)
	}
}

func TestDefaultConfigPath(t *testing.T) {
	// Test with XDG_CONFIG_HOME set
	t.Setenv("XDG_CONFIG_HOME", "/tmp/test-xdg")
	path := defaultConfigPath()
	expected := "/tmp/test-xdg/siftrank/config.yaml"
	if path != expected {
		t.Errorf("defaultConfigPath() = %q, want %q", path, expected)
	}
}

func TestDefaultConfigPath_NoXDG(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", "")
	path := defaultConfigPath()
	// Should end with .config/siftrank/config.yaml
	if filepath.Base(filepath.Dir(filepath.Dir(path))) != ".config" {
		t.Errorf("expected .config in default path, got %q", path)
	}
}
