package main

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// Config represents the siftrank config file structure.
type Config struct {
	Provider string            `yaml:"provider"`
	Model    string            `yaml:"model"`
	APIKey   string            `yaml:"api_key"`
	BaseURL  string            `yaml:"base_url"`
	APIKeys  map[string]string `yaml:"api_keys"`
}

// defaultConfigPath returns the default config file path following XDG Base Directory spec.
func defaultConfigPath() string {
	configHome := os.Getenv("XDG_CONFIG_HOME")
	if configHome == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		configHome = filepath.Join(home, ".config")
	}
	return filepath.Join(configHome, "siftrank", "config.yaml")
}

// loadConfig reads and parses a YAML config file.
// Returns an empty Config and nil error if the file doesn't exist.
// Returns an empty Config and prints a warning if the file is malformed.
func loadConfig(path string) Config {
	if path == "" {
		return Config{}
	}

	data, err := os.ReadFile(path) // #nosec G304 - user-provided config path
	if err != nil {
		if os.IsNotExist(err) {
			return Config{}
		}
		fmt.Fprintf(os.Stderr, "Warning: could not read config file %q: %v\n", path, err)
		return Config{}
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: malformed config file %q: %v\n", path, err)
		return Config{}
	}

	return cfg
}

// resolveAPIKeyFromConfig returns an API key for the given provider from the config's
// per-provider api_keys map. Returns empty string if not found.
func resolveAPIKeyFromConfig(cfg Config, provider string) string {
	if cfg.APIKeys == nil {
		return ""
	}
	return cfg.APIKeys[provider]
}
