package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/meganerd/siftrank/pkg/siftrank"
)

func TestBatchSubmit_NonOpenAIProvider(t *testing.T) {
	// Reset flags
	batchSubProvider = "anthropic"
	batchSubModel = "claude-sonnet-4-20250514"
	batchSubFile = "test.txt"
	batchSubPrompt = "Rank"
	batchSubSize = 10
	batchSubOutputDir = "."

	err := runBatchSubmit(batchSubmitCmd, nil)
	if err == nil {
		t.Fatal("expected error for non-openai provider")
	}
	if !strings.Contains(err.Error(), "only supports the openai provider") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestBatchSubmit_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.txt")
	if err := os.WriteFile(path, []byte(""), 0600); err != nil {
		t.Fatal(err)
	}

	batchSubProvider = "openai"
	batchSubModel = "gpt-4o-mini"
	batchSubFile = path
	batchSubPrompt = "Rank"
	batchSubSize = 10
	batchSubOutputDir = dir

	// Set a fake API key via env
	t.Setenv("OPENAI_API_KEY", "sk-test-fake-key")

	err := runBatchSubmit(batchSubmitCmd, nil)
	if err == nil {
		t.Fatal("expected error for empty file")
	}
	if !strings.Contains(err.Error(), "no documents found") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestLoadBatchDocuments(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.txt")
	content := "First line\nSecond line\n\nThird line\n"
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}

	docs, err := loadBatchDocuments(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(docs) != 3 {
		t.Fatalf("expected 3 documents, got %d", len(docs))
	}

	if docs[0].Value != "First line" {
		t.Errorf("doc[0].Value = %q, want %q", docs[0].Value, "First line")
	}
	if docs[1].Value != "Second line" {
		t.Errorf("doc[1].Value = %q, want %q", docs[1].Value, "Second line")
	}
	if docs[2].Value != "Third line" {
		t.Errorf("doc[2].Value = %q, want %q", docs[2].Value, "Third line")
	}

	// Verify IDs are deterministic and 8 chars
	for i, doc := range docs {
		if len(doc.ID) != 8 {
			t.Errorf("doc[%d].ID length = %d, want 8", i, len(doc.ID))
		}
		if doc.InputIndex != i {
			t.Errorf("doc[%d].InputIndex = %d, want %d", i, doc.InputIndex, i)
		}
	}
}

func TestLoadBatchDocuments_MissingFile(t *testing.T) {
	_, err := loadBatchDocuments("/nonexistent/file.txt")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestBatchResults_MappingFileRoundtrip(t *testing.T) {
	dir := t.TempDir()

	mapping := siftrank.BatchMapping{
		BatchID:   "batch_test123",
		FileID:    "file-test456",
		Model:     "gpt-4o-mini",
		Prompt:    "Rank by importance",
		BatchSize: 10,
		Documents: map[string][]siftrank.BatchDocInfo{
			"batch-0": {
				{ID: "abc12345", Value: "Document one", InputIndex: 0},
				{ID: "def67890", Value: "Document two", InputIndex: 1},
			},
		},
	}

	// Write mapping file
	path := filepath.Join(dir, ".siftrank-batch.json")
	data, err := json.MarshalIndent(mapping, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}

	// Read it back
	readData, err := os.ReadFile(path) // #nosec G304 - test path
	if err != nil {
		t.Fatal(err)
	}

	var restored siftrank.BatchMapping
	if err := json.Unmarshal(readData, &restored); err != nil {
		t.Fatal(err)
	}

	if restored.BatchID != "batch_test123" {
		t.Errorf("BatchID = %q, want %q", restored.BatchID, "batch_test123")
	}
	if restored.Model != "gpt-4o-mini" {
		t.Errorf("Model = %q, want %q", restored.Model, "gpt-4o-mini")
	}
	if len(restored.Documents["batch-0"]) != 2 {
		t.Errorf("batch-0 docs = %d, want 2", len(restored.Documents["batch-0"]))
	}
}
