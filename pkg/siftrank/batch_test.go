package siftrank

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestGenerateBatchInput_BasicOutput(t *testing.T) {
	docs := []BatchDocInfo{
		{ID: "doc1", Value: "Hello world", InputIndex: 0},
		{ID: "doc2", Value: "Goodbye world", InputIndex: 1},
		{ID: "doc3", Value: "Test document", InputIndex: 2},
	}

	output, mapping := GenerateBatchInput(docs, "Rank by relevance", "gpt-4o-mini", 2, nil)

	// Should produce 2 lines (batch of 2 + batch of 1)
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 JSONL lines, got %d", len(lines))
	}

	// Verify first line structure
	var line1 BatchRequestLine
	if err := json.Unmarshal([]byte(lines[0]), &line1); err != nil {
		t.Fatalf("failed to parse line 1: %v", err)
	}
	if line1.CustomID != "batch-0" {
		t.Errorf("line 1 custom_id = %q, want %q", line1.CustomID, "batch-0")
	}
	if line1.Method != "POST" {
		t.Errorf("line 1 method = %q, want %q", line1.Method, "POST")
	}
	if line1.URL != "/v1/chat/completions" {
		t.Errorf("line 1 url = %q, want %q", line1.URL, "/v1/chat/completions")
	}
	if line1.Body.Model != "gpt-4o-mini" {
		t.Errorf("line 1 model = %q, want %q", line1.Body.Model, "gpt-4o-mini")
	}
	if len(line1.Body.Messages) != 1 {
		t.Fatalf("line 1 messages count = %d, want 1", len(line1.Body.Messages))
	}
	if line1.Body.Messages[0].Role != "user" {
		t.Errorf("line 1 message role = %q, want %q", line1.Body.Messages[0].Role, "user")
	}

	// Verify prompt contains doc IDs and values
	content := line1.Body.Messages[0].Content
	if !strings.Contains(content, "doc1") {
		t.Error("line 1 prompt missing doc1 ID")
	}
	if !strings.Contains(content, "Hello world") {
		t.Error("line 1 prompt missing doc1 value")
	}
	if !strings.Contains(content, "doc2") {
		t.Error("line 1 prompt missing doc2 ID")
	}

	// Verify mapping
	if len(mapping) != 2 {
		t.Fatalf("expected 2 mapping entries, got %d", len(mapping))
	}
	batch0 := mapping["batch-0"]
	if len(batch0) != 2 {
		t.Errorf("batch-0 has %d docs, want 2", len(batch0))
	}
	batch1 := mapping["batch-1"]
	if len(batch1) != 1 {
		t.Errorf("batch-1 has %d docs, want 1", len(batch1))
	}
}

func TestGenerateBatchInput_WithSchema(t *testing.T) {
	docs := []BatchDocInfo{
		{ID: "doc1", Value: "Test", InputIndex: 0},
	}

	schema := map[string]interface{}{"type": "object"}
	output, _ := GenerateBatchInput(docs, "Rank", "gpt-4o-mini", 10, schema)

	var line BatchRequestLine
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(output))), &line); err != nil {
		t.Fatalf("failed to parse: %v", err)
	}

	if line.Body.ResponseFormat == nil {
		t.Fatal("expected response_format with schema")
	}
	if line.Body.ResponseFormat.Type != "json_schema" {
		t.Errorf("response_format type = %q, want %q", line.Body.ResponseFormat.Type, "json_schema")
	}
	if line.Body.ResponseFormat.JSONSchema == nil {
		t.Fatal("expected json_schema in response_format")
	}
	if !line.Body.ResponseFormat.JSONSchema.Strict {
		t.Error("expected strict: true")
	}
}

func TestGenerateBatchInput_NoSchema(t *testing.T) {
	docs := []BatchDocInfo{
		{ID: "doc1", Value: "Test", InputIndex: 0},
	}

	output, _ := GenerateBatchInput(docs, "Rank", "gpt-4o-mini", 10, nil)

	var line BatchRequestLine
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(output))), &line); err != nil {
		t.Fatalf("failed to parse: %v", err)
	}

	if line.Body.ResponseFormat != nil {
		t.Error("expected no response_format when schema is nil")
	}
}

func TestGenerateBatchInput_DefaultBatchSize(t *testing.T) {
	docs := make([]BatchDocInfo, 25)
	for i := range docs {
		docs[i] = BatchDocInfo{ID: "d" + string(rune('a'+i)), Value: "val", InputIndex: i}
	}

	output, mapping := GenerateBatchInput(docs, "Rank", "gpt-4o", 0, nil) // 0 -> default

	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	// 25 docs / 10 default = 3 batches
	if len(lines) != 3 {
		t.Errorf("expected 3 lines with default batch size, got %d", len(lines))
	}
	if len(mapping) != 3 {
		t.Errorf("expected 3 mapping entries, got %d", len(mapping))
	}
}

func TestGenerateBatchInput_PromptContainsDisclaimer(t *testing.T) {
	docs := []BatchDocInfo{
		{ID: "doc1", Value: "Test", InputIndex: 0},
	}

	output, _ := GenerateBatchInput(docs, "Custom prompt", "gpt-4o-mini", 10, nil)

	var line BatchRequestLine
	_ = json.Unmarshal([]byte(strings.TrimSpace(string(output))), &line)

	content := line.Body.Messages[0].Content
	if !strings.Contains(content, "Custom prompt") {
		t.Error("prompt missing user prompt")
	}
	if !strings.Contains(content, "REMEMBER to:") {
		t.Error("prompt missing disclaimer")
	}
	if !strings.Contains(content, "NEVER respond with the actual value") {
		t.Error("prompt missing disclaimer details")
	}
}

func TestBatchMapping_Roundtrip(t *testing.T) {
	mapping := BatchMapping{
		BatchID:   "batch_abc123",
		FileID:    "file-xyz789",
		Model:     "gpt-4o-mini",
		Prompt:    "Rank by relevance",
		BatchSize: 10,
		Documents: map[string][]BatchDocInfo{
			"batch-0": {
				{ID: "doc1", Value: "Hello", InputIndex: 0},
				{ID: "doc2", Value: "World", InputIndex: 1},
			},
			"batch-1": {
				{ID: "doc3", Value: "Test", InputIndex: 2},
			},
		},
	}

	data, err := json.Marshal(mapping)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var restored BatchMapping
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if restored.BatchID != mapping.BatchID {
		t.Errorf("BatchID = %q, want %q", restored.BatchID, mapping.BatchID)
	}
	if restored.FileID != mapping.FileID {
		t.Errorf("FileID = %q, want %q", restored.FileID, mapping.FileID)
	}
	if restored.Model != mapping.Model {
		t.Errorf("Model = %q, want %q", restored.Model, mapping.Model)
	}
	if restored.Prompt != mapping.Prompt {
		t.Errorf("Prompt = %q, want %q", restored.Prompt, mapping.Prompt)
	}
	if len(restored.Documents) != 2 {
		t.Errorf("Documents count = %d, want 2", len(restored.Documents))
	}
	if len(restored.Documents["batch-0"]) != 2 {
		t.Errorf("batch-0 docs = %d, want 2", len(restored.Documents["batch-0"]))
	}
	if restored.Documents["batch-0"][0].ID != "doc1" {
		t.Errorf("batch-0[0].ID = %q, want %q", restored.Documents["batch-0"][0].ID, "doc1")
	}
}

func TestParseBatchResults_ValidOutput(t *testing.T) {
	output := `{"id":"resp-1","custom_id":"batch-0","response":{"status_code":200,"body":{"choices":[{"message":{"content":"{\"docs\":[\"doc1\",\"doc2\"]}"}}],"usage":{"prompt_tokens":100,"completion_tokens":50}}},"error":null}
{"id":"resp-2","custom_id":"batch-1","response":{"status_code":200,"body":{"choices":[{"message":{"content":"{\"docs\":[\"doc3\"]}"}}],"usage":{"prompt_tokens":80,"completion_tokens":30}}},"error":null}
`

	results, err := ParseBatchResults([]byte(output))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	if results["batch-0"] != `{"docs":["doc1","doc2"]}` {
		t.Errorf("batch-0 content = %q", results["batch-0"])
	}
	if results["batch-1"] != `{"docs":["doc3"]}` {
		t.Errorf("batch-1 content = %q", results["batch-1"])
	}
}

func TestParseBatchResults_SkipsErrors(t *testing.T) {
	output := `{"id":"resp-1","custom_id":"batch-0","response":{"status_code":200,"body":{"choices":[{"message":{"content":"ok"}}]}},"error":null}
{"id":"resp-2","custom_id":"batch-1","response":{"status_code":500,"body":{}},"error":{"code":"server_error","message":"internal error"}}
`

	results, err := ParseBatchResults([]byte(output))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("expected 1 result (skipping error), got %d", len(results))
	}

	if _, ok := results["batch-0"]; !ok {
		t.Error("expected batch-0 in results")
	}
}

func TestParseBatchResults_EmptyOutput(t *testing.T) {
	results, err := ParseBatchResults([]byte(""))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results for empty output, got %d", len(results))
	}
}

func TestNewBatchClient(t *testing.T) {
	bc := NewBatchClient("sk-test-key", "")
	if bc == nil {
		t.Fatal("expected non-nil BatchClient")
	}
	if bc.client == nil {
		t.Fatal("expected non-nil client")
	}
}

func TestNewBatchClient_WithBaseURL(t *testing.T) {
	bc := NewBatchClient("sk-test-key", "https://custom-api.example.com")
	if bc == nil {
		t.Fatal("expected non-nil BatchClient")
	}
}
