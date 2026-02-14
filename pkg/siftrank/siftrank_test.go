package siftrank

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openai/openai-go"
)

func TestNewRanker(t *testing.T) {
	tests := []struct {
		name    string
		config  *Config
		wantErr bool
	}{
		{
			name: "valid config",
			config: &Config{
				InitialPrompt:   "test prompt",
				BatchSize:       5,
				NumTrials:       2,
				Concurrency:     20,
				OpenAIModel:     openai.ChatModelGPT4oMini,
				RefinementRatio: 0.5,
				OpenAIKey:       "test-key",
				Encoding:        "o200k_base",
				BatchTokens:     2000,
				DryRun:          true,
			},
			wantErr: false,
		},
		{
			name: "empty prompt",
			config: &Config{
				InitialPrompt: "",
				BatchSize:     5,
				NumTrials:     2,
				OpenAIModel:   openai.ChatModelGPT4oMini,
				BatchTokens:   1000,
				OpenAIKey:     "test-key",
				Encoding:      "o200k_base",
			},
			wantErr: true,
		},
		{
			name: "invalid batch size",
			config: &Config{
				InitialPrompt: "test",
				BatchSize:     1, // Less than minBatchSize (2)
				NumTrials:     2,
				OpenAIModel:   openai.ChatModelGPT4oMini,
				BatchTokens:   1000,
				OpenAIKey:     "test-key",
				Encoding:      "o200k_base",
			},
			wantErr: true,
		},
		{
			name: "missing OpenAI key",
			config: &Config{
				InitialPrompt: "test",
				BatchSize:     5,
				NumTrials:     2,
				OpenAIModel:   openai.ChatModelGPT4oMini,
				BatchTokens:   1000,
				OpenAIKey:     "", // Empty key
				Encoding:      "o200k_base",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ranker, err := NewRanker(tt.config)
			if tt.wantErr {
				if err == nil {
					t.Errorf("NewRanker() expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Errorf("NewRanker() unexpected error: %v", err)
				return
			}
			if ranker == nil {
				t.Errorf("NewRanker() returned nil ranker")
			}
		})
	}
}

func TestRankFromFile_DryRun(t *testing.T) {
	// Create a temporary test file
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.txt")
	content := "apple\nbanana\ncherry"
	if err := os.WriteFile(testFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	config := &Config{
		InitialPrompt:   "Rank by alphabetical order",
		BatchSize:       3, // Set to 3 to include all items
		NumTrials:       1,
		Concurrency:     20,
		OpenAIModel:     openai.ChatModelGPT4oMini,
		RefinementRatio: 0.0, // Set to 0.0 to disable refinement and keep all results
		OpenAIKey:       "test-key",
		Encoding:        "o200k_base",
		BatchTokens:     2000,
		DryRun:          true, // Use dry run to avoid actual API calls
	}

	ranker, err := NewRanker(config)
	if err != nil {
		t.Fatalf("NewRanker() unexpected error: %v", err)
	}

	results, err := ranker.RankFromFile(testFile, "{{.Data}}", false)
	if err != nil {
		t.Errorf("RankFromFile() unexpected error: %v", err)
		return
	}

	if len(results) == 0 {
		t.Error("RankFromFile() returned no results")
		return
	}

	// Should get at least 2 results (algorithm may batch/filter items)
	if len(results) < 2 {
		t.Errorf("RankFromFile() expected at least 2 results, got %d", len(results))
		return
	}

	// Verify all results have required fields
	for i, result := range results {
		if result.Key == "" {
			t.Errorf("Result %d missing Key", i)
		}
		if result.Value == "" {
			t.Errorf("Result %d missing Value", i)
		}
		if result.Rank == 0 {
			t.Errorf("Result %d missing Rank", i)
		}
		if result.Exposure == 0 {
			t.Errorf("Result %d missing Exposure", i)
		}
	}

	// Verify ranks are properly assigned (1-based) for the results we got
	for i, result := range results {
		expectedRank := i + 1
		if result.Rank != expectedRank {
			t.Errorf("Result %d expected rank %d, got %d", i, expectedRank, result.Rank)
		}
	}
}

func TestRankFromFile_WithSentencesData(t *testing.T) {
	sentencesFile := filepath.Join("..", "..", "testdata", "sentences.txt")

	// Check if the testdata file exists
	if _, err := os.Stat(sentencesFile); os.IsNotExist(err) {
		t.Skip("testdata/sentences.txt not found, skipping integration test")
	}

	config := &Config{
		InitialPrompt:   `Rank each of these items according to their relevancy to the concept of "time".`,
		BatchSize:       10,
		NumTrials:       3,
		Concurrency:     20,
		OpenAIModel:     openai.ChatModelGPT4oMini,
		RefinementRatio: 0.5,
		OpenAIKey:       "test-key", // This would normally come from environment
		Encoding:        "o200k_base",
		BatchTokens:     128000,
		DryRun:          true, // Use dry run to avoid actual API calls
	}

	// Allow integration testing with real OpenAI API if API key is provided
	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		config.OpenAIKey = apiKey
		config.DryRun = false
		// Optionally override the base URL (defaults to OpenAI's standard URL)
		if apiBase := os.Getenv("OPENAI_API_BASE"); apiBase != "" {
			config.OpenAIAPIURL = apiBase
		}
	}

	ranker, err := NewRanker(config)
	if err != nil {
		t.Fatalf("NewRanker() unexpected error: %v", err)
	}

	results, err := ranker.RankFromFile(sentencesFile, "{{.Data}}", false)
	if err != nil {
		t.Fatalf("RankFromFile() unexpected error: %v", err)
	}

	// Basic sanity checks
	if len(results) == 0 {
		t.Error("RankFromFile() returned no results")
		return
	}

	t.Logf("Successfully ranked %d items from sentences.txt", len(results))

	// Print top 10 results
	maxResults := 10
	if len(results) < maxResults {
		maxResults = len(results)
	}

	t.Log("Top 10 results by relevance to 'time':")
	for i := 0; i < maxResults; i++ {
		t.Logf("%d. %s", i+1, results[i].Value)
	}

	// If this is not a dry run (real API call), validate that at least one time-related
	// sentence appears in the top 3 results
	if !config.DryRun {
		timeRelatedSentences := []string{
			"The train arrived exactly on time.",
			"The clock ticked steadily on the wall.",
			"The old clock chimed twelve times.",
		}

		top3Results := results
		if len(results) > 3 {
			top3Results = results[:3]
		}

		foundTimeRelated := false
		for _, result := range top3Results {
			for _, timeSentence := range timeRelatedSentences {
				if result.Value == timeSentence {
					foundTimeRelated = true
					t.Logf("Found time-related sentence in top 3: %s (rank %d)", result.Value, result.Rank)
					break
				}
			}
			if foundTimeRelated {
				break
			}
		}

		if !foundTimeRelated {
			t.Logf("Warning: None of the expected time-related sentences found in top 3")
			t.Logf("Expected one of: %v", timeRelatedSentences)
			t.Logf("Got top 3: %v", []string{top3Results[0].Value, top3Results[1].Value, top3Results[2].Value})
			// Note: This is a warning, not a failure, since AI ranking can vary
		}
	}

	// Verify structure of results
	for i, result := range results[:maxResults] {
		if result.Key == "" {
			t.Errorf("Result %d missing Key", i)
		}
		if result.Value == "" {
			t.Errorf("Result %d missing Value", i)
		}
		if result.Rank != i+1 {
			t.Errorf("Result %d expected rank %d, got %d", i, i+1, result.Rank)
		}
	}
}

func TestRankFromFile_Errors(t *testing.T) {
	config := &Config{
		InitialPrompt:   "test prompt",
		BatchSize:       5,
		NumTrials:       3,
		Concurrency:     20,
		OpenAIModel:     openai.ChatModelGPT4oMini,
		RefinementRatio: 0.5,
		OpenAIKey:       "test-key",
		Encoding:        "o200k_base",
		BatchTokens:     2000,
		DryRun:          true,
	}

	ranker, err := NewRanker(config)
	if err != nil {
		t.Fatalf("NewRanker() unexpected error: %v", err)
	}

	// Test with non-existent file
	_, err = ranker.RankFromFile("nonexistent.txt", "{{.Data}}", false)
	if err == nil {
		t.Error("RankFromFile() with non-existent file should return error")
	}

	// Test with invalid template
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.txt")
	if err := os.WriteFile(testFile, []byte("test"), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	_, err = ranker.RankFromFile(testFile, "{{.InvalidField | badFunc}}", false)
	if err == nil {
		t.Error("RankFromFile() with invalid template should return error")
	}
}

// TestRankFromFiles_MultipleTextFiles tests directory input with multiple .txt files
func TestRankFromFiles_MultipleTextFiles(t *testing.T) {
	tmpDir := t.TempDir()

	// Create 3 text files
	file1 := filepath.Join(tmpDir, "file1.txt")
	file2 := filepath.Join(tmpDir, "file2.txt")
	file3 := filepath.Join(tmpDir, "file3.txt")

	if err := os.WriteFile(file1, []byte("apple\nbanana"), 0600); err != nil {
		t.Fatalf("Failed to create file1: %v", err)
	}
	if err := os.WriteFile(file2, []byte("cherry\ndate"), 0600); err != nil {
		t.Fatalf("Failed to create file2: %v", err)
	}
	if err := os.WriteFile(file3, []byte("elderberry"), 0600); err != nil {
		t.Fatalf("Failed to create file3: %v", err)
	}

	config := &Config{
		InitialPrompt:   "Sort alphabetically",
		BatchSize:       5,
		NumTrials:       1,
		Concurrency:     20,
		OpenAIModel:     openai.ChatModelGPT4oMini,
		RefinementRatio: 0.0,
		OpenAIKey:       "test-key",
		Encoding:        "o200k_base",
		BatchTokens:     2000,
		DryRun:          true,
	}

	ranker, err := NewRanker(config)
	if err != nil {
		t.Fatalf("NewRanker() unexpected error: %v", err)
	}

	filePaths := []string{file1, file2, file3}
	results, err := ranker.RankFromFiles(filePaths, "{{.Data}}", false)
	if err != nil {
		t.Fatalf("RankFromFiles() unexpected error: %v", err)
	}

	// 5 total documents: apple, banana, cherry, date, elderberry
	if len(results) != 5 {
		t.Errorf("RankFromFiles() expected 5 results, got %d", len(results))
	}
}

// TestRankFromFiles_MixedJSON tests JSON files aggregation
func TestRankFromFiles_MixedJSON(t *testing.T) {
	tmpDir := t.TempDir()

	// Create 2 JSON files
	file1 := filepath.Join(tmpDir, "data1.json")
	file2 := filepath.Join(tmpDir, "data2.json")

	json1 := `[{"id": 1, "text": "first"}, {"id": 2, "text": "second"}]`
	json2 := `[{"id": 3, "text": "third"}]`

	if err := os.WriteFile(file1, []byte(json1), 0600); err != nil {
		t.Fatalf("Failed to create file1: %v", err)
	}
	if err := os.WriteFile(file2, []byte(json2), 0600); err != nil {
		t.Fatalf("Failed to create file2: %v", err)
	}

	config := &Config{
		InitialPrompt:   "Rank by id",
		BatchSize:       5,
		NumTrials:       1,
		Concurrency:     20,
		OpenAIModel:     openai.ChatModelGPT4oMini,
		RefinementRatio: 0.0,
		OpenAIKey:       "test-key",
		Encoding:        "o200k_base",
		BatchTokens:     2000,
		DryRun:          true,
	}

	ranker, err := NewRanker(config)
	if err != nil {
		t.Fatalf("NewRanker() unexpected error: %v", err)
	}

	// Use proper template for JSON objects with "text" field
	results, err := ranker.RankFromFiles([]string{file1, file2}, "{{.text}}", false)
	if err != nil {
		t.Fatalf("RankFromFiles() unexpected error: %v", err)
	}

	// 3 total documents from the JSON arrays
	if len(results) != 3 {
		t.Errorf("RankFromFiles() expected 3 results, got %d", len(results))
	}
}

// TestRankFromFiles_EmptyDirectory tests no files provided
func TestRankFromFiles_EmptyDirectory(t *testing.T) {
	config := &Config{
		InitialPrompt:   "test",
		BatchSize:       5,
		NumTrials:       1,
		Concurrency:     20,
		OpenAIModel:     openai.ChatModelGPT4oMini,
		RefinementRatio: 0.0,
		OpenAIKey:       "test-key",
		Encoding:        "o200k_base",
		BatchTokens:     2000,
		DryRun:          true,
	}

	ranker, err := NewRanker(config)
	if err != nil {
		t.Fatalf("NewRanker() unexpected error: %v", err)
	}

	_, err = ranker.RankFromFiles([]string{}, "{{.Data}}", false)

	if err == nil {
		t.Error("RankFromFiles() with empty file list should return error")
	}

	if err != nil && !strings.Contains(err.Error(), "no documents loaded") {
		t.Errorf("RankFromFiles() error should contain 'no documents loaded', got: %v", err)
	}
}

// TestRankFromFiles_ExceedsDocumentLimit tests error when document count exceeds limit
func TestRankFromFiles_ExceedsDocumentLimit(t *testing.T) {
	tmpDir := t.TempDir()

	// Create file with 10001 lines (exceeds MaxDocuments limit of 10000)
	var lines []string
	for i := 0; i < 10001; i++ {
		lines = append(lines, fmt.Sprintf("document %d", i))
	}

	file1 := filepath.Join(tmpDir, "large.txt")
	if err := os.WriteFile(file1, []byte(strings.Join(lines, "\n")), 0600); err != nil {
		t.Fatal(err)
	}

	config := &Config{
		InitialPrompt:   "test",
		BatchSize:       5,
		NumTrials:       1,
		Concurrency:     20,
		OpenAIModel:     openai.ChatModelGPT4oMini,
		RefinementRatio: 0.0,
		OpenAIKey:       "test-key",
		Encoding:        "o200k_base",
		BatchTokens:     2000,
		DryRun:          true,
		Logger:          slog.New(slog.NewTextHandler(io.Discard, nil)),
		LogLevel:        slog.LevelInfo,
	}

	ranker, err := NewRanker(config)
	if err != nil {
		t.Fatal(err)
	}

	_, err = ranker.RankFromFiles([]string{file1}, "{{.Data}}", false)

	if err == nil {
		t.Fatal("Expected error for document count exceeding limit, got nil")
	}

	expectedMsg := "too many documents to rank (max 10000)"
	if !strings.Contains(err.Error(), expectedMsg) {
		t.Errorf("Expected error to contain %q, got: %v", expectedMsg, err)
	}
}

// TestRankFromFiles_AtDocumentLimit tests success when document count equals limit
func TestRankFromFiles_AtDocumentLimit(t *testing.T) {
	tmpDir := t.TempDir()

	// Create file with exactly 10000 lines (at MaxDocuments limit)
	var lines []string
	for i := 0; i < 10000; i++ {
		lines = append(lines, fmt.Sprintf("document %d", i))
	}

	file1 := filepath.Join(tmpDir, "large.txt")
	if err := os.WriteFile(file1, []byte(strings.Join(lines, "\n")), 0600); err != nil {
		t.Fatal(err)
	}

	config := &Config{
		InitialPrompt:   "test",
		BatchSize:       5,
		NumTrials:       1,
		Concurrency:     20,
		OpenAIModel:     openai.ChatModelGPT4oMini,
		RefinementRatio: 0.0,
		OpenAIKey:       "test-key",
		Encoding:        "o200k_base",
		BatchTokens:     2000,
		DryRun:          true,
		Logger:          slog.New(slog.NewTextHandler(io.Discard, nil)),
		LogLevel:        slog.LevelInfo,
	}

	ranker, err := NewRanker(config)
	if err != nil {
		t.Fatal(err)
	}

	results, err := ranker.RankFromFiles([]string{file1}, "{{.Data}}", false)

	if err != nil {
		t.Fatalf("Expected success for document count at limit, got error: %v", err)
	}

	if len(results) != 10000 {
		t.Errorf("Expected 10000 documents, got %d", len(results))
	}
}
