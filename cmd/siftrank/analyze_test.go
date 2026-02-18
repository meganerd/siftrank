package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const testTraceContent = `{"round":1,"trial":1,"trials_completed":1,"trials_remaining":4,"total_input_tokens":50000,"total_output_tokens":20000}
{"round":1,"trial":2,"trials_completed":2,"trials_remaining":3,"total_input_tokens":100000,"total_output_tokens":40000,"elbow_position":3}
{"round":1,"trial":3,"trials_completed":3,"trials_remaining":2,"total_input_tokens":150000,"total_output_tokens":60000,"elbow_position":3,"stable_trials_count":2}
`

func writeTestTrace(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "trace.jsonl")
	if err := os.WriteFile(path, []byte(testTraceContent), 0600); err != nil {
		t.Fatalf("failed to write test trace: %v", err)
	}
	return path
}

func TestAnalyze_RecommendFlag(t *testing.T) {
	tracePath := writeTestTrace(t)

	// Reset flags for test
	analyzeModel = "gpt-4o-mini"
	analyzeRecommend = true
	analyzeExcludeProvider = nil
	analyzeJSON = false

	// Capture stdout
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	err := runAnalyze(analyzeCmd, []string{tracePath})

	w.Close()
	os.Stdout = old

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	buf := make([]byte, 64*1024)
	n, _ := r.Read(buf)
	output := string(buf[:n])

	if !strings.Contains(output, "Model Recommendations") {
		t.Error("expected 'Model Recommendations' section in output")
	}
	if !strings.Contains(output, "Trace Analysis") {
		t.Error("expected 'Trace Analysis' header in output")
	}
}

func TestAnalyze_RecommendWithoutModel(t *testing.T) {
	tracePath := writeTestTrace(t)

	analyzeModel = ""
	analyzeRecommend = true
	analyzeExcludeProvider = nil
	analyzeJSON = false

	err := runAnalyze(analyzeCmd, []string{tracePath})
	if err == nil {
		t.Fatal("expected error when --recommend without --model")
	}
	if !strings.Contains(err.Error(), "--recommend requires --model") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestAnalyze_JSONOutput(t *testing.T) {
	tracePath := writeTestTrace(t)

	analyzeModel = "gpt-4o-mini"
	analyzeRecommend = true
	analyzeExcludeProvider = nil
	analyzeJSON = true

	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	err := runAnalyze(analyzeCmd, []string{tracePath})

	w.Close()
	os.Stdout = old

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	buf := make([]byte, 64*1024)
	n, _ := r.Read(buf)
	output := buf[:n]

	var report analyzeReport
	if err := json.Unmarshal(output, &report); err != nil {
		t.Fatalf("invalid JSON output: %v\noutput: %s", err, output)
	}

	if report.Rounds != 1 {
		t.Errorf("Rounds = %d, want 1", report.Rounds)
	}
	if report.Trials != 3 {
		t.Errorf("Trials = %d, want 3", report.Trials)
	}
	if report.InputTokens != 150000 {
		t.Errorf("InputTokens = %d, want 150000", report.InputTokens)
	}
	if report.Cost == nil {
		t.Fatal("expected cost section in JSON")
	}
	if report.Cost.TotalCost <= 0 {
		t.Errorf("expected positive cost, got %f", report.Cost.TotalCost)
	}
	if len(report.Recommend) == 0 {
		t.Error("expected recommendations in JSON output")
	}
}

func TestAnalyze_ExcludeProvider(t *testing.T) {
	tracePath := writeTestTrace(t)

	analyzeModel = "gpt-4o-mini"
	analyzeRecommend = true
	analyzeExcludeProvider = []string{"ollama", "openrouter"}
	analyzeJSON = true

	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	err := runAnalyze(analyzeCmd, []string{tracePath})

	w.Close()
	os.Stdout = old

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	buf := make([]byte, 64*1024)
	n, _ := r.Read(buf)

	var report analyzeReport
	if err := json.Unmarshal(buf[:n], &report); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	for _, r := range report.Recommend {
		if r.Provider == "ollama" || r.Provider == "openrouter" {
			t.Errorf("excluded provider %q still in recommendations", r.Provider)
		}
	}
}

func TestAnalyze_BasicOutputUnchanged(t *testing.T) {
	tracePath := writeTestTrace(t)

	// No new flags set — basic mode
	analyzeModel = ""
	analyzeRecommend = false
	analyzeExcludeProvider = nil
	analyzeJSON = false

	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	err := runAnalyze(analyzeCmd, []string{tracePath})

	w.Close()
	os.Stdout = old

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	buf := make([]byte, 64*1024)
	n, _ := r.Read(buf)
	output := string(buf[:n])

	// Should have basic output
	if !strings.Contains(output, "Trace Analysis") {
		t.Error("missing 'Trace Analysis' header")
	}
	if !strings.Contains(output, "Rounds:        1") {
		t.Error("missing rounds line")
	}
	// Should NOT have recommendation or cost sections
	if strings.Contains(output, "Model Recommendations") {
		t.Error("should not have recommendations without --recommend")
	}
	if strings.Contains(output, "Cost Estimate") {
		t.Error("should not have cost estimate without --model")
	}
}
