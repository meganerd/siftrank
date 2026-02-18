package eval

import (
	"strings"
	"testing"
)

func TestParseTrace_EmptyInput(t *testing.T) {
	_, err := ParseTrace(strings.NewReader(""))
	if err == nil {
		t.Fatal("expected error for empty trace, got nil")
	}
	if !strings.Contains(err.Error(), "empty") {
		t.Errorf("expected 'empty' in error, got %q", err.Error())
	}
}

func TestParseTrace_SingleTrialState(t *testing.T) {
	input := `{"round":1,"trial":1,"trials_completed":1,"trials_remaining":4,"total_input_tokens":500,"total_output_tokens":200}`
	summary, err := ParseTrace(strings.NewReader(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if summary.TotalRounds != 1 {
		t.Errorf("TotalRounds = %d, want 1", summary.TotalRounds)
	}
	if summary.TotalTrials != 1 {
		t.Errorf("TotalTrials = %d, want 1", summary.TotalTrials)
	}
	if summary.TotalInputTokens != 500 {
		t.Errorf("TotalInputTokens = %d, want 500", summary.TotalInputTokens)
	}
	if summary.TotalOutputTokens != 200 {
		t.Errorf("TotalOutputTokens = %d, want 200", summary.TotalOutputTokens)
	}
	if summary.Converged {
		t.Error("expected Converged=false for single trial with remaining > 0")
	}
	if summary.FinalElbow != -1 {
		t.Errorf("FinalElbow = %d, want -1 (not detected)", summary.FinalElbow)
	}
}

func TestParseTrace_MultipleTrials(t *testing.T) {
	lines := []string{
		`{"round":1,"trial":1,"trials_completed":1,"trials_remaining":4,"total_input_tokens":500,"total_output_tokens":200}`,
		`{"round":1,"trial":2,"trials_completed":2,"trials_remaining":3,"total_input_tokens":1000,"total_output_tokens":400,"elbow_position":3}`,
		`{"round":1,"trial":3,"trials_completed":3,"trials_remaining":2,"total_input_tokens":1500,"total_output_tokens":600,"elbow_position":3,"stable_trials_count":2}`,
	}
	input := strings.Join(lines, "\n")
	summary, err := ParseTrace(strings.NewReader(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if summary.TotalRounds != 1 {
		t.Errorf("TotalRounds = %d, want 1", summary.TotalRounds)
	}
	if summary.TotalTrials != 3 {
		t.Errorf("TotalTrials = %d, want 3", summary.TotalTrials)
	}
	// Token totals are cumulative — last line wins
	if summary.TotalInputTokens != 1500 {
		t.Errorf("TotalInputTokens = %d, want 1500", summary.TotalInputTokens)
	}
	if summary.TotalOutputTokens != 600 {
		t.Errorf("TotalOutputTokens = %d, want 600", summary.TotalOutputTokens)
	}
	if summary.FinalElbow != 3 {
		t.Errorf("FinalElbow = %d, want 3", summary.FinalElbow)
	}
	// stable_trials_count > 0 and trials_remaining > 0 → converged
	if !summary.Converged {
		t.Error("expected Converged=true when stable_trials_count > 0 and trials_remaining > 0")
	}
}

func TestParseTrace_ModelPerfEntries(t *testing.T) {
	lines := []string{
		`{"round":1,"trial":1,"trials_completed":1,"trials_remaining":2,"total_input_tokens":500,"total_output_tokens":200}`,
		`{"event_type":"model_perf","round":1,"trial":1,"models":[{"model_id":"gpt-4o-mini","call_count":5,"success_rate":1.0,"error_count":0,"avg_latency_ms":150,"p50_latency_ms":140,"p95_latency_ms":200,"p99_latency_ms":220,"total_tokens":700}]}`,
	}
	input := strings.Join(lines, "\n")
	summary, err := ParseTrace(strings.NewReader(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(summary.ModelStats) != 1 {
		t.Fatalf("ModelStats length = %d, want 1", len(summary.ModelStats))
	}
	stat := summary.ModelStats[0]
	if stat.ModelID != "gpt-4o-mini" {
		t.Errorf("ModelID = %q, want %q", stat.ModelID, "gpt-4o-mini")
	}
	if stat.CallCount != 5 {
		t.Errorf("CallCount = %d, want 5", stat.CallCount)
	}
	if stat.SuccessRate != 1.0 {
		t.Errorf("SuccessRate = %f, want 1.0", stat.SuccessRate)
	}
	if stat.AvgLatency != 150 {
		t.Errorf("AvgLatency = %d, want 150", stat.AvgLatency)
	}
}

func TestParseTrace_MultipleModelPerfKeepsLatest(t *testing.T) {
	lines := []string{
		`{"round":1,"trial":1,"trials_completed":1,"trials_remaining":2,"total_input_tokens":500,"total_output_tokens":200}`,
		`{"event_type":"model_perf","round":1,"trial":1,"models":[{"model_id":"gpt-4o-mini","call_count":5,"success_rate":1.0,"error_count":0,"avg_latency_ms":150,"p50_latency_ms":140,"p95_latency_ms":200,"p99_latency_ms":220,"total_tokens":700}]}`,
		`{"round":1,"trial":2,"trials_completed":2,"trials_remaining":1,"total_input_tokens":1000,"total_output_tokens":400}`,
		`{"event_type":"model_perf","round":1,"trial":2,"models":[{"model_id":"gpt-4o-mini","call_count":10,"success_rate":0.9,"error_count":1,"avg_latency_ms":160,"p50_latency_ms":150,"p95_latency_ms":210,"p99_latency_ms":230,"total_tokens":1400}]}`,
	}
	input := strings.Join(lines, "\n")
	summary, err := ParseTrace(strings.NewReader(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(summary.ModelStats) != 1 {
		t.Fatalf("ModelStats length = %d, want 1", len(summary.ModelStats))
	}
	// Latest model_perf snapshot should win (cumulative)
	stat := summary.ModelStats[0]
	if stat.CallCount != 10 {
		t.Errorf("CallCount = %d, want 10 (latest snapshot)", stat.CallCount)
	}
}

func TestParseTrace_MultiRound(t *testing.T) {
	lines := []string{
		`{"round":1,"trial":3,"trials_completed":3,"trials_remaining":0,"total_input_tokens":1500,"total_output_tokens":600,"elbow_position":5}`,
		`{"round":2,"trial":2,"trials_completed":2,"trials_remaining":1,"total_input_tokens":2500,"total_output_tokens":1000,"elbow_position":3}`,
	}
	input := strings.Join(lines, "\n")
	summary, err := ParseTrace(strings.NewReader(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if summary.TotalRounds != 2 {
		t.Errorf("TotalRounds = %d, want 2", summary.TotalRounds)
	}
	if summary.TotalTrials != 3 {
		t.Errorf("TotalTrials = %d, want 3 (max trial across all rounds)", summary.TotalTrials)
	}
	// Last line's cumulative tokens
	if summary.TotalInputTokens != 2500 {
		t.Errorf("TotalInputTokens = %d, want 2500", summary.TotalInputTokens)
	}
	// Last elbow position seen
	if summary.FinalElbow != 3 {
		t.Errorf("FinalElbow = %d, want 3", summary.FinalElbow)
	}
}

func TestParseTrace_InvalidJSON(t *testing.T) {
	input := `not valid json`
	_, err := ParseTrace(strings.NewReader(input))
	if err == nil {
		t.Fatal("expected error for invalid JSON, got nil")
	}
}

func TestParseTrace_SkipsBlankLines(t *testing.T) {
	lines := []string{
		`{"round":1,"trial":1,"trials_completed":1,"trials_remaining":2,"total_input_tokens":500,"total_output_tokens":200}`,
		``,
		`{"round":1,"trial":2,"trials_completed":2,"trials_remaining":1,"total_input_tokens":1000,"total_output_tokens":400}`,
	}
	input := strings.Join(lines, "\n")
	summary, err := ParseTrace(strings.NewReader(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if summary.TotalTrials != 2 {
		t.Errorf("TotalTrials = %d, want 2", summary.TotalTrials)
	}
}
