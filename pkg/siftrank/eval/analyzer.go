package eval

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// TraceEntry represents a single trial state line from a trace.jsonl file.
type TraceEntry struct {
	Round             int  `json:"round"`
	Trial             int  `json:"trial"`
	TrialsCompleted   int  `json:"trials_completed"`
	TrialsRemaining   int  `json:"trials_remaining"`
	TotalInputTokens  int  `json:"total_input_tokens"`
	TotalOutputTokens int  `json:"total_output_tokens"`
	ElbowPosition     *int `json:"elbow_position,omitempty"`
	StableTrialsCount int  `json:"stable_trials_count,omitempty"`
}

// ModelPerfEntry represents a model performance event from a trace.jsonl file.
type ModelPerfEntry struct {
	EventType string            `json:"event_type"`
	Round     int               `json:"round"`
	Trial     int               `json:"trial"`
	Models    []ModelPerfDetail `json:"models"`
}

// ModelPerfDetail contains per-model statistics from a trace event.
type ModelPerfDetail struct {
	ModelID     string  `json:"model_id"`
	CallCount   int     `json:"call_count"`
	SuccessRate float64 `json:"success_rate"`
	ErrorCount  int     `json:"error_count"`
	AvgLatency  int64   `json:"avg_latency_ms"`
	P50Latency  int64   `json:"p50_latency_ms"`
	P95Latency  int64   `json:"p95_latency_ms"`
	P99Latency  int64   `json:"p99_latency_ms"`
	TotalTokens int     `json:"total_tokens"`
}

// TraceSummary contains aggregated analysis of a trace file.
type TraceSummary struct {
	TotalRounds       int
	TotalTrials       int
	TotalInputTokens  int
	TotalOutputTokens int
	Converged         bool
	FinalElbow        int // -1 if not detected
	ModelStats        []ModelPerfDetail
}

// ParseTraceFile reads and parses a trace.jsonl file, returning a TraceSummary.
func ParseTraceFile(path string) (*TraceSummary, error) {
	f, err := os.Open(path) // #nosec G304 - user-provided path for analysis tool
	if err != nil {
		return nil, fmt.Errorf("failed to open trace file: %w", err)
	}
	defer f.Close()

	return ParseTrace(f)
}

// ParseTrace reads and parses trace JSONL from a reader.
func ParseTrace(r io.Reader) (*TraceSummary, error) {
	scanner := bufio.NewScanner(r)
	// Increase buffer size for large trace lines (rankings can be big)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)

	var (
		maxRound       int
		maxTrial       int
		lastInputToks  int
		lastOutputToks int
		lastElbow      = -1
		converged      bool
		latestPerf     []ModelPerfDetail
		lineCount      int
	)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		lineCount++

		// Peek at event_type to determine which struct to unmarshal into
		var peek struct {
			EventType string `json:"event_type"`
		}
		if err := json.Unmarshal(line, &peek); err != nil {
			return nil, fmt.Errorf("line %d: failed to parse JSON: %w", lineCount, err)
		}

		if peek.EventType == "model_perf" {
			var entry ModelPerfEntry
			if err := json.Unmarshal(line, &entry); err != nil {
				return nil, fmt.Errorf("line %d: failed to parse model_perf event: %w", lineCount, err)
			}
			// Keep the latest model perf snapshot (it's cumulative)
			latestPerf = entry.Models
		} else {
			var entry TraceEntry
			if err := json.Unmarshal(line, &entry); err != nil {
				return nil, fmt.Errorf("line %d: failed to parse trace entry: %w", lineCount, err)
			}

			if entry.Round > maxRound {
				maxRound = entry.Round
			}
			if entry.Trial > maxTrial {
				maxTrial = entry.Trial
			}

			// Token totals are cumulative within a round — use the latest values
			lastInputToks = entry.TotalInputTokens
			lastOutputToks = entry.TotalOutputTokens

			if entry.ElbowPosition != nil {
				lastElbow = *entry.ElbowPosition
			}
			if entry.StableTrialsCount > 0 && entry.TrialsRemaining > 0 {
				converged = true
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading trace: %w", err)
	}

	if lineCount == 0 {
		return nil, fmt.Errorf("trace file is empty")
	}

	return &TraceSummary{
		TotalRounds:       maxRound,
		TotalTrials:       maxTrial,
		TotalInputTokens:  lastInputToks,
		TotalOutputTokens: lastOutputToks,
		Converged:         converged,
		FinalElbow:        lastElbow,
		ModelStats:        latestPerf,
	}, nil
}
