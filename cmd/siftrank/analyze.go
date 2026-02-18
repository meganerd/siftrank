package main

import (
	"fmt"
	"os"

	"github.com/meganerd/siftrank/pkg/siftrank"
	"github.com/meganerd/siftrank/pkg/siftrank/eval"
	"github.com/spf13/cobra"
)

var analyzeModel string

var analyzeCmd = &cobra.Command{
	Use:   "analyze <trace-file>",
	Short: "Analyze a trace.jsonl file and display performance summary",
	Long:  "Parse a siftrank trace file (JSON Lines) and display a summary of rounds, trials, token usage, convergence, and optional cost estimate.",
	Args:  cobra.ExactArgs(1),
	RunE:  runAnalyze,
}

func init() {
	analyzeCmd.Flags().StringVarP(&analyzeModel, "model", "m", "", "model name for cost estimation (uses default pricing registry)")
	// Reset usage template so subcommand doesn't inherit rootCmd's flag-group template
	analyzeCmd.SetUsageTemplate(analyzeCmd.UsageTemplate())
	rootCmd.AddCommand(analyzeCmd)
}

func runAnalyze(cmd *cobra.Command, args []string) error {
	tracePath := args[0]

	// Validate path
	validPath, err := validatePath(tracePath)
	if err != nil {
		return fmt.Errorf("invalid trace file path: %w", err)
	}

	summary, err := eval.ParseTraceFile(validPath)
	if err != nil {
		return fmt.Errorf("failed to parse trace file: %w", err)
	}

	// Print summary
	fmt.Fprintf(os.Stdout, "Trace Analysis\n")
	fmt.Fprintf(os.Stdout, "==============\n")
	fmt.Fprintf(os.Stdout, "Rounds:        %d\n", summary.TotalRounds)
	fmt.Fprintf(os.Stdout, "Trials:        %d\n", summary.TotalTrials)
	fmt.Fprintf(os.Stdout, "Input tokens:  %d\n", summary.TotalInputTokens)
	fmt.Fprintf(os.Stdout, "Output tokens: %d\n", summary.TotalOutputTokens)
	fmt.Fprintf(os.Stdout, "Total tokens:  %d\n", summary.TotalInputTokens+summary.TotalOutputTokens)

	if summary.Converged {
		fmt.Fprintf(os.Stdout, "Converged:     yes\n")
	} else {
		fmt.Fprintf(os.Stdout, "Converged:     no\n")
	}

	if summary.FinalElbow >= 0 {
		fmt.Fprintf(os.Stdout, "Elbow pos:     %d\n", summary.FinalElbow)
	}

	// Cost estimate if model provided
	if analyzeModel != "" {
		registry := siftrank.DefaultPricingRegistry()
		pricing, found := registry.Get(analyzeModel)
		if found {
			usage := siftrank.Usage{
				InputTokens:  summary.TotalInputTokens,
				OutputTokens: summary.TotalOutputTokens,
			}
			cost := siftrank.CalculateCost(usage, pricing)
			fmt.Fprintf(os.Stdout, "\nCost Estimate (%s)\n", analyzeModel)
			fmt.Fprintf(os.Stdout, "  Input:  $%.6f\n", float64(summary.TotalInputTokens)*pricing.InputPricePerToken)
			fmt.Fprintf(os.Stdout, "  Output: $%.6f\n", float64(summary.TotalOutputTokens)*pricing.OutputPricePerToken)
			fmt.Fprintf(os.Stdout, "  Total:  $%.6f\n", cost)
		} else {
			fmt.Fprintf(os.Stderr, "Warning: no pricing data for model %q, skipping cost estimate\n", analyzeModel)
		}
	}

	// Per-model stats if available (compare mode traces)
	if len(summary.ModelStats) > 0 {
		fmt.Fprintf(os.Stdout, "\nPer-Model Statistics\n")
		fmt.Fprintf(os.Stdout, "====================\n")
		for _, m := range summary.ModelStats {
			fmt.Fprintf(os.Stdout, "\n  %s\n", m.ModelID)
			fmt.Fprintf(os.Stdout, "    Calls:        %d\n", m.CallCount)
			fmt.Fprintf(os.Stdout, "    Success rate: %.1f%%\n", m.SuccessRate*100)
			fmt.Fprintf(os.Stdout, "    Errors:       %d\n", m.ErrorCount)
			fmt.Fprintf(os.Stdout, "    Avg latency:  %dms\n", m.AvgLatency)
			fmt.Fprintf(os.Stdout, "    P50 latency:  %dms\n", m.P50Latency)
			fmt.Fprintf(os.Stdout, "    P95 latency:  %dms\n", m.P95Latency)
			fmt.Fprintf(os.Stdout, "    P99 latency:  %dms\n", m.P99Latency)
			fmt.Fprintf(os.Stdout, "    Total tokens: %d\n", m.TotalTokens)
		}
	}

	return nil
}
