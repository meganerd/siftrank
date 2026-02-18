package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/meganerd/siftrank/pkg/siftrank"
	"github.com/meganerd/siftrank/pkg/siftrank/eval"
	"github.com/spf13/cobra"
)

var (
	analyzeModel           string
	analyzeRecommend       bool
	analyzeExcludeProvider []string
	analyzeJSON            bool
)

var analyzeCmd = &cobra.Command{
	Use:   "analyze <trace-file>",
	Short: "Analyze a trace.jsonl file and display performance summary",
	Long:  "Parse a siftrank trace file (JSON Lines) and display a summary of rounds, trials, token usage, convergence, optional cost estimate, and model recommendations.",
	Args:  cobra.ExactArgs(1),
	RunE:  runAnalyze,
}

func init() {
	analyzeCmd.Flags().StringVarP(&analyzeModel, "model", "m", "", "model name for cost estimation and recommendations")
	analyzeCmd.Flags().BoolVar(&analyzeRecommend, "recommend", false, "show model recommendations based on trace workload (requires --model)")
	analyzeCmd.Flags().StringSliceVar(&analyzeExcludeProvider, "exclude-provider", nil, "exclude providers from recommendations (repeatable)")
	analyzeCmd.Flags().BoolVar(&analyzeJSON, "json", false, "output report as JSON")
	// Reset usage template so subcommand doesn't inherit rootCmd's flag-group template
	analyzeCmd.SetUsageTemplate(analyzeCmd.UsageTemplate())
	rootCmd.AddCommand(analyzeCmd)
}

// analyzeReport is the JSON-serializable report structure.
type analyzeReport struct {
	Rounds       int                    `json:"rounds"`
	Trials       int                    `json:"trials"`
	InputTokens  int                    `json:"input_tokens"`
	OutputTokens int                    `json:"output_tokens"`
	TotalTokens  int                    `json:"total_tokens"`
	Converged    bool                   `json:"converged"`
	ElbowPos     int                    `json:"elbow_position"`
	Cost         *analyzeCostReport     `json:"cost,omitempty"`
	ModelStats   []eval.ModelPerfDetail `json:"model_stats,omitempty"`
	Recommend    []analyzeRecReport     `json:"recommendations,omitempty"`
}

type analyzeCostReport struct {
	Model      string  `json:"model"`
	InputCost  float64 `json:"input_cost"`
	OutputCost float64 `json:"output_cost"`
	TotalCost  float64 `json:"total_cost"`
}

type analyzeRecReport struct {
	ModelID       string  `json:"model_id"`
	Provider      string  `json:"provider"`
	EstimatedCost float64 `json:"estimated_cost"`
	CostSavings   float64 `json:"cost_savings"`
	Rationale     string  `json:"rationale"`
}

func runAnalyze(cmd *cobra.Command, args []string) error {
	tracePath := args[0]

	// Validate flag dependencies
	if analyzeRecommend && analyzeModel == "" {
		return fmt.Errorf("--recommend requires --model to specify the baseline model")
	}

	// Validate path
	validPath, err := validatePath(tracePath)
	if err != nil {
		return fmt.Errorf("invalid trace file path: %w", err)
	}

	summary, err := eval.ParseTraceFile(validPath)
	if err != nil {
		return fmt.Errorf("failed to parse trace file: %w", err)
	}

	// Build report data
	report := analyzeReport{
		Rounds:       summary.TotalRounds,
		Trials:       summary.TotalTrials,
		InputTokens:  summary.TotalInputTokens,
		OutputTokens: summary.TotalOutputTokens,
		TotalTokens:  summary.TotalInputTokens + summary.TotalOutputTokens,
		Converged:    summary.Converged,
		ElbowPos:     summary.FinalElbow,
		ModelStats:   summary.ModelStats,
	}

	// Cost estimate
	catalog := siftrank.DefaultModelCatalog()
	if analyzeModel != "" {
		registry := catalog.PricingRegistry()
		pricing, found := registry.Get(analyzeModel)
		if found {
			usage := siftrank.Usage{
				InputTokens:  summary.TotalInputTokens,
				OutputTokens: summary.TotalOutputTokens,
			}
			report.Cost = &analyzeCostReport{
				Model:      analyzeModel,
				InputCost:  float64(summary.TotalInputTokens) * pricing.InputPricePerToken,
				OutputCost: float64(summary.TotalOutputTokens) * pricing.OutputPricePerToken,
				TotalCost:  siftrank.CalculateCost(usage, pricing),
			}
		} else if !analyzeJSON {
			fmt.Fprintf(os.Stderr, "Warning: no pricing data for model %q, skipping cost estimate\n", analyzeModel)
		}
	}

	// Recommendations
	if analyzeRecommend {
		var opts *siftrank.RecommendOptions
		if len(analyzeExcludeProvider) > 0 {
			excluded := make([]siftrank.ProviderType, len(analyzeExcludeProvider))
			for i, p := range analyzeExcludeProvider {
				excluded[i] = siftrank.ProviderType(p)
			}
			opts = &siftrank.RecommendOptions{ExcludeProviders: excluded}
		}

		recs := siftrank.Recommend(
			summary.TotalInputTokens,
			summary.TotalOutputTokens,
			analyzeModel,
			catalog,
			opts,
		)

		for _, r := range recs {
			report.Recommend = append(report.Recommend, analyzeRecReport{
				ModelID:       r.ModelID,
				Provider:      string(r.Provider),
				EstimatedCost: r.EstimatedCost,
				CostSavings:   r.CostSavings,
				Rationale:     r.Rationale,
			})
		}
	}

	// Output
	if analyzeJSON {
		return outputJSON(report)
	}
	return outputText(report, summary)
}

func outputJSON(report analyzeReport) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}

func outputText(report analyzeReport, summary *eval.TraceSummary) error {
	fmt.Fprintf(os.Stdout, "Trace Analysis\n")
	fmt.Fprintf(os.Stdout, "==============\n")
	fmt.Fprintf(os.Stdout, "Rounds:        %d\n", report.Rounds)
	fmt.Fprintf(os.Stdout, "Trials:        %d\n", report.Trials)
	fmt.Fprintf(os.Stdout, "Input tokens:  %d\n", report.InputTokens)
	fmt.Fprintf(os.Stdout, "Output tokens: %d\n", report.OutputTokens)
	fmt.Fprintf(os.Stdout, "Total tokens:  %d\n", report.TotalTokens)

	if report.Converged {
		fmt.Fprintf(os.Stdout, "Converged:     yes\n")
	} else {
		fmt.Fprintf(os.Stdout, "Converged:     no\n")
	}

	if report.ElbowPos >= 0 {
		fmt.Fprintf(os.Stdout, "Elbow pos:     %d\n", report.ElbowPos)
	}

	if report.Cost != nil {
		fmt.Fprintf(os.Stdout, "\nCost Estimate (%s)\n", report.Cost.Model)
		fmt.Fprintf(os.Stdout, "  Input:  $%.6f\n", report.Cost.InputCost)
		fmt.Fprintf(os.Stdout, "  Output: $%.6f\n", report.Cost.OutputCost)
		fmt.Fprintf(os.Stdout, "  Total:  $%.6f\n", report.Cost.TotalCost)
	}

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

	if len(report.Recommend) > 0 {
		fmt.Fprintf(os.Stdout, "\nModel Recommendations\n")
		fmt.Fprintf(os.Stdout, "=====================\n")
		for i, r := range report.Recommend {
			fmt.Fprintf(os.Stdout, "  %d. %s\n", i+1, r.Rationale)
		}
	}

	return nil
}
