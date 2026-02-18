package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/meganerd/siftrank/pkg/siftrank"
	"github.com/spf13/cobra"
)

var (
	batchSubModel     string
	batchSubProvider  string
	batchSubFile      string
	batchSubPrompt    string
	batchSubSize      int
	batchSubOutputDir string
)

var batchCmd = &cobra.Command{
	Use:   "batch",
	Short: "Submit and manage OpenAI Batch API jobs for cost-effective ranking",
	Long: `Use the OpenAI Batch API to rank documents at 50% lower cost.

Batch mode submits all ranking requests as a single batch job that completes
within 24 hours. Use this for large, cost-sensitive jobs where real-time
results are not required.

Note: Batch mode only supports the OpenAI provider.`,
}

var batchSubmitCmd = &cobra.Command{
	Use:   "submit",
	Short: "Submit a batch ranking job",
	Long: `Generate batch requests from input documents and submit to the OpenAI Batch API.

Saves a mapping file (.siftrank-batch.json) in the output directory for
correlating results later.`,
	RunE: runBatchSubmit,
}

var batchStatusCmd = &cobra.Command{
	Use:   "status <batch-id>",
	Short: "Check the status of a batch job",
	Args:  cobra.ExactArgs(1),
	RunE:  runBatchStatus,
}

var batchResultsCmd = &cobra.Command{
	Use:   "results <mapping-file>",
	Short: "Download and process results from a completed batch job",
	Args:  cobra.ExactArgs(1),
	RunE:  runBatchResults,
}

func init() {
	// Submit flags
	batchSubmitCmd.Flags().StringVarP(&batchSubModel, "model", "m", "gpt-4o-mini", "model name")
	batchSubmitCmd.Flags().StringVar(&batchSubProvider, "provider", "openai", "LLM provider (only openai supported)")
	batchSubmitCmd.Flags().StringVarP(&batchSubFile, "file", "f", "", "input file (required)")
	batchSubmitCmd.Flags().StringVarP(&batchSubPrompt, "prompt", "p", "", "ranking prompt (required)")
	batchSubmitCmd.Flags().IntVarP(&batchSubSize, "batch-size", "b", siftrank.DefaultBatchSize, "documents per batch request")
	batchSubmitCmd.Flags().StringVarP(&batchSubOutputDir, "output-dir", "o", ".", "directory for mapping file output")
	if err := batchSubmitCmd.MarkFlagRequired("file"); err != nil {
		panic(fmt.Sprintf("failed to mark flag: %v", err))
	}
	if err := batchSubmitCmd.MarkFlagRequired("prompt"); err != nil {
		panic(fmt.Sprintf("failed to mark flag: %v", err))
	}

	// Status flags — uses global --api-key or env var
	// Results flags — uses global --api-key or env var

	batchCmd.AddCommand(batchSubmitCmd)
	batchCmd.AddCommand(batchStatusCmd)
	batchCmd.AddCommand(batchResultsCmd)

	// Reset usage template so subcommand doesn't inherit rootCmd's flag-group template
	batchCmd.SetUsageTemplate(batchCmd.UsageTemplate())
	batchSubmitCmd.SetUsageTemplate(batchSubmitCmd.UsageTemplate())
	batchStatusCmd.SetUsageTemplate(batchStatusCmd.UsageTemplate())
	batchResultsCmd.SetUsageTemplate(batchResultsCmd.UsageTemplate())

	rootCmd.AddCommand(batchCmd)
}

func runBatchSubmit(cmd *cobra.Command, args []string) error {
	// Validate provider
	if batchSubProvider != "openai" {
		return fmt.Errorf("batch mode only supports the openai provider (got %q)", batchSubProvider)
	}

	// Resolve API key
	apiKey := resolveAPIKey(siftrank.ProviderTypeOpenAI)
	if apiKey == "" {
		return fmt.Errorf("OpenAI API key required: set OPENAI_API_KEY or use config file")
	}

	// Load prompt from file if prefixed with @
	prompt := batchSubPrompt
	if strings.HasPrefix(prompt, "@") {
		filePath := strings.TrimPrefix(prompt, "@")
		validPath, err := validatePath(filePath)
		if err != nil {
			return fmt.Errorf("invalid prompt file path: %w", err)
		}
		content, err := os.ReadFile(validPath) // #nosec G304 - validated path
		if err != nil {
			return fmt.Errorf("could not read prompt file: %w", err)
		}
		prompt = string(content)
	}

	// Load documents from input file
	validPath, err := validatePath(batchSubFile)
	if err != nil {
		return fmt.Errorf("invalid input file path: %w", err)
	}

	docs, err := loadBatchDocuments(validPath)
	if err != nil {
		return fmt.Errorf("failed to load documents: %w", err)
	}

	if len(docs) == 0 {
		return fmt.Errorf("no documents found in input file")
	}

	fmt.Fprintf(os.Stderr, "Loaded %d documents from %s\n", len(docs), batchSubFile)

	// Generate batch JSONL
	schema := siftrank.GenerateRankingSchema()
	jsonlContent, docMapping := siftrank.GenerateBatchInput(docs, prompt, batchSubModel, batchSubSize, schema)

	fmt.Fprintf(os.Stderr, "Generated %d batch requests\n", len(docMapping))

	// Upload and create batch
	client := siftrank.NewBatchClient(apiKey, appConfig.BaseURL)
	ctx := context.Background()

	fileID, err := client.UploadBatchFile(ctx, jsonlContent)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "Uploaded batch file: %s\n", fileID)

	batchID, err := client.CreateBatch(ctx, fileID)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "Created batch: %s\n", batchID)

	// Save mapping file
	mapping := siftrank.BatchMapping{
		BatchID:   batchID,
		FileID:    fileID,
		Model:     batchSubModel,
		Prompt:    prompt,
		BatchSize: batchSubSize,
		Documents: docMapping,
	}

	validOutDir, err := validatePath(batchSubOutputDir)
	if err != nil {
		// Output dir might not exist as a file, use as-is
		validOutDir = batchSubOutputDir
	}
	mappingPath := validOutDir
	info, statErr := os.Stat(mappingPath)
	if statErr == nil && info.IsDir() {
		mappingPath = mappingPath + "/.siftrank-batch.json"
	}

	mappingData, err := json.MarshalIndent(mapping, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal mapping: %w", err)
	}
	if err := os.WriteFile(mappingPath, mappingData, 0600); err != nil {
		return fmt.Errorf("failed to write mapping file: %w", err)
	}

	fmt.Fprintf(os.Stdout, "Batch submitted successfully!\n")
	fmt.Fprintf(os.Stdout, "  Batch ID:     %s\n", batchID)
	fmt.Fprintf(os.Stdout, "  Mapping file: %s\n", mappingPath)
	fmt.Fprintf(os.Stdout, "\nCheck status:\n")
	fmt.Fprintf(os.Stdout, "  siftrank batch status %s\n", batchID)
	fmt.Fprintf(os.Stdout, "\nGet results when complete:\n")
	fmt.Fprintf(os.Stdout, "  siftrank batch results %s\n", mappingPath)

	return nil
}

func runBatchStatus(cmd *cobra.Command, args []string) error {
	batchID := args[0]

	apiKey := resolveAPIKey(siftrank.ProviderTypeOpenAI)
	if apiKey == "" {
		return fmt.Errorf("OpenAI API key required: set OPENAI_API_KEY or use config file")
	}

	client := siftrank.NewBatchClient(apiKey, appConfig.BaseURL)
	ctx := context.Background()

	batch, err := client.GetBatch(ctx, batchID)
	if err != nil {
		return err
	}

	fmt.Fprintf(os.Stdout, "Batch Status\n")
	fmt.Fprintf(os.Stdout, "============\n")
	fmt.Fprintf(os.Stdout, "ID:        %s\n", batch.ID)
	fmt.Fprintf(os.Stdout, "Status:    %s\n", batch.Status)
	fmt.Fprintf(os.Stdout, "Endpoint:  %s\n", batch.Endpoint)

	fmt.Fprintf(os.Stdout, "\nRequest Counts\n")
	fmt.Fprintf(os.Stdout, "  Total:     %d\n", batch.RequestCounts.Total)
	fmt.Fprintf(os.Stdout, "  Completed: %d\n", batch.RequestCounts.Completed)
	fmt.Fprintf(os.Stdout, "  Failed:    %d\n", batch.RequestCounts.Failed)

	if batch.CreatedAt > 0 {
		fmt.Fprintf(os.Stdout, "\nTimestamps\n")
		fmt.Fprintf(os.Stdout, "  Created:   %s\n", time.Unix(batch.CreatedAt, 0).Format(time.RFC3339))
	}
	if batch.InProgressAt > 0 {
		fmt.Fprintf(os.Stdout, "  Started:   %s\n", time.Unix(batch.InProgressAt, 0).Format(time.RFC3339))
	}
	if batch.CompletedAt > 0 {
		fmt.Fprintf(os.Stdout, "  Completed: %s\n", time.Unix(batch.CompletedAt, 0).Format(time.RFC3339))
	}
	if batch.ExpiresAt > 0 {
		fmt.Fprintf(os.Stdout, "  Expires:   %s\n", time.Unix(batch.ExpiresAt, 0).Format(time.RFC3339))
	}

	if batch.OutputFileID != "" {
		fmt.Fprintf(os.Stdout, "\nOutput file: %s\n", batch.OutputFileID)
	}
	if batch.ErrorFileID != "" {
		fmt.Fprintf(os.Stdout, "Error file:  %s\n", batch.ErrorFileID)
	}

	return nil
}

func runBatchResults(cmd *cobra.Command, args []string) error {
	mappingPath := args[0]

	// Load mapping file
	validPath, err := validatePath(mappingPath)
	if err != nil {
		return fmt.Errorf("invalid mapping file path: %w", err)
	}
	mappingData, err := os.ReadFile(validPath) // #nosec G304 - validated path
	if err != nil {
		return fmt.Errorf("failed to read mapping file: %w", err)
	}

	var mapping siftrank.BatchMapping
	if err := json.Unmarshal(mappingData, &mapping); err != nil {
		return fmt.Errorf("failed to parse mapping file: %w", err)
	}

	// Resolve API key
	apiKey := resolveAPIKey(siftrank.ProviderTypeOpenAI)
	if apiKey == "" {
		return fmt.Errorf("OpenAI API key required: set OPENAI_API_KEY or use config file")
	}

	client := siftrank.NewBatchClient(apiKey, appConfig.BaseURL)
	ctx := context.Background()

	// Check batch status
	batch, err := client.GetBatch(ctx, mapping.BatchID)
	if err != nil {
		return err
	}

	if batch.Status != "completed" {
		return fmt.Errorf("batch is not complete (status: %s)", batch.Status)
	}

	if batch.OutputFileID == "" {
		return fmt.Errorf("batch has no output file")
	}

	// Download results
	fmt.Fprintf(os.Stderr, "Downloading results from batch %s...\n", mapping.BatchID)
	output, err := client.DownloadResults(ctx, batch.OutputFileID)
	if err != nil {
		return err
	}

	// Parse batch results
	results, err := siftrank.ParseBatchResults(output)
	if err != nil {
		return err
	}

	fmt.Fprintf(os.Stderr, "Processing %d batch responses...\n", len(results))

	// Score documents across all batches
	scores := make(map[string][]float64)
	docInfo := make(map[string]siftrank.BatchDocInfo)

	// Build doc lookup
	for _, docs := range mapping.Documents {
		for _, doc := range docs {
			docInfo[doc.ID] = doc
		}
	}

	// Process each batch result
	for customID, content := range results {
		batchDocs, ok := mapping.Documents[customID]
		if !ok {
			continue
		}

		// Parse the ranking response
		var resp struct {
			Documents []string `json:"docs"`
		}
		if err := json.Unmarshal([]byte(content), &resp); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: failed to parse response for %s: %v\n", customID, err)
			continue
		}

		// Assign positional scores (lower = better, like siftrank)
		for i, docID := range resp.Documents {
			// Find the actual doc ID (might use memorable IDs)
			found := false
			for _, bDoc := range batchDocs {
				if bDoc.ID == docID {
					scores[docID] = append(scores[docID], float64(i+1))
					found = true
					break
				}
			}
			if !found {
				// Try as-is (in case IDs match directly)
				scores[docID] = append(scores[docID], float64(i+1))
			}
		}
	}

	// Build ranked documents
	type scoredDoc struct {
		id       string
		avgScore float64
	}
	var ranked []scoredDoc
	for id, docScores := range scores {
		sum := 0.0
		for _, s := range docScores {
			sum += s
		}
		ranked = append(ranked, scoredDoc{id: id, avgScore: sum / float64(len(docScores))})
	}

	// Sort by average score (lower is better)
	for i := 0; i < len(ranked); i++ {
		for j := i + 1; j < len(ranked); j++ {
			if ranked[j].avgScore < ranked[i].avgScore {
				ranked[i], ranked[j] = ranked[j], ranked[i]
			}
		}
	}

	// Produce output in same format as siftrank
	var output_docs []siftrank.RankedDocument
	for i, r := range ranked {
		info, ok := docInfo[r.id]
		if !ok {
			continue
		}
		output_docs = append(output_docs, siftrank.RankedDocument{
			Key:        info.ID,
			Value:      info.Value,
			Document:   info.Document,
			Score:      r.avgScore,
			Rank:       i + 1,
			InputIndex: info.InputIndex,
		})
	}

	// Output JSON
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(output_docs)
}

// loadBatchDocuments reads an input file and returns BatchDocInfo entries.
// Supports text files (one document per line).
func loadBatchDocuments(path string) ([]siftrank.BatchDocInfo, error) {
	f, err := os.Open(path) // #nosec G304 - path already validated by caller
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var docs []siftrank.BatchDocInfo
	scanner := bufio.NewScanner(f)

	// Increase scanner buffer for large lines
	const maxLineSize = 1024 * 1024 // 1MB
	scanner.Buffer(make([]byte, maxLineSize), maxLineSize)

	idx := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		id := siftrank.ShortDeterministicID(line, 8)
		docs = append(docs, siftrank.BatchDocInfo{
			ID:         id,
			Value:      line,
			InputIndex: idx,
		})
		idx++
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	return docs, nil
}
