package siftrank

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// BatchClient wraps the OpenAI Files and Batch API for submitting batch jobs.
type BatchClient struct {
	client *openai.Client
}

// NewBatchClient creates a BatchClient using the given API key and optional base URL.
func NewBatchClient(apiKey, baseURL string) *BatchClient {
	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}
	if baseURL != "" {
		if !strings.HasSuffix(baseURL, "/") {
			baseURL += "/"
		}
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	client := openai.NewClient(opts...)
	return &BatchClient{client: &client}
}

// UploadBatchFile uploads JSONL content as a batch input file and returns the file ID.
func (bc *BatchClient) UploadBatchFile(ctx context.Context, content []byte) (string, error) {
	file, err := bc.client.Files.New(ctx, openai.FileNewParams{
		File:    bytes.NewReader(content),
		Purpose: openai.FilePurposeBatch,
	})
	if err != nil {
		return "", fmt.Errorf("failed to upload batch file: %w", err)
	}
	return file.ID, nil
}

// CreateBatch creates a batch job from an uploaded file and returns the batch ID.
func (bc *BatchClient) CreateBatch(ctx context.Context, fileID string) (string, error) {
	batch, err := bc.client.Batches.New(ctx, openai.BatchNewParams{
		InputFileID:      fileID,
		Endpoint:         openai.BatchNewParamsEndpointV1ChatCompletions,
		CompletionWindow: openai.BatchNewParamsCompletionWindow24h,
	})
	if err != nil {
		return "", fmt.Errorf("failed to create batch: %w", err)
	}
	return batch.ID, nil
}

// GetBatch retrieves the current state of a batch.
func (bc *BatchClient) GetBatch(ctx context.Context, batchID string) (*openai.Batch, error) {
	batch, err := bc.client.Batches.Get(ctx, batchID)
	if err != nil {
		return nil, fmt.Errorf("failed to get batch: %w", err)
	}
	return batch, nil
}

// DownloadResults downloads the output file for a completed batch and returns the raw content.
func (bc *BatchClient) DownloadResults(ctx context.Context, outputFileID string) ([]byte, error) {
	resp, err := bc.client.Files.Content(ctx, outputFileID)
	if err != nil {
		return nil, fmt.Errorf("failed to download batch results: %w", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read batch results: %w", err)
	}
	return data, nil
}

// GenerateRankingSchema returns the JSON schema used for structured ranking responses.
// This is the same schema used by the real-time ranking path (round 1, no relevance).
func GenerateRankingSchema() interface{} {
	return generateSchema[rankedDocumentResponseNoRelevance]()
}

// BatchRequestLine represents a single request in the OpenAI Batch API JSONL format.
type BatchRequestLine struct {
	CustomID string                `json:"custom_id"`
	Method   string                `json:"method"`
	URL      string                `json:"url"`
	Body     BatchRequestLineBody  `json:"body"`
}

// BatchRequestLineBody is the chat completion request body within a batch request line.
type BatchRequestLineBody struct {
	Model          string                  `json:"model"`
	Messages       []BatchMessage          `json:"messages"`
	ResponseFormat *BatchResponseFormat     `json:"response_format,omitempty"`
}

// BatchMessage represents a message in the batch request.
type BatchMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// BatchResponseFormat requests structured JSON output.
type BatchResponseFormat struct {
	Type       string              `json:"type"`
	JSONSchema *BatchJSONSchema    `json:"json_schema,omitempty"`
}

// BatchJSONSchema describes the expected JSON schema for structured output.
type BatchJSONSchema struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Schema      interface{} `json:"schema"`
	Strict      bool        `json:"strict"`
}

// BatchDocInfo stores document metadata for correlating batch results back to input.
type BatchDocInfo struct {
	ID         string      `json:"id"`
	Value      string      `json:"value"`
	Document   interface{} `json:"document,omitempty"`
	InputIndex int         `json:"input_index"`
}

// BatchMapping is the sidecar file that connects a batch job to its input documents.
type BatchMapping struct {
	BatchID   string                  `json:"batch_id"`
	FileID    string                  `json:"file_id"`
	Model     string                  `json:"model"`
	Prompt    string                  `json:"prompt"`
	BatchSize int                     `json:"batch_size"`
	Documents map[string][]BatchDocInfo `json:"documents"` // custom_id -> docs in that batch
}

// GenerateBatchInput creates JSONL content for the OpenAI Batch API from a set of documents.
// It splits documents into batches and generates one batch request line per batch.
// Returns the JSONL bytes and a mapping of custom_id to document info for result correlation.
func GenerateBatchInput(docs []BatchDocInfo, prompt, model string, batchSize int, schema interface{}) ([]byte, map[string][]BatchDocInfo) {
	if batchSize <= 0 {
		batchSize = DefaultBatchSize
	}

	var buf bytes.Buffer
	mapping := make(map[string][]BatchDocInfo)

	// Split documents into batches
	for batchNum := 0; batchNum*batchSize < len(docs); batchNum++ {
		start := batchNum * batchSize
		end := start + batchSize
		if end > len(docs) {
			end = len(docs)
		}
		batch := docs[start:end]

		// Build the prompt for this batch (same format as siftrank real-time)
		fullPrompt := prompt + "\n\nREMEMBER to:\n" +
			"- ALWAYS respond with the short 6-8 character ID of each item found above the value " +
			"(i.e., I'll provide you with `id: <ID>` above the value, and you should respond with that same ID in your response)\n" +
			"— NEVER respond with the actual value!\n" +
			"— NEVER include backticks around IDs in your response!\n" +
			"— NEVER include scores or a written reason/justification in your response!\n"

		for _, doc := range batch {
			fullPrompt += fmt.Sprintf("id: `%s`\nvalue:\n```\n%s\n```\n\n", doc.ID, doc.Value)
		}

		customID := fmt.Sprintf("batch-%d", batchNum)
		mapping[customID] = batch

		line := BatchRequestLine{
			CustomID: customID,
			Method:   "POST",
			URL:      "/v1/chat/completions",
			Body: BatchRequestLineBody{
				Model: model,
				Messages: []BatchMessage{
					{Role: "user", Content: fullPrompt},
				},
			},
		}

		// Add structured output if schema provided
		if schema != nil {
			line.Body.ResponseFormat = &BatchResponseFormat{
				Type: "json_schema",
				JSONSchema: &BatchJSONSchema{
					Name:        "structured_response",
					Description: "Structured JSON response",
					Schema:      schema,
					Strict:      true,
				},
			}
		}

		lineJSON, _ := json.Marshal(line)
		buf.Write(lineJSON)
		buf.WriteByte('\n')
	}

	return buf.Bytes(), mapping
}

// BatchResponseLine represents a single response line from the OpenAI Batch API output.
type BatchResponseLine struct {
	ID       string                    `json:"id"`
	CustomID string                    `json:"custom_id"`
	Response BatchResponseLineResponse `json:"response"`
	Error    *BatchResponseLineError   `json:"error"`
}

// BatchResponseLineResponse is the HTTP response within a batch result line.
type BatchResponseLineResponse struct {
	StatusCode int             `json:"status_code"`
	Body       json.RawMessage `json:"body"`
}

// BatchResponseLineError represents an error in a batch result line.
type BatchResponseLineError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// BatchCompletionBody is the chat completion response body within batch results.
type BatchCompletionBody struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// ParseBatchResults parses batch API output JSONL and correlates results with the mapping.
// Returns a map of custom_id to the parsed ranking response content.
func ParseBatchResults(output []byte) (map[string]string, error) {
	results := make(map[string]string)
	dec := json.NewDecoder(bytes.NewReader(output))

	for dec.More() {
		var line BatchResponseLine
		if err := dec.Decode(&line); err != nil {
			return nil, fmt.Errorf("failed to parse batch result line: %w", err)
		}

		if line.Error != nil {
			continue // Skip failed requests
		}

		if line.Response.StatusCode != 200 {
			continue // Skip non-200 responses
		}

		var body BatchCompletionBody
		if err := json.Unmarshal(line.Response.Body, &body); err != nil {
			continue // Skip unparseable responses
		}

		if len(body.Choices) > 0 {
			results[line.CustomID] = body.Choices[0].Message.Content
		}
	}

	return results, nil
}
