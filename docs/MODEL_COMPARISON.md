# Model Comparison Guide

The `--compare` flag enables head-to-head performance evaluation of multiple LLM models during ranking operations.

## Overview

Model comparison mode:
- Rotates between specified models in round-robin fashion
- Collects performance metrics for each model (latency, tokens, success rate)
- Writes detailed statistics to trace file for analysis
- Zero impact on ranking quality (same prompts sent to all models)

## Usage

### Basic Syntax

```bash
siftrank --file documents.txt --prompt "Rank by relevance" \
  --compare "provider:model,provider:model" \
  --trace comparison.jsonl
```

### Supported Providers

- **openai**: OpenAI models (requires `OPENAI_API_KEY`)
- **ollama**: Local Ollama models (requires `OLLAMA_BASE_URL`, defaults to `http://localhost:11434`)
- **openrouter**: OpenRouter models (requires `OPENROUTER_API_KEY`)

### Example Comparisons

**Compare OpenAI models:**
```bash
siftrank --file docs.txt --prompt "Rank by quality" \
  --compare "openai:gpt-4o-mini,openai:gpt-4o" \
  --trace comparison.jsonl
```

**Compare OpenAI vs Ollama:**
```bash
export OPENAI_API_KEY="your-key"
export OLLAMA_BASE_URL="http://localhost:11434"

siftrank --file docs.txt --prompt "Rank by relevance" \
  --compare "openai:gpt-4o-mini,ollama:qwen2.5-coder:32b" \
  --trace comparison.jsonl
```

**Three-way comparison:**
```bash
siftrank --file docs.txt --prompt "Rank technical docs" \
  --compare "openai:gpt-4o-mini,ollama:qwen2.5-coder:32b,ollama:llama3.2:latest" \
  --trace comparison.jsonl
```

## Trace File Format

When `--compare` is enabled, the trace file contains two types of events:

### 1. Standard Trial Events
```json
{
  "round": 1,
  "trial": 5,
  "trials_completed": 5,
  "trials_remaining": 45,
  "total_input_tokens": 12500,
  "total_output_tokens": 850,
  "rankings": [...]
}
```

### 2. Model Performance Events (NEW)
```json
{
  "event_type": "model_perf",
  "round": 1,
  "trial": 5,
  "models": [
    {
      "model_id": "openai:gpt-4o-mini",
      "call_count": 25,
      "success_rate": 1.0,
      "error_count": 0,
      "avg_latency_ms": 245,
      "p50_latency_ms": 230,
      "p95_latency_ms": 320,
      "p99_latency_ms": 380,
      "total_tokens": 6800
    },
    {
      "model_id": "ollama:qwen2.5-coder:32b",
      "call_count": 25,
      "success_rate": 1.0,
      "error_count": 0,
      "avg_latency_ms": 1250,
      "p50_latency_ms": 1200,
      "p95_latency_ms": 1450,
      "p99_latency_ms": 1580,
      "total_tokens": 7200
    }
  ]
}
```

## Performance Metrics

For each model, the following metrics are collected:

| Metric | Description |
|--------|-------------|
| `call_count` | Total number of API calls made |
| `success_rate` | Ratio of successful calls (0.0-1.0) |
| `error_count` | Number of failed calls |
| `avg_latency_ms` | Mean end-to-end latency in milliseconds |
| `p50_latency_ms` | Median latency (50th percentile) |
| `p95_latency_ms` | 95th percentile latency |
| `p99_latency_ms` | 99th percentile latency |
| `total_tokens` | Sum of input + output tokens consumed |

## Analyzing Results

### Extract Model Performance Stats

```bash
# Get final performance summary for each model
jq 'select(.event_type == "model_perf") | .models[] | select(.call_count > 0) | {model_id, call_count, success_rate, avg_latency_ms, p95_latency_ms, total_tokens}' comparison.jsonl | jq -s 'group_by(.model_id) | map({model: .[0].model_id, final_stats: .[-1]})'
```

### Calculate Cost Comparison

If you know your provider's pricing:

```bash
# Example pricing: OpenAI $0.15/1M tokens, Ollama free
jq 'select(.event_type == "model_perf") | .models[] | select(.model_id | startswith("openai")) | .total_tokens' comparison.jsonl | jq -s 'add' | awk '{print "OpenAI cost: $" $1 * 0.15 / 1000000}'
```

### Latency Distribution Comparison

```bash
# Compare P95 latencies
jq 'select(.event_type == "model_perf") | .models[] | {model: .model_id, p95: .p95_latency_ms}' comparison.jsonl | jq -s 'group_by(.model) | map({model: .[0].model, p95_latency: (map(.p95) | add / length)})'
```

## Architecture

The comparison system uses the **Decorator pattern**:

1. **EvalProvider** wraps the LLMProvider interface
2. **MetricsCollector** records call data (thread-safe)
3. **SessionAggregator** computes statistics (percentiles, averages)
4. **ProviderSelector** implements round-robin model rotation

### Performance Overhead

Metrics collection adds **<1ms overhead** per call (measured via benchmarks):
- Thread-safe metric recording: ~263 nanoseconds
- Zero impact on model responses or ranking quality
- Negligible memory footprint (metrics stored in memory during session)

## Important Notes

### Model Rotation

Models rotate in **round-robin order** across trials:
- Trial 1: Model A
- Trial 2: Model B
- Trial 3: Model C
- Trial 4: Model A (wraps around)

This ensures balanced comparison even if one model is faster/slower.

### API Keys

Each provider requires its own API key:
- OpenAI: `OPENAI_API_KEY`
- OpenRouter: `OPENROUTER_API_KEY`
- Ollama: No key needed (local), but requires `OLLAMA_BASE_URL` if non-default

### Trace File Required

The `--trace` flag **must be specified** when using `--compare`. Model performance events are written to the trace file after each trial completes.

### Ranking Consistency

Model comparison does NOT affect ranking results - all models receive identical prompts and their outputs are evaluated using the same scoring algorithm.

## Example Workflow

```bash
# 1. Run comparison
siftrank --file security-vulnerabilities.txt \
  --prompt "Rank by severity" \
  --compare "openai:gpt-4o-mini,ollama:qwen2.5-coder:32b" \
  --trace comparison.jsonl \
  --output ranked.json

# 2. Extract final stats
jq 'select(.event_type == "model_perf") | .models[]' comparison.jsonl | jq -s 'group_by(.model_id) | map({model: .[0].model_id, stats: .[-1]})'

# 3. Compare costs
echo "OpenAI tokens:" $(jq 'select(.event_type == "model_perf") | .models[] | select(.model_id == "openai:gpt-4o-mini") | .total_tokens' comparison.jsonl | jq -s 'max')
echo "Ollama tokens:" $(jq 'select(.event_type == "model_perf") | .models[] | select(.model_id == "ollama:qwen2.5-coder:32b") | .total_tokens' comparison.jsonl | jq -s 'max')

# 4. Compare latencies
jq 'select(.event_type == "model_perf") | .models[] | {model: .model_id, avg_ms: .avg_latency_ms, p95_ms: .p95_latency_ms}' comparison.jsonl
```

## Troubleshooting

### "no models configured for comparison"

Ensure the `--compare` string is properly formatted:
- Correct: `"openai:gpt-4o-mini,ollama:qwen2.5-coder:32b"`
- Incorrect: `"gpt-4o-mini,qwen2.5-coder:32b"` (missing provider prefix)

### "provider not found"

Check that:
1. Provider type is supported (openai, ollama, openrouter)
2. Required API keys are set in environment
3. Ollama is running (for ollama provider)

### Missing model performance events in trace

Verify:
1. `--trace` flag is specified
2. Trace file is writable
3. At least one trial has completed

## Future Enhancements (Phase 2+)

- Cost tracking (automatically calculate $ per ranking)
- Quality metrics (agreement rates between models)
- Dashboard visualization (real-time latency graphs)
- Automatic model selection (choose fastest/cheapest for workload)

---

For more information on SiftRank architecture, see the main [README](../README.md).
