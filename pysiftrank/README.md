# pysiftrank

Python wrapper for the [siftrank](https://github.com/meganerd/siftrank) document ranking CLI.

`pysiftrank` provides a Pythonic interface to siftrank by invoking the CLI binary as a subprocess and parsing the JSON output. It supports all ranking features including batch mode, model comparison, and cost tracking.

## Installation

```bash
pip install pysiftrank
```

**Prerequisite:** The `siftrank` binary must be installed and available on your `PATH`:

```bash
go install github.com/meganerd/siftrank/cmd/siftrank@latest
```

## Quick Start

```python
from pysiftrank import SiftRank, Provider

ranker = SiftRank(
    provider=Provider.OPENAI,
    model="gpt-4o-mini",
    api_key="sk-...",  # optional, falls back to OPENAI_API_KEY env var
)

results = ranker.rank(
    file="data.txt",
    prompt="Rank by relevance to security",
)

for doc in results:
    print(f"{doc.rank}. {doc.value} (score: {doc.score})")
```

## Advanced Usage

```python
results = ranker.rank(
    file="data.json",
    prompt="@prompt.txt",           # file-based prompt
    template="{{.title}}: {{.body}}",
    json=True,                      # force JSON parsing
    batch_size=15,
    max_trials=20,
    relevance=True,                 # include pros/cons justification
    output="results.json",          # save output to file
)

# Cost tracking
results = ranker.rank(
    file="data.txt",
    prompt="Rank items",
    report_cost=True,
)
print(f"Estimated cost: ${results.cost}")
print(f"Input tokens: {results.usage.input_tokens}")

# Model comparison
results = ranker.rank(
    file="data.txt",
    prompt="Rank",
    compare=["openai:gpt-4o-mini", "anthropic:claude-haiku-4-20250514"],
    trace="comparison.jsonl",
)

# Directory input with pattern filtering
results = ranker.rank(
    file="./data_dir",
    prompt="Rank all documents",
    pattern="*.json",
)
```

### All Ranking Options

| Parameter | CLI Flag | Description |
|-----------|----------|-------------|
| `batch_size` | `--batch-size` | Documents per batch (default: 10) |
| `max_trials` | `--max-trials` | Maximum ranking trials (default: 50) |
| `min_trials` | `--min-trials` | Minimum trials before convergence check |
| `stable_trials` | `--stable-trials` | Stable trials for convergence |
| `concurrency` | `--concurrency` | Max concurrent LLM calls (default: 50) |
| `template` | `--template` | Template for each document |
| `output` | `--output` | JSON output file path |
| `trace` | `--trace` | Trace file for execution state |
| `pattern` | `--pattern` | Glob pattern for directory input |
| `effort` | `--effort` | Reasoning effort level |
| `encoding` | `--encoding` | Tokenizer encoding |
| `tokens` | `--tokens` | Max tokens per batch |
| `ratio` | `--ratio` | Refinement ratio (0.0-1.0) |
| `elbow_method` | `--elbow-method` | Elbow detection method |
| `elbow_tolerance` | `--elbow-tolerance` | Elbow position tolerance |
| `json` | `--json` | Force JSON parsing |
| `relevance` | `--relevance` | Include relevance justification |
| `dry_run` | `--dry-run` | Log API calls without making them |
| `no_converge` | `--no-converge` | Disable convergence early stopping |
| `report_cost` | `--report-cost` | Include cost/usage in results |
| `compare` | `--compare` | List of `"provider:model"` strings |

## Batch Mode

Batch mode uses the OpenAI Batch API for 50% lower cost on large jobs (24-hour completion).

```python
# Submit a batch job
batch = ranker.batch_submit(
    file="large_dataset.txt",
    prompt="Rank by business value",
    output_dir="./output",
)
print(f"Batch ID: {batch.batch_id}")
print(f"Mapping file: {batch.mapping_file}")

# Check status
status = ranker.batch_status(batch.batch_id)
print(f"Status: {status.status}")  # "in_progress", "completed", etc.
print(f"Progress: {status.completed}/{status.total}")

# Download results when complete
if status.status == "completed":
    results = ranker.batch_results(batch.mapping_file)
    for doc in results:
        print(f"{doc.rank}. {doc.value}")
```

## Error Handling

```python
from pysiftrank import SiftRank, SiftRankError

ranker = SiftRank(binary="/usr/local/bin/siftrank")

try:
    results = ranker.rank(file="missing.txt", prompt="rank")
except SiftRankError as e:
    print(f"siftrank failed: {e}")
```

If the `siftrank` binary is not found, the constructor raises `RuntimeError`.

## API Reference

### `SiftRank(provider, model, api_key, base_url, binary)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | `"openai"` | LLM provider (or use `Provider` enum) |
| `model` | `str` | `"gpt-4o-mini"` | Model name |
| `api_key` | `str \| None` | `None` | API key (falls back to env var) |
| `base_url` | `str \| None` | `None` | Custom API base URL |
| `binary` | `str \| None` | `None` | Path to siftrank binary (auto-detected) |

### Data Classes

**`RankedDocument`** — A single ranked document.

| Field | Type | Description |
|-------|------|-------------|
| `key` | `str` | Document identifier |
| `value` | `str` | Document text content |
| `score` | `float` | Ranking score (0.0-1.0) |
| `rank` | `int` | Rank position (1-based) |
| `input_index` | `int` | Original position in input |
| `document` | `dict \| None` | Original JSON object (if JSON input) |
| `justification` | `dict \| None` | Pros/cons (if `--relevance` used) |

**`RankResult`** — Result from `rank()`. Iterable over documents.

| Field | Type | Description |
|-------|------|-------------|
| `documents` | `list[RankedDocument]` | Ranked documents |
| `usage` | `Usage \| None` | Token usage (if `report_cost=True`) |
| `cost` | `float \| None` | Estimated cost (if `report_cost=True`) |

Supports `len()`, indexing (`result[0]`), and iteration (`for doc in result`).

**`Usage`** — Token usage statistics.

| Field | Type | Description |
|-------|------|-------------|
| `input_tokens` | `int` | Input tokens consumed |
| `output_tokens` | `int` | Output tokens generated |
| `reasoning_tokens` | `int` | Reasoning tokens (default: 0) |

**`BatchSubmitResult`** — Result from `batch_submit()`.

| Field | Type | Description |
|-------|------|-------------|
| `batch_id` | `str` | OpenAI batch job ID |
| `mapping_file` | `str` | Path to mapping file for results |
| `file_id` | `str` | OpenAI file ID |

**`BatchStatus`** — Result from `batch_status()`.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Batch job ID |
| `status` | `str` | Job status (`"in_progress"`, `"completed"`, etc.) |
| `total` | `int` | Total requests in batch |
| `completed` | `int` | Completed requests |
| `failed` | `int` | Failed requests |

### Enums

**`Provider`** — Supported LLM providers: `OPENAI`, `ANTHROPIC`, `OPENROUTER`, `OLLAMA`, `GOOGLE`.

### Exceptions

**`SiftRankError`** — Raised when the siftrank CLI exits with a non-zero status code. The error message contains the CLI's stderr output.

## Requirements

- Python >= 3.10
- `siftrank` binary on PATH
