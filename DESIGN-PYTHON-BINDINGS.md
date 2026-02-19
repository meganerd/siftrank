# Python Binding Interface Design

## Executive Summary

This document evaluates approaches for exposing siftrank's document ranking capabilities to Python. After analyzing three approaches (CLI wrapper, cgo shared library, gRPC server), the **CLI subprocess wrapper** is recommended for its simplicity, maintainability, and zero-dependency deployment.

**Recommendation:** Approach 1 (CLI subprocess wrapper) with a thin `pysiftrank` package.

---

## Current Go API Surface

The public API that bindings need to expose:

```
pkg/siftrank:
  NewRanker(config *Config) → (*Ranker, error)
  Ranker.RankFromFile(path, fd, template, forceJSON) → ([]*RankedDocument, error)
  Ranker.RankFromFiles(paths, template, forceJSON) → ([]*RankedDocument, error)
  Ranker.RankFromReader(reader, template, isJSON) → ([]*RankedDocument, error)
  Ranker.TotalUsage() → Usage
  NewProvider(cfg ProviderConfig) → (LLMProvider, error)
  Config.Validate() → error

Types:
  Config (20+ fields)
  RankedDocument { Key, Value, Document, Score, Rank, InputIndex, Justification }
  ProviderConfig { Type, APIKey, Model, BaseURL, Encoding, Effort, Logger }
  Usage { InputTokens, OutputTokens, ReasoningTokens }
```

**Batch mode** is exposed via the CLI subcommand only (`siftrank batch submit/status/results`), not as a library API.

---

## Approach 1: CLI Subprocess Wrapper (RECOMMENDED)

### Concept

A Python package (`pysiftrank`) that invokes the `siftrank` CLI binary as a subprocess, passes arguments, and parses the JSON output.

### Python API Design

```python
from pysiftrank import SiftRank, Provider

# Basic usage
ranker = SiftRank(
    provider=Provider.OPENAI,      # or "openai"
    model="gpt-4o-mini",
    api_key="sk-...",              # optional, falls back to env var
)

results = ranker.rank(
    file="data.txt",
    prompt="Rank by relevance to security",
)

for doc in results:
    print(f"{doc.rank}. {doc.value} (score: {doc.score})")

# Advanced usage
results = ranker.rank(
    file="data.json",
    prompt="@prompt.txt",          # file-based prompt
    template="{{.title}}: {{.body}}",
    json=True,
    batch_size=15,
    max_trials=20,
    relevance=True,
    output="results.json",
)

# Cost tracking
print(f"Estimated cost: ${results.cost}")
print(f"Input tokens: {results.usage.input_tokens}")

# Batch mode
batch = ranker.batch_submit(
    file="large_dataset.txt",
    prompt="Rank by business value",
    output_dir="./output",
)
print(f"Batch ID: {batch.batch_id}")
print(f"Mapping file: {batch.mapping_file}")

status = ranker.batch_status(batch.batch_id)
print(f"Status: {status.status}")

if status.status == "completed":
    results = ranker.batch_results(batch.mapping_file)

# Model comparison
results = ranker.rank(
    file="data.txt",
    prompt="Rank",
    compare=["openai:gpt-4o-mini", "anthropic:claude-haiku-4-20250514"],
    trace="comparison.jsonl",
)

# Directory input
results = ranker.rank(
    file="./data_dir",
    prompt="Rank all documents",
    pattern="*.json",
)
```

### Data Classes

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    GOOGLE = "google"

@dataclass
class RankedDocument:
    key: str
    value: str
    score: float
    rank: int
    input_index: int
    document: Optional[dict] = None       # original JSON object if applicable
    justification: Optional[dict] = None  # pros/cons if --relevance used

@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0

@dataclass
class RankResult:
    documents: list[RankedDocument]
    usage: Optional[Usage] = None
    cost: Optional[float] = None

    def __iter__(self):
        return iter(self.documents)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]

@dataclass
class BatchSubmitResult:
    batch_id: str
    mapping_file: str
    file_id: str

@dataclass
class BatchStatus:
    id: str
    status: str  # "validating", "in_progress", "completed", "failed", etc.
    total: int
    completed: int
    failed: int
```

### Implementation Sketch

```python
import json
import subprocess
import shutil
from pathlib import Path

class SiftRank:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        binary: str | None = None,  # path to siftrank binary
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.binary = binary or shutil.which("siftrank")
        if not self.binary:
            raise RuntimeError(
                "siftrank binary not found. Install with: "
                "go install github.com/meganerd/siftrank/cmd/siftrank@latest"
            )

    def rank(self, file: str, prompt: str, **kwargs) -> RankResult:
        cmd = [self.binary, "-f", file, "-p", prompt]
        cmd += ["--provider", self.provider, "--model", self.model]

        if self.api_key:
            cmd += ["--api-key", self.api_key]
        if self.base_url:
            cmd += ["--base-url", self.base_url]
        if kwargs.get("report_cost"):
            cmd += ["--report-cost"]

        # Map kwargs to CLI flags
        flag_map = {
            "batch_size": "--batch-size",
            "max_trials": "--max-trials",
            "concurrency": "--concurrency",
            "template": "--template",
            "output": "--output",
            "trace": "--trace",
            "pattern": "--pattern",
            "effort": "--effort",
            "encoding": "--encoding",
            "tokens": "--tokens",
            "elbow_method": "--elbow-method",
            "elbow_tolerance": "--elbow-tolerance",
            "stable_trials": "--stable-trials",
            "min_trials": "--min-trials",
            "ratio": "--ratio",
        }
        bool_flags = {
            "json": "--json",
            "relevance": "--relevance",
            "dry_run": "--dry-run",
            "no_converge": "--no-converge",
        }

        for key, flag in flag_map.items():
            if key in kwargs and kwargs[key] is not None:
                cmd += [flag, str(kwargs[key])]
        for key, flag in bool_flags.items():
            if kwargs.get(key):
                cmd += [flag]

        if "compare" in kwargs:
            cmd += ["--compare", ",".join(kwargs["compare"])]

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
        )

        if result.returncode != 0:
            raise SiftRankError(result.stderr.strip())

        documents = [
            RankedDocument(**doc) for doc in json.loads(result.stdout)
        ]

        # Parse cost from stderr if --report-cost was used
        usage, cost = None, None
        if kwargs.get("report_cost"):
            usage, cost = _parse_cost_report(result.stderr)

        return RankResult(documents=documents, usage=usage, cost=cost)
```

### Advantages

- **Zero cgo complexity** — No shared library compilation, no platform-specific builds
- **Always in sync** — Wraps the same binary users install, no API drift
- **Simple distribution** — Pure Python package, `pip install pysiftrank`
- **All features included** — Batch mode, comparison, watch mode available automatically
- **Trivial maintenance** — New CLI flags automatically available via `**kwargs`

### Disadvantages

- **Process overhead** — ~50ms subprocess startup per call
- **No streaming** — Can't stream results during ranking (batch returns at end)
- **Binary dependency** — Requires `siftrank` binary installed separately
- **Error handling** — Limited to parsing stderr text

### Distribution

```
pysiftrank/
  __init__.py          # SiftRank, Provider, data classes
  _runner.py           # subprocess execution, output parsing
  _cost.py             # cost report parsing
  py.typed             # PEP 561 marker
pyproject.toml
README.md
tests/
  test_siftrank.py
  test_batch.py
```

Published to PyPI as `pysiftrank`. Requires `siftrank` binary on PATH.

### Estimated Effort: 4-8 hours

---

## Approach 2: cgo Shared Library

### Concept

Compile `pkg/siftrank` as a C shared library (`.so`/`.dylib`/`.dll`) via `cgo`, then call from Python via `ctypes` or `cffi`.

### Changes Required

```go
// bindings/python/siftrank_cgo.go
package main

import "C"
import (
    "encoding/json"
    "github.com/meganerd/siftrank/pkg/siftrank"
)

//export SiftrankRankFromFile
func SiftrankRankFromFile(
    provider *C.char, model *C.char, apiKey *C.char,
    filePath *C.char, prompt *C.char, batchSize C.int,
) *C.char {
    // Convert C types, create config, run ranking, marshal JSON
    // Return JSON string (caller must free)
}

func main() {} // Required for cgo shared library
```

Build: `go build -buildmode=c-shared -o libsiftrank.so ./bindings/python/`

### Advantages

- **In-process** — No subprocess overhead, direct function calls
- **Streaming possible** — Could expose callback-based API for progress
- **Tighter integration** — Python objects directly mapped to Go structs

### Disadvantages

- **cgo complexity** — Cross-compilation is fragile, platform-specific builds required
- **Memory management** — Must carefully manage C string allocation/deallocation
- **Distribution nightmare** — Need pre-built wheels for linux/macos/windows × amd64/arm64
- **Go runtime in Python process** — Adds ~30MB to process memory, goroutine scheduler conflicts possible
- **API drift risk** — Separate C API layer must be manually kept in sync with Go API
- **Thread safety** — Go runtime and Python GIL interaction is complex
- **Build dependency** — Requires Go toolchain to build from source

### Estimated Effort: 20-40 hours

---

## Approach 3: gRPC/HTTP Server

### Concept

Run siftrank as a long-running server (gRPC or REST), Python client sends ranking requests.

### Advantages

- **Language-agnostic** — Any language can use the API
- **Streaming** — gRPC streaming for progress updates
- **Stateful** — Server can cache models, connections

### Disadvantages

- **Architectural overhead** — Requires running a server process
- **Deployment complexity** — Server lifecycle management
- **Overkill** — siftrank is a CLI tool, not a service
- **Latency** — Network serialization overhead
- **State management** — Server must handle concurrent requests, cleanup

### Estimated Effort: 30-50 hours

---

## Comparison

| Criteria | CLI Wrapper | cgo Shared Library | gRPC Server |
|----------|-------------|-------------------|-------------|
| **Implementation effort** | 4-8 hours | 20-40 hours | 30-50 hours |
| **Distribution** | Pure Python (PyPI) | Platform wheels | Server + client |
| **Maintenance** | Minimal | High (C API sync) | Medium |
| **Performance** | ~50ms overhead | Near-native | Network overhead |
| **All features** | Yes (full CLI) | Subset only | Subset only |
| **Dependencies** | siftrank binary | Go toolchain + cgo | Server process |
| **Cross-platform** | Anywhere Go runs | Per-platform builds | Anywhere |
| **Batch mode** | Yes (CLI subcommand) | Would need new API | Would need new API |

---

## Recommendation: CLI Wrapper

**Approach 1 (CLI subprocess wrapper)** is recommended because:

1. **Simplest to build and maintain** — 4-8 hours vs 20-50 hours
2. **Feature-complete from day one** — Wraps the full CLI including batch mode, comparison, watch
3. **Zero distribution complexity** — Pure Python package on PyPI
4. **Always in sync** — No separate API layer to maintain
5. **Proven pattern** — Many Go CLI tools use this approach (terraform, kubectl wrappers)

The ~50ms subprocess overhead is negligible compared to the seconds-to-minutes LLM API calls that dominate siftrank's runtime.

### Implementation Plan (for siftrank-30)

1. Create `pysiftrank/` directory with package structure
2. Implement `SiftRank` class with `rank()`, `batch_submit()`, `batch_status()`, `batch_results()`
3. Implement data classes (`RankedDocument`, `RankResult`, `Usage`, etc.)
4. Add cost report parsing from stderr
5. Write tests using `--dry-run` mode
6. Add `pyproject.toml` with metadata
7. Write README with examples

---

**Document Status:** Research Complete
**Date:** 2026-02-19
**Issue:** siftrank-29
