<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/logo-dark.png">
    <img alt="logo" src="img/logo-light.png" width="500px">
  </picture>
  <br>
  Use LLMs for document ranking.
</p>

## Description

Got a bunch of data? Want to use an LLM to find the most "interesting" stuff? If you simply paste your data into an LLM chat session, you'll run into problems:
- Nondeterminism: Doesn't always respond with the same result
- Limited context: Can't pass in all the data at once, need to break it up
- Output constraints: Sometimes doesn't return all the data you asked it to review
- Scoring subjectivity: Struggles to assign a consistent numeric score to an individual item

`siftrank` is an implementation of the **Sift**Rank document ranking algorithm that uses LLMs to efficiently find the items in any dataset that are most relevant to a given prompt:
- **S**tochastic: Randomly samples the dataset into small batches.
- **I**nflective: Looks for a natural inflection point in the scores that distinguishes particularly relevant items from the rest.
- **F**ixed: Caps the maximum number of LLM calls so the computational complexity remains linear in the worst case.
- **T**rial: Repeatedly compares batched items until the relevance scores stabilize.

Use any LLM to rank anything. No fine-tuning. No domain-specific models. Just an off-the-shelf model and your ranking prompt. Typically runs in seconds and costs pennies.

### Supported Providers

`siftrank` is **provider-agnostic** and works with multiple LLM providers:

- **OpenAI** - GPT-4, GPT-4o, GPT-4o-mini (via `OPENAI_API_KEY`)
- **Anthropic** - Claude Opus, Claude Sonnet, Claude Haiku (via `ANTHROPIC_API_KEY`)
- **OpenRouter** - Access 200+ models from multiple providers (via `OPENROUTER_API_KEY`)
- **Ollama** - Local models like Llama, Mistral, Qwen (via local Ollama server)
- **Google** - Gemini Pro, Gemini Flash (via `GOOGLE_API_KEY`)

Select your provider with `--provider <name>` or use the default (OpenAI). Set the appropriate API key environment variable for your chosen provider.

## Getting started

### Install

```
go install github.com/noperator/siftrank/cmd/siftrank@latest
```

### Configure

Set the API key for your chosen provider:

```bash
# OpenAI (default provider)
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenRouter
export OPENROUTER_API_KEY="sk-or-..."

# Google
export GOOGLE_API_KEY="..."

# Ollama (runs locally, no API key needed)
# Ensure Ollama server is running: ollama serve
```

### Usage

```
siftrank -h

Options:
  -f, --file string       input file (required)
  -m, --model string      model name (default "gpt-4o-mini")
  -o, --output string     JSON output file
      --pattern string    glob pattern for filtering files in directory (default "*")
  -p, --prompt string     initial prompt (prefix with @ to use a file)
      --provider string   LLM provider: openai, anthropic, openrouter, ollama, google (default "openai")
  -r, --relevance         post-process each item by providing relevance justification (skips round 1)
      --compare string    compare multiple models (format: "provider:model,provider:model")

Visualization:
      --no-minimap   disable minimap panel in watch mode
      --watch        enable live terminal visualization (logs suppressed unless --log is specified)

Debug:
  -d, --debug          enable debug logging
      --dry-run        log API calls without making them
      --log string     write logs to file instead of stderr
      --trace string   trace file path for streaming trial execution state (JSON Lines format)

Advanced:
  -u, --base-url string         custom API base URL (for OpenAI-compatible APIs like vLLM)
  -b, --batch-size int          number of items per batch (default 10)
  -c, --concurrency int         max concurrent LLM calls across all trials (default 50)
  -e, --effort string           reasoning effort level: none, minimal, low, medium, high
      --elbow-method string     elbow detection method: curvature (default), perpendicular (default "curvature")
      --elbow-tolerance float   elbow position tolerance (0.05 = 5%) (default 0.05)
      --encoding string         tokenizer encoding (default "o200k_base")
      --json                    force JSON parsing regardless of file extension
      --max-trials int          maximum number of ranking trials (default 50)
      --min-trials int          minimum trials before checking convergence (default 5)
      --no-converge             disable early stopping based on convergence
      --ratio float             refinement ratio (0.0-1.0, e.g. 0.5 = top 50%) (default 0.5)
      --stable-trials int       stable trials required for convergence (default 5)
      --template string         template for each object (prefix with @ to use a file) (default "{{.Data}}")
      --tokens int              max tokens per batch (default 128000)

Flags:
  -h, --help   help for siftrank
```

### Quick Example

Compares 100 [sentences](https://github.com/noperator/siftrank/blob/main/testdata/sentences.txt) in 7 seconds using the default provider (OpenAI):

```bash
siftrank \
    -f testdata/sentences.txt \
    -p 'Rank each of these items according to their relevancy to the concept of "time".' |
    jq -r '.[:10] | map(.value)[]' |
    nl

   1  The train arrived exactly on time.
   2  The old clock chimed twelve times.
   3  The clock ticked steadily on the wall.
   4  The bell rang, signaling the end of class.
   5  The rooster crowed at the break of dawn.
   6  She climbed to the top of the hill to watch the sunset.
   7  He watched as the leaves fell one by one.
   8  The stars twinkled brightly in the clear night sky.
   9  He spotted a shooting star while stargazing.
  10  She opened the curtains to let in the morning light.
```

Use a different provider by specifying `--provider` and `--model`:

```bash
# Use Anthropic's Claude Sonnet
siftrank \
    --provider anthropic \
    --model claude-sonnet-4-20250514 \
    -f testdata/sentences.txt \
    -p 'Rank by relevancy to "time".'

# Use Ollama with a local model
siftrank \
    --provider ollama \
    --model llama3.3 \
    -f testdata/sentences.txt \
    -p 'Rank by relevancy to "time".'
```

### Multi-Provider Examples

Examples demonstrating different providers and use cases.

#### OpenAI

**Basic ranking with gpt-4o-mini (default):**
```bash
siftrank \
    -f logs/access.log \
    -p 'Find suspicious requests that might indicate an attack.' \
    -o suspicious_requests.json
```

**Using GPT-4o for complex analysis:**
```bash
siftrank \
    --provider openai \
    --model gpt-4o \
    -f cve_descriptions.txt \
    -p 'Rank vulnerabilities by exploitability and impact.'
```

**With reasoning effort (o1/o3 models):**
```bash
siftrank \
    --provider openai \
    --model o1-mini \
    --effort medium \
    -f security_findings.json \
    -p 'Prioritize findings by severity and likelihood of exploitation.'
```

#### Anthropic

**Claude Sonnet for balanced performance:**
```bash
siftrank \
    --provider anthropic \
    --model claude-sonnet-4-20250514 \
    -f research_papers.json \
    -p 'Rank papers by relevance to LLM security.' \
    --trace anthropic_trace.jsonl
```

**Claude Haiku for fast, cost-effective ranking:**
```bash
siftrank \
    --provider anthropic \
    --model claude-haiku-4-20250514 \
    -f user_feedback.txt \
    -p 'Identify feedback indicating bugs or usability issues.' \
    --watch
```

**Claude Opus for highest quality analysis:**
```bash
siftrank \
    --provider anthropic \
    --model claude-opus-4-20250514 \
    -f threat_intelligence.json \
    -p 'Rank threats by sophistication and potential impact to our infrastructure.'
```

#### OpenRouter

**Access multiple providers through one API:**
```bash
# Set OpenRouter API key
export OPENROUTER_API_KEY="sk-or-..."

# Use any model from OpenRouter's catalog
siftrank \
    --provider openrouter \
    --model anthropic/claude-sonnet-4 \
    -f documents.txt \
    -p 'Find documents related to incident response.'
```

**Compare frontier models:**
```bash
siftrank \
    --provider openrouter \
    --model google/gemini-2.0-flash-exp \
    -f code_review.json \
    -p 'Identify security vulnerabilities in this code.' \
    --compare "openrouter:anthropic/claude-sonnet-4,openrouter:openai/gpt-4o"
```

#### Ollama (Local Models)

**Run completely local with Llama:**
```bash
# Ensure Ollama is running: ollama serve
# Pull model if needed: ollama pull llama3.3

siftrank \
    --provider ollama \
    --model llama3.3 \
    -f sensitive_data.txt \
    -p 'Identify PII that needs redaction.' \
    -o redaction_candidates.json
```

**Use local model with custom Ollama server:**
```bash
siftrank \
    --provider ollama \
    --model qwen2.5-coder:7b \
    --base-url http://gpu-server:11434 \
    -f code_snippets.txt \
    -p 'Rank code by complexity and maintainability.'
```

**Local model for privacy-sensitive ranking:**
```bash
siftrank \
    --provider ollama \
    --model mistral:7b-instruct \
    -f employee_reviews.txt \
    -p 'Identify reviews mentioning management concerns.' \
    --no-converge \
    --max-trials 10
```

#### Model Comparison

**Compare cost vs performance:**
```bash
# Fast model vs quality model
siftrank \
    -f large_dataset.json \
    -p 'Rank by business value.' \
    --compare "openai:gpt-4o-mini,openai:gpt-4o" \
    --trace comparison_cost_quality.jsonl
```

**Compare across providers:**
```bash
# OpenAI vs Anthropic vs local
siftrank \
    -f documents.txt \
    -p 'Find documents about security best practices.' \
    --compare "openai:gpt-4o-mini,anthropic:claude-haiku-4-20250514,ollama:llama3.3" \
    --trace multi_provider_comparison.jsonl

# Analyze results
jq -s 'group_by(.model) | map({
    model: .[0].model,
    calls: length,
    avg_latency: (map(.latency_ms) | add / length),
    total_tokens: (map(.input_tokens + .output_tokens) | add)
})' multi_provider_comparison.jsonl
```

**Compare OpenRouter models:**
```bash
siftrank \
    -f research_questions.txt \
    -p 'Prioritize research questions by impact.' \
    --compare "openrouter:anthropic/claude-sonnet-4,openrouter:google/gemini-2.0-flash-exp,openrouter:meta-llama/llama-3.3-70b-instruct" \
    --trace openrouter_comparison.jsonl
```

<details><summary>Advanced usage</summary>

#### JSON support

If the input file is a JSON document, it will be read as an array of objects and each object will be used for ranking.

For instance, two objects would be loaded and ranked from this document:

```json
[
  {
    "path": "/foo",
    "code": "bar"
  },
  {
    "path": "/baz",
    "code": "nope"
  }
]
```

#### Templates

It is possible to include each element from the input file in a template using the [Go template syntax](https://pkg.go.dev/text/template) via the `--template "template string"` (or `--template @file.tpl`) argument.

For text input files, each line can be referenced in the template with the `Data` variable:

```
Anything you want with {{ .Data }}
```

For JSON input files, each object in the array can be referenced directly. For instance, elements of the previous JSON example can be referenced in the template code like so:

```
# {{ .path }}

{{ .code }}
```

Note in the following example that the resulting `value` key contains the actual value being presented for ranking (as described by the template), while the `object` key contains the entire original object from the input file for easy reference.

```
# Create some test JSON data.
seq 9 |
    paste -d @ - - - |
    parallel 'echo {} | tr @ "\n" | jo -a | jo nums=:/dev/stdin' |
    jo -a |
    tee input.json

[{"nums":[1,2,3]},{"nums":[4,5,6]},{"nums":[7,8,9]}]

# Use template to extract the first element of the nums array in each input object.
siftrank \
	-f input.json \
	-p 'Which is biggest?' \
	--template '{{ index .nums 0 }}' \
	--max-trials 1 |
	jq -c '.[]'

{"key":"eQJpm-Qs","value":"7","object":{"nums":[7,8,9]},"score":0,"exposure":1,"rank":1}
{"key":"SyJ3d9Td","value":"4","object":{"nums":[4,5,6]},"score":2,"exposure":1,"rank":2}
{"key":"a4ayc_80","value":"1","object":{"nums":[1,2,3]},"score":3,"exposure":1,"rank":3}
```

#### Token Usage and Performance Tracking

`siftrank` tracks token consumption and performance metrics for all LLM calls, enabling cost estimation and model comparison.

##### Token Tracking

Every LLM API call records:
- **Input tokens** (prompt tokens)
- **Output tokens** (completion tokens)
- **Reasoning tokens** (for o1/o3 models)

Token usage accumulates across all trials and is included in the trace file (see `--trace` flag).

##### Model Comparison with --compare

Compare multiple models side-by-side to evaluate performance and cost tradeoffs:

```bash
# Compare OpenAI vs Anthropic
siftrank \
    -f testdata/sentences.txt \
    -p 'Rank by relevancy to "time".' \
    --compare "openai:gpt-4o-mini,anthropic:claude-haiku-4-20250514" \
    --trace comparison.jsonl

# Compare multiple OpenRouter models
siftrank \
    -f testdata/sentences.txt \
    -p 'Rank by relevancy to "time".' \
    --compare "openrouter:anthropic/claude-sonnet-4,openrouter:openai/gpt-4o" \
    --trace comparison.jsonl
```

**Collected metrics per model:**
- **Call count** - Total number of API calls
- **Success rate** - Ratio of successful vs failed calls
- **Latency statistics** - Average, P50, P95, P99 (milliseconds)
- **Total tokens** - Sum of all input + output + reasoning tokens across all calls

##### Trace File Format

The `--trace <file>` flag writes JSON Lines output with detailed execution state:

```bash
siftrank -f data.txt -p 'Rank items' --trace trace.jsonl
```

Each line in the trace file contains:
```json
{
  "trial": 1,
  "round": 2,
  "model": "gpt-4o-mini",
  "batch_size": 10,
  "input_tokens": 1234,
  "output_tokens": 567,
  "reasoning_tokens": 0,
  "latency_ms": 850,
  "success": true,
  "elbow_detected": false
}
```

Use the trace file to:
- **Monitor progress** in real-time (`tail -f trace.jsonl`)
- **Analyze token consumption** patterns across trials
- **Compare model performance** when using `--compare`
- **Debug convergence** behavior with elbow detection data

##### Cost Estimation

To estimate costs from token usage:

1. **Extract token totals** from trace file:
```bash
jq -s 'map({model, input: .input_tokens, output: .output_tokens}) | group_by(.model) | map({model: .[0].model, total_input: (map(.input) | add), total_output: (map(.output) | add)})' trace.jsonl
```

2. **Apply provider pricing** (as of 2026-02):

| Provider | Model | Input (per 1M tokens) | Output (per 1M tokens) |
|----------|-------|-----------------------|------------------------|
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| OpenAI | gpt-4o | $2.50 | $10.00 |
| Anthropic | claude-haiku-4 | $0.25 | $1.25 |
| Anthropic | claude-sonnet-4 | $3.00 | $15.00 |
| OpenRouter | varies | varies | varies |
| Ollama | free (local) | $0.00 | $0.00 |

**Example cost calculation:**
```
Input tokens: 50,000
Output tokens: 10,000
Model: gpt-4o-mini

Cost = (50,000 / 1,000,000) × $0.15 + (10,000 / 1,000,000) × $0.60
     = $0.0075 + $0.0060
     = $0.0135 (~1.4 cents)
```

> **Note:** Built-in cost reporting (automatic $ calculation) is planned for a future release. Track progress in issue siftrank-20.

</details>

## Back matter

### Acknowledgements

I released the prototype of this tool, Raink, while at Bishop Fox. See the original [presentation](https://www.youtube.com/watch?v=IBuL1zY69tY), [blog post](https://bishopfox.com/blog/raink-llms-document-ranking) and [CLI tool](https://github.com/bishopfox/raink).

### See also

- [O(N) the Money: Scaling Vulnerability Research with LLMs](https://noperator.dev/posts/on-the-money/)
- [Using LLMs to solve security problems](https://noperator.dev/posts/ai-for-security/)
- [Hard problems that reduce to document ranking](https://noperator.dev/posts/document-ranking-for-complex-problems/)
- [Commentary: Critical Thinking - Bug Bounty Podcast](https://youtu.be/qd08UBNpu7k?si=pMVEYtmKnyuJkL9B&t=1511)
- [Discussion: Hacker News](https://news.ycombinator.com/item?id=43174910)
- [Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting](https://arxiv.org/html/2306.17563v2)

### To-do

- [ ] add python bindings?
- [ ] allow specifying an input _directory_ (where each file is distinct object)
- [ ] clarify when prompt included in token estimate
- [ ] factor LLM calls out into a separate package
- [ ] run openai batch mode
- [ ] report cost + token usage
- [ ] add more examples, use cases
- [ ] account for reasoning tokens separately

<details><summary>Completed</summary>

- [x] add visualization
- [x] support reasoning effort
- [x] add blog link
- [x] add parameter for refinement ratio
- [x] add ~boolean~ refinement ratio flag
- [x] alert if the incoming context window is super large
- [x] automatically calculate optimal batch size?
- [x] explore "tournament" sort vs complete exposure each time
- [x] make sure that each randomized run is evenly split into groups so each one gets included/exposed
- [x] parallelize openai calls for each run
- [x] remove token limit threshold? potentially confusing/unnecessary
- [x] save time by using shorter hash ids
- [x] separate package and cli tool
- [x] some batches near the end of a run (9?) are small for some reason
- [x] support non-OpenAI models

</details>

### License

This project is licensed under the [MIT License](LICENSE).
