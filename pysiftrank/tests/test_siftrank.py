"""Tests for pysiftrank core ranking functionality."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pysiftrank import (
    Provider,
    RankedDocument,
    RankResult,
    SiftRank,
    SiftRankError,
    Usage,
)


# --- Provider enum tests ---


def test_provider_values():
    assert Provider.OPENAI == "openai"
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENROUTER == "openrouter"
    assert Provider.OLLAMA == "ollama"
    assert Provider.GOOGLE == "google"


def test_provider_is_string():
    assert isinstance(Provider.OPENAI, str)
    assert Provider.OPENAI.value == "openai"


# --- Data class tests ---


def test_ranked_document():
    doc = RankedDocument(
        key="1", value="apple", score=0.95, rank=1, input_index=0
    )
    assert doc.key == "1"
    assert doc.value == "apple"
    assert doc.score == 0.95
    assert doc.rank == 1
    assert doc.input_index == 0
    assert doc.document is None
    assert doc.justification is None


def test_ranked_document_with_optionals():
    doc = RankedDocument(
        key="1",
        value="apple",
        score=0.95,
        rank=1,
        input_index=0,
        document={"name": "apple"},
        justification={"pros": ["tasty"], "cons": ["bruises"]},
    )
    assert doc.document == {"name": "apple"}
    assert doc.justification["pros"] == ["tasty"]


def test_rank_result_iteration():
    docs = [
        RankedDocument(key="1", value="a", score=0.9, rank=1, input_index=0),
        RankedDocument(key="2", value="b", score=0.5, rank=2, input_index=1),
    ]
    result = RankResult(documents=docs)
    assert len(result) == 2
    assert result[0].value == "a"
    assert result[1].value == "b"
    assert [d.value for d in result] == ["a", "b"]


def test_rank_result_with_cost():
    result = RankResult(
        documents=[],
        usage=Usage(input_tokens=1000, output_tokens=200),
        cost=0.0032,
    )
    assert result.usage.input_tokens == 1000
    assert result.cost == 0.0032


def test_usage_defaults():
    usage = Usage(input_tokens=100, output_tokens=50)
    assert usage.reasoning_tokens == 0


# --- SiftRank constructor tests ---


def test_binary_not_found():
    with patch("pysiftrank._runner.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="siftrank binary not found"):
            SiftRank(binary=None)


def test_custom_binary_path():
    sr = SiftRank(binary="/usr/local/bin/siftrank")
    assert sr.binary == "/usr/local/bin/siftrank"


def test_which_fallback():
    with patch("pysiftrank._runner.shutil.which", return_value="/usr/bin/siftrank"):
        sr = SiftRank()
        assert sr.binary == "/usr/bin/siftrank"


def test_provider_enum_accepted():
    sr = SiftRank(provider=Provider.ANTHROPIC, binary="/bin/siftrank")
    assert sr.provider == "anthropic"


# --- rank() tests ---


SAMPLE_OUTPUT = json.dumps([
    {
        "key": "1",
        "value": "apple",
        "score": 0.95,
        "rank": 1,
        "input_index": 0,
    },
    {
        "key": "2",
        "value": "banana",
        "score": 0.72,
        "rank": 2,
        "input_index": 1,
    },
    {
        "key": "3",
        "value": "cherry",
        "score": 0.41,
        "rank": 3,
        "input_index": 2,
    },
])


def _mock_run(stdout="", stderr="", returncode=0):
    mock = MagicMock(spec=subprocess.CompletedProcess)
    mock.stdout = stdout
    mock.stderr = stderr
    mock.returncode = returncode
    return mock


@patch("pysiftrank._runner.subprocess.run")
def test_rank_basic(mock_run):
    mock_run.return_value = _mock_run(stdout=SAMPLE_OUTPUT)
    sr = SiftRank(binary="/bin/siftrank")

    result = sr.rank(file="data.txt", prompt="rank by preference")

    assert len(result) == 3
    assert result[0].value == "apple"
    assert result[0].score == 0.95
    assert result[2].value == "cherry"

    # Check the command was built correctly
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/bin/siftrank"
    assert "-f" in cmd
    assert "data.txt" in cmd
    assert "-p" in cmd
    assert "rank by preference" in cmd
    assert "--provider" in cmd
    assert "openai" in cmd
    assert "--model" in cmd
    assert "gpt-4o-mini" in cmd


@patch("pysiftrank._runner.subprocess.run")
def test_rank_with_flags(mock_run):
    mock_run.return_value = _mock_run(stdout=SAMPLE_OUTPUT)
    sr = SiftRank(
        provider="anthropic",
        model="claude-haiku-4-20250514",
        api_key="sk-test",
        binary="/bin/siftrank",
    )

    sr.rank(
        file="data.json",
        prompt="rank items",
        json=True,
        batch_size=15,
        max_trials=20,
        relevance=True,
        template="{{.title}}",
        concurrency=10,
        effort="high",
        no_converge=True,
    )

    cmd = mock_run.call_args[0][0]
    assert "--provider" in cmd and "anthropic" in cmd
    assert "--model" in cmd and "claude-haiku-4-20250514" in cmd
    assert "--api-key" in cmd and "sk-test" in cmd
    assert "--json" in cmd
    assert "--batch-size" in cmd and "15" in cmd
    assert "--max-trials" in cmd and "20" in cmd
    assert "--relevance" in cmd
    assert "--template" in cmd and "{{.title}}" in cmd
    assert "--concurrency" in cmd and "10" in cmd
    assert "--effort" in cmd and "high" in cmd
    assert "--no-converge" in cmd


@patch("pysiftrank._runner.subprocess.run")
def test_rank_with_compare(mock_run):
    mock_run.return_value = _mock_run(stdout=SAMPLE_OUTPUT)
    sr = SiftRank(binary="/bin/siftrank")

    sr.rank(
        file="data.txt",
        prompt="rank",
        compare=["openai:gpt-4o-mini", "anthropic:claude-haiku-4-20250514"],
    )

    cmd = mock_run.call_args[0][0]
    assert "--compare" in cmd
    idx = cmd.index("--compare")
    assert cmd[idx + 1] == "openai:gpt-4o-mini,anthropic:claude-haiku-4-20250514"


@patch("pysiftrank._runner.subprocess.run")
def test_rank_with_report_cost(mock_run):
    stderr = (
        'level=INFO msg="Cost report" '
        "input_tokens=5000 output_tokens=1200 "
        "reasoning_tokens=0 estimated_cost=$0.0042"
    )
    mock_run.return_value = _mock_run(stdout=SAMPLE_OUTPUT, stderr=stderr)
    sr = SiftRank(binary="/bin/siftrank")

    result = sr.rank(file="data.txt", prompt="rank", report_cost=True)

    assert result.usage is not None
    assert result.usage.input_tokens == 5000
    assert result.usage.output_tokens == 1200
    assert result.cost == 0.0042
    assert "--report-cost" in mock_run.call_args[0][0]


@patch("pysiftrank._runner.subprocess.run")
def test_rank_error(mock_run):
    mock_run.return_value = _mock_run(
        stderr="Error: file not found: missing.txt", returncode=1
    )
    sr = SiftRank(binary="/bin/siftrank")

    with pytest.raises(SiftRankError, match="file not found"):
        sr.rank(file="missing.txt", prompt="rank")


@patch("pysiftrank._runner.subprocess.run")
def test_rank_empty_output(mock_run):
    mock_run.return_value = _mock_run(stdout="")
    sr = SiftRank(binary="/bin/siftrank")

    result = sr.rank(file="empty.txt", prompt="rank")
    assert len(result) == 0


@patch("pysiftrank._runner.subprocess.run")
def test_rank_with_output_file(mock_run):
    mock_run.return_value = _mock_run(stdout=SAMPLE_OUTPUT)
    sr = SiftRank(binary="/bin/siftrank")

    sr.rank(file="data.txt", prompt="rank", output="results.json")

    cmd = mock_run.call_args[0][0]
    assert "--output" in cmd and "results.json" in cmd


@patch("pysiftrank._runner.subprocess.run")
def test_rank_with_base_url(mock_run):
    mock_run.return_value = _mock_run(stdout=SAMPLE_OUTPUT)
    sr = SiftRank(
        provider="ollama",
        model="llama3",
        base_url="http://localhost:11434",
        binary="/bin/siftrank",
    )

    sr.rank(file="data.txt", prompt="rank")

    cmd = mock_run.call_args[0][0]
    assert "--base-url" in cmd and "http://localhost:11434" in cmd


@patch("pysiftrank._runner.subprocess.run")
def test_rank_document_with_justification(mock_run):
    output = json.dumps([
        {
            "key": "1",
            "value": "apple",
            "score": 0.95,
            "rank": 1,
            "input_index": 0,
            "justification": {
                "pros": ["nutritious", "portable"],
                "cons": ["bruises easily"],
            },
        }
    ])
    mock_run.return_value = _mock_run(stdout=output)
    sr = SiftRank(binary="/bin/siftrank")

    result = sr.rank(file="data.txt", prompt="rank", relevance=True)

    assert result[0].justification is not None
    assert "pros" in result[0].justification
