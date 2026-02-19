"""Tests for pysiftrank batch mode functionality."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pysiftrank import BatchStatus, BatchSubmitResult, SiftRank, SiftRankError


def _mock_run(stdout="", stderr="", returncode=0):
    mock = MagicMock(spec=subprocess.CompletedProcess)
    mock.stdout = stdout
    mock.stderr = stderr
    mock.returncode = returncode
    return mock


# --- batch_submit tests ---


@patch("pysiftrank._runner.subprocess.run")
def test_batch_submit(mock_run):
    output = json.dumps({
        "batch_id": "batch_abc123",
        "mapping_file": "./output/.siftrank-batch.json",
        "file_id": "file-xyz789",
    })
    mock_run.return_value = _mock_run(stdout=output)
    sr = SiftRank(binary="/bin/siftrank")

    result = sr.batch_submit(file="data.txt", prompt="rank by value")

    assert isinstance(result, BatchSubmitResult)
    assert result.batch_id == "batch_abc123"
    assert result.mapping_file == "./output/.siftrank-batch.json"
    assert result.file_id == "file-xyz789"

    cmd = mock_run.call_args[0][0]
    assert cmd[:3] == ["/bin/siftrank", "batch", "submit"]
    assert "-f" in cmd and "data.txt" in cmd
    assert "-p" in cmd and "rank by value" in cmd


@patch("pysiftrank._runner.subprocess.run")
def test_batch_submit_with_options(mock_run):
    output = json.dumps({
        "batch_id": "batch_abc123",
        "mapping_file": "./out/.siftrank-batch.json",
        "file_id": "file-xyz789",
    })
    mock_run.return_value = _mock_run(stdout=output)
    sr = SiftRank(binary="/bin/siftrank")

    sr.batch_submit(
        file="data.txt",
        prompt="rank",
        model="gpt-4o",
        batch_size=20,
        output_dir="./out",
    )

    cmd = mock_run.call_args[0][0]
    assert "--model" in cmd and "gpt-4o" in cmd
    assert "--batch-size" in cmd and "20" in cmd
    assert "--output-dir" in cmd and "./out" in cmd


@patch("pysiftrank._runner.subprocess.run")
def test_batch_submit_error(mock_run):
    mock_run.return_value = _mock_run(
        stderr="Error: batch mode only supports openai", returncode=1
    )
    sr = SiftRank(binary="/bin/siftrank", provider="anthropic")

    with pytest.raises(SiftRankError, match="only supports openai"):
        sr.batch_submit(file="data.txt", prompt="rank")


# --- batch_status tests ---


@patch("pysiftrank._runner.subprocess.run")
def test_batch_status(mock_run):
    output = json.dumps({
        "id": "batch_abc123",
        "status": "in_progress",
        "total": 50,
        "completed": 25,
        "failed": 0,
    })
    mock_run.return_value = _mock_run(stdout=output)
    sr = SiftRank(binary="/bin/siftrank")

    result = sr.batch_status("batch_abc123")

    assert isinstance(result, BatchStatus)
    assert result.id == "batch_abc123"
    assert result.status == "in_progress"
    assert result.total == 50
    assert result.completed == 25
    assert result.failed == 0

    cmd = mock_run.call_args[0][0]
    assert cmd == ["/bin/siftrank", "batch", "status", "batch_abc123"]


@patch("pysiftrank._runner.subprocess.run")
def test_batch_status_completed(mock_run):
    output = json.dumps({
        "id": "batch_abc123",
        "status": "completed",
        "total": 50,
        "completed": 50,
        "failed": 0,
    })
    mock_run.return_value = _mock_run(stdout=output)
    sr = SiftRank(binary="/bin/siftrank")

    result = sr.batch_status("batch_abc123")
    assert result.status == "completed"
    assert result.completed == result.total


# --- batch_results tests ---


@patch("pysiftrank._runner.subprocess.run")
def test_batch_results(mock_run):
    output = json.dumps([
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
    ])
    mock_run.return_value = _mock_run(stdout=output)
    sr = SiftRank(binary="/bin/siftrank")

    results = sr.batch_results(".siftrank-batch.json")

    assert len(results) == 2
    assert results[0].value == "apple"
    assert results[1].value == "banana"

    cmd = mock_run.call_args[0][0]
    assert cmd == ["/bin/siftrank", "batch", "results", ".siftrank-batch.json"]


@patch("pysiftrank._runner.subprocess.run")
def test_batch_results_error(mock_run):
    mock_run.return_value = _mock_run(
        stderr="Error: mapping file not found", returncode=1
    )
    sr = SiftRank(binary="/bin/siftrank")

    with pytest.raises(SiftRankError, match="mapping file not found"):
        sr.batch_results("missing.json")
