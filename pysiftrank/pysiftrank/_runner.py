"""Subprocess execution and output parsing for siftrank CLI."""

import json
import shutil
import subprocess
from typing import Optional

from ._cost import parse_cost_report
from ._types import (
    BatchStatus,
    BatchSubmitResult,
    RankedDocument,
    RankResult,
    SiftRankError,
    Usage,
)


# Maps Python kwarg names to CLI flags (value flags)
_FLAG_MAP = {
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
    "base_url": "--base-url",
    "log": "--log",
}

# Maps Python kwarg names to CLI flags (boolean flags)
_BOOL_FLAGS = {
    "json": "--json",
    "relevance": "--relevance",
    "dry_run": "--dry-run",
    "no_converge": "--no-converge",
    "report_cost": "--report-cost",
    "watch": "--watch",
    "no_minimap": "--no-minimap",
    "debug": "--debug",
}


class SiftRankRunner:
    """Handles subprocess execution of the siftrank CLI binary."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        binary: Optional[str] = None,
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
        """Run siftrank ranking and return parsed results."""
        cmd = [self.binary, "-f", file, "-p", prompt]
        cmd += ["--provider", self.provider, "--model", self.model]

        if self.api_key:
            cmd += ["--api-key", self.api_key]
        if self.base_url:
            cmd += ["--base-url", self.base_url]

        # Map kwargs to CLI flags
        for key, flag in _FLAG_MAP.items():
            if key in kwargs and kwargs[key] is not None:
                cmd += [flag, str(kwargs[key])]
        for key, flag in _BOOL_FLAGS.items():
            if kwargs.get(key):
                cmd += [flag]

        if "compare" in kwargs and kwargs["compare"]:
            cmd += ["--compare", ",".join(kwargs["compare"])]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            raise SiftRankError(result.stderr.strip())

        documents = []
        if result.stdout.strip():
            raw = json.loads(result.stdout)
            documents = [_parse_document(doc) for doc in raw]

        usage, cost = None, None
        if kwargs.get("report_cost"):
            usage, cost = parse_cost_report(result.stderr)

        return RankResult(documents=documents, usage=usage, cost=cost)

    def batch_submit(
        self,
        file: str,
        prompt: str,
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> BatchSubmitResult:
        """Submit a batch ranking job."""
        cmd = [self.binary, "batch", "submit", "-f", file, "-p", prompt]
        cmd += ["--model", model or self.model]

        if batch_size is not None:
            cmd += ["--batch-size", str(batch_size)]
        if output_dir is not None:
            cmd += ["--output-dir", output_dir]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            raise SiftRankError(result.stderr.strip())

        data = json.loads(result.stdout)
        return BatchSubmitResult(
            batch_id=data["batch_id"],
            mapping_file=data["mapping_file"],
            file_id=data["file_id"],
        )

    def batch_status(self, batch_id: str) -> BatchStatus:
        """Check the status of a batch job."""
        cmd = [self.binary, "batch", "status", batch_id]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            raise SiftRankError(result.stderr.strip())

        data = json.loads(result.stdout)
        return BatchStatus(
            id=data["id"],
            status=data["status"],
            total=data.get("total", 0),
            completed=data.get("completed", 0),
            failed=data.get("failed", 0),
        )

    def batch_results(self, mapping_file: str) -> list[RankedDocument]:
        """Download and process results from a completed batch job."""
        cmd = [self.binary, "batch", "results", mapping_file]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            raise SiftRankError(result.stderr.strip())

        raw = json.loads(result.stdout)
        return [_parse_document(doc) for doc in raw]


def _parse_document(doc: dict) -> RankedDocument:
    """Parse a JSON document dict into a RankedDocument."""
    return RankedDocument(
        key=doc["key"],
        value=doc["value"],
        score=doc["score"],
        rank=doc["rank"],
        input_index=doc["input_index"],
        document=doc.get("document"),
        justification=doc.get("justification"),
    )
