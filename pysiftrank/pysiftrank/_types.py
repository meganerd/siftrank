"""Data classes and types for pysiftrank."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Provider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    GOOGLE = "google"


class SiftRankError(Exception):
    """Error raised when the siftrank CLI returns a non-zero exit code."""


@dataclass
class RankedDocument:
    """A single ranked document from siftrank output."""

    key: str
    value: str
    score: float
    rank: int
    input_index: int
    document: Optional[dict] = None
    justification: Optional[dict] = None


@dataclass
class Usage:
    """Token usage statistics."""

    input_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0


@dataclass
class RankResult:
    """Result from a ranking operation. Iterable over documents."""

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
    """Result from submitting a batch job."""

    batch_id: str
    mapping_file: str
    file_id: str


@dataclass
class BatchStatus:
    """Status of a batch job."""

    id: str
    status: str
    total: int
    completed: int
    failed: int
