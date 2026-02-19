"""Cost report parsing from siftrank stderr output."""

import re
from typing import Optional

from ._types import Usage


def parse_cost_report(stderr: str) -> tuple[Optional[Usage], Optional[float]]:
    """Parse cost and usage information from siftrank --report-cost stderr.

    Returns a tuple of (Usage, cost_float) or (None, None) if not found.
    """
    usage = None
    cost = None

    input_match = re.search(r"input_tokens=(\d+)", stderr)
    output_match = re.search(r"output_tokens=(\d+)", stderr)
    reasoning_match = re.search(r"reasoning_tokens=(\d+)", stderr)

    if input_match and output_match:
        usage = Usage(
            input_tokens=int(input_match.group(1)),
            output_tokens=int(output_match.group(1)),
            reasoning_tokens=int(reasoning_match.group(1)) if reasoning_match else 0,
        )

    cost_match = re.search(r"estimated_cost=\$?([\d.]+)", stderr)
    if cost_match:
        cost = float(cost_match.group(1))

    return usage, cost
