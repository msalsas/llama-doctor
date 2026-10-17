"""Prometheus text-exposition parsing.

Values are validated against the Prometheus float grammar before conversion;
lines that do not match the contract regex or carry an invalid value are
skipped silently. Limitation: the label block is captured verbatim, so a
``}`` inside an escaped label value (e.g. ``l{a="}"}``) ends the block early
and the whole line is skipped.
"""

import re

_LINE_RE = re.compile(r"^([A-Za-z_:][\w:]*)(\{[^}]*\})?\s+(\S+)(?:\s+-?\d+)?\s*$")
_VALUE_RE = re.compile(
    r"^(?:[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?|NaN|[+-]Inf(inity)?)$"
)


def parse_prometheus(text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for line in text.splitlines():
        match = _LINE_RE.match(line)
        if match is None:
            continue
        name, labels, raw_value = match.groups()
        if _VALUE_RE.fullmatch(raw_value) is None:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        samples[name + (labels or "")] = value
    return samples
