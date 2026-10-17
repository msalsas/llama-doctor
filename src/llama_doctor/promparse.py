import re

_LINE_RE = re.compile(r"^([A-Za-z_:][\w:]*)(\{[^}]*\})?\s+(\S+)(?:\s+-?\d+)?\s*$")


def parse_prometheus(text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for line in text.splitlines():
        match = _LINE_RE.match(line)
        if match is None:
            continue
        name, labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        samples[name + (labels or "")] = value
    return samples
