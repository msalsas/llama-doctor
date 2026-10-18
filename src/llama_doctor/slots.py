"""Defensive parser for llama-server's ``GET /slots``.

The payload shape differs between server versions and carries prompt text and request
ids. Only a strict whitelist of numeric fields is read; everything else is dropped here.
"""

from typing import Any

from .models import SlotsSnapshot

MAX_SLOTS = 1024
MAX_TOKENS = 10**12


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= MAX_TOKENS else None


def _n_decoded(slot: dict[str, Any]) -> int | None:
    next_token = slot.get("next_token")
    if isinstance(next_token, list):
        next_token = next_token[0] if next_token else None
    return _count(next_token.get("n_decoded")) if isinstance(next_token, dict) else None


def parse_slots(payload: Any) -> SlotsSnapshot | None:
    """Sum the whitelisted numbers over slots; None when nothing usable is found."""
    if not isinstance(payload, list):
        return None
    used = capacity = n_slots = n_processing = 0
    for slot in payload[:MAX_SLOTS]:
        if not isinstance(slot, dict):
            continue
        n_ctx = _count(slot.get("n_ctx"))
        if n_ctx is None:
            continue
        n_slots += 1
        capacity += n_ctx
        used += (_count(slot.get("n_prompt_tokens")) or 0) + (_n_decoded(slot) or 0)
        n_processing += slot.get("is_processing") is True
    if n_slots == 0 or capacity <= 0:
        return None
    return SlotsSnapshot(used, capacity, n_slots, n_processing)
