import secrets
import time
from collections.abc import Callable

from .client import FetchError, LlamaClient
from .models import ProbeResult

PROBE_PROMPT_WORDS = 1400
WARMUP_N_PREDICT = 8

_PROMPT_WORDS = (
    "amber", "breeze", "candle", "dawn", "earth", "forest", "garden", "harbor",
    "island", "journey", "kindle", "lantern", "meadow", "north", "ocean", "path",
    "quiet", "river", "stone", "thunder", "valley", "window", "yellow", "zephyr",
    "artist", "bridge", "cloud", "distant", "ember", "field", "golden", "horizon",
    "ink", "juniper", "keeper", "light", "morning", "night", "orchard", "paper",
)


def build_prompt(words: int = PROBE_PROMPT_WORDS, nonce: str | None = None) -> str:
    """Build a synthetic prompt with a unique prefix-cache-busting nonce."""
    if nonce is None:
        nonce = secrets.token_hex(8)
    body = " ".join(_PROMPT_WORDS[index % len(_PROMPT_WORDS)] for index in range(words))
    return f"[{nonce}] {body}\nSummarize the text above in one sentence."


def run_probe(
    client: LlamaClient,
    *,
    n_predict: int = 64,
    clock: Callable[[], float] = time.monotonic,
) -> ProbeResult | None:
    """Run and measure a warm-up followed by one uncached synthetic completion."""
    try:
        started = clock()
        client.complete("Hello", WARMUP_N_PREDICT)
        warmup_s = clock() - started
        timings = client.complete(build_prompt(), n_predict)
    except FetchError:
        return None

    if (
        timings.prompt_n is None
        or timings.predicted_n is None
        or timings.prompt_ms is None
        or timings.predicted_ms is None
        or timings.prompt_ms <= 0
        or timings.predicted_ms <= 0
    ):
        return None
    return ProbeResult(
        prefill_tps=timings.prompt_n / (timings.prompt_ms / 1000),
        decode_tps=timings.predicted_n / (timings.predicted_ms / 1000),
        prompt_n=timings.prompt_n,
        predicted_n=timings.predicted_n,
        warmup_s=warmup_s,
    )
