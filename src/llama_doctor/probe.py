import secrets
import time
from collections.abc import Callable

from .client import FetchError, LlamaClient, ProbeRouteError
from .models import ProbeResult, ProbeSkip

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


def run_probe_detailed(
    client: LlamaClient,
    *,
    n_predict: int = 64,
    clock: Callable[[], float] = time.monotonic,
    model_id: str | None = None,
) -> ProbeResult | ProbeSkip:
    """Run and measure a warm-up followed by one uncached synthetic completion."""
    try:
        started = clock()
        client.complete("Hello", WARMUP_N_PREDICT, model_id=model_id)
        warmup_s = clock() - started
    except ProbeRouteError as exc:
        return ProbeSkip("discovery", "no usable completion route", None, exc.attempts)
    except FetchError as exc:
        return ProbeSkip("measure", exc.reason, exc.endpoint)
    try:
        timings = client.complete(build_prompt(), n_predict, model_id=model_id)
    except FetchError as exc:
        return ProbeSkip("measure", exc.reason, client.completion_route)

    if (
        timings.prompt_n is None
        or timings.predicted_n is None
        or timings.prompt_ms is None
        or timings.predicted_ms is None
        or timings.prompt_ms <= 0
        or timings.predicted_ms <= 0
    ):
        return ProbeSkip("timings", "unusable timings", client.completion_route)
    return ProbeResult(
        prefill_tps=timings.prompt_n / (timings.prompt_ms / 1000),
        decode_tps=timings.predicted_n / (timings.predicted_ms / 1000),
        prompt_n=timings.prompt_n,
        predicted_n=timings.predicted_n,
        warmup_s=warmup_s,
    )


def run_probe(
    client: LlamaClient,
    *,
    n_predict: int = 64,
    clock: Callable[[], float] = time.monotonic,
    model_id: str | None = None,
) -> ProbeResult | None:
    """Like :func:`run_probe_detailed`, but collapsed to ``ProbeResult | None``."""
    outcome = run_probe_detailed(client, n_predict=n_predict, clock=clock, model_id=model_id)
    if isinstance(outcome, ProbeSkip):
        return None
    return outcome


def describe_skip(skip: ProbeSkip) -> str:
    if skip.stage == "measure":
        return f"probe: measured prompt failed on {skip.route}: {skip.reason}, using lifetime counters"
    if skip.stage == "timings":
        return f"probe: unusable timings from {skip.route}, using lifetime counters"
    attempts = "; ".join(f"{attempt.endpoint}: {attempt.reason}" for attempt in skip.attempts)
    return f"probe: no usable completion route ({attempts}), using lifetime counters"
