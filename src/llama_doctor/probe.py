import math
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from .client import COMPLETION_ROUTES, FetchError, LlamaClient, ProbeRouteError
from .models import CompletionTimings, ProbeResult, ProbeSkip

PROBE_PROMPT_WORDS = 1400
WARMUP_N_PREDICT = 8
PROBE_BUDGET_S = 300.0
PROBE_REDUCED_PROMPT_WORDS = 1000
PROBE_REDUCED_N_PREDICT = 16
PROBE_TOKENS_PER_WORD = 1.25
PROBE_MIN_MEASURE_S = 1.0
PROBE_BUDGET_EXHAUSTED = "probe budget exhausted"

_PROMPT_WORDS = (
    "amber", "breeze", "candle", "dawn", "earth", "forest", "garden", "harbor",
    "island", "journey", "kindle", "lantern", "meadow", "north", "ocean", "path",
    "quiet", "river", "stone", "thunder", "valley", "window", "yellow", "zephyr",
    "artist", "bridge", "cloud", "distant", "ember", "field", "golden", "horizon",
    "ink", "juniper", "keeper", "light", "morning", "night", "orchard", "paper",
)


@dataclass(frozen=True)
class MeasurePlan:
    words: int
    n_predict: int
    timeout: float


def build_prompt(words: int = PROBE_PROMPT_WORDS, nonce: str | None = None) -> str:
    """Build a synthetic prompt with a unique prefix-cache-busting nonce."""
    if nonce is None:
        nonce = secrets.token_hex(8)
    body = " ".join(_PROMPT_WORDS[index % len(_PROMPT_WORDS)] for index in range(words))
    return f"[{nonce}] {body}\nSummarize the text above in one sentence."


def plan_measurement(
    warmup: CompletionTimings, remaining_s: float, *, n_predict: int = 64
) -> MeasurePlan | None:
    """Pick the measured prompt size from the warm-up decode rate and remaining time."""
    if math.isnan(remaining_s) or remaining_s < PROBE_MIN_MEASURE_S:
        return None
    decode_tps = None
    if (
        warmup.predicted_n is not None
        and warmup.predicted_ms is not None
        and warmup.predicted_n >= 1
        and warmup.predicted_ms > 0
    ):
        decode_tps = warmup.predicted_n / (warmup.predicted_ms / 1000)
    if decode_tps is not None and (
        PROBE_PROMPT_WORDS * PROBE_TOKENS_PER_WORD + n_predict
    ) / decode_tps <= remaining_s:
        return MeasurePlan(PROBE_PROMPT_WORDS, n_predict, remaining_s)
    return MeasurePlan(
        PROBE_REDUCED_PROMPT_WORDS, min(n_predict, PROBE_REDUCED_N_PREDICT), remaining_s
    )


def run_probe_detailed(
    client: LlamaClient,
    *,
    n_predict: int = 64,
    clock: Callable[[], float] = time.monotonic,
    model_id: str | None = None,
    budget_s: float = PROBE_BUDGET_S,
    on_plan: Callable[[MeasurePlan], None] | None = None,
) -> ProbeResult | ProbeSkip:
    """Run and measure a warm-up followed by one uncached synthetic completion."""
    if not (math.isfinite(budget_s) and budget_s > 0):
        raise ValueError("budget_s must be a finite number > 0")
    warmup_timeout = budget_s / len(COMPLETION_ROUTES)
    try:
        started = clock()
        warmup = client.complete(
            "Hello", WARMUP_N_PREDICT, model_id=model_id, timeout=warmup_timeout
        )
        warmup_s = clock() - started
    except ProbeRouteError as exc:
        return ProbeSkip("discovery", "no usable completion route", None, exc.attempts)
    except FetchError as exc:
        return ProbeSkip("measure", exc.reason, exc.endpoint)
    plan = plan_measurement(warmup, budget_s - warmup_s, n_predict=n_predict)
    if plan is None:
        return ProbeSkip("measure", PROBE_BUDGET_EXHAUSTED, client.completion_route)
    if on_plan is not None:
        on_plan(plan)
    try:
        timings = client.complete(
            build_prompt(plan.words), plan.n_predict, model_id=model_id,
            timeout=plan.timeout,
        )
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
    budget_s: float = PROBE_BUDGET_S,
) -> ProbeResult | None:
    """Like :func:`run_probe_detailed`, but collapsed to ``ProbeResult | None``."""
    outcome = run_probe_detailed(
        client, n_predict=n_predict, clock=clock, model_id=model_id, budget_s=budget_s
    )
    if isinstance(outcome, ProbeSkip):
        return None
    return outcome


def describe_skip(skip: ProbeSkip) -> str:
    if skip.stage == "measure":
        if skip.reason == PROBE_BUDGET_EXHAUSTED:
            return "probe: budget exhausted before the measured prompt, using lifetime counters"
        return f"probe: measured prompt failed on {skip.route}: {skip.reason}, using lifetime counters"
    if skip.stage == "timings":
        return f"probe: unusable timings from {skip.route}, using lifetime counters"
    attempts = "; ".join(f"{attempt.endpoint}: {attempt.reason}" for attempt in skip.attempts)
    return f"probe: no usable completion route ({attempts}), using lifetime counters"


def budget_hint(skip: ProbeSkip, budget_s: float) -> str | None:
    hinted = (
        skip.stage == "discovery"
        and bool(skip.attempts)
        and skip.attempts[-1].reason == "timed out"
    ) or (
        skip.stage == "measure"
        and skip.reason in ("timed out", PROBE_BUDGET_EXHAUSTED)
    )
    if not hinted:
        return None
    return (
        f"probe: the server did not answer within the {budget_s:g} s probe budget; "
        "retry with a larger --probe-budget"
    )
