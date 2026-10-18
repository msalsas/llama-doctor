import math
from dataclasses import dataclass
from itertools import pairwise

from .base import DiagnosisContext

MIN_PREFILL_TOKENS = 1000
CPU_BOUND_PREFILL_TPS = 100.0
MIN_PREFILL_DECODE_RATIO = 2.0


@dataclass(frozen=True)
class Rate:
    value: float
    source: str


def prefill_rate(ctx: DiagnosisContext) -> Rate | None:
    if ctx.probe is not None and ctx.probe.reliable:
        return Rate(ctx.probe.prefill_tps, "probe")
    if (
        ctx.metrics is not None
        and (ctx.metrics.prompt_tokens_total or 0) >= MIN_PREFILL_TOKENS
        and ctx.metrics.prefill_tps is not None
    ):
        return Rate(ctx.metrics.prefill_tps, "lifetime")
    return None


def decode_rate(ctx: DiagnosisContext) -> Rate | None:
    if ctx.probe is not None and ctx.probe.reliable and ctx.probe.predicted_n > 0:
        return Rate(ctx.probe.decode_tps, "probe")
    if ctx.metrics is not None and ctx.metrics.decode_tps is not None:
        return Rate(ctx.metrics.decode_tps, "lifetime")
    return None


def mean_prompt_tokens(ctx: DiagnosisContext) -> float | None:
    """Mean evaluated prompt tokens per completion step in the window, None without one.

    Counters only move when a request completes, so each rise of prompt_tokens_total
    between samples is one request's uncached prompt (or a few, if they finished together).
    """
    totals = [sample.prompt_tokens_total for sample in ctx.window]
    steps = [
        after - before
        for before, after in pairwise(totals)
        if before is not None and after is not None and after > before
    ]
    return sum(steps) / len(steps) if steps else None


def deferred_peak(ctx: DiagnosisContext) -> float | None:
    """Deepest queue seen: max of the deferred gauge over the window, else its current value."""
    samples = ctx.window or ((ctx.metrics,) if ctx.metrics else ())
    return max(
        (sample.requests_deferred for sample in samples if sample.requests_deferred is not None),
        default=None,
    )


def context_used_ratio(n_tokens_max: float | None, n_ctx: float | None) -> float | None:
    """Observed peak tokens in context over the configured window; a peak, not current use."""
    if n_tokens_max is None or n_ctx is None:
        return None
    if not (math.isfinite(n_tokens_max) and math.isfinite(n_ctx)) or n_tokens_max < 0 or n_ctx <= 0:
        return None
    return n_tokens_max / n_ctx


def is_cpu_bound(ctx: DiagnosisContext) -> bool | None:
    prefill = prefill_rate(ctx)
    if prefill is None:
        return None
    if prefill.value < CPU_BOUND_PREFILL_TPS:
        return True
    decode = decode_rate(ctx)
    if decode is None or not math.isfinite(decode.value) or decode.value <= 0:
        return False
    return prefill.value / decode.value < MIN_PREFILL_DECODE_RATIO
