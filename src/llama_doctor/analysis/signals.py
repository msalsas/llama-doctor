from dataclasses import dataclass

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


def is_cpu_bound(ctx: DiagnosisContext) -> bool | None:
    prefill = prefill_rate(ctx)
    if prefill is None:
        return None
    if prefill.value < CPU_BOUND_PREFILL_TPS:
        return True
    decode = decode_rate(ctx)
    return decode is not None and prefill.value / decode.value < MIN_PREFILL_DECODE_RATIO
