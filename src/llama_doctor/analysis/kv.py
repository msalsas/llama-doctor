import math

from .base import DiagnosisContext, Finding, Impact, Recommendation, Severity
from .signals import is_cpu_bound

SATURATED_RATIO = 0.9
OVERSIZED_MIN_CTX = 16384
OVERSIZED_PEAK_RATIO = 0.25

def analyze(ctx: DiagnosisContext) -> list[Finding]:
    metrics = ctx.metrics
    if metrics is None:
        return [Finding("kv", "kv.insufficient_data", Severity.INSUFFICIENT_DATA,
                        "KV cache data unavailable", "The metrics endpoint is unavailable.")]
    findings: list[Finding] = []
    if metrics.kv_cache_usage_ratio is not None and metrics.kv_cache_usage_ratio >= SATURATED_RATIO:
        findings.append(Finding("kv", "kv.saturated", Severity.WARN, "KV cache is saturated",
                                f"KV cache usage is at or above {SATURATED_RATIO * 100:.0f} % of capacity, "
                                "constraining requests.",
                                recommendations=(Recommendation("Reduce parallel requests.",
                                                                "Fewer concurrent requests reduce KV pressure.",
                                                                Impact.MEDIUM, "--parallel"),)))
    n_ctx = ctx.props.n_ctx if ctx.props else None
    n_tokens_max = metrics.n_tokens_max
    evaluable = (
        n_ctx is not None
        and n_tokens_max is not None
        and math.isfinite(n_ctx)
        and math.isfinite(n_tokens_max)
    )
    if evaluable and n_ctx >= OVERSIZED_MIN_CTX and n_tokens_max / n_ctx < OVERSIZED_PEAK_RATIO:
        suggested = max(4096, math.ceil(2 * n_tokens_max / 4096) * 4096)
        impact = Impact.HIGH if is_cpu_bound(ctx) is True else Impact.MEDIUM
        findings.append(Finding("kv", "kv.context_oversized", Severity.WARN, "Context is oversized",
                                f"Observed peak token use is under {OVERSIZED_PEAK_RATIO * 100:.0f} % of a "
                                f"configured context of {OVERSIZED_MIN_CTX} tokens or more.",
                                {"peak_ratio": n_tokens_max / n_ctx, "n_ctx": n_ctx,
                                 "n_tokens_max": n_tokens_max, "suggested_ctx": suggested},
                                (Recommendation(f"Set context to {suggested}.",
                                                "A smaller context frees memory.", impact, "-c"),)))
    if findings:
        return findings
    if evaluable:
        return [Finding("kv", "kv.ok", Severity.OK, "KV cache sizing is healthy",
                        "Context sizing is consistent with observed peak use.")]
    return [Finding("kv", "kv.insufficient_data", Severity.INSUFFICIENT_DATA,
                    "KV cache data unavailable", "Context size or peak token metrics are unavailable.")]
