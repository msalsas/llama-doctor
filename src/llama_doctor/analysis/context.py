import math

from .base import DiagnosisContext, Finding, Impact, Recommendation, Severity
from .signals import context_used_ratio, is_cpu_bound

NEAR_LIMIT_RATIO = 0.9
OVERSIZED_MIN_CTX = 16384
OVERSIZED_PEAK_RATIO = 0.25


def analyze(ctx: DiagnosisContext) -> list[Finding]:
    n_tokens_max = ctx.metrics.n_tokens_max if ctx.metrics else None
    n_ctx = ctx.props.n_ctx if ctx.props else None
    ratio = context_used_ratio(n_tokens_max, n_ctx)
    if ratio is None:
        return [Finding("context", "context.insufficient_data", Severity.INSUFFICIENT_DATA,
                        "Context use unavailable", "Context size or peak token metrics are unavailable.")]
    evidence = {"used_ratio": ratio, "n_tokens_max": n_tokens_max, "n_ctx": n_ctx}
    if ratio >= NEAR_LIMIT_RATIO:
        return [Finding("context", "context.near_limit", Severity.WARN, "Context is near its limit",
                        "The observed peak of tokens in context is close to the configured context size.",
                        evidence,
                        (Recommendation("Raise the context size or shorten prompts.",
                                        "Requests past n_ctx get truncated or have their context shifted.",
                                        Impact.MEDIUM, "-c"),))]
    if n_ctx >= OVERSIZED_MIN_CTX and ratio < OVERSIZED_PEAK_RATIO:
        suggested = max(4096, math.ceil(2 * n_tokens_max / 4096) * 4096)
        impact = Impact.HIGH if is_cpu_bound(ctx) is True else Impact.MEDIUM
        return [Finding("context", "context.oversized", Severity.WARN, "Context is oversized",
                        f"Observed peak token use is under {OVERSIZED_PEAK_RATIO * 100:.0f} % of a "
                        f"configured context of {OVERSIZED_MIN_CTX} tokens or more.",
                        {**evidence, "suggested_ctx": suggested},
                        (Recommendation(f"Set context to {suggested}.",
                                        "A smaller context frees memory.", impact, "-c"),))]
    return [Finding("context", "context.ok", Severity.OK, "Context has headroom",
                    "The observed peak of tokens in context stays below the configured size.", evidence)]
