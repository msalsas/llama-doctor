from .base import DiagnosisContext, Finding, Impact, Recommendation, Severity
from .signals import context_used_ratio

NEAR_LIMIT_RATIO = 0.9


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
    return [Finding("context", "context.ok", Severity.OK, "Context has headroom",
                    "The observed peak of tokens in context stays below the configured size.", evidence)]
