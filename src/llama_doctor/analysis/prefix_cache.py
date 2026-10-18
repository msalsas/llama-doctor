from .base import DiagnosisContext, Finding, Impact, Recommendation, Severity

MIN_TOKENS = 1000
LOW_HIT_RATIO = 0.2
MODERATE_HIT_RATIO = 0.5

def analyze(ctx: DiagnosisContext) -> list[Finding]:
    metrics = ctx.metrics
    ratio = metrics.cache_hit_ratio if metrics else None
    token_count = (
        (metrics.prompt_tokens_cached_total or 0) + (metrics.prompt_tokens_total or 0)
        if metrics else 0
    )
    if ratio is None or token_count < MIN_TOKENS:
        return [Finding("prefix_cache", "prefix_cache.insufficient_data", Severity.INSUFFICIENT_DATA,
                        "Prefix cache data unavailable",
                        f"Process at least {MIN_TOKENS} cached and prompt tokens.")]
    evidence = {"hit_ratio": ratio}
    if ratio < LOW_HIT_RATIO:
        return [Finding("prefix_cache", "prefix_cache.low_hit_ratio", Severity.WARN,
                        "Prefix cache hit rate is low", "Few prompt tokens are reused.", evidence, (
                            Recommendation("Keep the prompt prefix byte-identical and put variable content last.",
                                           "Stable prefixes maximize cache reuse.", Impact.HIGH),
                            Recommendation("Enable cache reuse.", "Reuse preserves compatible prompt prefixes.",
                                           Impact.MEDIUM, "--cache-reuse"),
                        ))]
    if ratio < MODERATE_HIT_RATIO:
        return [Finding("prefix_cache", "prefix_cache.moderate_hit_ratio", Severity.INFO,
                        "Prefix cache hit rate is moderate", "More stable prompt prefixes could improve reuse.",
                        evidence, (Recommendation("Keep prompt prefixes stable.", "Stable prefixes improve reuse.",
                                                  Impact.LOW),))]
    return [Finding("prefix_cache", "prefix_cache.ok", Severity.OK, "Prefix cache is effective",
                    "Prompt prefixes are being reused.", evidence)]
