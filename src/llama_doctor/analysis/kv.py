from .base import DiagnosisContext, Finding, Impact, Recommendation, Severity

SATURATED_RATIO = 0.9


def analyze(ctx: DiagnosisContext) -> list[Finding]:
    metrics = ctx.metrics
    if metrics is None:
        return [Finding("kv", "kv.insufficient_data", Severity.INSUFFICIENT_DATA,
                        "KV cache data unavailable", "The metrics endpoint is unavailable.")]
    usage = metrics.kv_cache_usage_ratio
    if usage is not None and usage >= SATURATED_RATIO:
        return [Finding("kv", "kv.saturated", Severity.WARN, "KV cache is saturated",
                        f"KV cache usage is at or above {SATURATED_RATIO * 100:.0f} % of capacity, "
                        "constraining requests.",
                        recommendations=(Recommendation("Reduce parallel requests.",
                                                        "Fewer concurrent requests reduce KV pressure.",
                                                        Impact.MEDIUM, "--parallel"),))]
    if usage is not None and 0 <= usage < SATURATED_RATIO:
        return [Finding("kv", "kv.ok", Severity.OK, "KV cache usage is well below capacity",
                        f"KV cache usage is under {SATURATED_RATIO * 100:.0f} % of capacity.")]
    slots = ctx.snapshot.slots
    if usage is None and slots is not None:
        return [Finding("kv", "kv.estimated", Severity.INFO, "KV cache usage is estimated",
                        "The KV cache usage metric is not exported; this estimate comes from /slots "
                        "(tokens held over slot capacity) and can read high with --cache-idle-slots. "
                        "It never triggers the saturation warning.",
                        {"estimated": True, "source": "/slots",
                         "usage_ratio": slots.live_used / slots.live_capacity,
                         "live_used": slots.live_used, "live_capacity": slots.live_capacity,
                         "slots": slots.n_slots})]
    return [Finding("kv", "kv.insufficient_data", Severity.INSUFFICIENT_DATA,
                    "KV cache data unavailable", "The KV cache usage metric is not exported.")]
