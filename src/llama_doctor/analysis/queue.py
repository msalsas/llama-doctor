from .base import DiagnosisContext, Finding, Severity
from .signals import deferred_peak


def analyze(ctx: DiagnosisContext) -> list[Finding]:
    metrics = ctx.metrics
    peak = deferred_peak(ctx)
    if metrics is None or peak is None:
        return [Finding("queue", "queue.insufficient_data", Severity.INSUFFICIENT_DATA,
                        "Queue data unavailable", "The requests_deferred gauge is not exported.")]
    if peak > 0:
        evidence = {"deferred_peak": peak,
                    "requests_deferred": metrics.requests_deferred,
                    "requests_processing": metrics.requests_processing}
        if ctx.window:
            evidence["window_s"] = ctx.window[-1].taken_at - ctx.window[0].taken_at
        return [Finding("queue", "queue.deferred", Severity.INFO,
                        "Requests are queued; all slots busy",
                        "Requests had to wait for a free slot because every slot was busy.",
                        evidence)]
    return [Finding("queue", "queue.ok", Severity.OK, "No requests queued",
                    "Every request found a free slot.", {"deferred_peak": peak})]
