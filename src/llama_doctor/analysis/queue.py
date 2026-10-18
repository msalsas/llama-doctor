from .base import DiagnosisContext, Finding, Impact, Recommendation, Severity
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
                        evidence,
                        (Recommendation("Reduce client concurrency.",
                                        "Fewer requests in flight leave a slot free for each new one.",
                                        Impact.MEDIUM),
                         Recommendation("Raise the slot count, and the context with it.",
                                        "More slots let more requests run at once, but -c divides across "
                                        "slots, so raise -c too to keep each slot's context.",
                                        Impact.MEDIUM, "--parallel")))]
    return [Finding("queue", "queue.ok", Severity.OK, "No requests queued",
                    "Every request found a free slot.", {"deferred_peak": peak})]
