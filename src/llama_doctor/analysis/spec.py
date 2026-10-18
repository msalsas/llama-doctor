from .base import DiagnosisContext, Finding, Impact, Recommendation, Severity
from .signals import is_cpu_bound

LOW_ACCEPTANCE = 0.4
OFFLOADED_MAX_DRAFT_LEN = 2.0
RAISE_DRAFT_ACCEPTANCE = 0.7
RAISE_DRAFT_MAX_LEN = 3.0


def analyze(ctx: DiagnosisContext) -> list[Finding]:
    metrics = ctx.metrics
    if metrics is None or metrics.draft_tokens_total is None or metrics.drafts_total is None:
        return [Finding("spec", "spec.insufficient_data", Severity.INSUFFICIENT_DATA,
                        "Speculative decoding data unavailable", "Draft metrics are unavailable.")]
    if metrics.drafts_total == 0:
        return [Finding("spec", "spec.inactive", Severity.INFO, "Speculative decoding inactive",
                        "No drafts have been recorded.")]
    average = metrics.avg_draft_len
    acceptance = metrics.draft_acceptance
    if acceptance is None or average is None:
        return [Finding("spec", "spec.insufficient_data", Severity.INSUFFICIENT_DATA,
                        "Speculative decoding data unavailable", "Draft acceptance metrics are incomplete.")]
    evidence = {"avg_draft_len": average, "acceptance": acceptance}
    if acceptance < LOW_ACCEPTANCE:
        return [Finding("spec", "spec.low_acceptance", Severity.WARN, "Draft acceptance is low",
                        "Most proposed draft tokens are rejected.", evidence,
                        (Recommendation("Reduce maximum draft length.", "Shorter drafts reduce rejected work.",
                                        Impact.MEDIUM, "--draft-max"),))]
    if is_cpu_bound(ctx) is True and average > OFFLOADED_MAX_DRAFT_LEN:
        return [Finding("spec", "spec.drafts_too_long_offloaded", Severity.WARN,
                        "Drafts are too long for CPU-offloaded work",
                        "CPU-bound inference benefits from shorter speculative drafts.", evidence,
                        (Recommendation("Set maximum draft length to 2.", "Short drafts limit CPU expert work.",
                                        Impact.MEDIUM, "--draft-max"),))]
    if (
        is_cpu_bound(ctx) is not True
        and acceptance >= RAISE_DRAFT_ACCEPTANCE
        and average < RAISE_DRAFT_MAX_LEN
    ):
        return [Finding("spec", "spec.raise_draft", Severity.INFO, "Draft length can increase",
                        "High acceptance suggests longer drafts may help.", evidence,
                        (Recommendation("Increase maximum draft length.", "High acceptance supports longer drafts.",
                                        Impact.MEDIUM, "--draft-max"),))]
    return [Finding("spec", "spec.ok", Severity.OK, "Speculative decoding is healthy",
                    "Draft length and acceptance are balanced.", evidence)]
