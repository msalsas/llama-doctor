from .base import DiagnosisContext, Finding, Impact, Recommendation, Severity
from .signals import decode_rate, is_cpu_bound

SLOW_DECODE_TPS = 10.0


def analyze(ctx: DiagnosisContext) -> list[Finding]:
    rate = decode_rate(ctx)
    if rate is None:
        return [Finding("decode", "decode.insufficient_data", Severity.INSUFFICIENT_DATA,
                        "Decode rate unavailable", "Generate tokens or run --probe.")]
    cpu_bound = is_cpu_bound(ctx)
    evidence = {"decode_tps": rate.value, "source": rate.source, "cpu_bound": cpu_bound}
    if rate.value < SLOW_DECODE_TPS:
        if cpu_bound:
            return [Finding("decode", "decode.slow", Severity.WARN, "Decode is slow",
                            "This is a symptom of the prefill CPU-bound finding.", evidence)]
        return [Finding("decode", "decode.slow", Severity.WARN, "Decode is slow",
                        "Decode throughput is below the expected range.", evidence,
                        (Recommendation("Offload more layers to the GPU.", "GPU offload improves decode.",
                                        Impact.MEDIUM, "-ngl"),))]
    return [Finding("decode", "decode.ok", Severity.OK, "Decode throughput is healthy",
                    "Decode throughput is within the expected range.", evidence)]
