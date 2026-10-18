from .base import DiagnosisContext, Finding, Impact, Recommendation, Severity
from .signals import decode_rate, is_cpu_bound, prefill_rate

SLOW_PREFILL_TPS = 300.0


def _evidence(ctx: DiagnosisContext) -> dict[str, float | str | None]:
    prefill = prefill_rate(ctx)
    decode = decode_rate(ctx)
    return {"prefill_tps": prefill.value if prefill else None, "decode_tps": decode.value if decode else None,
            "source": prefill.source if prefill else None}


def analyze(ctx: DiagnosisContext) -> list[Finding]:
    rate = prefill_rate(ctx)
    if rate is None:
        total = ctx.metrics.prompt_tokens_total if ctx.metrics else None
        return [Finding("prefill", "prefill.insufficient_data", Severity.INSUFFICIENT_DATA,
                        "Prefill rate unavailable", "Run --probe or process at least 1000 prompt tokens.",
                        {"prompt_tokens_total": total})]
    evidence = _evidence(ctx)
    if is_cpu_bound(ctx):
        if ctx.props is not None and ctx.props.is_moe is True:
            recommendations = (
                Recommendation("Use a smaller quantization.", "It can keep more experts in VRAM.", Impact.HIGH),
                Recommendation("Put N-gram/PLE tables on CPU or mmap.", "Use this on models that ship them.",
                               Impact.HIGH, "--ple-on-cpu"),
                Recommendation("Use MTP or speculative drafting at 2.", "Short drafts reduce CPU expert work.",
                               Impact.MEDIUM, "--draft-max"),
                Recommendation("Reduce context to free VRAM for experts.", "More VRAM improves offload.",
                               Impact.LOW, "-c"),
            )
            return [Finding("prefill", "prefill.moe_cpu_experts", Severity.CRITICAL,
                            "MoE experts are CPU-bound", "Slow prefill indicates CPU-resident experts.",
                            evidence, recommendations)]
        recommendations = (
            Recommendation("Offload more layers to the GPU.", "GPU offload improves prefill.", Impact.HIGH, "-ngl"),
            Recommendation("Use a smaller quantization.", "It can increase GPU offload.", Impact.HIGH),
            Recommendation("Reduce context to free VRAM.", "More VRAM can enable offload.", Impact.LOW, "-c"),
        )
        return [Finding("prefill", "prefill.cpu_bound", Severity.WARN, "Prefill is CPU-bound",
                        "Prefill throughput indicates CPU work.", evidence, recommendations)]
    if rate.value < SLOW_PREFILL_TPS:
        return [Finding("prefill", "prefill.slow", Severity.INFO, "Prefill is slow",
                        f"Prefill throughput is below {SLOW_PREFILL_TPS:g} tok/s.", evidence,
                        (Recommendation("Increase micro-batch size.", "Larger batches may improve prefill.",
                                        Impact.MEDIUM, "--ubatch-size"),))]
    return [Finding("prefill", "prefill.ok", Severity.OK, "Prefill throughput is healthy",
                    f"Prefill throughput is at or above {SLOW_PREFILL_TPS:g} tok/s.", evidence)]
