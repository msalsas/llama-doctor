from .base import DiagnosisContext, Finding, Impact, Recommendation, Severity
from .signals import MIN_PREFILL_DECODE_RATIO, decode_rate, is_cpu_bound, prefill_rate

SLOW_PREFILL_TPS = 300.0
HIGH_CACHE_HIT_RATIO = 0.5


def _evidence(ctx: DiagnosisContext) -> dict[str, float | str | None]:
    prefill = prefill_rate(ctx)
    decode = decode_rate(ctx)
    return {"prefill_tps": prefill.value if prefill else None, "decode_tps": decode.value if decode else None,
            "source": prefill.source if prefill else None}


def _versus_decode(prefill_tps: float, decode_tps: float | None) -> str:
    if decode_tps is None or decode_tps <= 0:
        return "there is no decode rate to compare it with"
    return (f"it is {prefill_tps / decode_tps:.1f}x decode ({decode_tps:.0f} tok/s), above the "
            f"{MIN_PREFILL_DECODE_RATIO:g}x line below which prefill counts as CPU-bound")


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
                Recommendation("If the loaded model uses PLE/N-gram tables, put them on CPU or mmap.",
                               "Only models that ship these tables (Qwen3-Next class) have them; skip it otherwise.",
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
        # The probe disables the prefix cache, so only lifetime rates are skewed by cache hits.
        hit = ctx.metrics.cache_hit_ratio if ctx.metrics and rate.source == "lifetime" else None
        evidence = {**evidence, "cache_hit_ratio": hit}
        versus_decode = _versus_decode(rate.value, evidence["decode_tps"])
        if hit is not None and hit >= HIGH_CACHE_HIT_RATIO:
            return [Finding("prefill", "prefill.ok", Severity.OK, "Prefill throughput is healthy",
                            f"Prefill measures {rate.value:.0f} tok/s, under {SLOW_PREFILL_TPS:g} tok/s, but "
                            f"{hit * 100:.0f} % of prompt tokens come from the prefix cache, so the lifetime "
                            f"rate only covers short uncached tails and reads low; {versus_decode}.",
                            evidence)]
        if rate.source == "probe":
            cache_note = "measured by the probe with the prefix cache disabled"
        elif hit is None:
            cache_note = "with no prefix-cache hit data to explain it"
        else:
            cache_note = f"with only {hit * 100:.0f} % prefix-cache hits to explain it"
        return [Finding("prefill", "prefill.slow", Severity.WARN, "Prefill is slow",
                        f"Prefill runs at {rate.value:.0f} tok/s, under {SLOW_PREFILL_TPS:g} tok/s, "
                        f"{cache_note}; {versus_decode}. The wait before the first token "
                        "grows with prompt length.", evidence)]
    return [Finding("prefill", "prefill.ok", Severity.OK, "Prefill throughput is healthy",
                    f"Prefill throughput is at or above {SLOW_PREFILL_TPS:g} tok/s.", evidence)]
