from .base import DiagnosisContext, Finding, Impact, Recommendation, Severity
from .signals import (
    HIGH_CACHE_HIT_RATIO,
    MIN_PREFILL_DECODE_RATIO,
    decode_rate,
    is_cpu_bound,
    mean_prompt_tokens,
    prefill_rate,
)

SLOW_PREFILL_TPS = 300.0
# Mean uncached prompt tokens per request above which a high cache hit no longer explains a low rate.
LARGE_PROMPT_TOKENS = 2048


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


def _gpu_layers(impact: Impact) -> Recommendation:
    return Recommendation("Check that every layer is on the GPU: set -ngl 999 and confirm the load log "
                          "offloads all layers.",
                          "Layers left on the CPU slow prompt processing the most.", impact, "-ngl")


def _moe_experts(impact: Impact) -> Recommendation:
    return Recommendation("Keep every layer on the GPU with -ngl 999 and move only expert tensors to the CPU: "
                          "--cpu-moe for all of them, or --n-cpu-moe N for the first N layers, "
                          "with N as low as VRAM allows.",
                          "Attention and shared weights stay on the GPU; only sparse expert weights run "
                          "from RAM.", impact, "--n-cpu-moe")


def analyze(ctx: DiagnosisContext) -> list[Finding]:
    rate = prefill_rate(ctx)
    if rate is None:
        total = ctx.metrics.prompt_tokens_total if ctx.metrics else None
        return [Finding("prefill", "prefill.insufficient_data", Severity.INSUFFICIENT_DATA,
                        "Prefill rate unavailable", "Run --probe or process at least 1000 prompt tokens.",
                        {"prompt_tokens_total": total})]
    evidence = _evidence(ctx)
    is_moe = ctx.props is not None and ctx.props.is_moe is True
    if is_cpu_bound(ctx):
        if is_moe:
            recommendations = (
                Recommendation("Use a smaller quantization.", "It can keep more experts in VRAM.", Impact.HIGH),
                _moe_experts(Impact.HIGH),
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
        measured = f"Prefill measures {rate.value:.0f} tok/s, under {SLOW_PREFILL_TPS:g} tok/s"
        if hit is not None and hit >= HIGH_CACHE_HIT_RATIO:
            # A high hit only explains a low rate when every uncached tail is short; one large
            # uncached prompt among many cached ones also gives a high hit.
            mean_tokens = mean_prompt_tokens(ctx)
            evidence = {**evidence, "mean_prompt_tokens": mean_tokens}
            cached = f"{hit * 100:.0f} % of prompt tokens come from the prefix cache"
            if mean_tokens is None:
                return [Finding("prefill", "prefill.insufficient_data", Severity.INSUFFICIENT_DATA,
                                "Prefill rate cannot be judged",
                                f"{measured}, but {cached}, so the lifetime rate mixes short uncached tails "
                                "with any large uncached prompt and cannot tell them apart. Run --probe for "
                                "a cache-free measurement, or watch it under serve, whose window shows the "
                                "uncached tokens per request.", evidence)]
            if mean_tokens > LARGE_PROMPT_TOKENS:
                return [Finding("prefill", "prefill.rate_unreliable", Severity.INSUFFICIENT_DATA,
                                "Prefill rate is dominated by large prompts",
                                f"{measured} and {cached}, yet requests in the window evaluated "
                                f"{mean_tokens:.0f} uncached tokens on average, above {LARGE_PROMPT_TOKENS}: "
                                "the average rate is dominated by large prompts, not short tails, so the "
                                "cache hit does not explain it. Run --probe to measure prefill without the "
                                "cache.", evidence)]
            return [Finding("prefill", "prefill.ok", Severity.OK, "Prefill throughput is healthy",
                            f"{measured}, but {cached} and requests in the window evaluated only "
                            f"{mean_tokens:.0f} uncached tokens on average, so the rate covers short tails "
                            f"and reads low; {versus_decode}.", evidence)]
        if rate.source == "probe":
            cache_note = "measured by the probe with the prefix cache disabled"
        elif hit is None:
            cache_note = "with no prefix-cache hit data to explain it"
        else:
            cache_note = f"with only {hit * 100:.0f} % prefix-cache hits to explain it"
        recommendations = (_gpu_layers(Impact.MEDIUM),) + ((_moe_experts(Impact.MEDIUM),) if is_moe else ())
        return [Finding("prefill", "prefill.slow", Severity.WARN, "Prefill is slow",
                        f"Prefill runs at {rate.value:.0f} tok/s, under {SLOW_PREFILL_TPS:g} tok/s, "
                        f"{cache_note}; {versus_decode}. The wait before the first token "
                        "grows with prompt length.", evidence, recommendations)]
    return [Finding("prefill", "prefill.ok", Severity.OK, "Prefill throughput is healthy",
                    f"Prefill throughput is at or above {SLOW_PREFILL_TPS:g} tok/s.", evidence)]
