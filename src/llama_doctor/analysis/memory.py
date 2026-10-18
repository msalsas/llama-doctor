import math

from .base import DiagnosisContext, Finding, Impact, Recommendation, Severity


def _valid(value: float | None, positive: bool = False) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value if value > 0 or (value == 0 and not positive) else None


def analyze(ctx: DiagnosisContext) -> list[Finding]:
    size_bytes = ctx.models.size_bytes if ctx.models else None
    if size_bytes is not None:
        weights, source = size_bytes / 2**30, "server"
    else:
        weights, source = _valid(ctx.hints.weights_gib, positive=True), "hint"
    vram = _valid(ctx.hints.vram_gib)
    if weights is None or vram is None:
        if weights is None and vram is None:
            detail = ("Provide --vram-gib and --weights-gib (the server does not report the model size), "
                      "and optionally --ram-gib.")
        elif weights is None:
            detail = "Provide --weights-gib; the server does not report the model size."
        else:
            detail = "Provide --vram-gib and optionally --ram-gib."
        return [Finding("memory", "memory.insufficient_data", Severity.INSUFFICIENT_DATA,
                        "Memory estimate unavailable", detail)]
    detail = "This counts weights only."
    if source == "hint":
        detail += " Weights come from --weights-gib, not from the server."
    evidence = {"weights_source": source}
    ram = _valid(ctx.hints.ram_gib)
    if ram is not None and weights > vram + ram:
        return [Finding("memory", "memory.exceeds_total", Severity.CRITICAL, "Weights exceed available memory",
                        detail, evidence, recommendations=(
                            Recommendation("Use a smaller quantization.", "Weights exceed VRAM and RAM.",
                                           Impact.HIGH),))]
    if weights > vram:
        return [Finding("memory", "memory.partial_offload", Severity.INFO, "Weights require partial offload",
                        detail, {"offloaded_gib": weights - vram, **evidence})]
    return [Finding("memory", "memory.fits_vram", Severity.OK, "Weights fit in VRAM", detail, evidence)]
