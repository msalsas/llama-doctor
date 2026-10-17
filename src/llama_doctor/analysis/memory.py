import math

from .base import DiagnosisContext, Finding, Impact, Recommendation, Severity


def analyze(ctx: DiagnosisContext) -> list[Finding]:
    size_bytes = ctx.models.size_bytes if ctx.models else None
    vram = ctx.hints.vram_gib
    if vram is not None and (not math.isfinite(vram) or vram < 0):
        vram = None
    if size_bytes is None or vram is None:
        return [Finding("memory", "memory.insufficient_data", Severity.INSUFFICIENT_DATA,
                        "Memory estimate unavailable", "Provide --vram-gib and optionally --ram-gib.")]
    weights = size_bytes / 2**30
    ram = ctx.hints.ram_gib
    if ram is not None and (not math.isfinite(ram) or ram < 0):
        ram = None
    if ram is not None and weights > vram + ram:
        return [Finding("memory", "memory.exceeds_total", Severity.CRITICAL, "Weights exceed available memory",
                        "This counts weights only.", recommendations=(
                            Recommendation("Use a smaller quantization.", "Weights exceed VRAM and RAM.",
                                           Impact.HIGH),))]
    if weights > vram:
        return [Finding("memory", "memory.partial_offload", Severity.INFO, "Weights require partial offload",
                        "This counts weights only.", {"offloaded_gib": weights - vram})]
    return [Finding("memory", "memory.fits_vram", Severity.OK, "Weights fit in VRAM",
                    "This counts weights only.")]
