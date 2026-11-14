from collections.abc import Sequence
from dataclasses import replace

from .base import AnalyzerSpec, DiagnosisContext, Finding, Report, Severity, requires_gpu
from .signals import GpuEvidence, gpu_evidence

_WITHHELD_UNKNOWN = ("GPU advice withheld: no evidence a GPU serves this model; "
                     "pass --vram-gib to include it.")
_WITHHELD_ZERO_VRAM = "GPU advice withheld: --vram-gib 0 declares no GPU."
_WITHHELD_ZERO_LAYERS = ("GPU advice withheld: the server reports 0 GPU layers; "
                         "pass --vram-gib if this host has a GPU.")


def _withheld_sentence(evidence: GpuEvidence) -> str:
    if evidence.source == "hint.vram_gib":
        return _WITHHELD_ZERO_VRAM
    if evidence.source == "props.n_gpu_layers":
        return _WITHHELD_ZERO_LAYERS
    return _WITHHELD_UNKNOWN


def guard_gpu(finding: Finding, evidence: GpuEvidence) -> Finding:
    if not any(requires_gpu(item) for item in finding.recommendations):
        return finding
    keys = {"gpu_status": evidence.status, "gpu_source": evidence.source}
    if evidence.status == "gpu":
        return replace(finding, evidence={**finding.evidence, **keys, "gpu_recs_withheld": 0})
    kept = tuple(item for item in finding.recommendations if not requires_gpu(item))
    withheld = len(finding.recommendations) - len(kept)
    return replace(
        finding,
        evidence={**finding.evidence, **keys, "gpu_recs_withheld": withheld},
        recommendations=kept,
        detail=f"{finding.detail} {_withheld_sentence(evidence)}",
    )


def run_analyzers(ctx: DiagnosisContext, analyzers: Sequence[AnalyzerSpec]) -> Report:
    evidence = gpu_evidence(ctx)
    indexed: list[tuple[int, Finding]] = []
    for index, spec in enumerate(analyzers):
        try:
            indexed.extend((index, guard_gpu(finding, evidence)) for finding in spec.func(ctx))
        except Exception as exc:  # noqa: BLE001
            indexed.append((
                index,
                guard_gpu(
                    Finding(
                        spec.id,
                        f"{spec.id}.error",
                        Severity.INSUFFICIENT_DATA,
                        f"{spec.title}: analyzer failed",
                        type(exc).__name__,
                    ),
                    evidence,
                ),
            ))
    indexed.sort(key=lambda item: (-item[1].severity, -item[1].max_impact, item[0]))
    return Report(tuple(finding for _, finding in indexed))
