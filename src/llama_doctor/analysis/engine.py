from collections.abc import Sequence

from .base import AnalyzerSpec, DiagnosisContext, Finding, Report, Severity


def run_analyzers(ctx: DiagnosisContext, analyzers: Sequence[AnalyzerSpec]) -> Report:
    indexed: list[tuple[int, Finding]] = []
    for index, spec in enumerate(analyzers):
        try:
            findings = spec.func(ctx)
        except Exception as exc:  # noqa: BLE001
            findings = [
                Finding(
                    spec.id,
                    f"{spec.id}.error",
                    Severity.INSUFFICIENT_DATA,
                    f"{spec.title}: analyzer failed",
                    type(exc).__name__,
                )
            ]
        indexed.extend((index, finding) for finding in findings)
    indexed.sort(key=lambda item: (-item[1].severity, -item[1].max_impact, item[0]))
    return Report(tuple(finding for _, finding in indexed))
