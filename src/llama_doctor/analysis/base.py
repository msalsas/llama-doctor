from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum

from llama_doctor.models import (
    MetricsSnapshot,
    ModelsMeta,
    ProbeResult,
    PropsSnapshot,
    ServerSnapshot,
)


class Severity(IntEnum):
    INSUFFICIENT_DATA = 0
    OK = 1
    INFO = 2
    WARN = 3
    CRITICAL = 4


class Impact(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


Evidence = Mapping[str, float | int | str | bool | None]


@dataclass(frozen=True)
class Recommendation:
    action: str
    rationale: str
    impact: Impact
    flag: str | None = None


@dataclass(frozen=True)
class Finding:
    analyzer_id: str
    code: str
    severity: Severity
    title: str
    detail: str
    evidence: Evidence = field(default_factory=dict)
    recommendations: tuple[Recommendation, ...] = ()

    @property
    def max_impact(self) -> int:
        return max((int(recommendation.impact) for recommendation in self.recommendations), default=0)


@dataclass(frozen=True)
class HostHints:
    vram_gib: float | None = None
    ram_gib: float | None = None


@dataclass(frozen=True)
class DiagnosisContext:
    snapshot: ServerSnapshot
    probe: ProbeResult | None = None
    hints: HostHints = HostHints()
    # Dashboard samples inside the rate window, oldest first and current last; empty for one-shot.
    window: tuple[MetricsSnapshot, ...] = ()

    @property
    def metrics(self) -> MetricsSnapshot | None:
        return self.snapshot.metrics

    @property
    def props(self) -> PropsSnapshot | None:
        return self.snapshot.server_props

    @property
    def models(self) -> ModelsMeta | None:
        return self.snapshot.models


AnalyzerFunc = Callable[[DiagnosisContext], list[Finding]]


@dataclass(frozen=True)
class AnalyzerSpec:
    id: str
    title: str
    func: AnalyzerFunc


@dataclass(frozen=True)
class Report:
    findings: tuple[Finding, ...]

    @property
    def overall(self) -> Severity:
        return max((finding.severity for finding in self.findings), default=Severity.INSUFFICIENT_DATA)

    @property
    def exit_code(self) -> int:
        if any(finding.severity == Severity.CRITICAL for finding in self.findings):
            return 2
        if any(finding.severity == Severity.WARN for finding in self.findings):
            return 1
        return 0

    def recommendations(self) -> list[Recommendation]:
        ordered = sorted(
            (recommendation for finding in self.findings for recommendation in finding.recommendations),
            key=lambda recommendation: -recommendation.impact,
        )
        unique: list[Recommendation] = []
        seen: set[str] = set()
        for recommendation in ordered:
            key = recommendation.flag if recommendation.flag is not None else recommendation.action
            if key not in seen:
                seen.add(key)
                unique.append(recommendation)
        return unique

    def by_code(self, code: str) -> Finding | None:
        return next((finding for finding in self.findings if finding.code == code), None)
