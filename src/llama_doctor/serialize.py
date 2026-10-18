import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from .analysis.base import Finding, Recommendation, Report, Severity
from .analysis.signals import context_used_ratio
from .models import EndpointError, MetricsSnapshot, ModelsMeta, ProbeResult, PropsSnapshot


def severity_name(severity: Severity) -> str:
    return {
        Severity.INSUFFICIENT_DATA: "insufficient_data",
        Severity.OK: "ok",
        Severity.INFO: "info",
        Severity.WARN: "warn",
        Severity.CRITICAL: "critical",
    }[severity]


def recommendation_to_dict(recommendation: Recommendation) -> dict[str, Any]:
    return {"action": recommendation.action, "rationale": recommendation.rationale,
            "impact": recommendation.impact.name.lower(), "flag": recommendation.flag}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def finding_to_dict(finding: Finding) -> dict[str, Any]:
    return {"analyzer": finding.analyzer_id, "code": finding.code,
            "severity": severity_name(finding.severity), "title": finding.title,
            "detail": finding.detail, "evidence": _json_safe(finding.evidence),
            "recommendations": [recommendation_to_dict(item) for item in finding.recommendations]}


def report_to_dict(report: Report) -> dict[str, Any]:
    return {"overall": severity_name(report.overall), "exit_code": report.exit_code,
            "findings": [finding_to_dict(item) for item in report.findings],
            "recommendations": [recommendation_to_dict(item) for item in report.recommendations()]}


def _snapshot_to_dict(snapshot: Any) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return _json_safe(asdict(snapshot))


def props_to_dict(props: PropsSnapshot | None) -> dict[str, Any] | None:
    return _snapshot_to_dict(props)


def models_to_dict(models: ModelsMeta | None) -> dict[str, Any] | None:
    return _snapshot_to_dict(models)


def rates_to_dict(metrics: MetricsSnapshot | None) -> dict[str, float | None]:
    keys = ("prefill_tps", "decode_tps", "cache_hit_ratio", "avg_draft_len", "draft_acceptance",
            "kv_usage_ratio", "n_tokens_max")
    if metrics is None:
        return {key: None for key in keys}
    return _json_safe({"prefill_tps": metrics.prefill_tps, "decode_tps": metrics.decode_tps,
            "cache_hit_ratio": metrics.cache_hit_ratio, "avg_draft_len": metrics.avg_draft_len,
            "draft_acceptance": metrics.draft_acceptance, "kv_usage_ratio": metrics.kv_cache_usage_ratio,
            "n_tokens_max": metrics.n_tokens_max})


def format_tokens(value: float | None) -> str | None:
    """Compact token count: 999, 78.5k, 1.0M."""
    if value is None or not math.isfinite(value):
        return None
    if abs(value) < 1000:
        return str(round(value))
    if abs(value) < 999_950:  # 999950 would print as 1000.0k
        return f"{value / 1000:.1f}k"
    return f"{value / 1_000_000:.1f}M"


def context_to_dict(metrics: MetricsSnapshot | None, props: PropsSnapshot | None) -> dict[str, Any]:
    n_tokens_max = metrics.n_tokens_max if metrics else None
    n_ctx = props.n_ctx if props else None
    ratio = context_used_ratio(n_tokens_max, n_ctx)
    label = f"{format_tokens(n_tokens_max)} / {format_tokens(n_ctx)}" if ratio is not None else None
    return _json_safe({"used_ratio": ratio, "n_tokens_max": n_tokens_max, "n_ctx": n_ctx, "label": label})


def probe_to_dict(probe: ProbeResult | None) -> dict[str, Any] | None:
    return _snapshot_to_dict(probe)


def errors_to_dict(errors: Mapping[str, EndpointError]) -> dict[str, str]:
    return {endpoint: error.reason for endpoint, error in errors.items()}
