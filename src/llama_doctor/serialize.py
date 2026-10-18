import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from .analysis.base import Finding, Recommendation, Report, Severity
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


def display_number(value: Any) -> Any:
    """Round a float for display: 1 decimal, or 3 below 1 so ratios stay readable."""
    if not isinstance(value, float):
        return value
    if not math.isfinite(value):
        return None
    return round(value, 1 if abs(value) >= 1 else 3)


def finding_to_dict(finding: Finding) -> dict[str, Any]:
    evidence = {key: display_number(value) for key, value in finding.evidence.items()}
    return {"analyzer": finding.analyzer_id, "code": finding.code,
            "severity": severity_name(finding.severity), "title": finding.title,
            "detail": finding.detail, "evidence": evidence,
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


def probe_to_dict(probe: ProbeResult | None) -> dict[str, Any] | None:
    return _snapshot_to_dict(probe)


def errors_to_dict(errors: Mapping[str, EndpointError]) -> dict[str, str]:
    return {endpoint: error.reason for endpoint, error in errors.items()}
