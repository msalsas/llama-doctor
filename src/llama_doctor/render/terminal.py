from collections.abc import Mapping
from typing import TextIO

from llama_doctor import __version__
from llama_doctor.analysis.base import Report, Severity
from llama_doctor.models import ProbeResult, ServerSnapshot

_LABELS = {
    Severity.CRITICAL: "CRIT",
    Severity.WARN: "WARN",
    Severity.INFO: "INFO",
    Severity.OK: "OK",
    Severity.INSUFFICIENT_DATA: "N/A",
}
_COLORS = {
    Severity.CRITICAL: "\x1b[1;31m",
    Severity.WARN: "\x1b[33m",
    Severity.INFO: "\x1b[36m",
    Severity.OK: "\x1b[32m",
    Severity.INSUFFICIENT_DATA: "\x1b[2m",
}


def use_color(stream: TextIO, env: Mapping[str, str]) -> bool:
    """Return whether terminal color should be emitted to stream."""
    return stream.isatty() and env.get("NO_COLOR", "") == "" and env.get("TERM") != "dumb"


def _display(value: object) -> str:
    return "-" if value is None else str(value)


def _rate(value: float | None, unit: str) -> str:
    return "n/a" if value is None else f"{value:.1f} {unit}"


def _draft_length(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _label(severity: Severity, color: bool) -> str:
    label = f"{_LABELS[severity]:<4}"
    if color:
        return f"{_COLORS[severity]}{label}\x1b[0m"
    return label


def render_report(
    report: Report,
    snapshot: ServerSnapshot,
    *,
    target: str,
    probe: ProbeResult | None = None,
    color: bool = False,
) -> str:
    """Render a concise terminal diagnosis report."""
    props = snapshot.server_props
    metrics = snapshot.metrics
    prefill = metrics.prefill_tps if metrics else None
    decode = metrics.decode_tps if metrics else None
    cache_ratio = metrics.cache_hit_ratio if metrics else None
    draft_length = metrics.avg_draft_len if metrics else None
    draft_acceptance = metrics.draft_acceptance if metrics else None

    lines = [
        f"llama-doctor {__version__}  {target}",
        (
            "model  "
            f"{_display(props.model_name if props else None)}  "
            f"quant {_display(props.quant if props else None)}  "
            f"ctx {_display(props.n_ctx if props else None)}  "
            f"slots {_display(props.total_slots if props else None)}  "
            f"build {_display(props.build_info if props else None)}"
        ),
        "",
        "rates (lifetime averages)",
        f"  prefill       {_rate(prefill, 'tok/s')}",
        f"  decode        {_rate(decode, 'tok/s')}",
        f"  prefix cache  {_rate(cache_ratio * 100 if cache_ratio is not None else None, '%')}",
        (
            "  drafts        "
            f"len {_draft_length(draft_length)}  "
            f"acceptance {_rate(draft_acceptance * 100 if draft_acceptance is not None else None, '%')}"
        ),
    ]
    if probe is not None:
        lines.append(
            f"  probe         prefill {probe.prefill_tps:.1f} tok/s  "
            f"decode {probe.decode_tps:.1f} tok/s  ({probe.prompt_n}-token prompt)"
        )

    lines.extend(("", "findings"))
    for finding in report.findings:
        lines.append(f"  {_label(finding.severity, color)} {finding.code}  {finding.title}")
        if finding.severity != Severity.OK:
            lines.append(f"        {finding.detail}")

    lines.append("recommendations (by impact)")
    recommendations = report.recommendations()
    if not recommendations:
        lines.append("  none")
    for index, recommendation in enumerate(recommendations, start=1):
        flag = f"  [{recommendation.flag}]" if recommendation.flag else ""
        lines.append(
            f"  {index}. {recommendation.impact.name.lower():<6} "
            f"{recommendation.action}{flag}"
        )
    lines.append(f"overall: {report.overall.name} (exit {report.exit_code})")
    return "\n".join(lines)
