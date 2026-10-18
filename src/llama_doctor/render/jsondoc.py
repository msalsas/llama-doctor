from typing import Any

from llama_doctor import __version__
from llama_doctor.analysis.base import Report
from llama_doctor.models import ProbeResult, ServerSnapshot
from llama_doctor.serialize import (
    errors_to_dict,
    models_to_dict,
    probe_to_dict,
    props_to_dict,
    rates_to_dict,
    report_to_dict,
)


def diagnosis_document(
    report: Report,
    snapshot: ServerSnapshot,
    *,
    target: str,
    probe: ProbeResult | None = None,
) -> dict[str, Any]:
    """Build the JSON-serializable diagnosis document."""
    serialized_report = report_to_dict(report)
    return {
        "tool": "llama-doctor",
        "version": __version__,
        "target": target,
        "taken_at": snapshot.taken_at,
        "overall": serialized_report["overall"],
        "exit_code": serialized_report["exit_code"],
        "server": props_to_dict(snapshot.server_props),
        "model": models_to_dict(snapshot.models),
        "rates": rates_to_dict(snapshot.metrics),
        "probe": probe_to_dict(probe),
        "findings": serialized_report["findings"],
        "recommendations": serialized_report["recommendations"],
        "errors": errors_to_dict(snapshot.errors),
    }
