from .base import DiagnosisContext, Finding, Impact, Recommendation, Severity


def analyze(ctx: DiagnosisContext) -> list[Finding]:
    if not ctx.snapshot.alive:
        detail = "; ".join(
            f"{endpoint}: {error.reason}" for endpoint, error in ctx.snapshot.errors.items()
        ) or "No endpoint responded."
        return [Finding("connectivity", "connectivity.unreachable", Severity.CRITICAL,
                        "Server unreachable", detail,
                        recommendations=(Recommendation("Check the server address and start the server.",
                                                        "No liveness endpoint responded.", Impact.HIGH),))]
    findings: list[Finding] = []
    metrics_error = next(
        (error for endpoint, error in ctx.snapshot.errors.items() if endpoint == "/metrics"),
        None,
    )
    if ctx.metrics is None:
        if metrics_error is not None and metrics_error.status in (404, 501):
            findings.append(Finding("connectivity", "connectivity.metrics_disabled", Severity.WARN,
                                    "Metrics disabled", "The server does not expose /metrics.",
                                    recommendations=(Recommendation("Enable metrics.", "Metrics enable diagnosis.",
                                                                    Impact.HIGH, "--metrics"),)))
        else:
            findings.append(Finding("connectivity", "connectivity.metrics_unavailable", Severity.WARN,
                                    "Metrics unavailable", "The /metrics endpoint did not provide data."))
    if ctx.props is None:
        findings.append(Finding("connectivity", "connectivity.props_unavailable", Severity.INFO,
                                "Properties unavailable", "The /props endpoint did not provide data."))
    return findings or [Finding("connectivity", "connectivity.ok", Severity.OK,
                                "Endpoints available", "Required server endpoints responded.")]
