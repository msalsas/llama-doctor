from .base import DiagnosisContext, Finding, Impact, Recommendation, Severity


def analyze(ctx: DiagnosisContext) -> list[Finding]:
    props = ctx.props
    if props is None:
        return [Finding("props", "props.insufficient_data", Severity.INSUFFICIENT_DATA,
                        "Properties unavailable", "The /props endpoint did not provide data.")]
    # Absent keys are listed in props.missing for the report; each analyzer that needs one
    # already says so, so they are not a finding of their own.
    findings: list[Finding] = []
    if (
        props.n_ctx is not None
        and ctx.models is not None
        and ctx.models.n_ctx_train is not None
        and props.n_ctx > ctx.models.n_ctx_train
    ):
        findings.append(Finding(
            "props", "props.ctx_exceeds_training", Severity.WARN,
            "Context exceeds training context", "Configured context exceeds the training limit.",
            recommendations=(Recommendation("Reduce context to the training limit.",
                                            "Stay within trained context.", Impact.MEDIUM, "-c"),),
        ))
    return findings or [Finding("props", "props.ok", Severity.OK, "Properties raise no issue",
                                "The configured context fits the model's training context or "
                                "cannot be compared with it.")]
