from .base import DiagnosisContext, Finding, Impact, Recommendation, Severity


def analyze(ctx: DiagnosisContext) -> list[Finding]:
    props = ctx.props
    if props is None:
        return [Finding("props", "props.insufficient_data", Severity.INSUFFICIENT_DATA,
                        "Properties unavailable", "The /props endpoint did not provide data.")]
    if props.missing:
        return [Finding("props", "props.minimal", Severity.INSUFFICIENT_DATA,
                        "Properties are minimal", "Wrapper builds such as unsloth-studio omit required fields.",
                        {"missing": ",".join(props.missing)})]
    if (
        props.n_ctx is not None
        and ctx.models is not None
        and ctx.models.n_ctx_train is not None
        and props.n_ctx > ctx.models.n_ctx_train
    ):
        return [Finding("props", "props.ctx_exceeds_training", Severity.WARN,
                        "Context exceeds training context", "Configured context exceeds the training limit.",
                        recommendations=(Recommendation("Reduce context to the training limit.",
                                                        "Stay within trained context.", Impact.MEDIUM, "-c"),))]
    return [Finding("props", "props.ok", Severity.OK, "Properties are complete",
                    "Server properties are sufficient for diagnosis.")]
