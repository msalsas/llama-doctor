from typing import Any

from llama_doctor import __version__

from .checks import AgentCheckResult, Outcome
from .family import Family
from .schema import _brief

_OUTCOME_ORDER = (
    Outcome.ACTIONABLE,
    Outcome.EMPTY,
    Outcome.THOUGHT_RUNAWAY,
    Outcome.ERROR,
)


def render_text(result: AgentCheckResult, *, target: str) -> str:
    family = result.family
    lines = [
        f"llama-doctor {__version__} agentcheck",
        f"target: {target}",
        f"family: {family.name.value} ({family.evidence})",
        f"model: {family.model_id or '-'}",
        "",
    ]
    conformance = result.conformance
    lines.append(
        f"tool-call conformance: {conformance.score}/{conformance.total} "
        f"{conformance.verdict.name}"
    )
    for case in conformance.cases:
        if case.passed:
            lines.append(f"  PASS {case.case_id}")
        else:
            called = ""
            if case.called:
                called = f" (called: {', '.join(_brief(name) for name in case.called)})"
            lines.append(f"  FAIL {case.case_id}: {case.detail}{called}")
    lines.append("")
    empty = result.empty_turn
    if empty.skipped is not None:
        lines.append(f"empty-turn detector: skipped {empty.verdict.name}")
        lines.append(f"  {empty.skipped}")
    else:
        counts = empty.counts
        total = sum(counts.values())
        lines.append(
            f"empty-turn detector: {counts[Outcome.ACTIONABLE]}/{total} "
            f"actionable {empty.verdict.name}"
        )
        for turn in empty.turns:
            turn_counts = {outcome: 0 for outcome in Outcome}
            max_prompt: int | None = None
            for replay in turn.replays:
                turn_counts[replay.outcome] += 1
                if replay.prompt_tokens is not None and (
                    max_prompt is None or replay.prompt_tokens > max_prompt
                ):
                    max_prompt = replay.prompt_tokens
            parts = [f"{turn_counts[outcome]} {outcome.value}"
                     for outcome in _OUTCOME_ORDER if turn_counts[outcome] > 0]
            prompt = f"{max_prompt}" if max_prompt is not None else "n/a"
            lines.append(f"  {turn.turn_id}: {', '.join(parts)} (prompt {prompt} tokens)")
    lines.append("")
    if result.remedies:
        if family.name is Family.UNKNOWN:
            lines.append("remediation (unknown server family: generic advice only)")
        else:
            lines.append(f"remediation ({family.name.value})")
        for remedy in result.remedies:
            lines.append(f"  {remedy.symptom.value}: {remedy.title}")
            lines.append(f"    {remedy.generic}")
            if remedy.fix is not None:
                lines.append(f"    fix: {remedy.fix}")
    else:
        lines.append("remediation: none needed")
    lines.append("")
    lines.append(f"overall: {result.overall.name} (exit {result.exit_code})")
    return "\n".join(lines)


def agentcheck_document(result: AgentCheckResult, *, target: str) -> dict[str, Any]:
    return {
        "tool": "llama-doctor",
        "command": "agentcheck",
        "version": __version__,
        "target": target,
        "overall": result.overall.name.lower(),
        "exit_code": result.exit_code,
        "family": {
            "name": result.family.name.value,
            "evidence": result.family.evidence,
            "model": result.family.model_id,
            "n_ctx": result.family.n_ctx,
        },
        "conformance": {
            "verdict": result.conformance.verdict.name.lower(),
            "score": result.conformance.score,
            "total": result.conformance.total,
            "cases": [
                {
                    "id": case.case_id,
                    "title": case.title,
                    "passed": case.passed,
                    "symptoms": [symptom.value for symptom in case.symptoms],
                    "detail": case.detail,
                    "called": list(case.called),
                }
                for case in result.conformance.cases
            ],
        },
        "empty_turn": {
            "verdict": result.empty_turn.verdict.name.lower(),
            "skipped": result.empty_turn.skipped,
            "counts": {
                outcome.value: result.empty_turn.counts[outcome] for outcome in Outcome
            },
            "turns": [
                {
                    "id": turn.turn_id,
                    "title": turn.title,
                    "replays": [
                        {
                            "outcome": replay.outcome.value,
                            "finish_reason": replay.finish_reason,
                            "prompt_tokens": replay.prompt_tokens,
                            "completion_tokens": replay.completion_tokens,
                            "error": replay.error,
                        }
                        for replay in turn.replays
                    ],
                }
                for turn in result.empty_turn.turns
            ],
        },
        "remediation": [
            {
                "symptom": remedy.symptom.value,
                "title": remedy.title,
                "generic": remedy.generic,
                "fix": remedy.fix,
            }
            for remedy in result.remedies
        ],
    }
