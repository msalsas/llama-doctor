import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, TextIO

from llama_doctor.client import FetchError

from .cases import (
    CONFORMANCE_CASES,
    TOOL_CATALOG,
    TOOL_NAMES,
    TOOL_SCHEMAS,
    ConformanceCase,
    ExpectedCall,
    agentic_turns,
    conformance_messages,
)
from .chat import ChatClient, ChatReply
from .errors import AgentCheckError
from .family import ServerFamily
from .remediation import AppliedRemedy, Symptom
from .schema import validate

CONFORMANCE_MAX_TOKENS = 2048
MIN_REPLAY_CTX = 12288
FATAL_STATUSES = frozenset({401, 403, 404, 405})
FATAL_REASONS = frozenset(
    {"connection refused", "unreachable", "invalid JSON", "no choices in response"}
)


class Verdict(IntEnum):
    PASS = 0
    WARN = 1
    FAIL = 2


class Outcome(Enum):
    ACTIONABLE = "actionable"
    EMPTY = "empty"
    THOUGHT_RUNAWAY = "thought_runaway"
    ERROR = "error"


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    title: str
    passed: bool
    symptoms: tuple[Symptom, ...]
    detail: str
    called: tuple[str, ...]


@dataclass(frozen=True)
class ConformanceResult:
    cases: tuple[CaseResult, ...]

    @property
    def score(self) -> int:
        return sum(1 for case in self.cases if case.passed)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def verdict(self) -> Verdict:
        return conformance_verdict(self.score, self.total)

    @property
    def symptoms(self) -> tuple[Symptom, ...]:
        seen: list[Symptom] = []
        for case in self.cases:
            for symptom in case.symptoms:
                if symptom not in seen:
                    seen.append(symptom)
        return tuple(seen)


@dataclass(frozen=True)
class ReplayResult:
    outcome: Outcome
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None = None


@dataclass(frozen=True)
class TurnResult:
    turn_id: str
    title: str
    replays: tuple[ReplayResult, ...]


@dataclass(frozen=True)
class EmptyTurnResult:
    turns: tuple[TurnResult, ...]
    skipped: str | None = None

    @property
    def counts(self) -> dict[Outcome, int]:
        counts = {outcome: 0 for outcome in Outcome}
        for turn in self.turns:
            for replay in turn.replays:
                counts[replay.outcome] += 1
        return counts

    @property
    def verdict(self) -> Verdict:
        if self.skipped is not None:
            return Verdict.WARN
        return empty_turn_verdict(self.counts)

    @property
    def symptoms(self) -> tuple[Symptom, ...]:
        if self.skipped is not None:
            return (Symptom.CONTEXT_TOO_SMALL,)
        counts = self.counts
        symptoms: list[Symptom] = []
        if counts[Outcome.EMPTY] > 0:
            symptoms.append(Symptom.EMPTY_TURN)
        if counts[Outcome.THOUGHT_RUNAWAY] > 0:
            symptoms.append(Symptom.THOUGHT_RUNAWAY)
        if counts[Outcome.ERROR] > 0:
            symptoms.append(Symptom.REQUEST_ERROR)
        return tuple(symptoms)


@dataclass(frozen=True)
class AgentCheckResult:
    family: ServerFamily
    conformance: ConformanceResult
    empty_turn: EmptyTurnResult
    remedies: tuple[AppliedRemedy, ...]

    @property
    def overall(self) -> Verdict:
        return Verdict(max(int(self.conformance.verdict), int(self.empty_turn.verdict)))

    @property
    def exit_code(self) -> int:
        return int(self.overall)


_MARKUP = re.compile(
    r"<tool_call>|\[TOOL_CALLS\]|<\|tool_call|<\|python_tag\|>|<function="
    r'|"name"\s*:\s*"(?:' + "|".join(TOOL_NAMES) + r')"'
)

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _brief(text: str) -> str:
    cleaned = " ".join(_CONTROL_CHARS.sub(" ", text).split())
    if len(cleaned) > 80:
        return cleaned[:80] + "…"
    return cleaned


def split_think(text: str | None) -> tuple[str, str]:
    if text is None:
        return ("", "")
    if "</think>" in text:
        before, _, after = text.rpartition("</think>")
        return (after.strip(), before.replace("<think>", "").strip())
    if "<think>" in text:
        before, _, after = text.partition("<think>")
        return (before.strip(), after.strip())
    return (text.strip(), "")


def has_tool_markup(text: str) -> bool:
    return _MARKUP.search(text) is not None


def classify(reply: ChatReply) -> Outcome:
    if reply.tool_calls:
        return Outcome.ACTIONABLE
    visible, inline_reasoning = split_think(reply.content)
    if visible:
        return Outcome.ACTIONABLE
    if (reply.reasoning or "").strip() or inline_reasoning:
        return Outcome.THOUGHT_RUNAWAY
    return Outcome.EMPTY


def evaluate_case(case: ConformanceCase, reply: ChatReply) -> CaseResult:
    symptoms: list[Symptom] = []
    messages: list[str] = []

    def _add(symptom: Symptom, message: str) -> None:
        if symptom not in symptoms:
            symptoms.append(symptom)
            messages.append(message)

    calls = reply.tool_calls
    called = tuple(_brief(call.name or "?") for call in calls)
    visible, inline_reasoning = split_think(reply.content)
    for call in calls:
        if call.name not in TOOL_SCHEMAS:
            _add(Symptom.HALLUCINATED_TOOL, f"called unknown tool {_brief(call.name or '?')!r}")
        elif call.error:
            _add(Symptom.MALFORMED_TOOL_ARGS, f"{call.name}: {call.error}")
        elif errors := validate(TOOL_SCHEMAS[call.name], call.arguments):
            _add(Symptom.MALFORMED_TOOL_ARGS, f"{call.name}: {errors[0]}")
    if not calls and has_tool_markup(visible):
        _add(Symptom.TEMPLATE_LEAK, "tool-call markup in reply text")
    if case.expected:
        if not calls and Symptom.TEMPLATE_LEAK not in symptoms:
            expected_names = ", ".join(exp.name for exp in case.expected)
            _add(
                _silent_symptom(reply, visible, inline_reasoning),
                f"expected {expected_names}, got no tool call",
            )
        elif case.parallel and 0 < len(calls) < len(case.expected):
            _add(
                Symptom.NO_PARALLEL_CALLS,
                f"expected {len(case.expected)} tool calls in one turn, got {len(calls)}",
            )
        elif calls and not _calls_match(case.expected, calls):
            expected_names = ", ".join(exp.name for exp in case.expected)
            if sorted(exp.name for exp in case.expected) == sorted(
                call.name or "?" for call in calls
            ):
                _add(
                    Symptom.WRONG_TOOL,
                    f"expected {expected_names} with the expected argument hints, "
                    f"got {', '.join(called)}",
                )
            else:
                _add(Symptom.WRONG_TOOL, f"expected {expected_names}, got {', '.join(called)}")
    else:
        allowed = [call for call in calls if case.refusal and call.name == "clarify"]
        stray = [call for call in calls if call not in allowed and call.name in TOOL_SCHEMAS]
        if stray:
            if case.tool_choice == "none":
                _add(Symptom.TOOL_CHOICE_IGNORED, "tool_choice none was ignored")
            else:
                names = ", ".join(_brief(call.name or "?") for call in stray)
                _add(Symptom.WRONG_TOOL, f"expected no tool call, got {names}")
        if not calls and not visible and Symptom.TEMPLATE_LEAK not in symptoms:
            _add(_silent_symptom(reply, visible, inline_reasoning), "no text and no tool call")
    passed = not symptoms
    detail = "ok" if passed else "; ".join(messages)
    return CaseResult(case.id, case.title, passed, tuple(symptoms), detail, called)


def _silent_symptom(reply: ChatReply, visible: str, inline_reasoning: str) -> Symptom:
    if (reply.reasoning or "").strip() or inline_reasoning:
        return Symptom.THOUGHT_RUNAWAY
    if visible:
        return Symptom.WRONG_TOOL
    return Symptom.EMPTY_TURN


def _calls_match(expected: tuple[ExpectedCall, ...], calls: tuple[Any, ...]) -> bool:
    remaining = list(calls)
    for exp in expected:
        hit = next(
            (
                call
                for call in remaining
                if call.name == exp.name
                and call.arguments is not None
                and _arg_matches(exp.args, call.arguments)
            ),
            None,
        )
        if hit is None:
            return False
        remaining.remove(hit)
    return True


def _arg_matches(expected: Mapping[str, str | int], actual: Mapping[str, Any]) -> bool:
    for key, want in expected.items():
        got = actual.get(key)
        if isinstance(want, str):
            if not isinstance(got, str) or want.lower() not in got.lower():
                return False
        elif not isinstance(got, int) or isinstance(got, bool) or got != want:
            return False
    return True


def conformance_verdict(score: int, total: int) -> Verdict:
    if score == total:
        return Verdict.PASS
    if score >= total - 2:
        return Verdict.WARN
    return Verdict.FAIL


def empty_turn_verdict(counts: Mapping[Outcome, int]) -> Verdict:
    total = sum(counts.values())
    if counts.get(Outcome.EMPTY, 0) > 0 or (
        total > 0 and counts.get(Outcome.ERROR, 0) == total
    ):
        return Verdict.FAIL
    if counts.get(Outcome.THOUGHT_RUNAWAY, 0) > 0 or counts.get(Outcome.ERROR, 0) > 0:
        return Verdict.WARN
    return Verdict.PASS


def is_fatal(exc: FetchError) -> bool:
    return exc.status in FATAL_STATUSES or (
        exc.status is None and exc.reason in FATAL_REASONS
    )


def _fatal_message(exc: FetchError) -> str:
    message = f"{exc.endpoint}: {exc.reason}"
    if exc.status in (401, 403):
        message += "; check --api-key-env"
    return message


def run_conformance(chat: ChatClient, progress: TextIO | None = None) -> ConformanceResult:
    results = []
    total = len(CONFORMANCE_CASES)
    for index, case in enumerate(CONFORMANCE_CASES, 1):
        if progress is not None:
            print(f"agentcheck: conformance {index}/{total} {case.id}", file=progress)
        try:
            reply = chat.chat(
                conformance_messages(case),
                TOOL_CATALOG,
                max_tokens=CONFORMANCE_MAX_TOKENS,
                tool_choice=case.tool_choice,
                parallel_tool_calls=True if case.parallel else None,
            )
        except FetchError as exc:
            if is_fatal(exc):
                raise AgentCheckError(_fatal_message(exc)) from None
            results.append(
                CaseResult(
                    case.id, case.title, False, (Symptom.REQUEST_ERROR,), exc.reason, ()
                )
            )
            continue
        results.append(evaluate_case(case, reply))
    return ConformanceResult(tuple(results))


def run_empty_turn(
    chat: ChatClient,
    *,
    replays: int,
    max_tokens: int,
    n_ctx: int | None,
    progress: TextIO | None = None,
) -> EmptyTurnResult:
    if n_ctx is not None and n_ctx < MIN_REPLAY_CTX:
        return EmptyTurnResult(
            (),
            skipped=f"context {n_ctx} tokens is below the {MIN_REPLAY_CTX} needed for the replay",
        )
    turns = []
    for turn in agentic_turns():
        replay_results = []
        for index in range(replays):
            if progress is not None:
                print(
                    f"agentcheck: empty-turn {turn.id} replay {index + 1}/{replays}",
                    file=progress,
                )
            try:
                reply = chat.chat(turn.messages, TOOL_CATALOG, max_tokens=max_tokens)
            except FetchError as exc:
                if is_fatal(exc):
                    raise AgentCheckError(_fatal_message(exc)) from None
                replay_results.append(
                    ReplayResult(Outcome.ERROR, None, None, None, exc.reason)
                )
                continue
            replay_results.append(
                ReplayResult(
                    classify(reply),
                    reply.finish_reason,
                    reply.prompt_tokens,
                    reply.completion_tokens,
                )
            )
        turns.append(TurnResult(turn.id, turn.title, tuple(replay_results)))
    return EmptyTurnResult(tuple(turns))
