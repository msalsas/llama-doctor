from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from .family import Family


class Symptom(Enum):
    EMPTY_TURN = "empty_turn"
    THOUGHT_RUNAWAY = "thought_runaway"
    MALFORMED_TOOL_ARGS = "malformed_tool_args"
    TEMPLATE_LEAK = "template_leak"
    HALLUCINATED_TOOL = "hallucinated_tool"
    WRONG_TOOL = "wrong_tool"
    TOOL_CHOICE_IGNORED = "tool_choice_ignored"
    NO_PARALLEL_CALLS = "no_parallel_calls"
    REQUEST_ERROR = "request_error"
    CONTEXT_TOO_SMALL = "context_too_small"


@dataclass(frozen=True)
class Remedy:
    symptom: Symptom
    title: str
    generic: str
    llama_cpp: str | None
    strata: str | None


@dataclass(frozen=True)
class AppliedRemedy:
    symptom: Symptom
    title: str
    generic: str
    fix: str | None


REMEDIES: dict[Symptom, Remedy] = {
    Symptom.EMPTY_TURN: Remedy(
        Symptom.EMPTY_TURN,
        "Empty turn: 200 OK with no text and no tool call",
        "The model ended its turn with nothing an agent can act on, so the agent stalls. "
        "This usually means the reply was spent inside reasoning that never closed. Cap the "
        "reasoning length on the server, then run agentcheck again.",
        "Relaunch with --reasoning-budget 2048.",
        "Set reasoning_close_retry to on (fixes a reply that ends inside thinking with no "
        "answer and no tool call) and set reasoning_budget_tokens (0 = off; the per-request "
        "key thinking_budget_tokens is also accepted).",
    ),
    Symptom.THOUGHT_RUNAWAY: Remedy(
        Symptom.THOUGHT_RUNAWAY,
        "Thought runaway: reasoning but no answer",
        "The model produced reasoning but no answer and no tool call before the token "
        "limit. Agents treat this as a stalled turn.",
        "Relaunch with --reasoning-budget 2048.",
        "Set reasoning_budget_tokens (0 = off; per-request thinking_budget_tokens also "
        "works), reasoning_close_retry to on, reasoning_loop_recovery to recover, and "
        "reasoning_effort to the default level you want.",
    ),
    Symptom.MALFORMED_TOOL_ARGS: Remedy(
        Symptom.MALFORMED_TOOL_ARGS,
        "Malformed tool arguments",
        "Tool-call arguments were not valid JSON or did not match the tool's schema. "
        "Agents reject these calls or run them with the wrong arguments.",
        "Ensure the server runs with --jinja, check the chat template, and try --grammar.",
        None,
    ),
    Symptom.TEMPLATE_LEAK: Remedy(
        Symptom.TEMPLATE_LEAK,
        "Tool-call markup leaked into the reply text",
        "Tool-call markup appeared in the reply text instead of the tool_calls field, so "
        "the agent never sees the call.",
        "Ensure the server runs with --jinja and check the chat template.",
        None,
    ),
    Symptom.HALLUCINATED_TOOL: Remedy(
        Symptom.HALLUCINATED_TOOL,
        "Call to a tool that does not exist",
        "The model called a tool that is not in the request. Agents return an error for "
        "it, which costs a turn. A model tuned for tool use usually fixes this.",
        None,
        None,
    ),
    Symptom.WRONG_TOOL: Remedy(
        Symptom.WRONG_TOOL,
        "Wrong tool or no tool",
        "The model picked the wrong tool, or answered when a tool call was expected. This "
        "is a model capability limit; try a model tuned for tool use.",
        None,
        None,
    ),
    Symptom.TOOL_CHOICE_IGNORED: Remedy(
        Symptom.TOOL_CHOICE_IGNORED,
        "tool_choice none ignored",
        "The model called a tool although the request set tool_choice to none. Agents that "
        "use none for summary turns will receive unexpected calls.",
        None,
        None,
    ),
    Symptom.NO_PARALLEL_CALLS: Remedy(
        Symptom.NO_PARALLEL_CALLS,
        "No parallel tool calls",
        "The model made one call where two were requested in a single turn. Agents still "
        "work, but each extra call costs a full round trip.",
        None,
        None,
    ),
    Symptom.REQUEST_ERROR: Remedy(
        Symptom.REQUEST_ERROR,
        "Requests failed",
        "Some requests returned an HTTP error or timed out; see the reason on each case. "
        "If replays timed out, raise --timeout.",
        "If every request with tools fails, ensure the server runs with --jinja.",
        None,
    ),
    Symptom.CONTEXT_TOO_SMALL: Remedy(
        Symptom.CONTEXT_TOO_SMALL,
        "Context too small for agent prompts",
        "The server context is below the 12288 tokens the replay needs. Coding agents "
        "routinely send prompts of 10k tokens or more.",
        "Relaunch with a larger -c.",
        None,
    ),
}


def remedies_for(
    symptoms: Iterable[Symptom], family: Family
) -> tuple[AppliedRemedy, ...]:
    wanted = set(symptoms)
    applied = []
    for symptom in Symptom:
        if symptom not in wanted:
            continue
        remedy = REMEDIES[symptom]
        if family is Family.LLAMA_CPP:
            fix = remedy.llama_cpp
        elif family is Family.STRATA:
            fix = remedy.strata
        else:
            fix = None
        applied.append(AppliedRemedy(symptom, remedy.title, remedy.generic, fix))
    return tuple(applied)
