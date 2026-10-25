import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any


def _p(type_: str, description: str, enum: tuple[str, ...] | None = None) -> dict[str, Any]:
    prop: dict[str, Any] = {"type": type_, "description": description}
    if enum is not None:
        prop["enum"] = list(enum)
    return prop


def _tool(
    name: str, description: str, properties: dict[str, Any], required: tuple[str, ...]
) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(required)}}}


TOOL_CATALOG: tuple[dict[str, Any], ...] = (
    _tool("run_command",
          "Run a shell command to completion and return its stdout, stderr and exit code.",
          {"command": _p("string", "Command to run."), "cwd": _p("string", "Working directory."),
           "timeout_s": _p("integer", "Timeout in seconds.")}, ("command",)),
    _tool("read_file", "Read a text file from the repository.",
          {"path": _p("string", "File path to read."), "offset": _p("integer", "Start line."),
           "limit": _p("integer", "Maximum lines.")}, ("path",)),
    _tool("patch_file", "Replace one exact occurrence of old_string with new_string in a file.",
          {"path": _p("string", "File to patch."), "old_string": _p("string", "Text to replace."),
           "new_string": _p("string", "Replacement text.")},
          ("path", "old_string", "new_string")),
    _tool("search_files", "Search file contents in the repository for a pattern.",
          {"pattern": _p("string", "Pattern to find."), "path": _p("string", "Directory."),
           "file_glob": _p("string", "File name glob filter.")}, ("pattern",)),
    _tool("write_file", "Create or overwrite a file with the given content.",
          {"path": _p("string", "File path to write."),
           "content": _p("string", "New file content.")},
          ("path", "content")),
    _tool("delegate_task", "Hand a self-contained subtask to a helper agent.",
          {"goal": _p("string", "What the helper must achieve."),
           "context": _p("string", "Subtask context."), "toolsets": _p("array", "Tool sets.")},
          ("goal",)),
    _tool("memory", "Store or remove a durable note.",
          {"action": _p("string", "Note action.", ("add", "replace", "remove")),
           "target": _p("string", "Note location.", ("memory", "user")),
           "content": _p("string", "Note content.")}, ("action", "target")),
    _tool("web_search", "Search the web and return result titles and URLs.",
          {"query": _p("string", "Search query."), "max_results": _p("integer", "Max results.")},
          ("query",)),
    _tool("browser_exec", "Open a URL in a headless browser and run a script on the page.",
          {"url": _p("string", "URL to open."), "script": _p("string", "Script to run.")},
          ("url",)),
    _tool("clarify", "Ask the user a clarifying question.",
          {"question": _p("string", "Question to ask."), "choices": _p("array", "Choices.")},
          ("question",)),
    _tool("skill_view", "Show the instructions of an installed skill.",
          {"name": _p("string", "Skill name.")}, ("name",)),
    _tool("terminal", "Send input to an already open interactive terminal session.",
          {"session_id": _p("string", "Terminal session id."), "input": _p("string", "Input.")},
          ("session_id", "input")),
    _tool("cronjob", "Create, list or delete scheduled jobs.",
          {"action": _p("string", "Job action.", ("create", "list", "delete")),
           "schedule": _p("string", "Cron schedule."), "command": _p("string", "Command.")},
          ("action",)),
    _tool("todo_list", "Manage the task list for the current session.",
          {"action": _p("string", "Task action.", ("add", "update", "remove", "list")),
           "id": _p("integer", "Task id."), "text": _p("string", "Task text."),
           "status": _p("string", "Task status.", ("pending", "in_progress", "completed"))},
          ("action",)),
    _tool("process_manage", "List, inspect or stop background processes.",
          {"action": _p("string", "Process action.", ("list", "status", "kill")),
           "pid": _p("integer", "Process id.")}, ("action",)),
)

TOOL_NAMES: tuple[str, ...] = tuple(t["function"]["name"] for t in TOOL_CATALOG)

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    t["function"]["name"]: t["function"]["parameters"] for t in TOOL_CATALOG
}

CONFORMANCE_SYSTEM = (
    "You are a coding agent working in a software repository. Use the provided tools when an "
    "action on files, commands or the web is required. Only call tools that are listed. If a "
    "request needs a tool that does not exist, say so briefly instead of calling a tool."
)

AGENT_SYSTEM = (
    "You are an autonomous coding agent operating in the user's repository. You complete tasks "
    "by calling the provided tools and reading their results. Inspect before you edit, keep "
    "changes minimal, and run the tests after each change. When the task is complete, reply "
    "with a short summary of what you changed. Never end a turn without either calling a tool "
    "or replying to the user."
)


@dataclass(frozen=True)
class ExpectedCall:
    name: str
    args: Mapping[str, str | int]


@dataclass(frozen=True)
class ConformanceCase:
    id: str
    title: str
    prompt: str
    expected: tuple[ExpectedCall, ...] = ()
    tool_choice: str = "auto"
    parallel: bool = False
    refusal: bool = False


CONFORMANCE_CASES: tuple[ConformanceCase, ...] = (
    ConformanceCase("select_read_file", "Select read_file for a file view",
                    "Show me the contents of src/app/main.py.",
                    (ExpectedCall("read_file", {"path": "src/app/main.py"}),)),
    ConformanceCase("select_run_command", "Select run_command for the test suite",
                    "Run the test suite with `pytest -q` from the repository root and tell me "
                    "what fails.", (ExpectedCall("run_command", {"command": "pytest"}),)),
    ConformanceCase("select_search_files", "Select search_files for a content search",
                    "Find every file in the project that contains the string TODO_REMOVE.",
                    (ExpectedCall("search_files", {"pattern": "TODO_REMOVE"}),)),
    ConformanceCase("args_required", "Fill every required patch_file argument",
                    "Use patch_file to change `debug: true` to `debug: false` in config.yaml.",
                    (ExpectedCall("patch_file",
                                  {"path": "config.yaml", "old_string": "debug: true",
                                   "new_string": "debug: false"}),)),
    ConformanceCase("args_enum", "Respect todo_list status enums",
                    "Mark item 3 on my todo list as completed.",
                    (ExpectedCall("todo_list",
                                  {"action": "update", "id": 3, "status": "completed"}),)),
    ConformanceCase("args_integer", "Pass max_results as an integer",
                    "Search the web for llama.cpp reasoning budget and return at most 5 results.",
                    (ExpectedCall("web_search",
                                  {"query": "reasoning budget", "max_results": 5}),)),
    ConformanceCase("parallel_calls", "Read both files in one turn",
                    "Read README.md and setup.cfg. Request both files in this single turn.",
                    (ExpectedCall("read_file", {"path": "README.md"}),
                     ExpectedCall("read_file", {"path": "setup.cfg"})), parallel=True),
    ConformanceCase("unknown_tool_refusal", "Refuse a tool that does not exist",
                    "Call the deploy_to_kubernetes tool to deploy the current build to "
                    "production.", refusal=True),
    ConformanceCase("tool_choice_none", "Summarize without tools",
                    "Read the file notes.txt and summarize it.", tool_choice="none"),
    ConformanceCase("tool_choice_auto_no_tool", "Answer a plain question",
                    "Explain in one sentence what a .gitignore file is for."),
)

SENTENCE_POOL: tuple[str, ...] = (
    "The ledger service stores every account movement as an immutable event in an append-only table.",
    "Balances are derived from events at read time and cached for a short interval per account.",
    "The ingestion worker batches incoming transfers and commits them inside a single database transaction.",
    "Each transfer carries an idempotency key so that retried requests never create duplicate entries.",
    "The reconciliation job compares daily totals against the bank statement and records any differences.",
    "Currency amounts are kept as integer minor units to avoid rounding errors in floating point math.",
    "The public API exposes accounts, transfers and statements through a small set of versioned endpoints.",
    "Authentication relies on short-lived tokens issued by the identity service and checked on every request.",
    "Rate limits apply per client and are enforced by a token bucket stored in the shared cache.",
    "The configuration loader reads defaults first and then overlays values from the environment.",
    "Logging uses structured records with a request identifier that is passed between services.",
    "Metrics for latency and error rates are exported to the monitoring stack and reviewed weekly.",
    "The schema migration runner applies changes in order and refuses to start when a step is missing.",
    "Integration tests start a disposable database container and load a small fixture of accounts.",
    "Unit tests cover the balance calculator, the fee rules and the currency conversion helpers.",
    "The fee engine evaluates rules from most specific to least specific and stops at the first match.",
    "Statements are rendered as plain text and as a compact machine readable format for partners.",
    "The archive process moves events older than the retention window into cheaper cold storage.",
    "A feature flag controls the new settlement path while it is rolled out to a subset of clients.",
    "The settlement module groups transfers by counterparty and sends one net payment per window.",
    "Errors returned to clients include a stable code, a short message and a link to documentation.",
    "The team prefers small pull requests with a clear description and a focused set of changes.",
    "Code review checks naming, error handling and whether new behavior is covered by tests.",
    "The deployment pipeline builds an image, runs the test suite and promotes the image by tag.",
    "Database connections are pooled, and the pool size is tuned to the number of worker processes.",
    "Background jobs record their progress so that a restarted worker resumes where it stopped.",
    "The audit trail keeps who changed what and when for every administrative action in the system.",
    "Timestamps are stored in coordinated universal time and converted only at the presentation layer.",
    "The parser for bank files accepts several historical layouts and normalizes them into one model.",
    "Validation happens at the boundary, so internal modules can assume well formed input values.",
    "The command line utility helps operators inspect accounts and replay events in a safe sandbox.",
    "Documentation lives next to the code and changes in the same commit as the behavior it describes.",
    "Dependency upgrades are grouped monthly and tested against the full integration suite first.",
    "The cache layer invalidates an account entry whenever a new event for that account is committed.",
    "Long running queries are moved to a read replica to keep the primary database responsive.",
    "The notification service sends a message to account owners when a large transfer settles.",
    "Retries use exponential backoff with jitter and give up after a bounded total waiting time.",
    "Partner integrations are isolated behind adapters so that a failing partner cannot block others.",
    "The data model separates accounts, holders and products to keep ownership rules simple.",
    "Every module exposes a narrow interface and keeps its storage details private to itself.",
    "Performance budgets are tracked for the slowest endpoints and checked before each release.",
    "The style guide asks for descriptive names, short functions and explicit error types.",
    "Incident reviews focus on contributing factors and end with a short list of follow up actions.",
    "The sandbox environment mirrors production settings but uses synthetic accounts and fake partners.",
    "Exchange rates are fetched from a provider once per hour and stored with their source and time.",
    "Report generation runs overnight and writes its output to a shared bucket for the finance team.",
    "Access to production data requires an approved ticket and is logged by the access gateway.",
    "The roadmap lists faster settlement, better partner onboarding and simpler statement exports.",
)

SYSTEM_FILLER_WORDS = 2350
RESULT_FILLER_WORDS = 2800


def filler(seed: int, words: int) -> str:
    state = seed
    paragraphs: list[str] = []
    sentences: list[str] = []
    count = 0
    while count < words:
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        sentence = SENTENCE_POOL[(state >> 33) % len(SENTENCE_POOL)]
        sentences.append(sentence)
        count += len(sentence.split())
        if len(sentences) == 6:
            paragraphs.append(" ".join(sentences))
            sentences = []
    if sentences:
        paragraphs.append(" ".join(sentences))
    return "\n\n".join(paragraphs)


def conformance_messages(case: ConformanceCase) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": CONFORMANCE_SYSTEM},
        {"role": "user", "content": case.prompt},
    ]


FEES_SOURCE = (
    "def apply_fee(amount: int, rate: float, minimum: int) -> int:\n"
    "    fee = int(amount * rate)\n"
    "    if fee < minimum:\n"
    "        return amount\n"
    "    return amount - fee\n"
)

SEARCH_RESULT = (
    "src/ledger/statements.py:41:        row.settlement_date = None\n"
    "src/ledger/statements.py:88:    columns = ('id', 'amount', 'settlement_date')\n"
    "tests/test_statements.py:17:    assert row.settlement_date is not None\n"
)


@dataclass(frozen=True)
class AgenticTurn:
    id: str
    title: str
    messages: tuple[dict[str, Any], ...]


def _system_message(seed: int) -> dict[str, Any]:
    return {"role": "system",
            "content": AGENT_SYSTEM + "\n\n## Project notes\n\n" + filler(seed, SYSTEM_FILLER_WORDS)}


def _assistant_call(name: str, args: Mapping[str, Any], call_id: str) -> dict[str, Any]:
    call = {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}
    return {"role": "assistant", "content": None, "tool_calls": [call]}


def _tool_message(call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _fix_failing_test_turn() -> tuple[dict[str, Any], ...]:
    user = {"role": "user", "content": "The test tests/test_fees.py::test_minimum_fee fails "
            "after the last change. Find the cause and fix it."}
    return (
        _system_message(100),
        user,
        _assistant_call("read_file", {"path": "docs/fees.md"}, "call_1"),
        _tool_message("call_1", filler(101, RESULT_FILLER_WORDS)),
        _assistant_call("read_file", {"path": "src/ledger/fees.py"}, "call_2"),
        _tool_message("call_2", FEES_SOURCE),
    )


def _parallel_investigation_turn() -> tuple[dict[str, Any], ...]:
    call_1 = {"id": "call_1", "type": "function",
              "function": {"name": "search_files",
                           "arguments": json.dumps({"pattern": "settlement_date"})}}
    call_2 = {"id": "call_2", "type": "function",
              "function": {"name": "read_file",
                           "arguments": json.dumps({"path": "docs/statements.md"})}}
    assistant = {"role": "assistant", "content": None, "tool_calls": [call_1, call_2]}
    user = {"role": "user", "content": "Statements for partner accounts are missing the "
            "settlement date. Investigate and propose a fix."}
    return (
        _system_message(200),
        user,
        assistant,
        _tool_message("call_1", SEARCH_RESULT),
        _tool_message("call_2", filler(201, RESULT_FILLER_WORDS)),
    )


def _after_command_turn() -> tuple[dict[str, Any], ...]:
    output = (
        "============ test session starts ============\n"
        + filler(301, RESULT_FILLER_WORDS)
        + "\nFAILED tests/test_rates.py::test_rate_expiry - AssertionError: rate expired too early\n"
        "1 failed, 214 passed in 9.81s"
    )
    return (
        _system_message(300),
        {"role": "user", "content": "Run the test suite and fix whatever fails."},
        _assistant_call("run_command", {"command": "pytest -q"}, "call_1"),
        _tool_message("call_1", output),
    )


@cache
def agentic_turns() -> tuple[AgenticTurn, ...]:
    return (
        AgenticTurn("fix_failing_test", "Fix a failing fee test", _fix_failing_test_turn()),
        AgenticTurn("parallel_investigation", "Investigate missing settlement dates",
                    _parallel_investigation_turn()),
        AgenticTurn("after_command", "Fix whatever the test run broke", _after_command_turn()),
    )
