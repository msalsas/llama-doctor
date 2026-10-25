# Design: `llama-doctor agentcheck URL`

This is contract **T6** and follows the format of `docs/contracts/T1–T5`.

## 0. Ground rules (the same as the rest of the repo)

- Python ≥ 3.10 and **stdlib only**. Every network call goes through `LlamaClient._fetch_json`, so redirects, size limits, error mapping and secret handling stay the same as today.
- **No `#` comments** in new code. One-line docstrings are allowed only on `run_agentcheck`, `detect_family` and `validate`, the same way the repo documents its public entry points.
- Use `unittest` only, with **no real network**. Tests run against the in-thread `FakeLlamaServer` from `tests/support.py`.
- Ruff-clean at `line-length = 100`. All dataclasses are `frozen=True`. `StrEnum` is not available on 3.10, so enums are plain `Enum` with `str` values.
- Exit codes: `0` pass, `1` warnings only, `2` any phase FAIL, `3` tool or config error (`EXIT_TOOL_ERROR`, already in `cli.py`).
- agentcheck never sends reasoning controls such as `thinking_budget_tokens` or `reasoning_effort`. It measures the server **as it is configured** and only tells the user which controls to change.

## 1. Module layout and signatures

```
src/llama_doctor/client.py                 (+2 public methods)
src/llama_doctor/cli.py                    (+parser, +dispatch, +AGENTCHECK_TIMEOUT)
src/llama_doctor/agentcheck/__init__.py    run_agentcheck, AgentCheckError
src/llama_doctor/agentcheck/chat.py        ChatClient, ChatReply, ToolCall, parse_reply
src/llama_doctor/agentcheck/family.py      Family, ServerFamily, detect_family
src/llama_doctor/agentcheck/schema.py      validate
src/llama_doctor/agentcheck/cases.py       corpus (catalog, conformance cases, agentic turns, filler)
src/llama_doctor/agentcheck/checks.py      phases A and B, classification, verdicts
src/llama_doctor/agentcheck/remediation.py Symptom, Remedy, REMEDIES, remedies_for
src/llama_doctor/agentcheck/report.py      render_text, agentcheck_document
tests/agentcheck_fakes.py                  canned replies and a responder for the fake server
tests/support.py                           (+FakeLlamaServer.set_responder)
tests/test_agentcheck_schema.py  test_agentcheck_cases.py  test_agentcheck_chat.py
tests/test_agentcheck_family.py  test_agentcheck_checks.py  test_agentcheck_remediation.py
tests/test_agentcheck_report.py  test_agentcheck_cli.py
```

### 1.1 `client.py` additions (public wrappers; behaviour unchanged)

```python
def get_json(self, path: str) -> Any:
    return self._fetch_json(path)

def post_json(self, path: str, payload: Mapping[str, Any]) -> Any:
    return self._fetch_json(path, data=json.dumps(payload).encode("utf-8"))
```

Add `from collections.abc import Callable, Mapping`. The existing `complete()` may optionally switch to `post_json`, but it does not have to.

### 1.2 `agentcheck/__init__.py`

```python
class AgentCheckError(Exception): ...

def run_agentcheck(
    client: LlamaClient,
    *,
    replays: int = 6,
    max_tokens: int = 512,
    as_json: bool = False,
    out: TextIO,
    err: TextIO,
) -> int:
    """Detect the server family, run both phases, print the report and return the exit code."""
```

`AgentCheckError` is defined in `__init__.py`. `chat.py`, `family.py` and `checks.py` import it from `llama_doctor.agentcheck`. To avoid an import cycle, put the class in `agentcheck/errors.py` and re-export it from `__init__`. `run_agentcheck` **raises** `AgentCheckError`, and `cli.main` maps it to exit 3. That keeps `EXIT_TOOL_ERROR` defined in one place.

### 1.3 `chat.py`

```python
CHAT_PATH = "/v1/chat/completions"
GREEDY_SEED = 0

@dataclass(frozen=True)
class ToolCall:
    id: str | None
    name: str | None
    arguments: dict[str, Any] | None
    error: str | None

@dataclass(frozen=True)
class ChatReply:
    content: str | None
    reasoning: str | None
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None

def parse_reply(payload: Any) -> ChatReply: ...

class ChatClient:
    def __init__(self, client: LlamaClient, model: str | None) -> None: ...
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        tool_choice: str = "auto",
        parallel_tool_calls: bool | None = None,
    ) -> ChatReply: ...
```

**Request payload**, built exactly like this:

```python
payload = {"messages": list(messages), "tools": list(tools), "tool_choice": tool_choice,
           "temperature": 0, "seed": GREEDY_SEED, "max_tokens": max_tokens, "stream": False}
if self._model is not None: payload["model"] = self._model
if parallel_tool_calls is not None: payload["parallel_tool_calls"] = parallel_tool_calls
```

**`parse_reply`**:

```
if payload is not a dict, or choices is not a non-empty list, or choices[0] is not a dict,
   or choices[0]["message"] is not a dict:
    raise FetchError(CHAT_PATH, "no choices in response")
content   = _text(message.get("content"))
            str -> str; list -> "".join of part["text"] for dict parts whose "text" is a str; else None
reasoning = _text(message.get("reasoning_content")) or _text(message.get("reasoning"))
calls     = tuple(_parse_call(c) for c in message.get("tool_calls") or [] if isinstance(c, dict))
            (if tool_calls is not a list, use [])
finish    = choice.get("finish_reason") if it is a str, else None
usage     = payload.get("usage") if it is a dict, else {}
prompt_tokens, completion_tokens = safe_int(usage.get(...))
```

Do **not** use `safe_str` for content. It truncates at 128 characters.

**`_parse_call(raw)`**:

```
fn   = raw.get("function") if dict else {}
name = fn.get("name") if str else None
args = fn.get("arguments")
dict           -> ToolCall(id, name, args, None)
str            -> json.loads; ValueError -> error "arguments are not valid JSON"
                   parsed not dict -> error "arguments are not a JSON object"
anything else  -> error "arguments missing"
id = raw.get("id") if str else None
```

### 1.4 `family.py`

```python
class Family(Enum):
    LLAMA_CPP = "llama.cpp"
    STRATA = "strata"
    UNKNOWN = "unknown"

@dataclass(frozen=True)
class ServerFamily:
    name: Family
    evidence: str
    model_id: str | None
    n_ctx: int | None

AUTH_STATUSES = frozenset({401, 403})
UNREACHABLE_REASONS = frozenset({"connection refused", "unreachable", "timed out"})

def detect_family(client: LlamaClient) -> ServerFamily:
    """Classify the server from GET /props and GET /v1/models without assuming a family."""
```

### 1.5 `schema.py`

```python
def validate(schema: Mapping[str, Any], value: Any, path: str = "$") -> list[str]:
    """Return validation errors for the type/properties/required/enum subset of JSON Schema."""
```

### 1.6 `cases.py` (public names)

```python
TOOL_NAMES: tuple[str, ...]
TOOL_CATALOG: tuple[dict[str, Any], ...]
TOOL_SCHEMAS: dict[str, dict[str, Any]]
CONFORMANCE_SYSTEM: str
AGENT_SYSTEM: str
SENTENCE_POOL: tuple[str, ...]
SYSTEM_FILLER_WORDS: int
RESULT_FILLER_WORDS: int

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

@dataclass(frozen=True)
class AgenticTurn:
    id: str
    title: str
    messages: tuple[dict[str, Any], ...]

CONFORMANCE_CASES: tuple[ConformanceCase, ...]

def filler(seed: int, words: int) -> str: ...
def conformance_messages(case: ConformanceCase) -> list[dict[str, Any]]: ...
def agentic_turns() -> tuple[AgenticTurn, ...]: ...
```

`TOOL_SCHEMAS` maps each tool name to its `parameters` schema. `agentic_turns()` is wrapped in `functools.cache`. `conformance_messages(case)` returns `[{"role": "system", "content": CONFORMANCE_SYSTEM}, {"role": "user", "content": case.prompt}]`.

### 1.7 `remediation.py`

```python
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

REMEDIES: dict[Symptom, Remedy]

def remedies_for(symptoms: Iterable[Symptom], family: Family) -> tuple[AppliedRemedy, ...]:
    ...
```

`remedies_for` deduplicates and orders by `tuple(Symptom)` declaration order. `fix` is `llama_cpp` or `strata` depending on the family, and `None` for `UNKNOWN`.

### 1.8 `checks.py`

```python
CONFORMANCE_MAX_TOKENS = 2048
MIN_REPLAY_CTX = 12288
FATAL_STATUSES = frozenset({401, 403, 404, 405})
FATAL_REASONS = frozenset({"connection refused", "unreachable", "invalid JSON", "no choices in response"})

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
    score: int          (property)
    total: int          (property)
    verdict: Verdict    (property)
    symptoms: tuple[Symptom, ...]  (property)

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
    counts: dict[Outcome, int]  (property)
    verdict: Verdict            (property)
    symptoms: tuple[Symptom, ...]  (property)

@dataclass(frozen=True)
class AgentCheckResult:
    family: ServerFamily
    conformance: ConformanceResult
    empty_turn: EmptyTurnResult
    remedies: tuple[AppliedRemedy, ...]
    overall: Verdict  (property)
    exit_code: int    (property)

def split_think(text: str | None) -> tuple[str, str]: ...
def has_tool_markup(text: str) -> bool: ...
def classify(reply: ChatReply) -> Outcome: ...
def evaluate_case(case: ConformanceCase, reply: ChatReply) -> CaseResult: ...
def conformance_verdict(score: int, total: int) -> Verdict: ...
def empty_turn_verdict(counts: Mapping[Outcome, int]) -> Verdict: ...
def is_fatal(exc: FetchError) -> bool: ...
def run_conformance(chat: ChatClient, progress: TextIO | None = None) -> ConformanceResult: ...
def run_empty_turn(
    chat: ChatClient, *, replays: int, max_tokens: int, n_ctx: int | None,
    progress: TextIO | None = None,
) -> EmptyTurnResult: ...
```

### 1.9 `report.py`

```python
def render_text(result: AgentCheckResult, *, target: str) -> str: ...
def agentcheck_document(result: AgentCheckResult, *, target: str) -> dict[str, Any]: ...
```

## 2. Algorithms (pseudocode)

### 2.1 Family detection

```
detect_family(client):
    props, props_err   = _try_get(client, "/props")        returns (payload, None) or (None, FetchError)
    models, models_err = _try_get(client, "/v1/models")
    for e in (props_err, models_err):
        if e and e.status in AUTH_STATUSES:
            raise AgentCheckError(f"{e.endpoint}: {e.reason}; check --api-key-env")
    if props_err and models_err and props_err.status is None and models_err.status is None
       and props_err.reason in UNREACHABLE_REASONS and models_err.reason in UNREACHABLE_REASONS:
        raise AgentCheckError(f"server unreachable (/props: {props_err.reason}, /v1/models: {models_err.reason})")
    model_id = ModelsMeta.from_payload(models).model_id if models is not None else None
    marker = _strata_marker(models)
    if marker: return ServerFamily(STRATA, marker, model_id, None)
    if isinstance(props, dict):
        return ServerFamily(LLAMA_CPP, "GET /props returned 200", model_id,
                            PropsSnapshot.from_payload(props).n_ctx)
    return ServerFamily(UNKNOWN, "no /props and no Strata markers in /v1/models", model_id, None)

_strata_marker(models):
    entries = [e for e in models["data"] if dict]   (None if the shape is wrong)
    any(isinstance(e.get("meta"), dict) and "instanceId" in e["meta"])
        -> "GET /v1/models lists meta.instanceId"
    any(isinstance(e.get("owned_by"), str) and e["owned_by"].strip().lower().startswith("strata"))
        -> "GET /v1/models owned_by strata"
    else None
```

Notes:
- A `/props` 200 with a non-JSON body (for example an HTML proxy page) raises `FetchError("invalid JSON")` with status None. That counts as **not** llama.cpp and detection falls through to the Strata check.
- Detection never infers a family from the model name or the port.

### 2.2 Error policy (applies to every chat request)

```
is_fatal(exc) = exc.status in FATAL_STATUSES or (exc.status is None and exc.reason in FATAL_REASONS)
fatal     -> raise AgentCheckError(f"{exc.endpoint}: {exc.reason}"
                                   + ("; check --api-key-env" if exc.status in (401, 403) else ""))
non-fatal -> recorded on the case or replay as REQUEST_ERROR / Outcome.ERROR with reason=exc.reason
```

Non-fatal errors are: timed out, HTTP 400/413/5xx, and response too large.

### 2.3 Phase A: tool-call conformance

```
run_conformance(chat):
    results = []
    for i, case in enumerate(CONFORMANCE_CASES, 1):
        print(f"agentcheck: conformance {i}/{len} {case.id}", file=progress)  if progress
        try:
            reply = chat.chat(conformance_messages(case), TOOL_CATALOG,
                              max_tokens=CONFORMANCE_MAX_TOKENS, tool_choice=case.tool_choice,
                              parallel_tool_calls=True if case.parallel else None)
        except FetchError as exc:
            if is_fatal(exc): raise AgentCheckError(...)
            results.append(CaseResult(case.id, case.title, False, (REQUEST_ERROR,), exc.reason, ()))
            continue
        results.append(evaluate_case(case, reply))
    return ConformanceResult(tuple(results))

evaluate_case(case, reply):
    symptoms = []   (ordered, no duplicates; use _add(sym, message))
    calls = reply.tool_calls
    called = tuple(c.name or "?" for c in calls)
    visible, inline_reasoning = split_think(reply.content)
    for c in calls:
        if c.name not in TOOL_SCHEMAS: _add(HALLUCINATED_TOOL, f"called unknown tool {c.name!r}")
        elif c.error: _add(MALFORMED_TOOL_ARGS, f"{c.name}: {c.error}")
        elif errs := validate(TOOL_SCHEMAS[c.name], c.arguments): _add(MALFORMED_TOOL_ARGS, f"{c.name}: {errs[0]}")
    if not calls and has_tool_markup(reply.content or ""):
        _add(TEMPLATE_LEAK, "tool-call markup in reply text")
    if case.expected:
        if not calls and TEMPLATE_LEAK not in symptoms:
            _add(_silent_symptom(reply, visible, inline_reasoning), f"expected {names(case)}, got no tool call")
        elif case.parallel and 0 < len(calls) < len(case.expected):
            _add(NO_PARALLEL_CALLS, f"expected {len(case.expected)} tool calls in one turn, got {len(calls)}")
        elif calls and not _calls_match(case.expected, calls):
            _add(WRONG_TOOL, f"expected {names(case)}, got {', '.join(called)}")
    else:
        allowed = [c for c in calls if case.refusal and c.name == "clarify"]
        stray = [c for c in calls if c not in allowed and c.name in TOOL_SCHEMAS]
        if stray:
            _add(TOOL_CHOICE_IGNORED if case.tool_choice == "none" else WRONG_TOOL,
                 "tool_choice none was ignored" if case.tool_choice == "none"
                 else f"expected no tool call, got {', '.join(c.name for c in stray)}")
        if not calls and not visible and TEMPLATE_LEAK not in symptoms:
            _add(_silent_symptom(reply, visible, inline_reasoning), "no text and no tool call")
    passed = not symptoms
    detail = "ok" if passed else "; ".join(messages)
    return CaseResult(case.id, case.title, passed, tuple(symptoms), detail, called)

_silent_symptom(reply, visible, inline_reasoning):
    if (reply.reasoning or "").strip() or inline_reasoning: return THOUGHT_RUNAWAY
    if visible: return WRONG_TOOL
    return EMPTY_TURN

_calls_match(expected, calls):
    remaining = list(calls)
    for exp in expected:
        hit = first c in remaining with c.name == exp.name and c.arguments is not None
              and _arg_matches(exp.args, c.arguments)
        if hit is None: return False
        remaining.remove(hit)
    return True                     (extra calls are tolerated)

_arg_matches(expected, actual):
    for key, want in expected.items():
        got = actual.get(key)
        str want: got must be a str and want.lower() in got.lower()
        int want: got must be an int, not a bool, and equal to want
    return True if every key matches

conformance_verdict(score, total):
    PASS if score == total; WARN if score >= total - 2; else FAIL     (10 -> PASS, 8-9 -> WARN, <=7 -> FAIL)
```

### 2.4 Phase B: empty-turn detector

```
run_empty_turn(chat, replays, max_tokens, n_ctx):
    if n_ctx is not None and n_ctx < MIN_REPLAY_CTX:
        return EmptyTurnResult((), skipped=f"context {n_ctx} tokens is below the {MIN_REPLAY_CTX} needed for the replay")
    turns = []
    for turn in agentic_turns():
        replay_results = []
        for i in range(replays):
            print(f"agentcheck: empty-turn {turn.id} replay {i + 1}/{replays}", file=progress)  if progress
            try:
                reply = chat.chat(turn.messages, TOOL_CATALOG, max_tokens=max_tokens)
            except FetchError as exc:
                if is_fatal(exc): raise AgentCheckError(...)
                replay_results.append(ReplayResult(Outcome.ERROR, None, None, None, exc.reason))
                continue
            replay_results.append(ReplayResult(classify(reply), reply.finish_reason,
                                               reply.prompt_tokens, reply.completion_tokens))
        turns.append(TurnResult(turn.id, turn.title, tuple(replay_results)))
    return EmptyTurnResult(tuple(turns))

classify(reply):
    if reply.tool_calls: return ACTIONABLE
    visible, inline_reasoning = split_think(reply.content)
    if visible: return ACTIONABLE
    if (reply.reasoning or "").strip() or inline_reasoning: return THOUGHT_RUNAWAY
    return EMPTY        (for any finish_reason; spec cases are "stop"/"length"; the reason is kept as evidence)

split_think(text):
    if text is None: return ("", "")
    if "</think>" in text:
        before, _, after = text.rpartition("</think>")
        return (after.strip(), before.replace("<think>", "").strip())
    if "<think>" in text:
        before, _, after = text.partition("<think>")
        return (before.strip(), after.strip())
    return (text.strip(), "")

empty_turn_verdict(counts):    (skipped -> WARN)
    total = sum(counts.values())
    FAIL if counts[EMPTY] > 0 or (total > 0 and counts[ERROR] == total)
    WARN if counts[THOUGHT_RUNAWAY] > 0 or counts[ERROR] > 0
    PASS otherwise

EmptyTurnResult.symptoms:
    skipped -> (CONTEXT_TOO_SMALL,)
    else EMPTY_TURN if EMPTY > 0, THOUGHT_RUNAWAY if THOUGHT_RUNAWAY > 0, REQUEST_ERROR if ERROR > 0
```

The `has_tool_markup` regex, compiled once:

```python
_MARKUP = re.compile(
    r"<tool_call>|\[TOOL_CALLS\]|<\|tool_call|<\|python_tag\|>|<function="
    r'|"name"\s*:\s*"(?:' + "|".join(TOOL_NAMES) + r')"'
)
```

### 2.5 Overall result and run orchestration

```
AgentCheckResult.overall   = Verdict(max(conformance.verdict, empty_turn.verdict))
AgentCheckResult.exit_code = int(overall)          (0 / 1 / 2)

run_agentcheck(client, replays, max_tokens, as_json, out, err):
    family = detect_family(client)
    print(f"agentcheck: server family {family.name.value} ({family.evidence})", file=err)
    chat = ChatClient(client, family.model_id)
    print(f"agentcheck: tool-call conformance, {len(CONFORMANCE_CASES)} cases", file=err)
    conformance = run_conformance(chat, err)
    print(f"agentcheck: empty-turn detector, {len(agentic_turns())} turns x {replays} replays", file=err)
    empty = run_empty_turn(chat, replays=replays, max_tokens=max_tokens, n_ctx=family.n_ctx, progress=err)
    remedies = remedies_for((*conformance.symptoms, *empty.symptoms), family.name)
    result = AgentCheckResult(family, conformance, empty, remedies)
    text = json.dumps(agentcheck_document(result, target=client.base_url), indent=2, allow_nan=False) \
           if as_json else render_text(result, target=client.base_url)
    print(text, file=out)
    return result.exit_code
```

Exit-code matrix (`conformance × empty_turn → exit`):

| A \ B | PASS | WARN | FAIL |
|---|---|---|---|
| PASS | 0 | 1 | 2 |
| WARN | 1 | 1 | 2 |
| FAIL | 2 | 2 | 2 |
| `AgentCheckError` / argparse error / missing key env | 3 | 3 | 3 |

## 3. Remediation table (exact text, stored without backticks)

| Symptom | title | generic | llama_cpp | strata |
|---|---|---|---|---|
| `empty_turn` | Empty turn: 200 OK with no text and no tool call | The model ended its turn with nothing an agent can act on, so the agent stalls. This usually means the reply was spent inside reasoning that never closed. Cap the reasoning length on the server, then run agentcheck again. | Relaunch with --reasoning-budget 2048. | Set reasoning_close_retry to on (fixes a reply that ends inside thinking with no answer and no tool call) and set reasoning_budget_tokens (0 = off; the per-request key thinking_budget_tokens is also accepted). |
| `thought_runaway` | Thought runaway: reasoning but no answer | The model produced reasoning but no answer and no tool call before the token limit. Agents treat this as a stalled turn. | Relaunch with --reasoning-budget 2048. | Set reasoning_budget_tokens (0 = off; per-request thinking_budget_tokens also works), reasoning_close_retry to on, reasoning_loop_recovery to recover, and reasoning_effort to the default level you want. |
| `malformed_tool_args` | Malformed tool arguments | Tool-call arguments were not valid JSON or did not match the tool's schema. Agents reject these calls or run them with the wrong arguments. | Ensure the server runs with --jinja, check the chat template, and try --grammar. | None |
| `template_leak` | Tool-call markup leaked into the reply text | Tool-call markup appeared in the reply text instead of the tool_calls field, so the agent never sees the call. | Ensure the server runs with --jinja and check the chat template. | None |
| `hallucinated_tool` | Call to a tool that does not exist | The model called a tool that is not in the request. Agents return an error for it, which costs a turn. A model tuned for tool use usually fixes this. | None | None |
| `wrong_tool` | Wrong tool or no tool | The model picked the wrong tool, or answered when a tool call was expected. This is a model capability limit; try a model tuned for tool use. | None | None |
| `tool_choice_ignored` | tool_choice none ignored | The model called a tool although the request set tool_choice to none. Agents that use none for summary turns will receive unexpected calls. | None | None |
| `no_parallel_calls` | No parallel tool calls | The model made one call where two were requested in a single turn. Agents still work, but each extra call costs a full round trip. | None | None |
| `request_error` | Requests failed | Some requests returned an HTTP error or timed out; see the reason on each case. If replays timed out, raise --timeout. | If every request with tools fails, ensure the server runs with --jinja. | None |
| `context_too_small` | Context too small for agent prompts | The server context is below the 12288 tokens the replay needs. Coding agents routinely send prompts of 10k tokens or more. | Relaunch with a larger -c. | None |

Rules:
- For `Family.UNKNOWN`, `fix` is always `None` and the renderer prints only `generic`.
- A `None` cell means no verified server setting exists for that symptom. The renderer then prints only `generic`. Never fill in a guessed flag.
- Slot thrash does not appear as a row because agentcheck cannot observe it.

## 4. Corpus construction (`cases.py`, under 400 lines)

### 4.1 Tool catalog: 15 tools, in this order

Build the catalog with two helpers so that each tool fits in 3–6 lines:

```python
def _p(type_: str, description: str, enum: tuple[str, ...] | None = None) -> dict[str, Any]
def _tool(name: str, description: str, properties: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]
    -> {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(required)}}}
```

In the table, `*` marks a required property.

| # | name | description | properties |
|---|---|---|---|
| 1 | `run_command` | Run a shell command to completion and return its stdout, stderr and exit code. | command* string, cwd string, timeout_s integer |
| 2 | `read_file` | Read a text file from the repository. | path* string, offset integer, limit integer |
| 3 | `patch_file` | Replace one exact occurrence of old_string with new_string in a file. | path* string, old_string* string, new_string* string |
| 4 | `search_files` | Search file contents in the repository for a pattern. | pattern* string, path string, file_glob string |
| 5 | `write_file` | Create or overwrite a file with the given content. | path* string, content* string |
| 6 | `delegate_task` | Hand a self-contained subtask to a helper agent. | goal* string, context string, toolsets array |
| 7 | `memory` | Store or remove a durable note. | action* enum[add, replace, remove], target* enum[memory, user], content string |
| 8 | `web_search` | Search the web and return result titles and URLs. | query* string, max_results integer |
| 9 | `browser_exec` | Open a URL in a headless browser and run a script on the page. | url* string, script string |
| 10 | `clarify` | Ask the user a clarifying question. | question* string, choices array |
| 11 | `skill_view` | Show the instructions of an installed skill. | name* string |
| 12 | `terminal` | Send input to an already open interactive terminal session. | session_id* string, input* string |
| 13 | `cronjob` | Create, list or delete scheduled jobs. | action* enum[create, list, delete], schedule string, command string |
| 14 | `todo_list` | Manage the task list for the current session. | action* enum[add, update, remove, list], id integer, text string, status enum[pending, in_progress, completed] |
| 15 | `process_manage` | List, inspect or stop background processes. | action* enum[list, status, kill], pid integer |

`TOOL_NAMES = tuple(t["function"]["name"] for t in TOOL_CATALOG)`.

### 4.2 Conformance cases (exactly 10, so the score is X/10)

`CONFORMANCE_SYSTEM = "You are a coding agent working in a software repository. Use the provided tools when an action on files, commands or the web is required. Only call tools that are listed. If a request needs a tool that does not exist, say so briefly instead of calling a tool."`

| id | prompt | expectation |
|---|---|---|
| `select_read_file` | Show me the contents of src/app/main.py. | read_file{path:"src/app/main.py"} |
| `select_run_command` | Run the test suite with `pytest -q` from the repository root and tell me what fails. | run_command{command:"pytest"} |
| `select_search_files` | Find every file in the project that contains the string TODO_REMOVE. | search_files{pattern:"TODO_REMOVE"} |
| `args_required` | Use patch_file to change `debug: true` to `debug: false` in config.yaml. | patch_file{path:"config.yaml", old_string:"debug: true", new_string:"debug: false"} |
| `args_enum` | Mark item 3 on my todo list as completed. | todo_list{action:"update", id:3, status:"completed"} |
| `args_integer` | Search the web for llama.cpp reasoning budget and return at most 5 results. | web_search{query:"reasoning budget", max_results:5} |
| `parallel_calls` | Read README.md and setup.cfg. Request both files in this single turn. | parallel=True; read_file{path:"README.md"}, read_file{path:"setup.cfg"} |
| `unknown_tool_refusal` | Call the deploy_to_kubernetes tool to deploy the current build to production. | refusal=True; no calls (a clarify call is allowed) and non-empty text |
| `tool_choice_none` | Read the file notes.txt and summarize it. | tool_choice="none"; no calls, non-empty text |
| `tool_choice_auto_no_tool` | Explain in one sentence what a .gitignore file is for. | no calls, non-empty text |

### 4.3 Sentence pool and filler

**Rules**, each enforced by a test:
- Exactly 48 sentences, each 10–18 words, ASCII only.
- No digits and none of the characters `<>{}[]|`\``.
- No tool names.
- None of these lowercase words: `tool`, `call`, `think`, `reasoning`, `ignore`, `instruction`, `you`, `assistant`.
- Declarative prose about a fictional "ledger service" with no imperatives. If a sentence breaks a rule, edit the sentence, not the rule.

The pool to paste:

```
The ledger service stores every account movement as an immutable event in an append-only table.
Balances are derived from events at read time and cached for a short interval per account.
The ingestion worker batches incoming transfers and commits them inside a single database transaction.
Each transfer carries an idempotency key so that retried requests never create duplicate entries.
The reconciliation job compares daily totals against the bank statement and records any differences.
Currency amounts are kept as integer minor units to avoid rounding errors in floating point math.
The public API exposes accounts, transfers and statements through a small set of versioned endpoints.
Authentication relies on short-lived tokens issued by the identity service and checked on every request.
Rate limits apply per client and are enforced by a token bucket stored in the shared cache.
The configuration loader reads defaults first and then overlays values from the environment.
Logging uses structured records with a request identifier that is passed between services.
Metrics for latency and error rates are exported to the monitoring stack and reviewed weekly.
The schema migration runner applies changes in order and refuses to start when a step is missing.
Integration tests start a disposable database container and load a small fixture of accounts.
Unit tests cover the balance calculator, the fee rules and the currency conversion helpers.
The fee engine evaluates rules from most specific to least specific and stops at the first match.
Statements are rendered as plain text and as a compact machine readable format for partners.
The archive process moves events older than the retention window into cheaper cold storage.
A feature flag controls the new settlement path while it is rolled out to a subset of clients.
The settlement module groups transfers by counterparty and sends one net payment per window.
Errors returned to clients include a stable code, a short message and a link to documentation.
The team prefers small pull requests with a clear description and a focused set of changes.
Code review checks naming, error handling and whether new behavior is covered by tests.
The deployment pipeline builds an image, runs the test suite and promotes the image by tag.
Database connections are pooled, and the pool size is tuned to the number of worker processes.
Background jobs record their progress so that a restarted worker resumes where it stopped.
The audit trail keeps who changed what and when for every administrative action in the system.
Timestamps are stored in coordinated universal time and converted only at the presentation layer.
The parser for bank files accepts several historical layouts and normalizes them into one model.
Validation happens at the boundary, so internal modules can assume well formed input values.
The command line utility helps operators inspect accounts and replay events in a safe sandbox.
Documentation lives next to the code and changes in the same commit as the behavior it describes.
Dependency upgrades are grouped monthly and tested against the full integration suite first.
The cache layer invalidates an account entry whenever a new event for that account is committed.
Long running queries are moved to a read replica to keep the primary database responsive.
The notification service sends a message to account owners when a large transfer settles.
Retries use exponential backoff with jitter and give up after a bounded total waiting time.
Partner integrations are isolated behind adapters so that a failing partner cannot block others.
The data model separates accounts, holders and products to keep ownership rules simple.
Every module exposes a narrow interface and keeps its storage details private to itself.
Performance budgets are tracked for the slowest endpoints and checked before each release.
The style guide asks for descriptive names, short functions and explicit error types.
Incident reviews focus on contributing factors and end with a short list of follow up actions.
The sandbox environment mirrors production settings but uses synthetic accounts and fake partners.
Exchange rates are fetched from a provider once per hour and stored with their source and time.
Report generation runs overnight and writes its output to a shared bucket for the finance team.
Access to production data requires an approved ticket and is logged by the access gateway.
The roadmap lists faster settlement, better partner onboarding and simpler statement exports.
```

**Filler generator.** It uses a fixed 64-bit LCG. Do not use the `random` module or `hash()`.

```
filler(seed, words):
    state = seed
    paragraphs, sentences, count = [], [], 0
    while count < words:
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        sentence = SENTENCE_POOL[(state >> 33) % len(SENTENCE_POOL)]
        sentences.append(sentence); count += len(sentence.split())
        if len(sentences) == 6: paragraphs.append(" ".join(sentences)); sentences = []
    if sentences: paragraphs.append(" ".join(sentences))
    return "\n\n".join(paragraphs)
```

### 4.4 Agentic turns (3 turns, Hermes-style; each ends on a `tool` message)

`AGENT_SYSTEM = "You are an autonomous coding agent operating in the user's repository. You complete tasks by calling the provided tools and reading their results. Inspect before you edit, keep changes minimal, and run the tests after each change. When the task is complete, reply with a short summary of what you changed. Never end a turn without either calling a tool or replying to the user."`

All three turns share:
- System message: `AGENT_SYSTEM + "\n\n## Project notes\n\n" + filler(turn_seed, SYSTEM_FILLER_WORDS)`.
- Assistant messages: `{"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": ..., "arguments": json.dumps(args)}}]}`.
- Tool messages: `{"role": "tool", "tool_call_id": "call_1", "content": ...}`.

| id (seed) | user | history after the user message |
|---|---|---|
| `fix_failing_test` (100) | The test tests/test_fees.py::test_minimum_fee fails after the last change. Find the cause and fix it. | call_1 read_file{path:"docs/fees.md"} → `filler(101, RESULT_FILLER_WORDS)`; call_2 read_file{path:"src/ledger/fees.py"} → `FEES_SOURCE` |
| `parallel_investigation` (200) | Statements for partner accounts are missing the settlement date. Investigate and propose a fix. | one assistant message with call_1 search_files{pattern:"settlement_date"} and call_2 read_file{path:"docs/statements.md"} → `SEARCH_RESULT`, `filler(201, RESULT_FILLER_WORDS)` |
| `after_command` (300) | Run the test suite and fix whatever fails. | call_1 run_command{command:"pytest -q"} → `"============ test session starts ============\n" + filler(301, RESULT_FILLER_WORDS) + "\nFAILED tests/test_rates.py::test_rate_expiry - AssertionError: rate expired too early\n1 failed, 214 passed in 9.81s"` |

Fixed snippets:

```python
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
```

**Sizes.** Start with `SYSTEM_FILLER_WORDS = 2000` and `RESULT_FILLER_WORDS = 2400`. Each turn's `len(json.dumps(messages)) + len(json.dumps(TOOL_CATALOG))` must land between **32,000 and 40,000 characters**, which is 8–10k tokens at roughly 4 characters per token. Tune both constants until the size test passes near 36,000. The report prints the real `usage.prompt_tokens`, so users can see the actual size.

## 5. CLI changes (`cli.py`)

Add a module constant after `EXIT_TOOL_ERROR`:

```python
AGENTCHECK_TIMEOUT = 300.0
```

If `_positive_int` and `_positive_float` do not already exist in `cli.py`, add them above `build_parser` (argparse `type=` callables raising `argparse.ArgumentTypeError` for values <= 0). Do not apply them to `serve` or `diagnose` flags in this branch.

In `build_parser`, after the `serve_parser` block and before `return parser`:

```python
    agentcheck_parser = commands.add_parser("agentcheck")
    agentcheck_parser.add_argument("url", metavar="URL")
    agentcheck_parser.add_argument("--json", action="store_true")
    agentcheck_parser.add_argument("--api-key-env", metavar="VAR")
    agentcheck_parser.add_argument(
        "--timeout", type=_positive_float, default=AGENTCHECK_TIMEOUT, metavar="SEC"
    )
    agentcheck_parser.add_argument("--replays", type=_positive_int, default=6, metavar="N")
    agentcheck_parser.add_argument("--max-tokens", type=_positive_int, default=512, metavar="N")
```

In `main`, right after `if client is None: return EXIT_TOOL_ERROR`:

```python
    if args.command == "agentcheck":
        from llama_doctor.agentcheck import AgentCheckError, run_agentcheck

        try:
            return run_agentcheck(
                client,
                replays=args.replays,
                max_tokens=args.max_tokens,
                as_json=args.json,
                out=sys.stdout,
                err=sys.stderr,
            )
        except AgentCheckError as exc:
            print(f"llama-doctor: {exc}", file=sys.stderr)
            return EXIT_TOOL_ERROR
```

Flag notes:
- `--max-tokens` applies to phase B only. Phase A uses `CONFORMANCE_MAX_TOKENS = 2048`, so a reasoning model is not marked down in tool selection just for thinking.
- There is no `--no-slots` flag, because agentcheck never reads `/slots`.

## 6. Report output

### 6.1 Text (exact layout)

```
llama-doctor 0.1.0 agentcheck
target: http://127.0.0.1:8080
family: llama.cpp (GET /props returned 200)
model: Qwen3-30B-A3B

tool-call conformance: 9/10 WARN
  PASS select_read_file
  ...
  FAIL parallel_calls: expected 2 tool calls in one turn, got 1

empty-turn detector: 17/18 actionable FAIL
  fix_failing_test: 5 actionable, 1 empty (prompt 9104 tokens)
  parallel_investigation: 6 actionable (prompt 8870 tokens)
  after_command: 6 actionable (prompt 8712 tokens)

remediation (llama.cpp)
  empty_turn: Empty turn: 200 OK with no text and no tool call
    The model ended its turn with nothing an agent can act on, ...
    fix: Relaunch with --reasoning-budget 2048.

overall: FAIL (exit 2)
```

Formatting rules:
- Use the repo's existing version constant everywhere the sample shows `0.1.0`; do not hardcode a version string.
- `model: -` when the model id is unknown.
- Per-turn counts list only non-zero outcomes, in the order actionable, empty, thought_runaway, error. The prompt size is the maximum `prompt_tokens` across replays, or `n/a`.
- When phase B is skipped, print `empty-turn detector: skipped WARN` followed by the indented `skipped` reason.
- With no remedies, print `remediation: none needed`.
- For the unknown family, the header is `remediation (unknown server family: generic advice only)` and no `fix:` lines appear.
- A remedy with `fix is None` prints no `fix:` line.
- There is no ANSI colour.

### 6.2 JSON (`agentcheck_document`)

```json
{"tool": "llama-doctor", "command": "agentcheck", "version": "0.1.0",
 "target": "http://...", "overall": "fail", "exit_code": 2,
 "family": {"name": "llama.cpp", "evidence": "GET /props returned 200", "model": "...", "n_ctx": 32768},
 "conformance": {"verdict": "warn", "score": 9, "total": 10,
   "cases": [{"id": "parallel_calls", "title": "...", "passed": false,
              "symptoms": ["no_parallel_calls"], "detail": "...", "called": ["read_file"]}]},
 "empty_turn": {"verdict": "fail", "skipped": null,
   "counts": {"actionable": 17, "empty": 1, "thought_runaway": 0, "error": 0},
   "turns": [{"id": "fix_failing_test", "title": "...",
              "replays": [{"outcome": "empty", "finish_reason": "stop",
                           "prompt_tokens": 9104, "completion_tokens": 0, "error": null}]}]},
 "remediation": [{"symptom": "empty_turn", "title": "...", "generic": "...", "fix": "..."}]}
```

Verdict values are lowercase names. Model text is never included, only counts and finish reasons.

## 7. Test plan (all offline)

### 7.1 Support changes

**`tests/support.py`**: add `FakeLlamaServer.set_responder(path, func)`. `func(body: bytes) -> tuple[int, str | bytes, str]` returns status, body and content type. In `_dispatch`, a responder takes precedence over a route.

**`tests/agentcheck_fakes.py`** provides:
- `reply(content=None, tool_calls=(), reasoning=None, finish="stop", prompt_tokens=9000) -> str`, an OpenAI-shaped JSON string.
- `call(name, args, id="c1")`, whose arguments are a JSON string.
- `perfect_reply(case)`, which builds the correct canned answer from `case.expected`, or polite text for no-call cases.
- `class ChatResponder`: it parses the request JSON. If the last message's role is `tool`, it pops from `self.agentic` (a list of reply strings, defaulting to an actionable reply). Otherwise it looks up the case by user prompt and returns `self.overrides.get(case.id) or perfect_reply(case)`. It records `self.payloads`.
- `agent_server(props=True, models=..., responder=...)`, a context manager that wires everything together.

### 7.2 Enumerated tests (52)

**test_agentcheck_schema.py**
1. `validate` accepts a fully valid `todo_list` object, returning `[]`.
2. A missing required property produces `"$: missing required property 'path'"`.
3. A wrong enum value produces `"$.status: 'done' not in enum"`.
4. Type errors are caught: `"5"` for an integer, `True` for an integer, `3` for a string, and a non-object at the root.
5. A `type` list (`["string", "null"]`) is accepted, nested `properties` are checked recursively, and unknown keywords are ignored.
6. Every catalog schema is well-formed: object type, `required ⊆ properties`, and enum values are strings.

**test_agentcheck_cases.py**

7. The catalog has exactly the 15 names in the specified order, all unique.
8. There are 10 conformance cases with unique ids. Every expected tool is in the catalog, and every expected arg set passes `validate` against that tool's schema.
9. The sentence pool follows every rule: 48 sentences, 10–18 words, ASCII, no digits or banned characters or words, no tool names.
10. `filler` is deterministic and seed-sensitive. Its word count is at least the requested count and less than the request plus 18.
11. Each agentic turn's size is within 32,000–40,000 characters and ends with a `tool` message. Every `tool_call_id` matches an earlier call id, and the turn ids are unique.
12. The corpus is identical across processes: a subprocess with `PYTHONHASHSEED=1` and `PYTHONPATH=src` produces the same sha256 of the corpus JSON.

**test_agentcheck_chat.py**

13. `parse_reply` decodes string arguments, accepts dict arguments, and sets `error` for invalid JSON and for non-object JSON.
14. `parse_reply` reads both `reasoning_content` and `reasoning`, joins list content parts, keeps content longer than 128 characters intact, and reads usage tokens.
15. `parse_reply` raises `FetchError("no choices in response")` for `{}`, for empty choices, and for a missing message.
16. The request payload has temperature 0, seed 0, stream false, `max_tokens` and `tool_choice`. `model` is omitted when None. `parallel_tool_calls` appears only when set. The Bearer header is sent.

**test_agentcheck_family.py**

17. `/props` 200 gives llama.cpp, with `n_ctx` from props and the model id from `/v1/models`.
18. `/props` 404 plus `meta.instanceId` gives strata, with evidence that names instanceId.
19. `/props` 404 plus `owned_by: "Strata"` gives strata.
20. `/props` 404 and plain models give unknown with `n_ctx` None.
21. `/props` 200 with an HTML body is not treated as llama.cpp and falls through to the Strata or unknown check.
22. `/props` 401 raises `AgentCheckError` with a message containing "HTTP 401" and "--api-key-env".
23. Both endpoints refusing connections raises `AgentCheckError("server unreachable ...")`.

**test_agentcheck_checks.py**

24. `classify` returns ACTIONABLE for text content, and for tool calls with empty content.
25. `classify` returns EMPTY for no content, no calls and `finish_reason="stop"`.
26. `classify` returns EMPTY for `finish_reason="length"` with `content=""`.
27. `classify` returns THOUGHT_RUNAWAY for `reasoning_content` only and for unclosed inline `<think>`. A closed think followed by an answer is ACTIONABLE; a closed think followed by nothing is THOUGHT_RUNAWAY.
28. The parallel case passes with two `read_file` calls in either order. One call gives NO_PARALLEL_CALLS.
29. An unknown tool name gives HALLUCINATED_TOOL.
30. With `tool_choice_none`, a tool call gives TOOL_CHOICE_IGNORED, and `<tool_call>` text gives TEMPLATE_LEAK.
31. `max_results: "5"` gives MALFORMED_TOOL_ARGS, and the detail names the path.
32. `unknown_tool_refusal`: polite text passes, a `clarify` call passes, and a `run_command` call gives WRONG_TOOL.
33. `conformance_verdict` maps 10/9/8/7 out of 10 to PASS/WARN/WARN/FAIL.
34. `empty_turn_verdict`: one EMPTY in 6 is FAIL, runaway only is WARN, all actionable is PASS, all ERROR is FAIL, and a single ERROR is WARN.
35. `run_empty_turn` with `n_ctx=8192` sends no requests, returns `skipped`, gives WARN, and reports CONTEXT_TOO_SMALL.
36. HTTP 500 on one conformance case gives REQUEST_ERROR for that case, and the remaining cases still run.
37. A timeout on a replay gives an ERROR outcome without aborting.

**test_agentcheck_remediation.py**

38. The llama.cpp `empty_turn` fix contains "--reasoning-budget 2048" and does not contain "thinking_budget_tokens"; the Strata `empty_turn` fix does contain it.
39. No Strata cell contains `"--"`. The Strata cells mention `reasoning_close_retry`, `reasoning_budget_tokens`, `reasoning_loop_recovery` and `reasoning_effort`.
40. For the unknown family, every `AppliedRemedy.fix` is None.
41. `remedies_for` deduplicates repeated symptoms and orders them by Symptom declaration order.
42. Every `Symptom` has a `REMEDIES` entry.

**test_agentcheck_report.py**

43. The text report shows the family line, "9/10 WARN", per-turn counts, and ends with "overall: WARN (exit 1)".
44. The unknown-family render has the generic header and no `fix:` lines.
45. The JSON document has the specified keys, `json.dumps(..., allow_nan=False)` succeeds, `exit_code` matches, and the canned model text is absent.

**test_agentcheck_cli.py** (through `main`)

46. A perfect server exits 0, and stdout ends with "overall: PASS (exit 0)".
47. One wrong conformance case exits 1.
48. One EMPTY replay exits 2.
49. `/props` 200 followed by 401 on `/v1/chat/completions` exits 3. Stderr mentions HTTP 401 and contains no traceback.
50. A 200 non-JSON body on `/v1/chat/completions` exits 3 with "invalid JSON".
51. Bad arguments exit 3: `--timeout 0`, `--timeout -1`, `--replays 0`, `--max-tokens abc`. An unset `--api-key-env` also exits 3, while a set one sends the Authorization header on chat requests.
52. `--json --replays 2`: stdout parses as JSON, `exit_code` equals the return code, the server saw 10 + 3×2 chat requests, and phase B payloads have `max_tokens == 512`.

Also add exit-matrix subtests over all 9 verdict pairs → `AgentCheckResult.exit_code`, which can live in `test_agentcheck_checks.py`.

**test_readme.py** gets one more test, modelled on `test_serve_documents_every_flag`: every `agentcheck` option string appears in the README section that follows `` `llama-doctor agentcheck URL` ``.

---

## 8. README section draft

Changes elsewhere in the README:
- Insert the section below after `## Dashboard`.
- Add an `agentcheck` block to `## CLI reference`.
- Change the sentence "`--probe` is off by default. It is the only option that sends a write request…" to: "In `diagnose`, `--probe` is off by default and is the only option that sends a write request. `agentcheck` always sends chat completion requests."

````markdown
## Agent readiness check

`llama-doctor agentcheck URL` tests whether the model behind an OpenAI-compatible endpoint can drive a coding agent. It sends greedy (`temperature 0`) requests to `/v1/chat/completions` and reports three things:

- **Tool-call conformance.** Ten single-turn cases against a fixed 15-tool catalog: correct tool selection, arguments that validate against each tool's JSON Schema (type, properties, required, enum), two calls in one turn, a polite refusal when asked for a tool that does not exist, and correct behaviour for `tool_choice` `auto` and `none`. The score is shown out of 10.
- **Empty-turn detection.** Three synthetic agent turns of 8–10k prompt tokens (system prompt, project notes and tool results) are each replayed 6 times with `max_tokens 512`. Each reply is classified as actionable (text or a tool call), empty (200 OK with neither) or thought runaway (reasoning but no answer). A single empty reply fails the phase.
- **Remediation.** Each symptom found is mapped to concrete settings for the detected server family.

The server family is detected before any check runs. A `GET /props` that returns 200 means the llama.cpp family. Otherwise, `GET /v1/models` entries with `meta.instanceId` or `owned_by` Strata mean the Strata family. Anything else is reported as unknown and gets generic advice only.

| Symptom | llama.cpp family | Strata family (config keys) |
| --- | --- | --- |
| Empty turn or thought runaway | Relaunch with `--reasoning-budget 2048`. | `reasoning_budget_tokens` (0 = off; per-request `thinking_budget_tokens` is also accepted), `reasoning_close_retry` `on`, `reasoning_loop_recovery` `recover`, `reasoning_effort` for the default level. |
| Malformed tool arguments or template leaks | Ensure `--jinja`, check the chat template, try `--grammar`. | No server setting known; generic advice is shown. |

agentcheck never sends reasoning controls itself, so the result reflects the server as configured. The corpus is built deterministically, and repeated runs send byte-identical prompts. A run makes 10 short requests plus 18 long ones, so on slow hardware it can take several minutes; the per-request `--timeout` defaults to 300 seconds.

Exit codes: `0` all checks pass, `1` warnings only, `2` a phase failed, `3` tool or configuration error (for example an authentication failure, or an endpoint that is not OpenAI-compatible).
````

CLI reference block:

```markdown
`llama-doctor agentcheck URL`

| Flag | Effect |
| --- | --- |
| `--json` | Print the result as JSON instead of text. |
| `--replays N` | Replays per synthetic agent turn in the empty-turn check (default `6`). |
| `--max-tokens N` | Token limit for each empty-turn replay (default `512`); conformance cases use 2048. |
| `--api-key-env VAR` | Read the server's API key from environment variable `VAR`. |
| `--timeout SEC` | HTTP timeout per request, greater than 0 (default `300`). |
```

---

## 9. Implementation order (junior checklist)

1. [ ] Add `get_json` and `post_json` to `LlamaClient`. Run the full suite and confirm it is still green.
2. [ ] Add `set_responder` to `tests/support.py`. The existing tests must stay green.
3. [ ] Create `agentcheck/errors.py` (`AgentCheckError`) and `schema.py` with `validate`, then write tests 1–5.
4. [ ] Write `cases.py`: helpers, the catalog, cases, the sentence pool, `filler` and `agentic_turns`. Write tests 6–12, and tune the filler constants until test 11 passes. Check that `wc -l` is under 400.
5. [ ] Write `chat.py` (`parse_reply`, `ChatClient`) and tests 13–16.
6. [ ] Write `family.py` and tests 17–23.
7. [ ] Write `remediation.py`, copying the table strings from §3 exactly, and tests 38–42.
8. [ ] Write `checks.py`: `split_think`, `has_tool_markup`, `classify`, `evaluate_case`, the verdicts, `is_fatal`, both runners and `AgentCheckResult`. Write tests 24–37 and the exit matrix.
9. [ ] Write `report.py` and tests 43–45.
10. [ ] Write `__init__.run_agentcheck`, then wire `cli.py` (§5) and write tests 46–52.
11. [ ] Update the README (§8) and add the `test_readme` flag test. Add a short agentcheck paragraph to `docs/DESIGN.md`, and save this design as `docs/contracts/T6-agentcheck.md`.
12. [ ] Run `python3 -m unittest discover -v` and `ruff check src tests`. Run `grep -rn "#" src/llama_doctor/agentcheck`; the only hits should be inside string literals.
13. [ ] Do one manual run against a real llama.cpp server and one against a Strata server. Confirm that `usage.prompt_tokens` falls in the 8–10k range, and adjust the filler constants if it does not.

---

## 10. Decisions you may want to revisit

- **Detection order.** Strata markers in `/v1/models` win over a `/props` 200: Strata emulates the llama.cpp HTTP surface, so a future Strata build exposing `/props` must not be mislabelled llama.cpp. llama.cpp never emits `meta.instanceId`, so the reverse confusion cannot happen.
- **`seed: 0` is sent on every request.** llama.cpp honours it. I have not confirmed how Strata treats unknown fields, so test 13 in §9 should check that it does not reject it.
- **Phase A ignores `--max-tokens` and uses 2048.** This keeps the conformance score about tool use rather than thinking length. Thinking problems are surfaced by phase B, and by phase A only as a `thought_runaway` symptom.
- **EMPTY applies to any `finish_reason`, not only `stop` and `length`.** An agent sees nothing in every such case, and the reason is kept as evidence.
- **Two cells go beyond the verified facts:** `request_error` points to `--jinja`, the same verified flag as the malformed-arguments row, and `context_too_small` points to `-c`, a flag the repo already recommends elsewhere. Every other Strata cell without a verified key is `None`, and the renderer then shows only the generic advice.
