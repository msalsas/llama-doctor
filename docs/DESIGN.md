# llama-doctor design

llama-doctor explains why a llama.cpp-family server is slow by reading two
endpoints it already exposes: `/metrics` (Prometheus text) and `/props`
(JSON). `/v1/models` is read as a liveness fallback and for model size. The
tool does not change server state and does not benchmark. The one exception
is opt-in: `--probe` sends a warm-up and one synthetic prompt, and every rate
it produces is labeled `probe`.

Runtime: Python >= 3.10, stdlib only.

## Module map

```
src/llama_doctor/
  __init__.py            __version__                                   (T0)
  __main__.py            python -m llama_doctor                        (T3)
  redact.py              Secret, safe_str, safe_int, safe_float        (T1)
  promparse.py           Prometheus text -> dict[str, float]           (T1)
  models.py              typed snapshots, built from whitelists        (T1)
  client.py              LlamaClient, FetchError, URL normalization    (T1)
  analysis/
    base.py              Severity, Impact, Recommendation, Finding,
                         HostHints, DiagnosisContext, AnalyzerSpec, Report (T2)
    signals.py           rate helpers shared by analyzers              (T2)
    engine.py            run_analyzers                                 (T2)
    connectivity.py prefill.py decode.py prefix_cache.py kv.py
    context.py queue.py spec.py memory.py props_check.py               (T2)
    __init__.py          BUILTIN_ANALYZERS, diagnose()                 (T2)
  serialize.py           JSON-safe dicts for reports and snapshots     (T2)
  catalog.py             metric explanations served at /api/catalog
  probe.py               opt-in live measurement                       (T3)
  render/terminal.py     ANSI report                                   (T3)
  render/jsondoc.py      --json document                               (T3)
  cli.py                 argparse entry point                          (T3)
  dashboard/
    sampler.py           background sampler, delta rates, ring buffer  (T4)
    server.py            ThreadingHTTPServer, /api/state, static files (T4)
    static/              index.html app.css app.js vendor/chart.umd.js (T5)
tests/
  support.py             load_fixture, FakeLlamaServer, make_context   (T1)
  fixtures/              synthetic /metrics, /props, /v1/models        (T1)
```

The vendored Chart.js lives inside the package (`dashboard/static/vendor/`)
rather than in a repo-root `assets/` directory, so `pip install` ships it and
the dashboard works offline with no path lookups outside the package.

## Data flow

```
LlamaClient.snapshot()
  GET /metrics   -> promparse -> MetricsSnapshot.from_samples
  GET /props     -> json      -> PropsSnapshot.from_payload   (whitelist)
  GET /v1/models -> json      -> ModelsMeta.from_payload      (whitelist)
  failures       -> errors[endpoint] = EndpointError
        |
  ServerSnapshot  (+ ProbeResult | None, + HostHints, + dashboard window)  = DiagnosisContext
        |
  diagnose(ctx) -> each AnalyzerSpec.func(ctx) -> list[Finding] -> Report
        |
  render/terminal | serialize -> --json | dashboard /api/state
```

Raw JSON payloads exist only as local variables inside `client.py`. No public
function returns them, stores them, or puts them in an exception.

## Typed models (T1, `models.py`, all frozen dataclasses)

`MetricsSnapshot`: `taken_at: float`, then `float | None` for each of
`prompt_tokens_total, prompt_seconds_total, tokens_predicted_total,
tokens_predicted_seconds_total, prompt_tokens_cached_total, n_decode_total,
n_busy_slots_per_decode, n_tokens_max, kv_cache_usage_ratio, kv_cache_tokens,
requests_processing, requests_deferred, draft_tokens_total, drafts_total,
draft_tokens_accepted_total`. Derived properties (None when an input is
missing or the denominator is <= 0): `prefill_tps`, `decode_tps`,
`cache_hit_ratio`, `avg_draft_len`, `draft_acceptance`.

Each field is looked up through `METRIC_ALIASES` (field -> candidate names,
tried as `llamacpp:<name>` then `<name>`). Forks name speculative-decoding
counters differently, so adding a new spelling means one line in that table.

Rates come from counters only. Gauges such as `predicted_tokens_seconds` drop
to 0 when the server is idle and are never used for rates.

`PropsSnapshot`: `model_name, quant, build_info, cache_type_k, cache_type_v:
str | None`; `n_ctx, total_slots, n_gpu_layers, n_threads: int | None`;
`has_vision, is_moe: bool | None`; `spec_types: tuple[str, ...] | None`
(`default_generation_settings.params["speculative.types"]` without `none`;
None when unpublished); `missing: tuple[str, ...]` (whitelisted keys the
server did not send, a report note rather than a finding). `n_gpu_layers`,
`n_threads` and `cache_type_k/v` are startup flags no official llama-server
publishes over HTTP, so they are read when present but never required. `quant`
comes from the filename, else from `model_ftype` (`Q4_K - Medium` gives
`Q4_K_M`). `is_moe` is a filename heuristic: True on a match, None otherwise,
never False.

`ModelsMeta`: `model_id: str | None`, `n_params, size_bytes, n_ctx_train:
int | None`.

`EndpointError`: `endpoint: str`, `status: int | None`, `reason: str`.

`ServerSnapshot`: `taken_at, metrics, props, models, errors: Mapping[str,
EndpointError]`; `alive` is true when `/metrics` or `/v1/models` answered.
`/health` is never used: custom wrappers 404 it while healthy.

`ProbeResult`: `prefill_tps, decode_tps: float`, `prompt_n, predicted_n:
int`, `warmup_s: float`; `reliable` is `prompt_n >= 1000`.

## Analyzer contract (T2)

```python
AnalyzerFunc = Callable[[DiagnosisContext], list[Finding]]
@dataclass(frozen=True)
class AnalyzerSpec: id: str; title: str; func: AnalyzerFunc
```

- Pure: no I/O, no clock, no globals that change. Same input, same output.
- Must not raise on missing data. When inputs are absent, return one Finding
  with `Severity.INSUFFICIENT_DATA` that names what was missing. Do not guess.
- The engine still wraps every call. An exception becomes an
  `INSUFFICIENT_DATA` finding coded `<id>.error` that carries only the
  exception type name, because the message could contain data.
- Thresholds are module-level constants so tests and readers can see them.
- Rates go through `signals.prefill_rate/decode_rate`, which prefer a
  reliable probe and fall back to lifetime counters. Lifetime prefill needs
  >= 1000 observed prompt tokens, because below that, fixed per-request
  overhead dominates the number.

Registration is the tuple `analysis.BUILTIN_ANALYZERS`. Adding an analyzer
means adding one module and one tuple entry. The CLI and the dashboard only
call `diagnose()` and render `Report`, so neither changes.

## Severity and ordering

`Severity(IntEnum)`: `INSUFFICIENT_DATA=0, OK=1, INFO=2, WARN=3,
CRITICAL=4`. `Impact(IntEnum)`: `LOW=1, MEDIUM=2, HIGH=3`.

`Finding`: `analyzer_id, code, severity, title, detail, evidence: Mapping[str,
float | int | str | bool | None], recommendations: tuple[Recommendation,
...]`. `Recommendation`: `action, rationale, impact, flag: str | None`.

The report sorts findings by severity desc, then by their highest
recommendation impact desc, then by registry order.
`Report.recommendations()` flattens them in finding order and sorts stably by
impact desc, so a playbook's order survives within one impact level. It then
dedupes on the `(action, flag)` pair, keeping the first occurrence. Two
analyzers that suggest `-c` with different actions (shrink it, raise it)
therefore both show up instead of one silently hiding the other.

Exit codes for `diagnose`: 2 if any CRITICAL, 1 if any WARN, else 0.
INSUFFICIENT_DATA never fails CI. 3 means a tool or usage error (bad URL,
API-key env var unset). argparse's own exit code 2 is overridden so that 2
always means critical.

## Root-cause model

Prefill comes first. A CPU-bound prefill is defined as `prefill_tps < 100` or
`prefill_tps / decode_tps < 2.0`. On a model whose name marks it as MoE, that
almost always means expert tensors are executing from system RAM: CRITICAL
`prefill.moe_cpu_experts`. Slow decode is reported as a symptom that points
back to this finding. Remedies, in impact order: smaller quant;
keep every layer on the GPU (`-ngl 999`) and move only expert tensors to the
CPU (`--cpu-moe`, or `--n-cpu-moe N` for the first N layers); MTP/speculative drafting at `--draft-max 2`; smaller `-c` to free VRAM for
experts.

The speculative-decoding analyzer reads the same offload signal. Once layers
run on CPU, acceptance falls off after draft position 1-2, so on those rigs
it recommends `--draft-max 2` instead of longer drafts.

## Dashboard API (T4)

Binds `127.0.0.1` by default. If the bind address is loopback, the `Host`
header must be loopback too (`127.0.0.1`, `localhost`, `[::1]`), which blocks
DNS rebinding. Every response carries
`X-Content-Type-Options: nosniff`, and HTML carries a strict CSP
(`default-src 'self'`, no inline script or style).

- `GET /` -> `index.html`; `GET /static/<name>` -> fixed allowlist only.
- `GET /api/catalog` -> `catalog.METRICS`: label, description and
  interpretation per metric id, used by the dashboard tooltips. The texts are
  model-agnostic: no absolute ranges tied to a model, quant or GPU.
- `GET /api/state` -> `application/json`, `Cache-Control: no-store`:

```
{ "version", "target", "interval_s", "rate_window_s", "updated_at": float|null,
  "server":   props_to_dict | null,
  "lifetime": rates (counter totals; kv_usage_ratio and n_tokens_max are current gauges),
  "history":  [ point, ... ]    oldest first, ring buffer (default 300)
  "context":  { used_ratio, n_tokens_max, n_ctx, label },
  "report":   report_to_dict | null,
  "errors":   { endpoint: reason },
  "notes":    { kv_unexported, cache_hit_unexported, decode_idle, prefill_idle } }
point = { "t", "decode_tps", "prefill_tps", "cache_hit_ratio",
          "avg_draft_len", "draft_acceptance", "kv_usage_ratio",
          "requests_processing", "requests_deferred", "context_used_ratio" }
```

History rates are deltas against the newest sample at or before the start of
the rate window (default 30 s, `--rate-window`), because llama.cpp counters
only move when a request completes. If a counter did not move (idle server),
the rate is `null` and the chart shows a gap instead of a misleading 0. If a
counter went backwards (server restart), the window resets. `kv_usage_ratio`
and `requests_*` are gauges copied from the current sample;
`context_used_ratio` is the `n_tokens_max` peak over `n_ctx`, not live use.
The analyzers also receive the samples inside the window, which `queue` uses
for the deepest queue seen.

## Security notes

`/props` includes `media_marker`, a sentinel string that multimodal servers
search for in chat text. If that value ever lands in an LLM conversation
(say a user pastes llama-doctor output into a chat with the same server),
the session breaks and every later request returns HTTP 400. `chat_template`
is large, rarely useful, and can contain the same marker. So:

1. Redaction happens at parse time, as a whitelist. `PropsSnapshot.from_payload`
   copies a fixed set of scalar keys through `safe_str`/`safe_int`. Any other
   key is never read, so a new server field cannot leak by default.
2. `safe_str` drops a value that contains the payload's `media_marker`,
   strips control characters, and truncates to 128 characters.
3. Fetch errors carry the endpoint, the HTTP status, and a fixed category
   string. They never carry a body, headers, or a URL with credentials.
4. API keys are read from an environment variable named by
   `--api-key-env` and never from argv, where `ps` would show them. They are
   wrapped in `Secret`, whose `repr`/`str` print `Secret('***')`, and sent
   only as `Authorization: Bearer`. Redirects are refused so the header
   cannot follow one to another host.
5. Tests feed fixtures with sentinel values for `media_marker`/
   `chat_template`, then assert those sentinels are absent from every repr,
   terminal output, `--json` output, and `/api/state` body.

## Extension guide

- New metric spelling: add it to `METRIC_ALIASES` and to a fixture.
- New `/props` field: add it to the `PropsSnapshot` whitelist with a typed
  extractor and a test proving that a sentinel next to it does not leak.
- New analyzer: create `analysis/<name>.py` exposing `analyze(ctx) ->
  list[Finding]`, give codes as `<id>.<slug>`, add an `AnalyzerSpec` to
  `BUILTIN_ANALYZERS`, and add fixture tests that cover the OK, WARN or
  CRITICAL, and INSUFFICIENT_DATA paths.
