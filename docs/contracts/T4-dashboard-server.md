# T4: dashboard server, sampler, `/api/state`

Context: T1 gives `llama_doctor.client.LlamaClient` and `llama_doctor.models`.
T2 gives `llama_doctor.analysis` (`diagnose`, `BUILTIN_ANALYZERS`,
`DiagnosisContext`) and `llama_doctor.serialize`. T3 gives `llama_doctor.cli`,
whose `serve` handler lazily calls `llama_doctor.dashboard.server.run`.
`tests/support.py` has `load_fixture` and `make_context`. Read them first.
**Stdlib only** (`http.server`, `threading`, `collections.deque`, `json`,
`ipaddress`, `importlib.resources`). This task serves the frontend files but
does not write them; T5 does.

## Files to create

```
src/llama_doctor/dashboard/__init__.py   (empty)
src/llama_doctor/dashboard/sampler.py
src/llama_doctor/dashboard/server.py
src/llama_doctor/dashboard/static/       (directory; add an empty .gitkeep if T5 has not run)
tests/test_sampler.py  tests/test_server.py
```

## sampler.py

```python
DEFAULT_INTERVAL = 2.0
DEFAULT_HISTORY = 300
DEFAULT_RATE_WINDOW = 30.0
POINT_KEYS = ("t", "decode_tps", "prefill_tps", "cache_hit_ratio", "avg_draft_len",
              "draft_acceptance", "kv_usage_ratio", "requests_processing",
              "requests_deferred", "context_used_ratio")
def delta_point(prev: MetricsSnapshot | None, cur: MetricsSnapshot, t: float,
                n_ctx: int | None = None) -> dict[str, float | None]
class Sampler:
    def __init__(self, fetch: Callable[[], ServerSnapshot], *, target: str,
                 interval: float = DEFAULT_INTERVAL, history: int = DEFAULT_HISTORY,
                 rate_window: float = DEFAULT_RATE_WINDOW,
                 analyzers: Sequence[AnalyzerSpec] = BUILTIN_ANALYZERS,
                 hints: HostHints = HostHints()) -> None
    def tick(self) -> None
    def state(self) -> dict[str, Any]
    def start(self) -> None
    def stop(self, timeout: float = 5.0) -> None
```
- `delta_point` computes each rate as `delta(num) / delta(den)` between
  `prev` and `cur`. The rate is None when `prev` is None, when either value
  is missing, when `delta(den) <= 0` (an idle server must show a gap in the
  chart, not 0), or when either delta is negative (a counter reset after a
  server restart). The pairs:
  prefill = prompt_tokens / prompt_seconds; decode = tokens_predicted /
  tokens_predicted_seconds; cache_hit_ratio = d_cached / (d_cached +
  d_prompt_tokens); avg_draft_len = d_draft_tokens / d_drafts;
  draft_acceptance = d_accepted / d_draft_tokens. `kv_usage_ratio`,
  `requests_processing` and `requests_deferred` are gauges copied from `cur`;
  `context_used_ratio` is `cur.n_tokens_max / n_ctx` (a peak, not live use).
  The keys are exactly `POINT_KEYS`.
- `tick()` calls `fetch()` and records the metrics in a buffer of recent
  samples. The rate baseline is the newest sample at or before
  `taken_at - rate_window` (else the oldest one), because llama.cpp counters
  only move when a request completes; a counter that goes backwards clears
  the buffer. It then runs `diagnose(DiagnosisContext(snapshot, hints=hints,
  window=<samples inside the window>), analyzers)` (so `serve --vram-gib`
  and `--weights-gib` reach the memory analyzer) and, if the snapshot has
  metrics, appends `delta_point(baseline, metrics, snapshot.taken_at, n_ctx)`
  to a `deque(maxlen=history)`. Any exception from `fetch` or `diagnose` is
  caught and stored as `errors = {"sampler": type(exc).__name__}`, and the
  thread keeps running. All shared state sits behind one `threading.Lock`.
- `state()` returns a new JSON-ready dict with exactly these keys:
  `version` (`llama_doctor.__version__`), `target`, `interval_s`,
  `rate_window_s`, `updated_at` (taken_at of the last successful tick, else
  None), `server` (`props_to_dict`), `lifetime` (`rates_to_dict` of the latest
  metrics), `history` (list, oldest first), `context` (`context_to_dict`:
  `used_ratio`, `n_tokens_max`, `n_ctx`, `label`, plus `live_used` and
  `live_capacity` only when `/slots` was read), `kv` (`kv_to_dict`: `usage_ratio`,
  `estimated`, `evidence`; the estimate is live_used/live_capacity from `/slots`
  with `estimated: true` and never feeds `kv.saturated`), `report` (`report_to_dict` or None),
  `errors` (`errors_to_dict`, or the sampler error dict), and `notes`
  (booleans for chart notes). `kv_unexported` / `cache_hit_unexported`:
  the latest metrics lack `kv_cache_usage_ratio` /
  `prompt_tokens_cached_total`. `slots_unavailable`: `/slots` was requested
  (no `--no-slots`) and failed or was unusable; it is a sticky boolean, so the
  warning is not repeated per poll. `decode_idle` / `prefill_idle`: the lifetime
  rate is known, history has at least 2 points, and every window rate is null.
- `start()` launches a daemon thread named `llama-doctor-sampler`. Its loop
  is `tick()` then `stop_event.wait(interval)`. `stop()` sets the event and
  joins. A second `start()` while running is a no-op.

## server.py

```python
STATIC_ROUTES: dict[str, str] = {
    "/": "index.html", "/static/app.css": "app.css", "/static/app.js": "app.js",
    "/static/vendor/chart.umd.js": "vendor/chart.umd.js"}
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
def is_loopback(host: str) -> bool        # "localhost" or ipaddress.ip_address(host).is_loopback
def default_static_dir() -> Traversable    # importlib.resources.files("llama_doctor.dashboard") / "static"
def make_server(state: Callable[[], dict[str, Any]], *, host: str = "127.0.0.1",
                port: int = 8765, static_dir: Traversable | Path | None = None) -> ThreadingHTTPServer
def run(client: LlamaClient, *, host: str, port: int, interval: float, history: int,
        out: TextIO, rate_window: float = DEFAULT_RATE_WINDOW,
        hints: HostHints = HostHints()) -> int   # hints go to the Sampler unchanged
```
Handler rules (a `BaseHTTPRequestHandler` subclass built in `make_server`):
- Only GET and HEAD. Anything else gets 405 with `Allow: GET, HEAD`.
- When the bind host is loopback, the `Host` header's hostname (port
  stripped, `[::1]` brackets removed) must be in `LOOPBACK_NAMES`, or the
  request gets 403. This blocks DNS rebinding. A missing `Host` header gets
  400.
- The path is matched with the query string removed. `/api/state` returns
  200 with `json.dumps(state(), allow_nan=False)`, `Content-Type:
  application/json; charset=utf-8`, and `Cache-Control: no-store`. If
  `state()` raises, the response is 500 with body `internal error`, with no
  exception text.
- `/api/catalog` returns 200 with `json.dumps(catalog.METRICS)` (encoded once
  at startup) and `Content-Type: application/json; charset=utf-8`.
- Static files come only from `STATIC_ROUTES`. No other path ever reads the
  filesystem, and any other path, or a missing file, gets 404. Content
  types: `.html` text/html, `.css` text/css, and `.js` text/javascript, all
  `; charset=utf-8`. HTML responses also send `Content-Security-Policy: CSP`
  and `Referrer-Policy: no-referrer`.
- Every response carries `X-Content-Type-Options: nosniff` and
  `Content-Length`. `log_message` is overridden to print nothing.
- `run`: call `make_server` first. On `OSError`, write `llama-doctor: cannot
  bind <host>:<port>: address unavailable` to `out` and return 3. Otherwise
  create `Sampler(client.snapshot, target=client.base_url, ...)`, `start()`
  it, and write `dashboard: http://<host>:<port>/  target: <base_url>`.
  For a non-loopback host, also write `warning: dashboard has no
  authentication and is reachable from the network`. Then `serve_forever()`.
  On `KeyboardInterrupt`, stop the sampler, close the server, and return 0.

## Test cases (`assertAlmostEqual(..., places=6)` for floats)

Build snapshots with `MetricsSnapshot.from_samples` and `ServerSnapshot(...)`
directly. Prev samples: prompt_tokens_total 1000, prompt_seconds_total 2,
tokens_predicted_total 500, tokens_predicted_seconds_total 10,
prompt_tokens_cached_total 0, draft_tokens_total 100, drafts_total 50,
draft_tokens_accepted_total 80. Cur samples: 3000, 4, 700, 14, 2000, 160,
70, 125.

test_sampler: `test_delta_point_rates` (prefill 1000.0, decode 50.0,
cache_hit 0.5, avg_draft 3.0, acceptance 0.75); `test_delta_point_idle_is_none`
(prev == cur gives every rate None); `test_delta_point_counter_reset_is_none`;
`test_delta_point_first_sample_none`; `test_delta_point_gauges_passthrough`
(healthy fixture: kv 0.42, requests_processing 1.0); `test_point_keys_exact`;
`test_tick_appends_history` (a fake fetch yields prev then cur: len 2,
history[0] decode None, history[1] decode 50.0, `t` equals taken_at);
`test_history_bounded` (history=5 and 8 ticks with taken_at 0..7 give len 5,
first `t` 3.0); `test_state_keys_exact`; `test_state_moe_report`
(`report.overall == "critical"`, `server.quant == "UD-Q4_K_XL"`);
`test_state_has_no_sentinel` (`json.dumps(state())` lacks `SENTINEL` for
both props fixtures); `test_tick_survives_fetch_exception` (fetch raises
`RuntimeError("SENTINEL")`: errors `{"sampler": "RuntimeError"}`, no
SENTINEL in the JSON); `test_unreachable_snapshot_keeps_history_empty`
(no metrics; report contains `connectivity.unreachable`);
`test_start_stop` (interval 0.01; within a 2 s deadline, fetch is called at
least twice; after `stop()` the thread is not alive).

test_server (`make_server(..., port=0)` with a temp static dir holding all
four files; `serve_forever` in a thread; `shutdown()` in tearDown; requests
made with `http.client.HTTPConnection`):
`test_api_state_json` (200; body equals the state dict; content type and
no-store header); `test_index_csp_headers` (CSP header equals `CSP`;
`nosniff`); `test_static_allowlist` (`/static/app.js` gives 200 and
text/javascript; `/static/../server.py`, `/static/other.js`, and
`/etc/passwd` give 404); `test_missing_static_file_404`;
`test_query_string_ignored`; `test_foreign_host_rejected` (`Host:
evil.example` gives 403); `test_localhost_host_allowed` (`Host:
localhost:<port>` gives 200); `test_post_405`; `test_state_error_500_without_text`;
`test_is_loopback` (`127.0.0.1`, `::1`, and `localhost` True; `0.0.0.0` and
`192.168.1.5` False); `test_run_port_in_use_returns_3` (keep a socket bound
to a port, then call `run` on it: result 3 and `cannot bind` in out);
`test_cli_serve_wired` (patch `llama_doctor.dashboard.server.run` to return
0; `cli.main(["serve", url, "--port", "9999"]) == 0`; called with host
`127.0.0.1`, port 9999, interval 2.0, history 300).

## Acceptance

`python3 -m unittest discover -v` passes (all earlier tests included) and
`ruff check .` is clean. Tests must finish in under 10 s and must not leave
threads or sockets open.
