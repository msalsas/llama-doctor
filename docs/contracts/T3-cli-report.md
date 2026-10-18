# T3: terminal report, JSON document, probe, CLI `diagnose`

Context: T1 gives `llama_doctor.client` (`LlamaClient`, `FetchError`),
`llama_doctor.models`, and `llama_doctor.redact.Secret`. T2 gives
`llama_doctor.analysis` (`diagnose`, `DiagnosisContext`, `HostHints`,
`Severity`, `Report`) and `llama_doctor.serialize`. `tests/support.py` has
`load_fixture`, `FakeLlamaServer`, and `make_context`. Read them first.
Stdlib only (argparse, json, secrets, sys, time). Type hints on public names.
No emoji. Plain, terse output.

## Files to create

```
src/llama_doctor/probe.py
src/llama_doctor/render/__init__.py      (empty)
src/llama_doctor/render/terminal.py
src/llama_doctor/render/jsondoc.py
src/llama_doctor/cli.py
src/llama_doctor/__main__.py             raise SystemExit(main())
tests/test_probe.py  tests/test_terminal.py  tests/test_cli.py
```
The console entry point `llama-doctor = llama_doctor.cli:main` is assumed to
be in `pyproject.toml`. Add it if missing.

## probe.py

```python
PROBE_PROMPT_WORDS = 1400
WARMUP_N_PREDICT = 8
def build_prompt(words: int = PROBE_PROMPT_WORDS, nonce: str | None = None) -> str
def run_probe(client: LlamaClient, *, n_predict: int = 64,
              clock: Callable[[], float] = time.monotonic) -> ProbeResult | None
```
- `build_prompt`: starts with `f"[{nonce}] "`, where nonce defaults to
  `secrets.token_hex(8)` so the server's prefix cache cannot serve the
  prompt. Then `words` words cycled from a fixed tuple of about 40 plain
  English words, then `"\nSummarize the text above in one sentence."`.
- `run_probe`: first a warm-up, `client.complete("Hello", WARMUP_N_PREDICT)`,
  with `warmup_s` measured by `clock()` before and after. The first request
  after idle runs about 3x slower, so it is discarded. Then one
  `client.complete(build_prompt(), n_predict)`. Rates are
  `prompt_n / (prompt_ms / 1000)` and `predicted_n / (predicted_ms / 1000)`.
  Return None on `FetchError` or when any timing is None or <= 0. Never
  print anything.

## render/terminal.py

```python
def use_color(stream: TextIO, env: Mapping[str, str]) -> bool
    # stream.isatty() and env.get("NO_COLOR", "") == "" and env.get("TERM") != "dumb"
def render_report(report: Report, snapshot: ServerSnapshot, *, target: str,
                  probe: ProbeResult | None = None, color: bool = False) -> str
```
Output layout (two-space indents). A missing value prints `-` in the
`model` line and `n/a` in the rates section; when a rate prints `n/a`, its
unit is dropped (`  prefix cache  n/a`). The `rates (...)` heading reads
`lifetime averages` whatever the probe state; probe numbers get their own
line.
```
llama-doctor <__version__>  <target>
model  <model_name>  quant <quant>  ctx <n_ctx>  slots <total_slots>  build <build_info>

rates (lifetime averages)
  prefill       <%.1f> tok/s
  decode        <%.1f> tok/s
  prefix cache  <%.1f> %
  drafts        len <%.2f>  acceptance <%.1f> %
  probe         prefill <%.1f> tok/s  decode <%.1f> tok/s  (<prompt_n>-token prompt)   # only if probe

findings
  <LABEL> <code>  <title>
        <detail>
recommendations (by impact)
  <n>. <impact:6> <action>  [<flag>]     # "[flag]" only when flag is set
overall: <SEVERITY NAME> (exit <code>)
```
Labels are CRIT, WARN, INFO, OK, N/A, all padded to 4 chars. With
`color=True`, wrap only the label: CRIT `\x1b[1;31m`, WARN `\x1b[33m`, INFO
`\x1b[36m`, OK `\x1b[32m`, N/A `\x1b[2m`, each reset with `\x1b[0m`. Without
color the output must contain no `\x1b`. Findings keep the report order. OK
findings print without a detail line. If there are no recommendations, the
section prints `  none`.

## render/jsondoc.py

```python
def diagnosis_document(report: Report, snapshot: ServerSnapshot, *, target: str,
                       probe: ProbeResult | None = None) -> dict[str, Any]
```
Keys: `tool` ("llama-doctor"), `version`, `target`, `taken_at`, `overall`
(severity name), `exit_code`, `server` (`props_to_dict`), `model`
(`models_to_dict`), `rates` (`rates_to_dict`), `probe` (`probe_to_dict`),
`findings`, `recommendations`, `errors` (`errors_to_dict`). Values come only
from the T2 serializers.

## cli.py

```python
EXIT_TOOL_ERROR = 3
def build_parser() -> argparse.ArgumentParser
def main(argv: Sequence[str] | None = None) -> int
```
- The parser is an `ArgumentParser` subclass whose `error()` prints usage and
  the message to stderr, then raises `SystemExit(3)`. Exit code 2 means
  critical and nothing else.
- `--version` prints `llama-doctor <__version__>`.
- `diagnose URL [--json] [--api-key-env VAR] [--timeout SEC (5.0)] [--probe]
  [--vram-gib N] [--ram-gib N] [--no-color]`
- `serve URL [--host 127.0.0.1] [--port 8765] [--interval 2.0] [--history 300]
  [--rate-window SEC (30.0, must be > 0)] [--api-key-env VAR] [--timeout SEC]`.
  Its handler builds the client and calls
  `llama_doctor.dashboard.server.run(client, host=..., port=..., interval=...,
  history=..., rate_window=..., out=sys.stderr) -> int`, imported inside the
  handler. T4 implements `run`; this task only wires the parser and the
  lazy call.
- There is no flag that takes a key value directly. `--api-key-env VAR`
  reads `Secret.from_env(VAR)`. If it is unset or empty, write
  `llama-doctor: environment variable VAR is not set or empty` to stderr
  and return 3.
- A `ValueError` from `normalize_base_url` gives `llama-doctor: <message>`
  on stderr and return 3.
- diagnose flow: `snapshot()`. If `--probe` and the server is alive, write
  `probe: warm-up plus one synthetic prompt (prefix cache disabled)` to
  stderr and run `run_probe`. If that returns None, write `probe: failed,
  using lifetime counters` to stderr. Then build the
  `DiagnosisContext(snapshot, probe, HostHints(vram, ram))`, run `diagnose`,
  and print either `json.dumps(diagnosis_document(...), indent=2,
  allow_nan=False)` or `render_report(..., color=use_color(sys.stdout,
  os.environ) and not args.no_color)`. Return `report.exit_code`.
- `target` is `client.base_url`. Neither the key nor any raw payload may
  reach stdout or stderr.

## Test cases (`assertAlmostEqual(..., places=6)` for floats)

test_probe (uses a fake client object with `complete()` recording calls):
`test_rates_from_timings` (`CompletionTimings(1200, 1500.0, 64, 3200.0)`
gives prefill 800.0, decode 20.0, prompt_n 1200, reliable True);
`test_warmup_runs_first` (first call is `("Hello", 8)`; second has
n_predict 64); `test_warmup_seconds_from_clock` (clock yields 10.0, 10.5,
... gives warmup_s 0.5); `test_returns_none_on_fetch_error`;
`test_returns_none_on_zero_ms`; `test_prompt_size_and_nonce`
(`len(build_prompt().split()) >= 1400`; two default prompts differ;
`build_prompt(nonce="abc")` starts with `[abc] `).

test_terminal (contexts via `make_context`, reports via `diagnose`):
`test_moe_report_contents` (contains `Qwen3-30B-A3B-UD-Q4_K_XL`,
`quant UD-Q4_K_XL`, `ctx 32768`, `40.0 tok/s`, `7.0 tok/s`, `11.1 %`,
`len 3.00`, `prefill.moe_cpu_experts`, `overall: CRITICAL (exit 2)`);
`test_no_ansi_without_color`; `test_color_wraps_labels` (contains
`\x1b[1;31mCRIT\x1b[0m`); `test_crit_before_warn`;
`test_recommendations_numbered_by_impact` (first rec line starts with
`  1. high`); `test_minimal_report_degrades` (unsloth context: `quant -`,
`ctx -`, `prefix cache  n/a`, contains `N/A`, `overall: OK (exit 0)`);
`test_probe_line_shown`; `test_no_sentinel_in_output` (all props fixtures);
`test_use_color_rules` (tty with empty env True; `NO_COLOR=1` False;
`TERM=dumb` False; non-tty False).

test_cli (FakeLlamaServer with fixture routes; capture output with
`contextlib.redirect_stdout`/`redirect_stderr`):
`test_diagnose_moe_exit_2`; `test_diagnose_healthy_exit_0` (healthy metrics,
props_full, models_full); `test_diagnose_exit_1_on_warn` (healthy metrics,
`/props` = `{"default_generation_settings": {"n_ctx": 65536}, "model_path":
"m-Q8_0.gguf"}` gives `kv.context_oversized`, exit 1); `test_json_output`
(parses; `exit_code == 2`, `rates.prefill_tps == 40.0`, `server.quant ==
"UD-Q4_K_XL"`, no `SENTINEL` anywhere in stdout); `test_unreachable_exit_2`
(closed port; `connectivity.unreachable` in output); `test_api_key_from_env`
(`patch.dict(os.environ, {"MY_KEY": "k3y"})`: the server saw `Bearer k3y`,
and `k3y` appears in neither stdout nor stderr); `test_missing_key_env_exit_3`;
`test_invalid_url_exit_3`; `test_usage_error_exit_3` (`SystemExit.code == 3`);
`test_version` (`SystemExit.code == 0`, version printed);
`test_probe_overrides_lifetime` (MOE routes plus `/completion` returning
timings 1200/1500.0/64/3200.0: `probe.prefill_tps == 800.0`,
`prefill.moe_cpu_experts` absent from the finding codes, exactly two POSTs
to `/completion`); `test_probe_failure_falls_back` (no `/completion` route:
stderr mentions `probe: failed`, exit 2).

## Acceptance

`python3 -m unittest discover -v` passes (all earlier tests included) and
`ruff check .` is clean. Manual check: `python3 -m llama_doctor diagnose
http://127.0.0.1:1` prints a CRITICAL unreachable report and exits 2.
