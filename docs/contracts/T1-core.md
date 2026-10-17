# T1: fetch layer, typed models, redaction

Context: llama-doctor reads `/metrics`, `/props` and `/v1/models` from a
llama.cpp-family server and diagnoses performance. This task builds the
layer everything else depends on. Python >= 3.10, **stdlib only** (no
third-party imports anywhere under `src/`). Type hints on every public
name. Docstrings only where behavior is not obvious. No emoji. Comments only
where they explain why.

Assumed scaffold: `pyproject.toml` (src layout), `src/llama_doctor/__init__.py`
defining `__version__`, and `tests/__init__.py` that inserts `<repo>/src` at
the front of `sys.path`. If `tests/__init__.py` is missing, create it with
exactly that insertion and nothing else.

## Files to create

```
src/llama_doctor/redact.py
src/llama_doctor/promparse.py
src/llama_doctor/models.py
src/llama_doctor/client.py
tests/support.py
tests/fixtures/metrics_moe_offload.txt
tests/fixtures/metrics_healthy.txt
tests/fixtures/metrics_minimal.txt
tests/fixtures/props_full.json
tests/fixtures/props_unsloth_minimal.json
tests/fixtures/models_full.json
tests/test_redact.py  tests/test_promparse.py  tests/test_models.py  tests/test_client.py
```

## Public API

`redact.py`
```python
class Secret:                      # __slots__; repr() and str() -> "Secret('***')"
    def __init__(self, value: str) -> None: ...
    def reveal(self) -> str: ...
    @classmethod
    def from_env(cls, var: str) -> "Secret | None": ...   # unset or empty -> None
def safe_str(value: object, *, forbidden: tuple[str, ...] = (), max_len: int = 128) -> str | None
    # non-str -> None; remove chars U+0000-U+001F and U+007F; truncate to max_len;
    # None if any non-empty forbidden substring is present; "" -> None
def safe_int(value: object) -> int | None     # bool -> None; int; float only if integral
def safe_float(value: object) -> float | None # bool -> None; int/float; NaN/inf -> None
```

`promparse.py`
```python
def parse_prometheus(text: str) -> dict[str, float]
```
Line regex `^([A-Za-z_:][\w:]*)(\{[^}]*\})?\s+(\S+)(?:\s+-?\d+)?\s*$`. The key
is the name plus the label block exactly as written. The value goes through
`float()`, which accepts `NaN`, `+Inf`, `-Inf`. Comment and blank lines are
skipped. A line that does not match, or whose value `float()` rejects, is
skipped silently. When a key repeats, the last value wins.

`models.py` (all `@dataclass(frozen=True)`; field lists exactly as below)
```python
METRIC_ALIASES: dict[str, tuple[str, ...]]
# every MetricsSnapshot float field maps to (field_name,), except:
#   draft_tokens_total          -> ("spec_decode_num_draft_tokens_total", "draft_tokens_total", "n_drafted_total")
#   drafts_total                -> ("spec_decode_num_drafts_total", "drafts_total", "n_drafts_total")
#   draft_tokens_accepted_total -> ("spec_decode_num_accepted_tokens_total", "draft_tokens_accepted_total", "n_drafted_accepted_total")
# first candidates are verified against an unsloth-studio build (2026-10); the rest are fallbacks.
# spec_decode_num_accepted_tokens_per_pos_total is deliberately out of scope for v0.1 (per-position, needs label parsing).
# lookup: for each alias in order, try "llamacpp:" + alias, then bare alias.

def ratio(num: float | None, den: float | None) -> float | None  # None if either None or den <= 0

class MetricsSnapshot:
    taken_at: float = 0.0
    prompt_tokens_total, prompt_seconds_total, tokens_predicted_total,
    tokens_predicted_seconds_total, prompt_tokens_cached_total, n_decode_total,
    n_busy_slots_per_decode, n_tokens_max, kv_cache_usage_ratio, kv_cache_tokens,
    requests_processing, requests_deferred, draft_tokens_total, drafts_total,
    draft_tokens_accepted_total: float | None = None
    @classmethod
    def from_samples(cls, samples: Mapping[str, float], taken_at: float = 0.0) -> "MetricsSnapshot"
        # NaN, inf, and negative values -> None
    prefill_tps     = ratio(prompt_tokens_total, prompt_seconds_total)            # @property
    decode_tps      = ratio(tokens_predicted_total, tokens_predicted_seconds_total)
    cache_hit_ratio = ratio(cached, cached + prompt_tokens_total)  # None if either missing
    avg_draft_len   = ratio(draft_tokens_total, drafts_total)
    draft_acceptance= ratio(draft_tokens_accepted_total, draft_tokens_total)
```
`cache_hit_ratio` assumes `prompt_tokens_total` counts only evaluated tokens,
which is what llama.cpp does. Put that in a one-line comment.

```python
PROPS_KEYS = ("model_path", "n_ctx", "total_slots", "build_info", "n_gpu_layers",
              "n_threads", "cache_type_k", "cache_type_v", "modalities")
class PropsSnapshot:
    model_name: str | None = None   # basename of model_path ('/' and '\\'), ".gguf" suffix removed
    quant: str | None = None
    n_ctx: int | None = None        # default_generation_settings.n_ctx, else top-level n_ctx
    total_slots: int | None = None
    build_info: str | None = None
    n_gpu_layers: int | None = None
    n_threads: int | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    has_vision: bool | None = None  # modalities.vision if it is a bool
    is_moe: bool | None = None
    missing: tuple[str, ...] = ()   # PROPS_KEYS absent from payload, in PROPS_KEYS order
    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PropsSnapshot"
```
`from_payload` reads only the keys above, plus `media_marker`, which it uses
solely as a `forbidden` value for `safe_str` on every string field. It never
reads `chat_template` and never stores the payload. A non-dict payload gives
`PropsSnapshot(missing=PROPS_KEYS)`.

```python
def parse_quant(name: str) -> str | None   # upper-cased match or None
_QUANT_RE = r"(UD-)?(IQ\d+_[A-Z0-9]+(?:_[A-Z0-9]+)*|Q\d+(?:_[A-Z0-9]+)*|MXFP4(?:_MOE)?|BF16|F16|F32)(?=$|[.-])"  # re.I, search
def guess_moe(name: str) -> bool | None    # True on match else None (never False)
_MOE_RE = r"[-_]A\d+(?:\.\d+)?B(?=$|[-_.])|moe|mixtral|gpt-oss|deepseek|qwen3-next"      # re.I

class ModelsMeta:      # from /v1/models: data[0].id and data[0].meta.{n_params,size,n_ctx_train}
    model_id: str | None = None; n_params: int | None = None
    size_bytes: int | None = None; n_ctx_train: int | None = None
    @classmethod
    def from_payload(cls, payload: Any) -> "ModelsMeta"
class CompletionTimings:  # from response["timings"]
    prompt_n: int | None; prompt_ms: float | None; predicted_n: int | None; predicted_ms: float | None
class EndpointError:  endpoint: str; status: int | None; reason: str
class ProbeResult:
    prefill_tps: float; decode_tps: float; prompt_n: int; predicted_n: int; warmup_s: float
    reliable: bool  # @property, prompt_n >= 1000
class ServerSnapshot:
    taken_at: float; metrics: MetricsSnapshot | None; props: PropsSnapshot | None
    models: ModelsMeta | None; errors: Mapping[str, EndpointError]
    alive: bool  # @property, metrics is not None or models is not None
```

`client.py`
```python
DEFAULT_TIMEOUT = 5.0
MAX_BODY_BYTES = 4 * 1024 * 1024
class FetchError(Exception):
    def __init__(self, endpoint: str, reason: str, status: int | None = None) -> None
    endpoint: str; reason: str; status: int | None       # str(e) == f"{endpoint}: {reason}"
    def to_endpoint_error(self) -> EndpointError
def normalize_base_url(url: str) -> str
class LlamaClient:
    def __init__(self, base_url: str, api_key: Secret | None = None,
                 timeout: float = DEFAULT_TIMEOUT, clock: Callable[[], float] = time.time) -> None
    base_url: str  # read-only property, normalized
    def fetch_metrics(self) -> MetricsSnapshot          # GET /metrics, taken_at = clock()
    def fetch_props(self) -> PropsSnapshot              # GET /props
    def fetch_models(self) -> ModelsMeta                # GET /v1/models
    def snapshot(self) -> ServerSnapshot                # all three; FetchError -> errors[endpoint]
    def complete(self, prompt: str, n_predict: int) -> CompletionTimings
        # POST /completion {"prompt", "n_predict", "cache_prompt": false, "temperature": 0, "stream": false}
```
- `normalize_base_url`: strip whitespace and trailing `/`. Require scheme
  `http`/`https` and a host. Reject userinfo, query, and fragment with
  `ValueError`. Keep any path prefix (servers behind a reverse proxy).
- Use `urllib.request` with an opener whose redirect handler returns None,
  so a 3xx surfaces as `HTTPError`. Read at most `MAX_BODY_BYTES + 1` bytes.
- `reason` is always one of: `HTTP <code>`, `connection refused`,
  `timed out`, `unreachable`, `response too large`, `invalid JSON`,
  `invalid response`, `no timings in response`. Never put a body, a header,
  the API key, or `str()` of an underlying exception into a message. Raise
  with `from None` so tracebacks do not chain the original exception.
- Only private helpers may hold raw JSON, and only as locals. `/health` is
  never requested. Error keys are the paths: `/metrics`, `/props`, `/v1/models`.

## Fixtures (exact content; `# HELP`/`# TYPE` lines may be added, nothing else)

`metrics_moe_offload.txt`: one unlabeled `llamacpp:` sample per line:
prompt_tokens_total 48000, prompt_seconds_total 1200, tokens_predicted_total 8400,
tokens_predicted_seconds_total 1200, prompt_tokens_cached_total 6000,
n_decode_total 9000, n_busy_slots_per_decode 1, n_tokens_max 6144,
prompt_tokens_seconds 0, predicted_tokens_seconds 0, requests_processing 0,
requests_deferred 0, spec_decode_num_draft_tokens_total 9000,
spec_decode_num_drafts_total 3000, spec_decode_num_accepted_tokens_total 3600.

`metrics_healthy.txt`: prompt_tokens_total 200000, prompt_seconds_total 100,
tokens_predicted_total 50000, tokens_predicted_seconds_total 1000,
prompt_tokens_cached_total 600000, `llamacpp:n_decode_total 50000 1700000000000`
(with timestamp), n_tokens_max 12000, kv_cache_usage_ratio 0.42,
kv_cache_tokens NaN, requests_processing 1, requests_deferred 0,
spec_decode_num_draft_tokens_total 20000, spec_decode_num_drafts_total 10000,
spec_decode_num_accepted_tokens_total 16000,
`llamacpp:slot_state{slot="0",state="idle"} 1`, plus one blank line.

`metrics_minimal.txt`: prompt_tokens_total 600, prompt_seconds_total 2,
tokens_predicted_total 300, tokens_predicted_seconds_total 10.

`props_full.json`:
`{"default_generation_settings": {"n_ctx": 32768, "params": {"temperature": 0.8}},
"total_slots": 1, "model_path": "/models/Qwen3-30B-A3B-UD-Q4_K_XL.gguf",
"chat_template": "SENTINEL_TEMPLATE {{ messages }}", "media_marker": "<__media_SENTINEL__>",
"modalities": {"vision": true, "audio": false}, "build_info": "b6500-abc1234",
"n_gpu_layers": 99, "n_threads": 8, "cache_type_k": "q8_0", "cache_type_v": "q8_0"}`

`props_unsloth_minimal.json`:
`{"model_path": "unsloth/Qwen3-30B-A3B-GGUF", "total_slots": 1, "build_info": "unsloth-studio",
"media_marker": "<__media_SENTINEL__>", "chat_template": "SENTINEL_TEMPLATE"}`

`models_full.json`:
`{"object": "list", "data": [{"id": "Qwen3-30B-A3B-UD-Q4_K_XL.gguf", "object": "model",
"meta": {"n_params": 30532122624, "size": 17716740096, "n_ctx_train": 40960}}]}`

## tests/support.py

```python
def load_fixture(name: str) -> str                 # tests/fixtures/<name>, utf-8
class FakeLlamaServer:   # ThreadingHTTPServer on 127.0.0.1:0 in a daemon thread
    def __init__(self) -> None
    url: str                                        # "http://127.0.0.1:<port>"
    requests: list[tuple[str, str, dict[str, str], bytes]]  # method, path, headers, body
    def set_route(self, path: str, status: int, body: str | bytes,
                  content_type: str = "text/plain", headers: dict[str, str] | None = None) -> None
    def __enter__(self) -> "FakeLlamaServer"; def __exit__(self, *exc: object) -> None
```
Routes match the path exactly, for any method. An unknown path returns 404
with an empty body. Request logging is silenced.

## Test cases (use `assertAlmostEqual(x, y, places=6)` for floats)

test_redact: `test_secret_repr_masks`, which checks that repr and str equal
`Secret('***')` and `reveal()` returns the value. `test_secret_from_env`
covers set, unset, and empty (`unittest.mock.patch.dict(os.environ)`).
`test_safe_str_strips_control_and_truncates`: `"a\x1b[31mb\n"` gives
`"a[31mb"` and 200 x's give length 128. `test_safe_str_forbidden`.
`test_safe_int_rules`: True gives None, 3.5 gives None, 4.0 gives 4, "4"
gives None. `test_safe_float_rejects_nonfinite`.

test_promparse: `test_parses_colon_prefixed_names` (`llamacpp:prompt_tokens_total`
== 200000.0 in the healthy fixture), `test_skips_comments_and_blank`
(no key starts with `#`), `test_keeps_label_block_in_key`,
`test_ignores_trailing_timestamp` (n_decode_total == 50000.0),
`test_special_floats` (NaN gives isnan, `+Inf` gives inf),
`test_skips_malformed_lines` (`"garbage line here\n{bad} 1\nfoo 1\n"` gives
`{"foo": 1.0}`).

test_models: `test_moe_fixture_rates` (prefill 40.0, decode 7.0, cache_hit
0.1111111, avg_draft 3.0, acceptance 0.4); `test_healthy_fixture_rates`
(2000.0, 50.0, 0.75, 2.0, 0.8, kv_cache_usage_ratio 0.42, kv_cache_tokens
None); `test_minimal_fixture_degrades` (prefill 300.0, decode 30.0, and None
for cache_hit, avg_draft, acceptance, n_tokens_max); `test_zero_denominator_is_none`;
`test_negative_counter_is_none`; `test_alias_lookup` (`{"n_drafted_total": 50,
"llamacpp:n_drafts_total": 25}` gives avg 2.0); `test_props_full_whitelist`
(every field: name `Qwen3-30B-A3B-UD-Q4_K_XL`, quant `UD-Q4_K_XL`, n_ctx 32768,
slots 1, build `b6500-abc1234`, ngl 99, threads 8, k/v `q8_0`, vision True,
is_moe True, missing ()); `test_props_unsloth_minimal` (name
`Qwen3-30B-A3B-GGUF`, quant None, n_ctx None, slots 1, build `unsloth-studio`,
is_moe True, missing `("n_ctx","n_gpu_layers","n_threads","cache_type_k",
"cache_type_v","modalities")`); `test_props_never_hold_sentinels` (for both
props fixtures, `"SENTINEL"` absent from repr and `str(asdict())`);
`test_props_drops_field_containing_media_marker`; `test_props_non_dict_payload`;
`test_parse_quant` (subTests: `Qwen3-30B-A3B-UD-Q4_K_XL` gives `UD-Q4_K_XL`,
`gemma-3-12b-it-Q8_0` gives `Q8_0`, `model-IQ2_XXS` gives `IQ2_XXS`,
`gpt-oss-20b-MXFP4` gives `MXFP4`, `Llama-3-8B-Q4_K_M-00001-of-00002` gives
`Q4_K_M`, `llama-7b` gives None); `test_guess_moe` (`Qwen3-30B-A3B-Q4_K_M`,
`Mixtral-8x7B-v0.1`, and `gpt-oss-20b` give True; `Llama-3.1-8B-Instruct` and
`gemma-3-12b-it` give None); `test_models_meta_full` (size 17716740096,
n_params 30532122624, n_ctx_train 40960); `test_models_meta_empty` (`{"data": []}`
gives all None); `test_probe_result_reliable` (999 gives False, 1000 gives True);
`test_server_snapshot_alive`.

test_client (all against `FakeLlamaServer`): `test_normalize_base_url`
(`" http://h:8080/ "` gives `http://h:8080`, `http://h/proxy/` gives
`http://h/proxy`; `ftp://h`, `http://u:p@h`, `http://h?x=1`, and `h:8080`
raise ValueError); `test_fetch_metrics` (prefill 40.0, taken_at == clock value);
`test_fetch_props_whitelisted`; `test_bearer_header_sent`
(`Authorization == "Bearer k3y"`); `test_no_auth_header_without_key`;
`test_http_error_hides_body` (500 with body `SENTINEL boom` gives status 500,
reason `HTTP 500`, and no `SENTINEL` in str or repr);
`test_invalid_json_reason`; `test_connection_refused` (bind a socket to get
a free port, close it, then fetch: reason `connection refused`, status None);
`test_redirect_refused` (302 to `/elsewhere` gives status 302, and
`/elsewhere` never appears in `requests`); `test_api_key_absent_from_errors`
(key `SENTINELKEY`, 401); `test_snapshot_collects_errors` (only `/metrics`
routed: props None, errors keys `{"/props", "/v1/models"}`, each status 404,
alive True, taken_at == 1234.5 from `clock=lambda: 1234.5`);
`test_snapshot_alive_without_metrics` (`/metrics` 501 with models OK gives
alive True and `errors["/metrics"].status == 501`); `test_never_requests_health`;
`test_complete_returns_timings_only` (response `{"content": "SENTINEL",
"timings": {"prompt_n": 1200, "prompt_ms": 3000.0, "predicted_n": 64,
"predicted_ms": 8000.0}}` gives exact timings, the request JSON has
`cache_prompt is False` and `n_predict == 64`, and no SENTINEL in repr);
`test_complete_without_timings` (reason `no timings in response`).

## Acceptance

`python3 -m unittest discover -v` passes and `ruff check .` is clean.
