# T2: analyzer engine and built-in analyzers

Context: T1 already provides `llama_doctor.models` (`MetricsSnapshot`,
`PropsSnapshot`, `ModelsMeta`, `EndpointError`, `ProbeResult`,
`ServerSnapshot`), `llama_doctor.promparse.parse_prometheus`,
`tests/support.py` (`load_fixture`, `FakeLlamaServer`), and the fixtures in
`tests/fixtures/`. Read those files before starting. Stdlib only, type hints
on public names, no emoji, comments only where they explain why. Analyzers
are pure: no I/O, no clock, and they never raise on missing data.

## Files to create

```
src/llama_doctor/analysis/__init__.py    BUILTIN_ANALYZERS, diagnose
src/llama_doctor/analysis/base.py
src/llama_doctor/analysis/signals.py
src/llama_doctor/analysis/engine.py
src/llama_doctor/analysis/{connectivity,prefill,decode,prefix_cache,kv,context,queue,spec,memory,props_check}.py
src/llama_doctor/serialize.py
tests/test_signals.py  tests/test_analyzers.py  tests/test_engine.py  tests/test_serialize.py
```
Append `make_context` (below) to `tests/support.py`.

## base.py

```python
class Severity(IntEnum): INSUFFICIENT_DATA = 0; OK = 1; INFO = 2; WARN = 3; CRITICAL = 4
class Impact(IntEnum): LOW = 1; MEDIUM = 2; HIGH = 3
Evidence = Mapping[str, float | int | str | bool | None]
@dataclass(frozen=True)
class Recommendation: action: str; rationale: str; impact: Impact; flag: str | None = None
@dataclass(frozen=True)
class Finding:
    analyzer_id: str; code: str; severity: Severity; title: str; detail: str
    evidence: Evidence = field(default_factory=dict)
    recommendations: tuple[Recommendation, ...] = ()
    max_impact: int  # @property: max rec impact, 0 if none
@dataclass(frozen=True)
class HostHints: vram_gib: float | None = None; ram_gib: float | None = None
@dataclass(frozen=True)
class DiagnosisContext:
    snapshot: ServerSnapshot; probe: ProbeResult | None = None; hints: HostHints = HostHints()
    window: tuple[MetricsSnapshot, ...] = ()   # dashboard rate window, current last; empty one-shot
    metrics / props / models  # @property shortcuts to snapshot fields
AnalyzerFunc = Callable[[DiagnosisContext], list[Finding]]
@dataclass(frozen=True)
class AnalyzerSpec: id: str; title: str; func: AnalyzerFunc
@dataclass(frozen=True)
class Report:
    findings: tuple[Finding, ...]
    overall: Severity          # @property: max severity; INSUFFICIENT_DATA if empty
    exit_code: int             # @property: 2 any CRITICAL, 1 any WARN, else 0
    def recommendations(self) -> list[Recommendation]
    def by_code(self, code: str) -> Finding | None
```
`recommendations()` flattens recs in finding order and sorts stably by
impact desc. It then dedupes on `flag`, or on `action` when `flag` is None,
keeping the first occurrence.

## engine.py

`run_analyzers(ctx: DiagnosisContext, analyzers: Sequence[AnalyzerSpec]) -> Report`.
Call each func. If one raises, record `Finding(spec.id, f"{spec.id}.error",
INSUFFICIENT_DATA, f"{spec.title}: analyzer failed", type(exc).__name__)`;
never put the exception message in it. Sort by
`(-severity, -max_impact, registry_index)`. `analysis/__init__.py` exports
`BUILTIN_ANALYZERS` in this order: connectivity, prefill, decode,
prefix_cache, kv, context, queue, spec, memory, props. `context` gives WARN
`context.near_limit` when `n_tokens_max / n_ctx >= 0.9` (evidence `used_ratio`,
`n_tokens_max`, `n_ctx`), OK `context.ok` below that, and
`context.insufficient_data` without both values. It also exports `diagnose(ctx,
analyzers=BUILTIN_ANALYZERS) -> Report`.

## signals.py

```python
MIN_PREFILL_TOKENS = 1000; CPU_BOUND_PREFILL_TPS = 100.0; MIN_PREFILL_DECODE_RATIO = 2.0
@dataclass(frozen=True)
class Rate: value: float; source: str        # "probe" | "lifetime"
def prefill_rate(ctx) -> Rate | None  # reliable probe first; else lifetime if prompt_tokens_total >= 1000
def decode_rate(ctx) -> Rate | None   # reliable probe with predicted_n > 0; else lifetime decode_tps
def is_cpu_bound(ctx) -> bool | None  # None if no prefill rate; True if prefill < 100
                                      # or (decode finite and > 0, and prefill/decode < 2.0)
                                      # for invalid decode rates, the ratio test is skipped
```

## Analyzers: one `analyze(ctx) -> list[Finding]` per module, `analyzer_id` = id

Codes are exact. Recommendation text is free, but impact and flag are exact,
and so is the order within a list.

- **connectivity**: if not alive, CRITICAL `connectivity.unreachable`
  (detail lists `endpoint: reason`; rec HIGH, flag None). If metrics is None
  and the `/metrics` status is in (404, 501), WARN
  `connectivity.metrics_disabled` (rec HIGH `--metrics`). Any other metrics
  failure is WARN `connectivity.metrics_unavailable`. Props None while alive
  adds INFO `connectivity.props_unavailable`. With no findings, return OK
  `connectivity.ok`.
- **prefill**: if there is no rate, INSUFFICIENT `prefill.insufficient_data`
  (evidence `prompt_tokens_total`; detail suggests `--probe`). If cpu-bound
  and `props.is_moe is True`, CRITICAL `prefill.moe_cpu_experts` with recs in
  this order: HIGH None (smaller quant); HIGH `--ple-on-cpu` (N-gram/PLE
  tables to CPU/mmap, on models that ship them); MEDIUM `--draft-max` (MTP or
  speculative drafting at 2); LOW `-c` (smaller context frees VRAM for
  experts). If cpu-bound otherwise, WARN `prefill.cpu_bound` with recs HIGH
  `-ngl`, HIGH None (smaller quant), LOW `-c`. If prefill < 300 (`SLOW_PREFILL_TPS`),
  INFO `prefill.slow` with rec MEDIUM `--ubatch-size`. Otherwise OK
  `prefill.ok`. Evidence on all rate paths: `prefill_tps`, `decode_tps` (or
  None), `source`.
- **decode**: if there is no rate, INSUFFICIENT `decode.insufficient_data`.
  If decode < 10.0 (`SLOW_DECODE_TPS`), WARN `decode.slow`. When cpu-bound is
  True, it gets no recs and the detail points to the prefill finding as the
  root cause. Otherwise it gets one MEDIUM rec, flag `-ngl`. Above the
  threshold, OK `decode.ok`. Evidence: `decode_tps`, `source`, `cpu_bound`.
- **prefix_cache**: if `cache_hit_ratio` is None, or cached + prompt tokens
  < 1000, INSUFFICIENT `prefix_cache.insufficient_data`. Ratio < 0.2 gives
  WARN `prefix_cache.low_hit_ratio` with recs HIGH None (keep the prompt
  prefix byte-identical and put variable content last) and MEDIUM
  `--cache-reuse`. Ratio < 0.5 gives INFO `prefix_cache.moderate_hit_ratio`
  with one rec LOW None. Otherwise OK `prefix_cache.ok`. Evidence:
  `hit_ratio`.
- **kv**: if metrics is None, INSUFFICIENT `kv.insufficient_data`. Rule A:
  `kv_cache_usage_ratio >= 0.9` gives WARN `kv.saturated` (rec MEDIUM
  `--parallel`); queued requests alone are `queue.deferred`, not KV. Rule B needs `n_tokens_max` and
  `props.n_ctx`, with `n_ctx >= 16384` and `n_tokens_max / n_ctx < 0.25`. It
  gives WARN `kv.context_oversized` with evidence `peak_ratio`, `n_ctx`,
  `n_tokens_max`, and `suggested_ctx = max(4096, ceil(2*n_tokens_max/4096)*4096)`.
  Non-finite context or peak-token values are not evaluable.
  Its rec, flag `-c`, names the suggested value; impact is HIGH when
  cpu-bound is True and MEDIUM otherwise. With no findings, return OK
  `kv.ok` ("KV cache usage is well below capacity") only when
  `0 <= kv_cache_usage_ratio < 0.9`, else INSUFFICIENT `kv.insufficient_data`
  (ratio missing or NaN). `kv.ok` speaks only of the KV usage gauge; window
  sizing against the observed peak belongs to `context`, so `kv.ok` never
  sits beside `context.near_limit` because of `n_ctx` / `n_tokens_max`.
- **queue**: `requests_processing` and `requests_deferred` are instantaneous
  gauges, never deltas. `deferred_peak` is the max of `requests_deferred` over
  `ctx.window` (dashboard rate window, current sample last), or the current
  value when the window is empty (one-shot diagnose). Missing metrics or gauge
  gives INSUFFICIENT `queue.insufficient_data`; `deferred_peak > 0` gives INFO
  `queue.deferred` with evidence `deferred_peak`, `requests_deferred`,
  `requests_processing`, plus `window_s` when windowed; otherwise OK `queue.ok`.
- **spec**: if draft_tokens_total or drafts_total is None, INSUFFICIENT
  `spec.insufficient_data`. If drafts_total == 0, INFO `spec.inactive`. If
  acceptance is None, INSUFFICIENT. Then the first matching rule wins:
  acceptance < 0.4 gives WARN `spec.low_acceptance` (MEDIUM `--draft-max`).
  Cpu-bound True with avg_len > 2.0 gives WARN `spec.drafts_too_long_offloaded`
  (MEDIUM `--draft-max`, which sets 2). Cpu-bound not True, acceptance >= 0.7,
  and avg_len < 3.0 give INFO `spec.raise_draft` (MEDIUM `--draft-max`).
  Otherwise OK `spec.ok`. Evidence: `avg_draft_len`, `acceptance`.
- **memory**: if `models.size_bytes` or `hints.vram_gib` is None,
  INSUFFICIENT `memory.insufficient_data` (detail names `--vram-gib`/`--ram-gib`).
  Negative or non-finite VRAM/RAM hints are treated as absent.
  Let `weights_gib = size_bytes / 2**30`. If ram is given and
  weights > vram + ram, CRITICAL `memory.exceeds_total` (rec HIGH None,
  smaller quant). If weights > vram, INFO `memory.partial_offload` with
  evidence `offloaded_gib = weights - vram`. Otherwise OK `memory.fits_vram`.
  Detail states this counts weights only.
- **props** (module `props_check.py`): if props is None, INSUFFICIENT
  `props.insufficient_data`. Non-empty `missing` gives INSUFFICIENT
  `props.minimal` (evidence `missing` = comma-joined; detail mentions wrapper
  builds such as unsloth-studio). If `n_ctx > models.n_ctx_train`, also emit WARN
  `props.ctx_exceeds_training` (MEDIUM `-c`). With no findings, OK `props.ok`.

## serialize.py

```python
def severity_name(s: Severity) -> str     # "insufficient_data" | "ok" | "info" | "warn" | "critical"
def recommendation_to_dict(r) -> dict     # action, rationale, impact ("low"/"medium"/"high"), flag
def finding_to_dict(f) -> dict            # analyzer, code, severity, title, detail, evidence, recommendations
def report_to_dict(r: Report) -> dict     # overall, exit_code, findings, recommendations
def props_to_dict(p: PropsSnapshot | None) -> dict | None   # asdict, tuples -> lists
def models_to_dict(m: ModelsMeta | None) -> dict | None
def rates_to_dict(m: MetricsSnapshot | None) -> dict
    # keys prefill_tps, decode_tps, cache_hit_ratio, avg_draft_len, draft_acceptance,
    # kv_usage_ratio, n_tokens_max; all None when m is None
def probe_to_dict(p: ProbeResult | None) -> dict | None
def errors_to_dict(e: Mapping[str, EndpointError]) -> dict[str, str]   # endpoint -> reason
```

All serializer output replaces non-finite floats with `None`, recursively in
evidence and dataclass-derived values, so it is always valid for
`json.dumps(..., allow_nan=False)`.

## tests/support.py addition

```python
def make_context(metrics: str | None = "metrics_moe_offload.txt",
                 props: str | None = "props_full.json",
                 models: str | None = "models_full.json", *,
                 errors: Mapping[str, EndpointError] | None = None,
                 probe: ProbeResult | None = None,
                 hints: HostHints = HostHints()) -> DiagnosisContext
```
Each fixture goes through the T1 parsers. `None` leaves that part None.
`taken_at = 0.0`.

## Test cases (`assertAlmostEqual(..., places=6)` for floats)

Context names below: MOE is the default; HEALTHY is `metrics_healthy.txt`;
MINIMAL is `metrics_minimal.txt` with `props_unsloth_minimal.json` and
models None.

test_signals: `test_prefill_rate_lifetime` (MOE gives 40.0, "lifetime");
`test_prefill_rate_insufficient_below_1000_tokens` (MINIMAL gives None);
`test_probe_preferred_when_reliable` (probe 850.0 with prompt_n 1200 gives
850.0, "probe"); `test_unreliable_probe_ignored` (prompt_n 500 falls back to
lifetime); `test_is_cpu_bound` (MOE True, HEALTHY False, MINIMAL None);
`test_cpu_bound_by_ratio` (probe prefill 150, decode 100 gives True).

test_analyzers, one test per row, each asserting codes, severity, and the
evidence given:
`test_connectivity_ok`; `test_connectivity_unreachable` (all three errors
gives CRITICAL); `test_connectivity_metrics_disabled_501`;
`test_connectivity_props_unavailable`;
`test_prefill_moe_cpu_experts` (MOE gives CRITICAL, prefill_tps 40.0,
decode_tps 7.0, and rec flags exactly `[None, "--ple-on-cpu", "--draft-max", "-c"]`
with impacts `[HIGH, HIGH, MEDIUM, LOW]`); `test_prefill_cpu_bound_dense`
(MOE context whose props are swapped via `dataclasses.replace` for
`PropsSnapshot.from_payload({"model_path": "Llama-3.1-8B-Q4_K_M.gguf"})`
gives WARN `prefill.cpu_bound`); `test_prefill_ok_healthy`
(2000.0); `test_prefill_insufficient_minimal`; `test_prefill_slow` (probe
250 / 20); `test_decode_slow_is_symptom` (MOE gives WARN, 7.0, recs ());
`test_decode_ok_minimal` (30.0, OK); `test_prefix_cache_low_moe` (WARN,
hit_ratio 0.1111111); `test_prefix_cache_ok_healthy` (0.75);
`test_prefix_cache_insufficient_minimal`; `test_kv_context_oversized_moe`
(WARN, peak_ratio 0.1875, suggested_ctx 12288, rec impact HIGH);
`test_kv_ok_healthy` (title "KV cache usage is well below capacity");
`test_kv_without_usage_ratio_is_insufficient`; `test_kv_non_finite_usage_ratio_is_insufficient`;
`test_kv_just_below_saturation_is_ok` (0.89); `test_kv_saturated_does_not_emit_ok` (0.9);
`test_kv_insufficient_minimal`; `test_kv_saturated`
(samples with `requests_deferred 3`); `test_spec_drafts_too_long_offloaded_moe`
(avg 3.0, acceptance 0.4); `test_spec_raise_draft_healthy` (INFO, 2.0, 0.8);
`test_spec_low_acceptance` (accepted 2000 of 9000); `test_spec_insufficient_minimal`;
`test_spec_inactive`; `test_memory_partial_offload` (vram 8, ram 32 gives
INFO, offloaded_gib 8.5); `test_memory_exceeds_total` (vram 8, ram 4 gives
CRITICAL); `test_memory_fits` (vram 24 gives OK); `test_memory_insufficient_without_hints`;
`test_props_minimal_unsloth` (INSUFFICIENT; missing evidence equals the T1
tuple joined by ","); `test_props_ctx_exceeds_training` (models n_ctx_train
16384 gives WARN); `test_props_ok_full`;
`test_no_analyzer_raises_on_empty_context` (every BUILTIN func with metrics,
props, and models all None returns a non-empty list and raises nothing).

test_engine: `test_moe_report_overall_critical_exit_2`;
`test_healthy_report_exit_0` (no WARN/CRITICAL; overall INFO);
`test_minimal_report_exit_0`; `test_findings_sorted` (custom MOE WARN
findings are ordered by maximum recommendation impact, not registration
order); `test_recommendations_deduped_by_flag` (unflagged recommendations
dedupe by action; same-flag recommendations retain the first, highest-impact
occurrence; retained recommendations are impact-descending across findings);
`test_analyzer_exception_isolated` (a spec whose func raises
`ValueError("SENTINEL")` gives `<id>.error` INSUFFICIENT, detail `ValueError`,
no SENTINEL); `test_custom_registry` (a one-analyzer tuple gives only its
findings); `test_kv_ok_never_coexists_with_context_near_limit_when_kv_ratio_high_or_missing`
(n_ctx 80128, n_tokens_max 78501, KV ratio None, NaN, inf and 0.9 to 1.0 give
`context.near_limit` and no `kv.ok`);
`test_kv_ok_beside_context_near_limit_only_speaks_of_the_kv_gauge` (same
context with KV ratio 0.0 to 0.89: `kv.ok` text names neither sizing,
context nor peak).

test_serialize: `test_report_json_roundtrip` (`json.dumps(report_to_dict(r),
allow_nan=False)` succeeds for the MOE, HEALTHY, and MINIMAL reports);
`test_severity_names`; `test_props_dict_has_no_sentinel` (both props
fixtures); `test_rates_none_when_no_metrics`; `test_errors_to_dict`.

## Acceptance

`python3 -m unittest discover -v` passes (T1 tests included) and
`ruff check .` is clean.
