# T10: hardware guard

Context: T2 provides `llama_doctor.analysis` (`Recommendation`, `Finding`, `HostHints`, `DiagnosisContext`, `run_analyzers`, `diagnose`, `signals`). `catalog.py` and every analyzer module are already in the repo. Read `analysis/base.py`, `analysis/engine.py`, `analysis/signals.py`, `analysis/prefill.py`, `analysis/decode.py`, `analysis/memory.py` and `models.PropsSnapshot` before starting. Stdlib only, type hints on public names, terse English copy, no em dashes. Analyzers stay pure. Raw JSON stays in `client.py` and the parsers. The guard reads only parsed snapshots and hints.

## Problem

Some recommendations tell the user to move work onto a GPU: `-ngl` offload, `--n-cpu-moe` / `--cpu-moe` expert placement, and "free VRAM" context cuts. They are emitted whenever the rates look CPU-bound, and nothing checks that the host has a GPU at all. On a CPU-only server, or a server whose payloads say nothing about devices, this advice is wrong. The user chases a flag that does nothing or breaks their launch. Wrong advice is worse than no advice.

`catalog.py` is not a recommendation catalog. It holds static metric explanations (`METRICS`) served at `GET /api/catalog`. Every prescription comes from an analyzer `Recommendation`.

## Decision

### 1. GPU evidence: what the code can observe today

| Source | Field | Observable? | Reading |
|---|---|---|---|
| `/props` | `n_gpu_layers` (`PropsSnapshot.n_gpu_layers`, `safe_int`) | Only on builds that publish it. Official llama-server does not (see the `PROPS_KEYS` comment). `props_full.json` has 99. | `>= 1`: GPU. `== 0`: no layers offloaded. Negative (`-1` is "auto" in recent llama.cpp) or None: no evidence. |
| `/props` | `build_info` | Always a string like `b6500-abc1234` or `unsloth-studio/v0.1.908-beta` | Names no backend. **No evidence.** Never pattern-match it. |
| `/props` | devices, tensor split, main GPU | Not parsed. No fixture or known payload carries them. | **No evidence.** Out of scope (OQ1). |
| `/metrics` | all `METRIC_ALIASES` | Counters and gauges only | No device series. **No evidence.** |
| `/v1/models` | `meta.size` | Model size | Says nothing about the device. **No evidence.** |
| probe / lifetime rates | `prefill_tps`, `decode_tps` | Yes | Circular, since CPU-bound rates are what trigger the advice. **Never evidence.** |
| `HostHints` | `vram_gib` (`--vram-gib` on `diagnose` and `serve`) | Only if the user passes it | Finite `> 0`: user declares a GPU. `== 0`: user declares none. Negative or non-finite: absent (same rule as `memory`). |
| Studio | `/props` without `n_gpu_layers` | Yes | **No evidence.** |
| Strata gateway | only `/v1/chat/completions` is known (`test_strata_style_falls_to_chat`). `/props` may be absent. | Yes | **No evidence.** |

The code has no family detection, and T10 adds none. The guard depends only on evidence.

`signals.py` adds:

```python
@dataclass(frozen=True)
class GpuEvidence:
    status: str          # "gpu" | "cpu" | "unknown"
    source: str | None   # "props.n_gpu_layers" | "hint.vram_gib" | None

def gpu_evidence(ctx: DiagnosisContext) -> GpuEvidence
```

The first matching rule wins:

1. `ctx.props.n_gpu_layers >= 1` gives `("gpu", "props.n_gpu_layers")`.
2. Valid `hints.vram_gib > 0` gives `("gpu", "hint.vram_gib")`.
3. Valid `hints.vram_gib == 0` gives `("cpu", "hint.vram_gib")`.
4. `ctx.props.n_gpu_layers == 0` gives `("cpu", "props.n_gpu_layers")`.
5. Otherwise `("unknown", None)`.

Server-proven GPU layers beat the hint, as server size beats `--weights-gib` in `memory`. A VRAM hint beats `n_gpu_layers == 0`, because 0 layers on a host with a GPU is exactly when `-ngl` is right. `gpu_evidence` never raises, including when props is None.

### 2. Where the guard lives: one chokepoint in `engine.run_analyzers`

`diagnose`, the CLI, JSON and the dashboard sampler all go through `run_analyzers`. Filtering there means no current or future analyzer can emit GPU advice without passing the guard.

`base.py`:

```python
GPU_FLAGS = frozenset({"-ngl", "--n-gpu-layers", "--n-cpu-moe", "--cpu-moe"})

@dataclass(frozen=True)
class Recommendation:
    action: str; rationale: str; impact: Impact; flag: str | None = None
    gpu_only: bool = False   # new, last, so existing positional calls still work

def requires_gpu(r: Recommendation) -> bool:  # r.gpu_only or r.flag in GPU_FLAGS
```

A GPU flag is caught with no marker needed. `gpu_only` covers advice whose flag is device-neutral but whose purpose is GPU-only, such as `-c` to free VRAM.

`engine.py`:

```python
def guard_gpu(finding: Finding, evidence: GpuEvidence) -> Finding
```

`run_analyzers` computes `gpu_evidence(ctx)` once and applies `guard_gpu` to every collected finding. Analyzer-error findings are included and pass through unchanged. The guard runs before the sort, so `max_impact` reflects what survived. Raw analyzer funcs are not changed, so tests that call `spec.func` directly still see the full recommendation list.

`guard_gpu` rules:
- A finding with no `requires_gpu` recommendation is returned unchanged, object identity included. No evidence keys are added.
- Otherwise it returns `replace(finding, evidence=..., recommendations=..., detail=...)`:
  - **status "gpu"**: recommendations unchanged and detail unchanged. Evidence adds `gpu_status: "gpu"`, `gpu_source: <source>`, `gpu_recs_withheld: 0`.
  - **status "cpu" or "unknown"**: drop every `requires_gpu` recommendation and keep the rest in their original order. Evidence adds `gpu_status`, `gpu_source` and `gpu_recs_withheld: <count dropped>`. Append one space and the matching sentence to the detail.
- Severity, code and title never change.

### 3. One rule without GPU evidence: drop, and say so in the finding

The diagnosis stays (prefill really is CPU-bound). Only the GPU prescription is removed, and the finding states why with one fixed sentence:

| status / source | Appended sentence (exact) |
|---|---|
| unknown / None | `GPU advice withheld: no evidence a GPU serves this model; pass --vram-gib to include it.` |
| cpu / hint.vram_gib | `GPU advice withheld: --vram-gib 0 declares no GPU.` |
| cpu / props.n_gpu_layers | `GPU advice withheld: the server reports 0 GPU layers; pass --vram-gib if this host has a GPU.` |

Device-neutral rewording was rejected because it would silently change advice per host. A separate insufficient-data finding was rejected because it would detach the reason from the finding it explains.

### 4. Copy fixes in scope

These recommendations stay unguarded because their actions are device-neutral, but their rationales cite VRAM. Reword them so they hold on any host:
- `prefill.moe_cpu_experts` smaller quant: `"It can keep more experts in VRAM."` becomes `"Fewer bits per weight move fewer bytes per token on any device."`
- `prefill.cpu_bound` smaller quant: `"It can increase GPU offload."` becomes the same sentence.
- `memory.exceeds_total`: `"Weights exceed VRAM and RAM."` becomes `"Weights exceed the memory given."`

### 5. Serialization

`recommendation_to_dict` keeps its 4 keys. `gpu_only` is not serialized. The audit trail is the three finding evidence keys, which are str, int or None and therefore JSON-safe as they are.

## GPU-dependent catalog table

Every `Recommendation` emitted today:

| Finding code | Rec (impact, flag) | Guarded | Why |
|---|---|---|---|
| `prefill.moe_cpu_experts` | (HIGH, None) smaller quant | no | neutral after the copy fix |
| `prefill.moe_cpu_experts` | (HIGH, `--n-cpu-moe`) experts to CPU, `-ngl 999` | **yes**, flag | needs a GPU to keep layers on |
| `prefill.moe_cpu_experts` | (MEDIUM, `--draft-max`) drafts at 2 | no | cuts CPU expert work on any host |
| `prefill.moe_cpu_experts` | (LOW, `-c`) free VRAM for experts | **yes**, `gpu_only=True` | VRAM only |
| `prefill.cpu_bound` | (HIGH, `-ngl`) offload layers | **yes**, flag | GPU offload |
| `prefill.cpu_bound` | (HIGH, None) smaller quant | no | neutral after the copy fix |
| `prefill.cpu_bound` | (LOW, `-c`) free VRAM | **yes**, `gpu_only=True` | VRAM only |
| `prefill.slow` | (MEDIUM, `-ngl`) check all layers on GPU | **yes**, flag | GPU offload |
| `prefill.slow` (MoE) | (MEDIUM, `--n-cpu-moe`) | **yes**, flag | GPU expert placement |
| `decode.slow` (not cpu-bound) | (MEDIUM, `-ngl`) | **yes**, flag | GPU offload |
| `context.oversized` | (`-c`) set suggested | no | frees memory on any device |
| `context.near_limit` | (MEDIUM, `-c`) raise | no | truncation is device-neutral |
| `props.ctx_exceeds_training` | (MEDIUM, `-c`) | no | model property |
| `kv.saturated` | (MEDIUM, `--parallel`) | no | slot pressure |
| `queue.deferred` | (MEDIUM, None) and (MEDIUM, `--parallel`) | no | concurrency |
| `prefix_cache.low_hit_ratio` | (HIGH, None) and (MEDIUM, `--cache-reuse`) | no | prompt shape |
| `prefix_cache.moderate_hit_ratio` | (LOW, None) | no | prompt shape |
| `spec.low_acceptance`, `spec.drafts_too_long_offloaded`, `spec.raise_draft` | (MEDIUM, `--draft-max`) | no | speculative tuning on any host |
| `spec.inactive` | (LOW, None) | no | log check |
| `memory.exceeds_total` | (HIGH, None) | no | neutral after the copy fix |
| `connectivity.unreachable` / `metrics_disabled` | (HIGH, None / `--metrics`) | no | reachability |

`catalog.METRICS` ids (`decode_tps`, `prefill_tps`, `prefix_cache_hit`, `draft_length`, `acceptance`, `queue_processing`, `queue_deferred`, `quant`, `context_tokens`, `slots`, `context_used`) are not guarded. They are static descriptions with no context and no prescribed GPU flag. `decode_tps` mentions GPU spill only as one possible cause (OQ2).

## Behavior per family

MOE means `metrics_moe_offload.txt`, which is CPU-bound at 40 tok/s prefill.

| Family / setup | `gpu_evidence` | What the user sees on `prefill.moe_cpu_experts` |
|---|---|---|
| llama-server, build publishing `n_gpu_layers: 99` | gpu, `props.n_gpu_layers` | all 4 recs; detail unchanged |
| llama-server with GPU, official build, `--vram-gib 24` | gpu, `hint.vram_gib` | all 4 recs; detail unchanged |
| llama-server with GPU, official build, no hint | unknown, None | 2 recs (quant, `--draft-max`); detail ends with the "pass --vram-gib" sentence |
| llama-server CPU-only, `--vram-gib 0` | cpu, `hint.vram_gib` | 2 recs; detail ends with "--vram-gib 0 declares no GPU." |
| llama-server publishing `n_gpu_layers: 0`, no hint | cpu, `props.n_gpu_layers` | 2 recs; detail ends with the "0 GPU layers" sentence |
| Unsloth Studio (`build_info` `unsloth-studio...`), no hint | unknown, None | 2 recs plus the unknown sentence; with `--vram-gib N>0`, all 4 |
| Strata gateway (props None) | unknown, None, unless a hint is given | same as Studio; `connectivity.props_unavailable` still INFO |

## Expected outcomes

Fixture names follow T2. UNSLOTH_PROPS is `props_unsloth_minimal.json` (MoE by name, no `n_gpu_layers`). DENSE_PROPS is `PropsSnapshot.from_payload({"model_path": "Llama-3.1-8B-Q4_K_M.gguf"})`. All reports come from `diagnose(ctx)`.

- **R1** `gpu_evidence(make_context())` gives `GpuEvidence("gpu", "props.n_gpu_layers")`.
- **R2** `gpu_evidence(make_context(props="props_unsloth_minimal.json"))` gives `GpuEvidence("unknown", None)`. With `props=None` the result is the same.
- **R3** UNSLOTH_PROPS with `HostHints(vram_gib=24.0)` gives `("gpu", "hint.vram_gib")`. With `vram_gib=0.0` it gives `("cpu", "hint.vram_gib")`. With `vram_gib` of -1.0, `nan` or `inf` it gives `("unknown", None)`.
- **R4** `props_full.json` (99 layers) with `vram_gib=0.0` gives `("gpu", "props.n_gpu_layers")`.
- **R5** Props `{"model_path": "x.gguf", "n_gpu_layers": 0}`: no hint gives `("cpu", "props.n_gpu_layers")`, and `vram_gib=8.0` gives `("gpu", "hint.vram_gib")`. `n_gpu_layers: -1` with no hint gives `("unknown", None)`.
- **R6** MOE default report, `prefill.moe_cpu_experts`:
  - flags `[None, "--n-cpu-moe", "--draft-max", "-c"]`, impacts `[3, 3, 2, 1]`
  - detail exactly `"Slow prefill indicates CPU-resident experts."`
  - evidence `gpu_status "gpu"`, `gpu_source "props.n_gpu_layers"`, `gpu_recs_withheld 0`, plus the existing `prefill_tps 40.0`, `decode_tps 7.0`, `source "lifetime"`.
- **R7** MOE with UNSLOTH_PROPS, no hints, `prefill.moe_cpu_experts`:
  - severity CRITICAL
  - flags `[None, "--draft-max"]`, impacts `[3, 2]`
  - evidence `gpu_status "unknown"`, `gpu_source None`, `gpu_recs_withheld 2`
  - detail exactly `"Slow prefill indicates CPU-resident experts. GPU advice withheld: no evidence a GPU serves this model; pass --vram-gib to include it."`
  - `report.exit_code == 2`.
- **R8** Same as R7 with `vram_gib=0.0`: same recs, `gpu_status "cpu"`, `gpu_source "hint.vram_gib"`, detail ends with `"GPU advice withheld: --vram-gib 0 declares no GPU."`
- **R9** Same as R7 with `vram_gib=24.0`: 4 recs as in R6, `gpu_source "hint.vram_gib"`, `gpu_recs_withheld 0`, detail unchanged.
- **R10** MOE with DENSE_PROPS, no hints, `prefill.cpu_bound`: WARN, recs `[(3, None)]`, `gpu_recs_withheld 2`.
- **R11** MOE with UNSLOTH_PROPS and `probe=ProbeResult(500, 7, 1200, 4, 0)`, `decode.slow`:
  - severity WARN
  - recommendations `()`, `gpu_recs_withheld 1`
  - evidence still has `cpu_bound False`
  - detail exactly `"Decode throughput is below 10 tok/s. GPU advice withheld: no evidence a GPU serves this model; pass --vram-gib to include it."`
- **R12** MOE with UNSLOTH_PROPS and `probe=ProbeResult(250, 20, 1200, 4, 0)`, `prefill.slow`: WARN, recs `()`, `gpu_recs_withheld 2`.
- **R13** In the R7 report, `report.recommendations()` contains no rec with `requires_gpu` True. In the R6 report, it contains flags `--n-cpu-moe` and `-c`, the latter with action `"Reduce context to free VRAM for experts."`.
- **R14** In the R7 report, findings without GPU recs carry no `gpu_*` key: `context.oversized` evidence keys are exactly `{used_ratio, n_tokens_max, n_ctx, suggested_ctx}`, and `prefix_cache.low_hit_ratio` keys are exactly `{hit_ratio}`. `guard_gpu(f, GpuEvidence("unknown", None)) is f` for such a finding.
- **R15** Sort uses guarded impact. Use custom specs A (WARN, one rec `(HIGH, "-ngl")`, registered first) and B (WARN, one rec `(MEDIUM, None)`) with unknown evidence. The finding order is `[B, A]`, A has `max_impact 0`, and with `vram_gib=8.0` the order is `[A, B]`.
- **R16** A spec whose func raises still gives `<id>.error` INSUFFICIENT_DATA with detail `ValueError` and no `gpu_*` keys.
- **R17** `recommendation_to_dict` keys stay exactly `{action, rationale, impact, flag}`. `json.dumps(report_to_dict(r), allow_nan=False)` succeeds for the R6, R7 and R8 reports, and for R7 the finding dict evidence has `"gpu_source": null`.
- **R18** Copy fixes:
  - the smaller-quant rationale in `prefill.moe_cpu_experts` and `prefill.cpu_bound` is exactly `"Fewer bits per weight move fewer bytes per token on any device."`
  - the `memory.exceeds_total` rationale is exactly `"Weights exceed the memory given."`
- **R19** Lint invariant. Collect every recommendation that a raw `spec.func` emits for the contexts in R6, R10, R11 and R12 (probe 250/20 with `props_full`), plus `memory.exceeds_total` (vram 8, ram 4). Each one whose `action + " " + rationale` matches `(?i)\bGPU\b|\bVRAM\b|-ngl|--n-gpu-layers|--cpu-moe|--n-cpu-moe` has `requires_gpu(r) is True`.
- **R20** Existing T2 tests that call `spec.func` directly (`test_prefill_cpu_bound_dense`, `test_prefill_slow`, `test_decode_slow_offload_recommendation`, and the rest) pass unmodified.

## Tests to write

Gate: `PYTHONPATH=src:tests python3 -m unittest discover -s tests -t .`

- `tests/test_signals.py`: `test_gpu_evidence_from_props_layers` (R1), `test_gpu_evidence_unknown_without_props_or_hint` (R2), `test_gpu_evidence_vram_hint` (R3), `test_gpu_evidence_server_layers_beat_zero_hint` (R4), `test_gpu_evidence_zero_layers_and_hint` (R5).
- `tests/test_engine.py`: `test_guard_keeps_gpu_advice_with_server_evidence` (R6), `test_guard_withholds_without_evidence` (R7), `test_guard_withholds_on_zero_vram_hint` (R8), `test_guard_keeps_with_vram_hint` (R9), `test_guard_dense_cpu_bound` (R10), `test_guard_empties_decode_slow` (R11), `test_guard_empties_prefill_slow` (R12), `test_report_recommendations_respect_guard` (R13), `test_guard_leaves_neutral_findings_untouched` (R14), `test_sort_uses_guarded_impact` (R15), `test_guard_skips_error_findings` (R16).
- `tests/test_serialize.py`: `test_recommendation_dict_keys_unchanged` and `test_guarded_reports_roundtrip_json` (R17).
- `tests/test_analyzers.py`: `test_device_neutral_rationales` (R18) and `test_gpu_text_implies_requires_gpu` (R19).
- R20 is covered by the existing suite staying green.

## Open questions

1. **Parse device fields (`devices`, `tensor_split`) from `/props` if a build adds them?** Recommended default: no. No known payload or fixture carries them. Add them in a later contract with a real fixture, as a new `gpu_evidence` rule placed ahead of the hint.
2. **Reword `catalog.METRICS["decode_tps"]` "layers spilled off the GPU" for CPU hosts?** Recommended default: leave it. It is a static description that names GPU spill as one of three possible causes, prescribes no flag, and is served without a context to guard against.
3. **Should a GPU-evidence host with `n_gpu_layers >= 999`-style full offload also suppress "offload more layers"?** Recommended default: out of scope for T10. That is advice correctness on GPU hosts, not hardware presence. Track it as its own contract.
