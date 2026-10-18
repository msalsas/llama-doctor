# T5: dashboard frontend

Context: T4 serves these files from `src/llama_doctor/dashboard/static/`
through a fixed allowlist (`/`, `/static/app.css`, `/static/app.js`,
`/static/vendor/chart.umd.js`). Every HTML response carries `CSP =
"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:;
connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"`.
That rules out inline scripts, inline `style` attributes, `<style>`
blocks, and any remote resource. The page polls `GET /api/state`; the
schema is below. Read `src/llama_doctor/dashboard/server.py`,
`sampler.py`, and `serialize.py` first. There is no build step, no npm at
runtime, and no framework: plain ES2020 in a single strict-mode IIFE. No
emoji anywhere, including the UI. Copy should be terse and technical.

## Files to create

```
src/llama_doctor/dashboard/static/index.html
src/llama_doctor/dashboard/static/app.css
src/llama_doctor/dashboard/static/app.js
src/llama_doctor/dashboard/static/vendor/chart.umd.js
tests/test_static.py
```
Remove `static/.gitkeep` if it exists. Vendoring: if the scaffold put
`assets/chart.umd.js` at the repo root, `git mv` it to
`static/vendor/chart.umd.js` and delete the empty `assets/`. Otherwise take
`package/dist/chart.umd.js` from `npm pack chart.js@4` (dev machine only)
and commit it unmodified, license banner included. Make `pyproject.toml`
ship `static/*.html|css|js` and `static/vendor/*.js` as package data. With
setuptools that is `[tool.setuptools.package-data] "llama_doctor.dashboard" =
["static/*", "static/vendor/*"]`. Check it with
`python3 -m pip wheel . --no-deps -w /tmp/w && python3 -m zipfile -l /tmp/w/*.whl`.

## /api/state schema (produced by T4)

```
version, target, interval_s, rate_window_s, updated_at: float|null,
server: {model_name, quant, n_ctx, total_slots, build_info, n_gpu_layers, n_threads,
         cache_type_k, cache_type_v, has_vision, is_moe, missing: [..]} | null,
lifetime: {prefill_tps, decode_tps, cache_hit_ratio, avg_draft_len, draft_acceptance,
           kv_usage_ratio, n_tokens_max},          // kv_usage_ratio, n_tokens_max: current gauges
history: [{t, decode_tps, prefill_tps, cache_hit_ratio, avg_draft_len, draft_acceptance,
           kv_usage_ratio, requests_processing, requests_deferred, context_used_ratio}],
                                                       // null = no data, draw a gap;
                                                       // kv_usage_ratio, requests_* are
                                                       // instantaneous gauges;
                                                       // context_used_ratio is a peak
context: {used_ratio, n_tokens_max, n_ctx, label},     // label like "78.5k / 80.1k"
report: {overall, exit_code, findings: [{analyzer, code, severity, title, detail,
         evidence: {k: v}, recommendations: [..]}], recommendations: [{action, rationale,
         impact, flag}]} | null,
errors: {endpoint: reason},
notes: {kv_unexported, cache_hit_unexported, decode_idle, prefill_idle}  // show note when true
severity in {"critical","warn","info","ok","insufficient_data"}; impact in {"high","medium","low"}
```

## index.html structure (each id appears exactly once)

- `<head>`: charset, viewport, `<title>llama-doctor</title>`,
  `<link rel="stylesheet" href="/static/app.css">`. Scripts go at the end
  of `<body>`: `/static/vendor/chart.umd.js`, then `/static/app.js`.
- Header bar: product name, `hdr-target`, an `overall-badge` (severity label
  text, never color alone), and `updated-at` ("updated 3 s ago" or "no data
  yet"). A model strip of label/value pairs: `hdr-model`, `hdr-quant`,
  `hdr-ctx`, `hdr-slots`, `hdr-build`.
- `conn-banner`: hidden by default. It shows `dashboard disconnected,
  retrying` when the fetch fails, or `server unreachable` plus the endpoint
  errors when `errors` is non-empty and `history` is empty.
- Stat tiles, in this order: `tile-decode` (Decode tok/s), `tile-prefill`
  (Prefill tok/s), `tile-cache` (Prefix-cache hit), `tile-kv` (KV usage),
  `tile-context` (Context), `tile-queue` (Queue). Decode, prefill and cache
  show the latest non-null live value from `history` as the hero number,
  with a smaller secondary line `lifetime <value>`. With no live value, the
  hero shows the lifetime value, labeled `lifetime avg`. `n/a` when both are
  null. KV usage is a gauge with no lifetime average: its hero is the latest
  `kv_usage_ratio`, secondary `current`, or `not exported` when
  `notes.kv_unexported`. Context shows `context.used_ratio` as a percent with
  `context.label` below it. Queue shows `requests_processing` with
  `<requests_deferred> waiting` below it.
- Info tooltips: text comes only from `GET /api/catalog`. Tiles get one
  button (Queue combines `queue_processing` and `queue_deferred`); each chart
  header gets one per series; `hdr-quant`, `hdr-ctx` and `hdr-slots` get one.
- Chart panels, each a `<canvas>` with a descriptive `aria-label` and a
  `<details>` that holds a table of the last 20 points. `chart-decode`
  (tok/s) and `chart-prefill` (tok/s) are separate charts, because their
  scales differ by 10-100x and one chart gets one y axis.
  `chart-kv-cache` has three series in percent, KV usage, prefix-cache
  hit and Context used (dashed, a peak), with a legend. The speculative-decoding panel holds
  `chart-draft-len` (tokens) and `chart-acceptance` (percent) as two small
  stacked charts.
- Findings panel: `findings-list` holds non-OK, non-insufficient findings in
  report order. Each row has a severity badge (text label CRIT, WARN, INFO),
  the code in monospace, the title, the detail, and evidence as a compact
  `key value` list. `recs-list` is an `<ol>` of `report.recommendations`,
  each with an impact tag, the action, the flag in `<code>`, and the
  rationale in secondary text. `skipped-list` sits inside a `<details>`
  summarized as `R run, S skipped`: R counts every finding that is not
  insufficient data, S the insufficient ones. The list shows the skipped
  checks plus the OK checks marked as passed.

## app.css (dark theme, tokens as CSS custom properties on `:root`)

```
--bg #121211  --surface #1a1a19  --surface-2 #232321  --border #2e2e2b  --grid #2a2a28
--text #f0efec  --text-2 #c3c2b7  --text-3 #8f8e86
--s-decode #3987e5  --s-prefill #d95926  --s-kv #199e70  --s-cache #c98500
--s-draft #d55181  --s-accept #9085e9  --s-ctx (Context used series)
--ok #0ca30c  --warn #fab219  --crit #d03b3b  --info var(--text-2)
```
- `color-scheme: dark`. Font stacks: `system-ui, sans-serif` for text,
  `ui-monospace, monospace` for codes and flags. Numbers use
  `font-variant-numeric: tabular-nums`.
- Layout: CSS grid. Two columns at >= 1100px, one column below. Panels have
  8px radius, a 1px `--border`, and 16px padding.
- Severity classes: `.sev-critical`, `.sev-warn`, `.sev-info`, `.sev-ok`,
  `.sev-insufficient_data`. A badge is outlined in its status color with
  text in `--text`, so text never wears the series or status color as its
  only cue.
- `.hidden { display: none }`. Keep shadows and gradients to a minimum.

## app.js

- Build the DOM with `document.createElement` and `textContent` only. Never
  use `innerHTML`, `outerHTML`, `insertAdjacentHTML`, `document.write`,
  `eval`, or `new Function`. All server strings are treated as untrusted.
- Read colors via `getComputedStyle(document.documentElement)`, so the CSS
  tokens are the single source of truth.
- Chart.js options for every chart: `animation: false`, `responsive: true`,
  `maintainAspectRatio: false`, `spanGaps: false` (nulls show as gaps), line
  `borderWidth: 2`, point radius 0 except 3 on a point isolated between
  gaps (so it stays visible), `pointHoverRadius: 4`, `tension: 0`,
  `interaction: {mode: "index", intersect: false}`, y `beginAtZero: true`
  (percent charts fixed 0-100), grid color `--grid`, tick color `--text-3`.
  Show the legend only on `chart-kv-cache`. The tooltip shows the value with
  its unit (`tok/s`, `%`, `tokens`). X labels are local `HH:MM:SS` from `t`.
  Ratios are multiplied by 100 for display.
- Create charts once, then on each poll replace `data.labels` and
  `datasets[i].data` and call `chart.update("none")`.
- Polling: `fetch("/api/state", {cache: "no-store"})` every
  `interval_s * 1000` ms (2000 before the first response) via chained
  `setTimeout`. On failure, show `conn-banner` and back off by doubling, up
  to 10 s. The first success resets the interval. Update `updated-at` once
  per second from `updated_at`.
- Formatting: rates `toFixed(1)`, percents `toFixed(1) + " %"`, draft
  length `toFixed(2)`. Null prints `n/a` and `-` in the header.
- The overall badge uses `report.overall` and falls back to
  `insufficient_data` when `report` is null.

## Test cases (`tests/test_static.py`, read files via `default_static_dir()`)

- `test_static_files_exist`: all four files exist and are non-empty.
- `test_index_references_local_assets_only`: it contains
  `href="/static/app.css"`, `src="/static/vendor/chart.umd.js"`, and
  `src="/static/app.js"`. Every `src=`/`href=` value starts with `/static/`
  (regex).
- `test_index_has_no_inline_code`: there is no `<style`, no ` style=`, no
  `<script>` without `src`, and nothing matching `\son[a-z]+=`.
- `test_required_ids_once`: each of `hdr-target hdr-model hdr-quant hdr-ctx
  hdr-slots hdr-build overall-badge updated-at conn-banner tile-decode
  tile-prefill tile-cache tile-kv tile-context tile-queue chart-decode
  chart-prefill chart-kv-cache chart-draft-len chart-acceptance findings-list
  recs-list skipped-list note-decode note-prefill note-kv note-cache-hit`
  appears exactly once as `id="..."`.
- `test_vendor_script_order`: the index position of chart.umd.js comes
  before app.js.
- `test_app_js_avoids_html_sinks`: none of the forbidden tokens listed above
  appear.
- `test_no_remote_urls_in_app_code`: `http://` and `https://` are absent
  from app.js, app.css, and index.html.
- `test_app_js_polls_state_endpoint`: it contains `"/api/state"` and
  `no-store`.
- `test_css_severity_classes`: all five `.sev-*` selectors are present.
- `test_css_tokens_present`: every token listed above is defined.
- `test_vendor_is_chartjs_v4_umd`: the first 1024 chars match
  `Chart\.js v4\.` and contain `MIT License`. Size > 100000 bytes.
- `test_served_by_dashboard_server`: `make_server(lambda: {}, port=0)` with
  the default static dir. `GET /` gives 200 with `id="findings-list"`;
  `GET /static/vendor/chart.umd.js` gives 200, `text/javascript`.

## Acceptance

`python3 -m unittest discover -v` passes (all earlier tests included) and
`ruff check .` is clean. Manual check: in a Python shell, start
`tests.support.FakeLlamaServer` with the MoE fixtures, then run
`python3 -m llama_doctor serve <fake url>`. Confirm that the browser
console shows no errors and no CSP violations, the network panel shows no
requests leaving the origin, the layout holds at 1280px and 768px wide,
idle periods appear as gaps rather than drops to zero, and the CRIT finding
`prefill.moe_cpu_experts` with its four recommendations is on screen.
