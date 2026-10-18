# llama-doctor

A command-line tool that reads a local LLM server's metrics and tells you what is misconfigured.

[![CI](https://github.com/msalsas/llama-doctor/actions/workflows/ci.yml/badge.svg)](https://github.com/msalsas/llama-doctor/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Dashboard](docs/assets/dashboard.png)

The live dashboard on a Qwen3-30B MoE rig with offloaded experts.

## What it does

- Reads `/metrics`, `/props` and `/v1/models` over HTTP. It never attaches to, restarts or reconfigures the server process.
- Runs 8 analyzers that emit 35 distinct findings across connectivity, memory, cache, context sizing, speculative decoding, quantization and expert offload.
- Ranks recommendations by impact and names the server flag to change.
- Returns an exit code per severity, so you can gate CI jobs or cron checks on it.
- Optionally serves a local dashboard with live charts and findings.

It works with llama.cpp's server and forks that expose the same `llamacpp:` metrics namespace. It has been tested against Unsloth Studio's llama.cpp builds.

## Install

Requires Python 3.10 or newer. There are no runtime dependencies; it uses the standard library only.

```sh
git clone https://github.com/msalsas/llama-doctor.git
cd llama-doctor
pip install .
```

To run it without installing anything:

```sh
PYTHONPATH=src python3 -m llama_doctor diagnose http://127.0.0.1:8080
```

## Quick start

Point it at a running llama.cpp server:

```sh
llama-doctor diagnose http://127.0.0.1:8080
```

```text
llama-doctor 0.1.0  http://127.0.0.1:8080
model  Qwen3-30B-A3B-UD-Q4_K_XL  quant UD-Q4_K_XL  ctx 32768  slots 1  build b6500-abc1234

rates (lifetime averages)
  prefill       8422.9 tok/s
  decode        1457.9 tok/s
  prefix cache  10.3 %
  drafts        len 3.00  acceptance 58.3 %

findings
  WARN prefix_cache.low_hit_ratio  Prefix cache hit rate is low
        Few prompt tokens are reused.
  WARN kv.context_oversized  Context is oversized
        Observed peak token use is far below configured context.
  OK   connectivity.ok  Endpoints available
  OK   prefill.ok  Prefill throughput is healthy
  OK   decode.ok  Decode throughput is healthy
  OK   spec.ok  Speculative decoding is healthy
  OK   props.ok  Properties are complete
  N/A  memory.insufficient_data  Memory estimate unavailable
        Provide --vram-gib and optionally --ram-gib.
recommendations (by impact)
  1. high   Keep the prompt prefix byte-identical and put variable content last.
  2. medium Enable cache reuse.  [--cache-reuse]
  3. medium Set context to 12288.  [-c]
overall: WARN (exit 1)
```

Exit codes: `0` = OK, `1` = WARN, `2` = CRIT (including server unreachable), `3` = invalid usage or arguments.

## Dashboard

```sh
llama-doctor serve http://127.0.0.1:8080
```

`serve` runs a local HTTP server on port 8765; open `http://127.0.0.1:8765` in a browser. It polls the llama.cpp server every 2 s and keeps the last 300 points in memory.

![Dashboard findings](docs/assets/dashboard-findings.png)

The findings and recommendations panels re-run every analyzer on each poll, so a change to your server flags shows up within one interval. The dashboard binds to `127.0.0.1` by default and loads no external resources: no CDNs, fonts or analytics.

## CLI reference

`llama-doctor diagnose URL`

| Flag | Effect |
| --- | --- |
| `--json` | Print findings as JSON instead of text. |
| `--probe` | Send a warm-up and one synthetic completion request to measure real latency. |
| `--vram-gib N` | GPU memory available, used by the memory estimate. |
| `--ram-gib N` | System RAM available for offloaded weights. |
| `--api-key-env VAR` | Read the server's API key from environment variable `VAR`. |
| `--timeout SEC` | HTTP timeout per request. |
| `--no-color` | Disable ANSI colour in text output. |

`--probe` is off by default. It is the only option that sends a write request to the server; everything else is a read-only GET.

`llama-doctor serve URL`

| Flag | Effect |
| --- | --- |
| `--host` | Address to bind (default `127.0.0.1`). |
| `--port` | Port to listen on (default `8765`). |
| `--interval` | Seconds between polls (default `2.0`). |
| `--history` | Number of points kept for the charts (default `300`). |

`serve` also accepts `--api-key-env` and `--timeout`.

## Checks

The analyzers cover connectivity, server properties, prefill throughput, decode throughput, prefix cache, KV cache and context sizing, speculative decoding, and memory (including quantization and expert offload). The modules in [`src/llama_doctor/analysis/`](src/llama_doctor/analysis/) are the source of truth for each finding and its threshold.

Two findings from the output above:

- `prefix_cache.low_hit_ratio`: only 10.3 % of prompt tokens came from the cache. The fix is to keep the prompt prefix byte-identical across requests and enable `--cache-reuse`.
- `kv.context_oversized`: the server reserves a 32768-token context, but observed peak use is far lower. Setting `-c 12288` frees KV cache memory.

## Development

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e . ruff
python3 -m unittest discover -v
ruff check src tests
```

Pull requests are welcome.

## License

MIT. See [LICENSE](LICENSE).
