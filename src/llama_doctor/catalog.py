"""Static explanations for the dashboard metrics, served at ``GET /api/catalog``.

Ranges are rough guides for a single-GPU 16 GB rig running llama-server.
"""

METRICS: dict[str, dict[str, str]] = {
    "decode_tps": {
        "label": "Decode throughput",
        "description": "Tokens generated per second while answering, after the prompt is read.",
        "interpretation": (
            "On 16 GB, 7-14B models fully on GPU at Q4 reach roughly 40-100 tok/s, 20-32B or MoE "
            "with experts on CPU 10-30 tok/s; under 5 tok/s usually means layers spilled to RAM."
        ),
    },
    "prefill_tps": {
        "label": "Prefill throughput",
        "description": "Prompt tokens evaluated per second before the first output token.",
        "interpretation": (
            "GPU-resident models on 16 GB typically prefill 1000-5000 tok/s; under 300 points to "
            "CPU offload or a small -ub batch, and under 100 the server is CPU-bound."
        ),
    },
    "prefix_cache_hit": {
        "label": "Prefix-cache hit",
        "description": "Share of prompt tokens reused from the cached prefix instead of recomputed.",
        "interpretation": (
            "Chat and agent workloads should stay above 50 %; under 20 % the prompt prefix changes "
            "every turn, so keep system prompt and tools byte-identical and put variable text last."
        ),
    },
    "kv_usage": {
        "label": "KV usage",
        "description": "Fraction of KV cache cells currently holding context tokens.",
        "interpretation": (
            "Under 80 % is comfortable; sustained above 90 % risks truncation or deferred requests, "
            "while a peak that never passes 25 % means the context is oversized and wastes VRAM."
        ),
    },
    "draft_length": {
        "label": "Draft length",
        "description": "Average number of tokens the draft model proposes per speculative step.",
        "interpretation": (
            "2-8 tokens is typical; long drafts only pay off when acceptance stays high, so lower "
            "--draft-max if acceptance drops as drafts grow."
        ),
    },
    "acceptance": {
        "label": "Draft acceptance",
        "description": "Fraction of drafted tokens that the main model accepts.",
        "interpretation": (
            "Above 70 % speculative decoding is paying off, 40-70 % is marginal, and under 40 % "
            "the draft model costs more than it saves on a 16 GB card."
        ),
    },
    "queue_processing": {
        "label": "Requests processing",
        "description": "Requests currently in flight across all server slots.",
        "interpretation": (
            "A single-user rig shows 0 or 1; a value pinned at the slot count (--parallel) means "
            "every slot is busy and new requests have to wait."
        ),
    },
    "queue_deferred": {
        "label": "Requests deferred",
        "description": "Requests waiting right now because all slots are busy.",
        "interpretation": (
            "It stays at 0 for one client and drops back to 0 once a burst drains; if it rarely "
            "returns to 0, demand exceeds the slots, so raise --parallel when KV memory allows or "
            "accept queueing."
        ),
    },
    "quant": {
        "label": "Quantization",
        "description": "Weight format of the loaded model, such as Q4_K_M or Q8_0.",
        "interpretation": (
            "On 16 GB, Q4_K_M or UD-Q4_K_XL is the sweet spot for 14-24B, Q6_K or Q8_0 fits 8B "
            "comfortably, and below Q3 output quality drops noticeably."
        ),
    },
    "context_tokens": {
        "label": "Context size",
        "description": "Configured context window per slot (n_ctx), in tokens.",
        "interpretation": (
            "KV memory grows linearly with it: 16k-32k suits a 14B Q4 model on 16 GB with f16 KV, "
            "and 64k or more usually needs a q8_0 KV cache or a smaller model."
        ),
    },
}
