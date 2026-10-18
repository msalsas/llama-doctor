"""Static explanations for the dashboard metrics, served at ``GET /api/catalog``.

Texts hold for any loaded model: they explain what each value measures and how to read
it relative to its own history, not absolute ranges tied to a model size, quant or GPU.
"""

METRICS: dict[str, dict[str, str]] = {
    "decode_tps": {
        "label": "Decode throughput",
        "description": "Tokens generated per second while answering, after the prompt is read.",
        "interpretation": (
            "Higher is faster output; its scale depends on the model and hardware, so "
            "compare it with this server's own lifetime average. A window clearly below "
            "lifetime points to contention, longer contexts or layers spilled off the GPU."
        ),
    },
    "prefill_tps": {
        "label": "Prefill throughput",
        "description": "Prompt tokens evaluated per second before the first output token.",
        "interpretation": (
            "Higher means a shorter wait before the first token. Prefill is batched, so "
            "it normally runs several times faster than decode on the same server; when "
            "it is barely above decode, prompt processing is compute-starved, typically "
            "by CPU offload or a small -ub."
        ),
    },
    "prefix_cache_hit": {
        "label": "Prefix-cache hit",
        "description": "Share of prompt tokens reused from the cached prefix instead of recomputed.",
        "interpretation": (
            "Higher means less prompt recomputation and a faster first token. A falling "
            "hit rate means the prompt prefix changes between turns, so keep system "
            "prompt and tools byte-identical and put variable text last."
        ),
    },
    "kv_usage": {
        "label": "KV usage",
        "description": "Fraction of KV cache cells currently holding context tokens.",
        "interpretation": (
            "It is a share of the cache's own capacity: sustained near 100 % risks "
            "truncation or deferred requests, while a peak that stays far below it means "
            "the context is oversized and wastes memory. Some servers, such as Unsloth "
            "Studio, do not export it."
        ),
    },
    "draft_length": {
        "label": "Draft length",
        "description": "Average number of tokens the draft model proposes per speculative step.",
        "interpretation": (
            "Longer drafts only pay off while acceptance stays high; if acceptance drops "
            "as drafts grow, lower --draft-max."
        ),
    },
    "acceptance": {
        "label": "Draft acceptance",
        "description": "Fraction of drafted tokens that the main model accepts.",
        "interpretation": (
            "Higher means speculative decoding saves more main-model steps; when it "
            "falls, the draft model's cost eats the gain, so check that decode "
            "throughput still beats running without a draft."
        ),
    },
    "queue_processing": {
        "label": "Requests processing",
        "description": "Requests currently in flight across all server slots.",
        "interpretation": (
            "Read it against the slot count (--parallel): below it there is spare "
            "capacity, and a value pinned at it means every slot is busy and new "
            "requests have to wait."
        ),
    },
    "queue_deferred": {
        "label": "Requests deferred",
        "description": "Requests waiting right now because all slots are busy.",
        "interpretation": (
            "It should drop back to 0 once a burst drains; if it rarely returns to 0, "
            "demand exceeds the slots, so raise --parallel when KV memory allows or "
            "accept queueing."
        ),
    },
    "quant": {
        "label": "Quantization",
        "description": "Weight format of the loaded model, such as Q4_K_M or Q8_0.",
        "interpretation": (
            "Fewer bits per weight shrink memory use and usually speed up decode, at the "
            "cost of output quality; a higher-bit quant of the same model is more "
            "faithful but heavier."
        ),
    },
    "context_tokens": {
        "label": "Context size",
        "description": "Configured context window per slot (n_ctx), in tokens, as reported by /props.",
        "interpretation": (
            "KV memory grows linearly with it, so a larger window costs memory that "
            "could hold weights; size it from the Context used peak rather than the "
            "model's maximum."
        ),
    },
    "context_used": {
        "label": "Context used",
        "description": (
            "Peak tokens held in context since the server started (n_tokens_max) as a share of "
            "n_ctx; it is the highest point observed, not the instantaneous use of the current request."
        ),
        "interpretation": (
            "At 90 % or more the longest conversation nearly hit the window, so raise -c or trim "
            "prompts; under 25 % the window is oversized. The peak only resets when llama-server "
            "restarts, and builds such as Unsloth export only n_tokens_max, so live per-request use "
            "is not visible."
        ),
    },
}
