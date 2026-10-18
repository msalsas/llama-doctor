import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .redact import safe_int, safe_str

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "prompt_tokens_total": ("prompt_tokens_total",),
    "prompt_seconds_total": ("prompt_seconds_total",),
    "tokens_predicted_total": ("tokens_predicted_total",),
    "tokens_predicted_seconds_total": ("tokens_predicted_seconds_total",),
    "prompt_tokens_cached_total": ("prompt_tokens_cached_total",),
    "n_decode_total": ("n_decode_total",),
    "n_busy_slots_per_decode": ("n_busy_slots_per_decode",),
    "n_tokens_max": ("n_tokens_max",),
    "kv_cache_usage_ratio": ("kv_cache_usage_ratio",),
    "kv_cache_tokens": ("kv_cache_tokens",),
    "requests_processing": ("requests_processing",),
    "requests_deferred": ("requests_deferred",),
    "draft_tokens_total": (
        "spec_decode_num_draft_tokens_total",
        "draft_tokens_total",
        "n_drafted_total",
    ),
    "drafts_total": ("spec_decode_num_drafts_total", "drafts_total", "n_drafts_total"),
    "draft_tokens_accepted_total": (
        "spec_decode_num_accepted_tokens_total",
        "draft_tokens_accepted_total",
        "n_drafted_accepted_total",
    ),
}


def ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return num / den


# Request counts add up across label sets (one series per model or slot); ratios do not.
SUMMABLE_METRICS = frozenset({"requests_processing", "requests_deferred"})


def _sample_value(
    samples: Mapping[str, float], aliases: tuple[str, ...], summable: bool = False
) -> float | None:
    """Unlabeled series first; else a lone labeled series, or their sum when ``summable``."""
    for alias in aliases:
        for key in ("llamacpp:" + alias, alias):
            if key in samples:
                return samples[key]
            labeled = [value for name, value in samples.items() if name.startswith(key + "{")]
            if len(labeled) == 1 or (labeled and summable):
                return sum(labeled)
    return None


@dataclass(frozen=True)
class MetricsSnapshot:
    taken_at: float = 0.0
    prompt_tokens_total: float | None = None
    prompt_seconds_total: float | None = None
    tokens_predicted_total: float | None = None
    tokens_predicted_seconds_total: float | None = None
    prompt_tokens_cached_total: float | None = None
    n_decode_total: float | None = None
    n_busy_slots_per_decode: float | None = None
    n_tokens_max: float | None = None
    kv_cache_usage_ratio: float | None = None
    kv_cache_tokens: float | None = None
    requests_processing: float | None = None
    requests_deferred: float | None = None
    draft_tokens_total: float | None = None
    drafts_total: float | None = None
    draft_tokens_accepted_total: float | None = None

    @classmethod
    def from_samples(
        cls, samples: Mapping[str, float], taken_at: float = 0.0
    ) -> "MetricsSnapshot":
        fields: dict[str, float | None] = {}
        for field_name, aliases in METRIC_ALIASES.items():
            raw = _sample_value(samples, aliases, field_name in SUMMABLE_METRICS)
            if raw is None or not math.isfinite(raw) or raw < 0:
                fields[field_name] = None
            else:
                fields[field_name] = raw
        return cls(taken_at=taken_at, **fields)

    @property
    def prefill_tps(self) -> float | None:
        return ratio(self.prompt_tokens_total, self.prompt_seconds_total)

    @property
    def decode_tps(self) -> float | None:
        return ratio(self.tokens_predicted_total, self.tokens_predicted_seconds_total)

    @property
    def cache_hit_ratio(self) -> float | None:
        # prompt_tokens_total counts only evaluated tokens, which is what llama.cpp does.
        if self.prompt_tokens_cached_total is None or self.prompt_tokens_total is None:
            return None
        return ratio(
            self.prompt_tokens_cached_total,
            self.prompt_tokens_cached_total + self.prompt_tokens_total,
        )

    @property
    def avg_draft_len(self) -> float | None:
        return ratio(self.draft_tokens_total, self.drafts_total)

    @property
    def draft_acceptance(self) -> float | None:
        return ratio(self.draft_tokens_accepted_total, self.draft_tokens_total)


# Keys a llama-server /props is expected to carry. Startup settings such as n_gpu_layers,
# n_threads and cache_type_k/v are read when present but no official build publishes them.
PROPS_KEYS: tuple[str, ...] = (
    "model_path",
    "n_ctx",
    "total_slots",
    "build_info",
    "modalities",
)


@dataclass(frozen=True)
class PropsSnapshot:
    model_name: str | None = None
    quant: str | None = None
    n_ctx: int | None = None
    total_slots: int | None = None
    build_info: str | None = None
    n_gpu_layers: int | None = None
    n_threads: int | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    has_vision: bool | None = None
    is_moe: bool | None = None
    # default_generation_settings.params["speculative.types"] without "none": () when the
    # server says nothing is configured, None when it does not say.
    spec_types: tuple[str, ...] | None = None
    missing: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PropsSnapshot":
        if not isinstance(payload, dict):
            return cls(missing=PROPS_KEYS)
        marker = payload.get("media_marker")
        forbidden = (marker,) if isinstance(marker, str) else ()
        raw_path = payload.get("model_path")
        model_name = None
        quant = None
        is_moe = None
        if isinstance(raw_path, str):
            base = raw_path.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".gguf")
            if base.strip():
                model_name = safe_str(base, forbidden=forbidden)
                if model_name is not None:
                    quant = parse_quant(model_name)
                    is_moe = guess_moe(model_name)
        modalities = payload.get("modalities")
        has_vision = None
        if isinstance(modalities, dict):
            vision = modalities.get("vision")
            if isinstance(vision, bool):
                has_vision = vision
        if quant is None:
            quant = parse_ftype(payload.get("model_ftype"), forbidden=forbidden)
        n_ctx = None
        spec_types = None
        defaults = payload.get("default_generation_settings")
        if isinstance(defaults, dict):
            n_ctx = safe_int(defaults.get("n_ctx"))
            params = defaults.get("params")
            if isinstance(params, dict):
                spec_types = parse_spec_types(params.get("speculative.types"))
        if n_ctx is None:
            n_ctx = safe_int(payload.get("n_ctx"))
        total_slots = safe_int(payload.get("total_slots"))
        build_info = safe_str(payload.get("build_info"), forbidden=forbidden)
        n_gpu_layers = safe_int(payload.get("n_gpu_layers"))
        n_threads = safe_int(payload.get("n_threads"))
        cache_type_k = safe_str(payload.get("cache_type_k"), forbidden=forbidden)
        cache_type_v = safe_str(payload.get("cache_type_v"), forbidden=forbidden)
        missing = tuple(
            key
            for key in PROPS_KEYS
            if key not in payload
            and not (key == "n_ctx" and isinstance(defaults, dict) and "n_ctx" in defaults)
        )
        return cls(
            model_name=model_name,
            quant=quant,
            n_ctx=n_ctx,
            total_slots=total_slots,
            build_info=build_info,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            has_vision=has_vision,
            is_moe=is_moe,
            spec_types=spec_types,
            missing=missing,
        )


_QUANT_RE = (
    r"(UD-)?(IQ\d+_[A-Z0-9]+(?:_[A-Z0-9]+)*|Q\d+(?:_[A-Z0-9]+)*|MXFP4(?:_MOE)?|"
    r"BF16|F16|F32)(?=$|[.-])"
)


def parse_quant(name: str) -> str | None:
    match = re.search(_QUANT_RE, name, re.IGNORECASE)
    if match is None:
        return None
    return match.group(0).upper()


_FTYPE_SIZES = {"small": "S", "medium": "M", "large": "L"}


def parse_ftype(value: Any, forbidden: tuple[str, ...] = ()) -> str | None:
    """Quant tag from llama.cpp's ftype name: ``Q4_K - Medium`` gives ``Q4_K_M``."""
    text = safe_str(value, forbidden=forbidden)
    if text is None:
        return None
    base, _, suffix = text.partition(" - ")
    tag = "_".join(base.split()).upper()
    size = _FTYPE_SIZES.get(suffix.strip().lower())
    if size is not None:
        tag += "_" + size
    return tag if re.fullmatch(r"[A-Z0-9_]+", tag) else None


def parse_spec_types(value: Any) -> tuple[str, ...] | None:
    """Configured speculative types, without ``none``; None when the value is unreadable."""
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        items = value
    else:
        return None
    names = (item.strip().lower() for item in items)
    return tuple(name for name in names if re.fullmatch(r"[a-z0-9_.-]+", name) and name != "none")


_MOE_RE = r"[-_]A\d+(?:\.\d+)?B(?=$|[-_.])|moe|mixtral|gpt-oss|deepseek|qwen3-next"


def guess_moe(name: str) -> bool | None:
    if re.search(_MOE_RE, name, re.IGNORECASE):
        return True
    return None


def _model_key(name: str) -> str:
    """Case-folded model name without path, ``.gguf``/``-GGUF`` suffix or quant tag."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1].lower().removesuffix(".gguf")
    base = base.removesuffix("-gguf")
    return re.sub(r"[-_.]?" + _QUANT_RE, "", base, count=1, flags=re.IGNORECASE)


def _same_model(model_id: str | None, model_name: str | None) -> bool:
    if model_id is None or model_name is None:
        return False
    return _model_key(model_id) == _model_key(model_name)


@dataclass(frozen=True)
class ModelsMeta:
    model_id: str | None = None
    n_params: int | None = None
    size_bytes: int | None = None
    n_ctx_train: int | None = None
    quant: str | None = None
    entry_loaded: bool = False

    @classmethod
    def from_payload(cls, payload: Any) -> "ModelsMeta":
        if not isinstance(payload, dict):
            return cls()
        data = payload.get("data")
        if not isinstance(data, list):
            return cls()
        entries = [entry for entry in data if isinstance(entry, dict)]
        if not entries:
            return cls()
        # Unsloth Studio lists every downloaded model; only one is loaded:true.
        entry = next((e for e in entries if e.get("loaded") is True), entries[0])
        meta = entry.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        return cls(
            model_id=safe_str(entry.get("id")),
            n_params=safe_int(meta.get("n_params")),
            size_bytes=safe_int(meta.get("size")),
            n_ctx_train=safe_int(meta.get("n_ctx_train")),
            quant=safe_str(entry.get("quant")) or safe_str(meta.get("quant")),
            entry_loaded=entry.get("loaded") is True,
        )


@dataclass(frozen=True)
class CompletionTimings:
    prompt_n: int | None = None
    prompt_ms: float | None = None
    predicted_n: int | None = None
    predicted_ms: float | None = None


@dataclass(frozen=True)
class EndpointError:
    endpoint: str
    status: int | None
    reason: str


@dataclass(frozen=True)
class ProbeResult:
    prefill_tps: float
    decode_tps: float
    prompt_n: int
    predicted_n: int
    warmup_s: float

    @property
    def reliable(self) -> bool:
        return self.prompt_n >= 1000


@dataclass(frozen=True)
class SlotsSnapshot:
    """Numbers summed over the server's slots; never any prompt text or ids."""

    live_used: int
    live_capacity: int
    n_slots: int
    n_processing: int
    slots_truncated: bool = False


@dataclass(frozen=True)
class ServerSnapshot:
    taken_at: float
    metrics: MetricsSnapshot | None
    props: PropsSnapshot | None
    models: ModelsMeta | None
    errors: Mapping[str, EndpointError]
    slots: SlotsSnapshot | None = None
    # Why /slots yielded nothing; None when it was skipped or succeeded.
    slots_error: str | None = None

    @property
    def alive(self) -> bool:
        return self.metrics is not None or self.models is not None

    @property
    def server_props(self) -> PropsSnapshot | None:
        """Props for display, with name/quant/MoE gaps filled from /v1/models.

        Unsloth Studio's /props reports a repo directory as model_path. The quant
        comes from props, then a ``...-UD-Q4_K_XL.gguf`` style model id, then the
        per-entry ``quant`` field of /v1/models. When no entry is ``loaded: true``
        the first entry is only trusted if its id names the same model as props.
        """
        props = self.props
        models = self.models
        if props is None or models is None:
            return props
        stem = models.model_id.removesuffix(".gguf") if models.model_id else None
        if not models.entry_loaded and not _same_model(stem, props.model_name):
            return props
        updates: dict[str, Any] = {}
        if props.model_name is None and stem is not None:
            updates["model_name"] = stem
            if props.is_moe is None and guess_moe(stem) is not None:
                updates["is_moe"] = guess_moe(stem)
        if props.quant is None:
            quant = (parse_quant(stem) if stem else None) or models.quant
            if quant is not None:
                updates["quant"] = quant
        return replace(props, **updates) if updates else props
