from collections import deque
from collections.abc import Callable, Sequence
from math import ceil, isfinite
from threading import Event, Lock, Thread, current_thread
from typing import Any

from llama_doctor import __version__
from llama_doctor.analysis import BUILTIN_ANALYZERS, diagnose
from llama_doctor.analysis.base import AnalyzerSpec, DiagnosisContext, Report
from llama_doctor.analysis.signals import context_used_ratio
from llama_doctor.models import MetricsSnapshot, PropsSnapshot, ServerSnapshot
from llama_doctor.serialize import (
    context_to_dict,
    errors_to_dict,
    props_to_dict,
    rates_to_dict,
    report_to_dict,
)

DEFAULT_INTERVAL = 2.0
DEFAULT_HISTORY = 300
DEFAULT_RATE_WINDOW = 30.0
POINT_KEYS = (
    "t",
    "decode_tps",
    "prefill_tps",
    "cache_hit_ratio",
    "avg_draft_len",
    "draft_acceptance",
    "kv_usage_ratio",
    "requests_processing",
    "requests_deferred",
    "context_used_ratio",
)


def _delta_rate(
    previous_numerator: float | None,
    current_numerator: float | None,
    previous_denominator: float | None,
    current_denominator: float | None,
) -> float | None:
    if (
        previous_numerator is None
        or current_numerator is None
        or previous_denominator is None
        or current_denominator is None
    ):
        return None
    if not all(
        isfinite(value)
        for value in (
            previous_numerator,
            current_numerator,
            previous_denominator,
            current_denominator,
        )
    ):
        return None
    numerator_delta = current_numerator - previous_numerator
    denominator_delta = current_denominator - previous_denominator
    if numerator_delta < 0 or denominator_delta <= 0:
        return None
    rate = numerator_delta / denominator_delta
    return rate if isfinite(rate) else None


def delta_point(
    prev: MetricsSnapshot | None, cur: MetricsSnapshot, t: float, n_ctx: int | None = None
) -> dict[str, float | None]:
    """Rates between a baseline snapshot and ``cur``; ``prev`` is the window baseline.

    llama.cpp counters only move when a request completes, so the sampler passes the
    newest snapshot at or before the window start rather than the adjacent sample.
    ``context_used_ratio`` is the observed peak over ``n_ctx``, not instantaneous use.
    """
    point: dict[str, float | None] = {
        "t": t,
        "decode_tps": None,
        "prefill_tps": None,
        "cache_hit_ratio": None,
        "avg_draft_len": None,
        "draft_acceptance": None,
        "kv_usage_ratio": cur.kv_cache_usage_ratio,
        "requests_processing": cur.requests_processing,
        "requests_deferred": cur.requests_deferred,
        "context_used_ratio": context_used_ratio(cur.n_tokens_max, n_ctx),
    }
    if prev is None:
        return point

    point["prefill_tps"] = _delta_rate(
        prev.prompt_tokens_total,
        cur.prompt_tokens_total,
        prev.prompt_seconds_total,
        cur.prompt_seconds_total,
    )
    point["decode_tps"] = _delta_rate(
        prev.tokens_predicted_total,
        cur.tokens_predicted_total,
        prev.tokens_predicted_seconds_total,
        cur.tokens_predicted_seconds_total,
    )
    if (
        prev.prompt_tokens_cached_total is not None
        and cur.prompt_tokens_cached_total is not None
        and prev.prompt_tokens_total is not None
        and cur.prompt_tokens_total is not None
    ):
        cached_tokens = cur.prompt_tokens_cached_total - prev.prompt_tokens_cached_total
        prompt_tokens = cur.prompt_tokens_total - prev.prompt_tokens_total
        denominator = cached_tokens + prompt_tokens
        if cached_tokens >= 0 and prompt_tokens >= 0 and denominator > 0:
            point["cache_hit_ratio"] = cached_tokens / denominator
    point["avg_draft_len"] = _delta_rate(
        prev.draft_tokens_total,
        cur.draft_tokens_total,
        prev.drafts_total,
        cur.drafts_total,
    )
    point["draft_acceptance"] = _delta_rate(
        prev.draft_tokens_accepted_total,
        cur.draft_tokens_accepted_total,
        prev.draft_tokens_total,
        cur.draft_tokens_total,
    )
    return point


def _counters_reset(prev: MetricsSnapshot, cur: MetricsSnapshot) -> bool:
    """True when a monotonic token counter went backwards (llama-server restarted)."""
    return any(
        before is not None and after is not None and after < before
        for before, after in (
            (prev.tokens_predicted_total, cur.tokens_predicted_total),
            (prev.prompt_tokens_total, cur.prompt_tokens_total),
        )
    )


class Sampler:
    def __init__(
        self,
        fetch: Callable[[], ServerSnapshot],
        *,
        target: str,
        interval: float = DEFAULT_INTERVAL,
        history: int = DEFAULT_HISTORY,
        rate_window: float = DEFAULT_RATE_WINDOW,
        analyzers: Sequence[AnalyzerSpec] = BUILTIN_ANALYZERS,
    ) -> None:
        self._fetch = fetch
        self._target = target
        self._interval = interval
        self._rate_window = rate_window
        self._analyzers = analyzers
        self._history: deque[dict[str, float | None]] = deque(maxlen=history)
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._recent: deque[tuple[float, MetricsSnapshot]] = deque(
            maxlen=max(4, ceil(rate_window / interval) + 2)
        )
        self._metrics: MetricsSnapshot | None = None
        self._props: PropsSnapshot | None = None
        self._report: Report | None = None
        self._errors: dict[str, Any] = {}
        self._updated_at: float | None = None

    def tick(self) -> None:
        try:
            snapshot = self._fetch()
            with self._lock:
                baseline, window = self._advance_window(snapshot)
            context = DiagnosisContext(snapshot, window=window)
            report = diagnose(context, self._analyzers)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._errors = {"sampler": type(exc).__name__}
            return

        with self._lock:
            self._props = snapshot.server_props
            self._metrics = snapshot.metrics
            self._report = report
            self._errors = errors_to_dict(snapshot.errors)
            self._updated_at = snapshot.taken_at
            if snapshot.metrics is not None:
                n_ctx = snapshot.server_props.n_ctx if snapshot.server_props else None
                self._history.append(
                    delta_point(baseline, snapshot.metrics, snapshot.taken_at, n_ctx)
                )

    def _advance_window(
        self, snapshot: ServerSnapshot
    ) -> tuple[MetricsSnapshot | None, tuple[MetricsSnapshot, ...]]:
        """Record ``snapshot``; return the rate baseline and the samples inside the window."""
        if snapshot.metrics is None:
            return None, ()
        if self._recent and _counters_reset(self._recent[-1][1], snapshot.metrics):
            self._recent.clear()
        self._recent.append((snapshot.taken_at, snapshot.metrics))
        cutoff = snapshot.taken_at - self._rate_window
        window = tuple(metrics for sampled_at, metrics in self._recent if sampled_at >= cutoff)
        return self._baseline(snapshot.taken_at), window

    def _baseline(self, taken_at: float) -> MetricsSnapshot | None:
        if len(self._recent) < 2:
            return None
        cutoff = taken_at - self._rate_window
        return next(
            (metrics for sampled_at, metrics in reversed(self._recent) if sampled_at <= cutoff),
            self._recent[0][1],
        )

    def _notes(self, lifetime: dict[str, float | None]) -> dict[str, bool]:
        """Chart notes: gauges the server never exports, and idle windows after past traffic.

        Idle needs at least two points, since the first sample has no window baseline.
        """
        metrics = self._metrics

        def idle(key: str) -> bool:
            return (
                lifetime[key] is not None
                and len(self._history) >= 2
                and all(point[key] is None for point in self._history)
            )

        return {
            "kv_unexported": metrics is not None and metrics.kv_cache_usage_ratio is None,
            "cache_hit_unexported": metrics is not None
            and metrics.prompt_tokens_cached_total is None,
            "decode_idle": idle("decode_tps"),
            "prefill_idle": idle("prefill_tps"),
        }

    def state(self) -> dict[str, Any]:
        with self._lock:
            lifetime = rates_to_dict(self._metrics)
            return {
                "version": __version__,
                "target": self._target,
                "interval_s": self._interval,
                "rate_window_s": self._rate_window,
                "updated_at": self._updated_at,
                "server": props_to_dict(self._props),
                "lifetime": lifetime,
                "history": list(self._history),
                "context": context_to_dict(self._metrics, self._props),
                "report": report_to_dict(self._report) if self._report is not None else None,
                "errors": dict(self._errors),
                "notes": self._notes(lifetime),
            }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.tick()
            self._stop_event.wait(self._interval)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = Thread(
                target=self._run,
                name="llama-doctor-sampler",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            thread = self._thread
            self._stop_event.set()
        if thread is not None and thread is not current_thread():
            thread.join(timeout)
