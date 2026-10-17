from collections import deque
from collections.abc import Callable, Sequence
from threading import Event, Lock, Thread, current_thread
from typing import Any

from llama_doctor import __version__
from llama_doctor.analysis import BUILTIN_ANALYZERS, diagnose
from llama_doctor.analysis.base import AnalyzerSpec, DiagnosisContext, Report
from llama_doctor.models import MetricsSnapshot, PropsSnapshot, ServerSnapshot
from llama_doctor.serialize import errors_to_dict, props_to_dict, rates_to_dict, report_to_dict

DEFAULT_INTERVAL = 2.0
DEFAULT_HISTORY = 300
POINT_KEYS = (
    "t",
    "decode_tps",
    "prefill_tps",
    "cache_hit_ratio",
    "avg_draft_len",
    "draft_acceptance",
    "kv_usage_ratio",
    "requests_processing",
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
    numerator_delta = current_numerator - previous_numerator
    denominator_delta = current_denominator - previous_denominator
    if numerator_delta < 0 or denominator_delta <= 0:
        return None
    return numerator_delta / denominator_delta


def delta_point(
    prev: MetricsSnapshot | None, cur: MetricsSnapshot, t: float
) -> dict[str, float | None]:
    point: dict[str, float | None] = {
        "t": t,
        "decode_tps": None,
        "prefill_tps": None,
        "cache_hit_ratio": None,
        "avg_draft_len": None,
        "draft_acceptance": None,
        "kv_usage_ratio": cur.kv_cache_usage_ratio,
        "requests_processing": cur.requests_processing,
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


class Sampler:
    def __init__(
        self,
        fetch: Callable[[], ServerSnapshot],
        *,
        target: str,
        interval: float = DEFAULT_INTERVAL,
        history: int = DEFAULT_HISTORY,
        analyzers: Sequence[AnalyzerSpec] = BUILTIN_ANALYZERS,
    ) -> None:
        self._fetch = fetch
        self._target = target
        self._interval = interval
        self._analyzers = analyzers
        self._history: deque[dict[str, float | None]] = deque(maxlen=history)
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._prev_metrics: MetricsSnapshot | None = None
        self._metrics: MetricsSnapshot | None = None
        self._props: PropsSnapshot | None = None
        self._report: Report | None = None
        self._errors: dict[str, Any] = {}
        self._updated_at: float | None = None

    def tick(self) -> None:
        try:
            snapshot = self._fetch()
            report = diagnose(DiagnosisContext(snapshot), self._analyzers)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._errors = {"sampler": type(exc).__name__}
            return

        with self._lock:
            self._props = snapshot.props
            self._metrics = snapshot.metrics
            self._report = report
            self._errors = errors_to_dict(snapshot.errors)
            self._updated_at = snapshot.taken_at
            if snapshot.metrics is not None:
                self._history.append(
                    delta_point(self._prev_metrics, snapshot.metrics, snapshot.taken_at)
                )
                self._prev_metrics = snapshot.metrics

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": __version__,
                "target": self._target,
                "interval_s": self._interval,
                "updated_at": self._updated_at,
                "server": props_to_dict(self._props),
                "lifetime": rates_to_dict(self._metrics),
                "history": list(self._history),
                "report": report_to_dict(self._report) if self._report is not None else None,
                "errors": dict(self._errors),
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
