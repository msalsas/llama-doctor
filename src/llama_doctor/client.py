import http.client
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from .models import (
    CompletionTimings,
    EndpointError,
    MetricsSnapshot,
    ModelsMeta,
    PropsSnapshot,
    ServerSnapshot,
    SlotsSnapshot,
)
from .promparse import parse_prometheus
from .redact import Secret, safe_float, safe_int
from .slots import parse_slots

DEFAULT_TIMEOUT: float = 5.0
MAX_BODY_BYTES: int = 4 * 1024 * 1024
_ORIGIN_SUFFIXES: tuple[str, ...] = ("/chat/completions", "/v1")
_ENDPOINT_SEGMENTS: frozenset[str] = frozenset(
    {"chat", "completion", "completions", "embeddings", "health", "metrics", "models", "props", "slots"}
)


class FetchError(Exception):
    def __init__(self, endpoint: str, reason: str, status: int | None = None) -> None:
        self.endpoint = endpoint
        self.reason = reason
        self.status = status
        super().__init__(f"{endpoint}: {reason}")

    def to_endpoint_error(self) -> EndpointError:
        return EndpointError(endpoint=self.endpoint, status=self.status, reason=self.reason)


class ProbeRouteError(FetchError):
    def __init__(self, attempts: tuple[EndpointError, ...]) -> None:
        self.attempts = attempts
        super().__init__(attempts[-1].endpoint, "no usable completion route", status=None)


COMPLETION_ROUTES: tuple[str, ...] = ("/v1/completions", "/v1/chat/completions", "/completion")
_FALLTHROUGH_STATUSES: frozenset[int] = frozenset({400, 404, 405, 422, 501})
_FALLTHROUGH_REASONS: frozenset[str] = frozenset(
    {"invalid JSON", "no timings in response", "unusable timings"}
)


def _completion_body(
    route: str, prompt: str, n_predict: int, model_id: str | None
) -> dict[str, Any]:
    if route == "/v1/completions":
        body: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": n_predict,
            "temperature": 0,
            "stream": False,
        }
        if model_id is not None:
            body["model"] = model_id
        return body
    if route == "/v1/chat/completions":
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_predict,
            "temperature": 0,
            "stream": False,
        }
        if model_id is not None:
            body["model"] = model_id
        return body
    return {
        "prompt": prompt,
        "n_predict": n_predict,
        "cache_prompt": False,
        "temperature": 0,
        "stream": False,
    }


def _parse_timings(route: str, decoded: Any) -> CompletionTimings:
    timings = decoded.get("timings") if isinstance(decoded, dict) else None
    if not isinstance(timings, dict):
        raise FetchError(route, "no timings in response")
    prompt_n = safe_int(timings.get("prompt_n"))
    prompt_ms = safe_float(timings.get("prompt_ms"))
    predicted_n = safe_int(timings.get("predicted_n"))
    predicted_ms = safe_float(timings.get("predicted_ms"))
    if None in (prompt_n, prompt_ms, predicted_n, predicted_ms):
        raise FetchError(route, "unusable timings")
    return CompletionTimings(
        prompt_n=prompt_n,
        prompt_ms=prompt_ms,
        predicted_n=predicted_n,
        predicted_ms=predicted_ms,
    )


def _falls_through(exc: FetchError) -> bool:
    return exc.status in _FALLTHROUGH_STATUSES or exc.reason in _FALLTHROUGH_REASONS


def normalize_base_url(url: str) -> str:
    if "?" in url or "#" in url:
        raise ValueError("base URL must not contain '?' or '#'")
    cleaned = url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(cleaned)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("base URL scheme must be http or https")
    if not parsed.hostname:
        raise ValueError("base URL must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base URL must not include userinfo")
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("base URL port must be digits only") from None
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("base URL port must be between 1 and 65535")
    path = parsed.path
    for suffix in _ORIGIN_SUFFIXES:
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
    last = path.rpartition("/")[2].lower()
    if last in _ENDPOINT_SEGMENTS or re.fullmatch(r"v[0-9]+", last):
        raise ValueError(
            "base URL path must be a server origin or proxy prefix, not an API path such as /v1/models"
        )
    return cleaned[: len(cleaned) - len(parsed.path)] + path


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LlamaClient:
    def __init__(
        self,
        base_url: str,
        api_key: Secret | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        clock: Callable[[], float] = time.time,
        model: str | None = None,
    ) -> None:
        self._base_url = normalize_base_url(base_url)
        if model is not None and model.strip() == "":
            raise ValueError("model must not be empty")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._clock = clock
        self._opener = urllib.request.build_opener(_NoRedirect())
        self._completion_route: str | None = None

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str | None:
        return self._model

    @property
    def completion_route(self) -> str | None:
        return self._completion_route

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = "Bearer " + self._api_key.reveal()
        return headers

    def _fetch_bytes(
        self, path: str, data: bytes | None = None, query: str = "",
        timeout: float | None = None,
    ) -> bytes:
        headers = self._headers()
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self._base_url + path + query, data=data, headers=headers
        )
        effective = self._timeout if timeout is None else timeout
        try:
            with self._opener.open(request, timeout=effective) as response:
                body = response.read(MAX_BODY_BYTES + 1)
        except urllib.error.HTTPError as exc:
            exc.close()
            raise FetchError(path, f"HTTP {exc.code}", status=exc.code) from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise FetchError(path, "timed out") from None
            if isinstance(exc.reason, ConnectionRefusedError):
                raise FetchError(path, "connection refused") from None
            raise FetchError(path, "unreachable") from None
        except TimeoutError:
            raise FetchError(path, "timed out") from None
        except http.client.HTTPException:
            raise FetchError(path, "invalid response") from None
        except (OverflowError, RecursionError):
            raise FetchError(path, "invalid response") from None
        except OSError:
            raise FetchError(path, "unreachable") from None
        if len(body) > MAX_BODY_BYTES:
            raise FetchError(path, "response too large")
        return body

    def _fetch_json(
        self, path: str, data: bytes | None = None, timeout: float | None = None
    ) -> Any:
        try:
            return json.loads(self._fetch_bytes(path, data, timeout=timeout))
        except ValueError:
            raise FetchError(path, "invalid JSON") from None
        except RecursionError:
            raise FetchError(path, "invalid response") from None

    def get_json(self, path: str) -> Any:
        return self._fetch_json(path)

    def post_json(self, path: str, payload: Mapping[str, Any]) -> Any:
        return self._fetch_json(path, data=json.dumps(payload).encode("utf-8"))

    def fetch_metrics(self) -> MetricsSnapshot:
        query = (
            "?model=" + urllib.parse.quote(self._model, safe="")
            if self._model is not None
            else ""
        )
        body = self._fetch_bytes("/metrics", query=query)
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            raise FetchError("/metrics", "invalid response") from None
        return MetricsSnapshot.from_samples(parse_prometheus(text), taken_at=self._clock())

    def fetch_props(self) -> PropsSnapshot:
        return PropsSnapshot.from_payload(self._fetch_json("/props"))

    def fetch_models(self) -> ModelsMeta:
        return ModelsMeta.from_payload(self._fetch_json("/v1/models"))

    def fetch_slots(self) -> SlotsSnapshot:
        slots = parse_slots(self._fetch_json("/slots"))
        if slots is None:
            raise FetchError("/slots", "unusable response")
        return slots

    def snapshot(self, include_slots: bool = False) -> ServerSnapshot:
        taken_at = self._clock()
        metrics: MetricsSnapshot | None = None
        props: PropsSnapshot | None = None
        models: ModelsMeta | None = None
        errors: dict[str, EndpointError] = {}
        try:
            metrics = self.fetch_metrics()
        except FetchError as exc:
            errors["/metrics"] = exc.to_endpoint_error()
        try:
            props = self.fetch_props()
        except FetchError as exc:
            errors["/props"] = exc.to_endpoint_error()
        try:
            models = self.fetch_models()
        except FetchError as exc:
            errors["/v1/models"] = exc.to_endpoint_error()
        slots: SlotsSnapshot | None = None
        slots_error: str | None = None
        if include_slots:
            try:
                slots = self.fetch_slots()
            except FetchError as exc:
                slots_error = exc.reason
        return ServerSnapshot(
            taken_at=taken_at,
            metrics=metrics,
            props=props,
            models=models,
            errors=errors,
            slots=slots,
            slots_error=slots_error,
        )

    def complete(
        self, prompt: str, n_predict: int, *, model_id: str | None = None,
        timeout: float | None = None,
    ) -> CompletionTimings:
        if timeout is not None and not (math.isfinite(timeout) and timeout > 0):
            raise ValueError("timeout must be a finite number > 0")
        if self._completion_route is not None:
            return self._complete_on(self._completion_route, prompt, n_predict, model_id, timeout)
        attempts: list[EndpointError] = []
        for route in COMPLETION_ROUTES:
            try:
                timings = self._complete_on(route, prompt, n_predict, model_id, timeout)
            except FetchError as exc:
                attempts.append(exc.to_endpoint_error())
                if not _falls_through(exc):
                    raise ProbeRouteError(tuple(attempts)) from None
                continue
            self._completion_route = route
            return timings
        raise ProbeRouteError(tuple(attempts))

    def _complete_on(
        self, route: str, prompt: str, n_predict: int, model_id: str | None,
        timeout: float | None = None,
    ) -> CompletionTimings:
        payload = _completion_body(route, prompt, n_predict, model_id)
        decoded = self._fetch_json(route, data=json.dumps(payload).encode("utf-8"), timeout=timeout)
        return _parse_timings(route, decoded)
