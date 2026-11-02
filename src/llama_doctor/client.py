import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
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


class FetchError(Exception):
    def __init__(self, endpoint: str, reason: str, status: int | None = None) -> None:
        self.endpoint = endpoint
        self.reason = reason
        self.status = status
        super().__init__(f"{endpoint}: {reason}")

    def to_endpoint_error(self) -> EndpointError:
        return EndpointError(endpoint=self.endpoint, status=self.status, reason=self.reason)


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
    return cleaned


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

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str | None:
        return self._model

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = "Bearer " + self._api_key.reveal()
        return headers

    def _fetch_bytes(self, path: str, data: bytes | None = None, query: str = "") -> bytes:
        headers = self._headers()
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self._base_url + path + query, data=data, headers=headers
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
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

    def _fetch_json(self, path: str, data: bytes | None = None) -> Any:
        try:
            return json.loads(self._fetch_bytes(path, data))
        except ValueError:
            raise FetchError(path, "invalid JSON") from None
        except RecursionError:
            raise FetchError(path, "invalid response") from None

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

    def complete(self, prompt: str, n_predict: int) -> CompletionTimings:
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "cache_prompt": False,
            "temperature": 0,
            "stream": False,
        }
        decoded = self._fetch_json(
            "/completion", data=json.dumps(payload).encode("utf-8")
        )
        timings = decoded.get("timings") if isinstance(decoded, dict) else None
        if not isinstance(timings, dict):
            raise FetchError("/completion", "no timings in response")
        return CompletionTimings(
            prompt_n=safe_int(timings.get("prompt_n")),
            prompt_ms=safe_float(timings.get("prompt_ms")),
            predicted_n=safe_int(timings.get("predicted_n")),
            predicted_ms=safe_float(timings.get("predicted_ms")),
        )
