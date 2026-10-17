import json
import os
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llama_doctor.analysis.base import DiagnosisContext, HostHints
from llama_doctor.models import (
    EndpointError,
    MetricsSnapshot,
    ModelsMeta,
    ProbeResult,
    PropsSnapshot,
    ServerSnapshot,
)
from llama_doctor.promparse import parse_prometheus


def load_fixture(name: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "fixtures", name)
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def make_context(
    metrics: str | None = "metrics_moe_offload.txt",
    props: str | None = "props_full.json",
    models: str | None = "models_full.json",
    *,
    errors: Mapping[str, EndpointError] | None = None,
    probe: ProbeResult | None = None,
    hints: HostHints = HostHints(),  # noqa: B008
) -> DiagnosisContext:
    metric_snapshot = (
        MetricsSnapshot.from_samples(parse_prometheus(load_fixture(metrics)), taken_at=0.0)
        if metrics is not None
        else None
    )
    props_snapshot = PropsSnapshot.from_payload(json.loads(load_fixture(props))) if props else None
    models_snapshot = ModelsMeta.from_payload(json.loads(load_fixture(models))) if models else None
    return DiagnosisContext(
        ServerSnapshot(0.0, metric_snapshot, props_snapshot, models_snapshot, errors or {}),
        probe,
        hints,
    )


class FakeLlamaServer:
    def __init__(self) -> None:
        self._routes: dict[str, tuple[int, bytes, str, dict[str, str]]] = {}
        self.requests: list[tuple[str, str, dict[str, str], bytes]] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler_class())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def set_route(
        self,
        path: str,
        status: int,
        body: str | bytes,
        content_type: str = "text/plain",
        headers: dict[str, str] | None = None,
    ) -> None:
        if isinstance(body, str):
            body = body.encode("utf-8")
        self._routes[path] = (status, body, content_type, dict(headers or {}))

    def __enter__(self) -> "FakeLlamaServer":  # noqa: PYI034
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def _dispatch(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                outer.requests.append((self.command, self.path, dict(self.headers), body))
                route = outer._routes.get(self.path)
                if route is None:
                    status, payload, content_type, extra = 404, b"", "text/plain", {}
                else:
                    status, payload, content_type, extra = route
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                for name, value in extra.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(payload)

            do_GET = _dispatch
            do_POST = _dispatch
            do_PUT = _dispatch
            do_DELETE = _dispatch
            do_PATCH = _dispatch

        return Handler
