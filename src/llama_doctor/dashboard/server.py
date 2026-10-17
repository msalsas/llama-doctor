import ipaddress
import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, TextIO

from llama_doctor.client import LlamaClient

from .sampler import Sampler

STATIC_ROUTES: dict[str, str] = {
    "/": "index.html",
    "/static/app.css": "app.css",
    "/static/app.js": "app.js",
    "/static/vendor/chart.umd.js": "vendor/chart.umd.js",
}
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def default_static_dir() -> Traversable:
    return files("llama_doctor.dashboard") / "static"


def _host_name(header: str) -> str:
    host = header.strip().lower()
    if host.startswith("["):
        closing = host.find("]")
        suffix = host[closing + 1 :] if closing != -1 else ""
        if closing == -1 or (
            suffix and (not suffix.startswith(":") or not suffix[1:].isdigit())
        ):
            return host
        return host[1:closing]
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    name, separator, port = host.rpartition(":")
    if separator and port.isdigit():
        return name
    return host


def make_server(
    state: Callable[[], dict[str, Any]],
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    static_dir: Traversable | Path | None = None,
) -> ThreadingHTTPServer:
    root = default_static_dir() if static_dir is None else static_dir
    require_loopback_host = is_loopback(host)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def _send(
            self,
            status: int,
            body: bytes = b"",
            content_type: str = "text/plain; charset=utf-8",
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            if headers:
                for name, value in headers.items():
                    self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _host_allowed(self) -> bool:
            if not require_loopback_host:
                return True
            header = self.headers.get("Host")
            if header is None:
                self._send(400, b"bad request")
                return False
            if _host_name(header) not in LOOPBACK_NAMES:
                self._send(403, b"forbidden")
                return False
            return True

        def _get(self) -> None:
            if not self._host_allowed():
                return
            path = self.path.partition("?")[0]
            if path == "/api/state":
                try:
                    body = json.dumps(state(), allow_nan=False).encode("utf-8")
                except Exception:  # noqa: BLE001
                    self._send(500, b"internal error")
                    return
                self._send(
                    200,
                    body,
                    "application/json; charset=utf-8",
                    {"Cache-Control": "no-store"},
                )
                return
            filename = STATIC_ROUTES.get(path)
            if filename is None:
                self._send(404, b"not found")
                return
            try:
                body = (root / filename).read_bytes()
            except OSError:
                self._send(404, b"not found")
                return
            content_type = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
            }[Path(filename).suffix]
            headers = (
                {"Content-Security-Policy": CSP, "Referrer-Policy": "no-referrer"}
                if filename.endswith(".html")
                else None
            )
            self._send(200, body, content_type, headers)

        do_GET = _get
        do_HEAD = _get

        def _method_not_allowed(self) -> None:
            self._send(405, b"method not allowed", headers={"Allow": "GET, HEAD"})

        do_POST = _method_not_allowed
        do_PUT = _method_not_allowed
        do_PATCH = _method_not_allowed
        do_DELETE = _method_not_allowed
        do_OPTIONS = _method_not_allowed
        do_TRACE = _method_not_allowed
        do_CONNECT = _method_not_allowed

        def __getattr__(self, name: str) -> Any:
            if name.startswith("do_"):
                return self._method_not_allowed
            raise AttributeError(name)

    return ThreadingHTTPServer((host, port), Handler)


def run(
    client: LlamaClient,
    *,
    host: str,
    port: int,
    interval: float,
    history: int,
    out: TextIO,
) -> int:
    try:
        server = make_server(lambda: sampler.state(), host=host, port=port)
    except OSError:
        out.write(f"llama-doctor: cannot bind {host}:{port}: address unavailable\n")
        return 3
    sampler = Sampler(
        client.snapshot,
        target=client.base_url,
        interval=interval,
        history=history,
    )
    try:
        sampler.start()
        out.write(f"dashboard: http://{host}:{port}/  target: {client.base_url}\n")
        if not is_loopback(host):
            out.write(
                "warning: dashboard has no authentication and is reachable from the network\n"
            )
        out.flush()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    finally:
        sampler.stop()
        server.server_close()
    return 0
