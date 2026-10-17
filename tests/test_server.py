import http.client
import io
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from llama_doctor.cli import main
from llama_doctor.dashboard.server import CSP, is_loopback, make_server, run


class TestServer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        (root / "vendor").mkdir()
        (root / "index.html").write_text("<h1>dashboard</h1>", encoding="utf-8")
        (root / "app.css").write_text("body {}", encoding="utf-8")
        (root / "app.js").write_text("console.log('app')", encoding="utf-8")
        (root / "vendor" / "chart.umd.js").write_text("chart", encoding="utf-8")
        self.state = {"answer": 42}
        self.server = make_server(lambda: self.state, port=0, static_dir=root)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp_dir.cleanup()

    def _request(self, method: str, path: str, headers: dict[str, str] | None = None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        body = response.read()
        headers_out = dict(response.getheaders())
        connection.close()
        return response.status, headers_out, body

    def test_api_state_json(self):
        status, headers, body = self._request("GET", "/api/state")

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), self.state)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Cache-Control"], "no-store")

    def test_index_csp_headers(self):
        status, headers, _ = self._request("GET", "/")

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Security-Policy"], CSP)
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")

    def test_static_allowlist(self):
        status, headers, _ = self._request("GET", "/static/app.js")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/javascript; charset=utf-8")
        for path in ("/static/../server.py", "/static/other.js", "/etc/passwd"):
            self.assertEqual(self._request("GET", path)[0], 404)

    def test_missing_static_file_404(self):
        Path(self.temp_dir.name, "app.css").unlink()

        self.assertEqual(self._request("GET", "/static/app.css")[0], 404)

    def test_query_string_ignored(self):
        status, _, body = self._request("GET", "/?unused=value")

        self.assertEqual(status, 200)
        self.assertEqual(body, b"<h1>dashboard</h1>")

    def test_foreign_host_rejected(self):
        self.assertEqual(
            self._request("GET", "/", {"Host": "evil.example"})[0],
            403,
        )

    def test_localhost_host_allowed(self):
        self.assertEqual(
            self._request("GET", "/", {"Host": f"localhost:{self.port}"})[0],
            200,
        )

    def test_post_405(self):
        status, headers, _ = self._request("POST", "/")

        self.assertEqual(status, 405)
        self.assertEqual(headers["Allow"], "GET, HEAD")

    def test_unknown_method_405(self):
        status, headers, _ = self._request("TRACE", "/")

        self.assertEqual(status, 405)
        self.assertEqual(headers["Allow"], "GET, HEAD")

    def test_state_error_500_without_text(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.server = make_server(
            Mock(side_effect=RuntimeError("SENTINEL")),
            port=0,
            static_dir=Path(self.temp_dir.name),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

        status, _, body = self._request("GET", "/api/state")
        self.assertEqual(status, 500)
        self.assertEqual(body, b"internal error")
        self.assertNotIn(b"SENTINEL", body)

    def test_is_loopback(self):
        for host in ("127.0.0.1", "::1", "localhost"):
            self.assertTrue(is_loopback(host))
        for host in ("0.0.0.0", "192.168.1.5"):
            self.assertFalse(is_loopback(host))

    def test_run_port_in_use_returns_3(self):
        occupied = socket.socket()
        occupied.bind(("127.0.0.1", 0))
        port = occupied.getsockname()[1]
        client = Mock()
        client.base_url = "http://server"
        out = io.StringIO()
        try:
            self.assertEqual(
                run(client, host="127.0.0.1", port=port, interval=2.0, history=300, out=out),
                3,
            )
        finally:
            occupied.close()
        self.assertIn("cannot bind", out.getvalue())

    def test_cli_serve_wired(self):
        with patch("llama_doctor.dashboard.server.run", return_value=0) as dashboard_run:
            self.assertEqual(main(["serve", "http://server", "--port", "9999"]), 0)

        _, kwargs = dashboard_run.call_args
        self.assertEqual(
            kwargs,
            {
                "host": "127.0.0.1",
                "port": 9999,
                "interval": 2.0,
                "history": 300,
                "out": kwargs["out"],
            },
        )


if __name__ == "__main__":
    unittest.main()
