import json
import socket
import threading
import unittest

from llama_doctor.client import FetchError, LlamaClient, normalize_base_url
from llama_doctor.redact import Secret

from .support import FakeLlamaServer, load_fixture


class _RawResponseServer:
    def __init__(self, response: bytes) -> None:
        self._response = response
        self._stop = threading.Event()
        self._socket = socket.socket()
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen(1)
        self._port = self._socket.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def close(self) -> None:
        self._stop.set()
        self._socket.close()

    def _serve(self) -> None:
        self._socket.settimeout(0.1)
        while not self._stop.is_set():
            try:
                connection, _address = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with connection:
                try:
                    connection.recv(65536)
                    connection.sendall(self._response)
                except OSError:
                    pass


class TestNormalizeBaseUrl(unittest.TestCase):
    def test_normalize_base_url(self):
        self.assertEqual(normalize_base_url(" http://h:8080/ "), "http://h:8080")
        self.assertEqual(normalize_base_url("http://h/proxy/"), "http://h/proxy")
        bad = (
            "ftp://h",
            "http://u:p@h",
            "http://h?x=1",
            "h:8080",
            "http://h?",
            "http://h#frag",
            "http://h:abc",
            "http://h:99999",
            "http://h:0",
        )
        for url in bad:
            with self.subTest(url=url), self.assertRaises(ValueError):
                normalize_base_url(url)


class TestLlamaClient(unittest.TestCase):
    def test_fetch_metrics(self):
        with FakeLlamaServer() as server:
            server.set_route("/metrics", 200, load_fixture("metrics_moe_offload.txt"))
            client = LlamaClient(server.url, clock=lambda: 1234.5)
            snap = client.fetch_metrics()
            self.assertAlmostEqual(snap.prefill_tps, 40.0, places=6)
            self.assertEqual(snap.taken_at, 1234.5)
            self.assertEqual(server.requests[0][1], "/metrics")

    def test_fetch_props_whitelisted(self):
        with FakeLlamaServer() as server:
            server.set_route(
                "/props", 200, load_fixture("props_full.json"), content_type="application/json"
            )
            client = LlamaClient(server.url)
            snap = client.fetch_props()
            self.assertEqual(snap.model_name, "Qwen3-30B-A3B-UD-Q4_K_XL")
            self.assertEqual(snap.quant, "UD-Q4_K_XL")
            self.assertEqual(snap.n_ctx, 32768)
            self.assertNotIn("SENTINEL", repr(snap))

    def test_bearer_header_sent(self):
        with FakeLlamaServer() as server:
            server.set_route("/metrics", 200, load_fixture("metrics_minimal.txt"))
            client = LlamaClient(server.url, api_key=Secret("k3y"))
            client.fetch_metrics()
            headers = server.requests[0][2]
            self.assertEqual(headers.get("Authorization"), "Bearer k3y")

    def test_no_auth_header_without_key(self):
        with FakeLlamaServer() as server:
            server.set_route("/metrics", 200, load_fixture("metrics_minimal.txt"))
            client = LlamaClient(server.url)
            client.fetch_metrics()
            headers = server.requests[0][2]
            self.assertNotIn("Authorization", headers)

    def test_http_error_hides_body(self):
        with FakeLlamaServer() as server:
            server.set_route("/metrics", 500, "SENTINEL boom")
            client = LlamaClient(server.url)
            with self.assertRaises(FetchError) as ctx:
                client.fetch_metrics()
            err = ctx.exception
            self.assertEqual(err.status, 500)
            self.assertEqual(err.reason, "HTTP 500")
            self.assertNotIn("SENTINEL", str(err))
            self.assertNotIn("SENTINEL", repr(err))

    def test_invalid_json_reason(self):
        with FakeLlamaServer() as server:
            server.set_route("/props", 200, "not json {", content_type="application/json")
            client = LlamaClient(server.url)
            with self.assertRaises(FetchError) as ctx:
                client.fetch_props()
            self.assertEqual(ctx.exception.reason, "invalid JSON")
            self.assertIsNone(ctx.exception.status)

    def test_connection_refused(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        client = LlamaClient(f"http://127.0.0.1:{port}")
        with self.assertRaises(FetchError) as ctx:
            client.fetch_metrics()
        self.assertEqual(ctx.exception.reason, "connection refused")
        self.assertIsNone(ctx.exception.status)

    def test_malformed_status_line_is_invalid_response(self):
        server = _RawResponseServer(b"NOT-HTTP GARBAGE\r\n\r\n")
        try:
            client = LlamaClient(server.url)
            with self.assertRaises(FetchError) as ctx:
                client.fetch_metrics()
            self.assertEqual(ctx.exception.reason, "invalid response")
            self.assertIsNone(ctx.exception.status)
            snap = client.snapshot()
            self.assertEqual(snap.errors["/metrics"].reason, "invalid response")
        finally:
            server.close()

    def test_redirect_refused(self):
        with FakeLlamaServer() as server:
            server.set_route("/metrics", 302, "", headers={"Location": "/elsewhere"})
            client = LlamaClient(server.url)
            with self.assertRaises(FetchError) as ctx:
                client.fetch_metrics()
            self.assertEqual(ctx.exception.status, 302)
            self.assertEqual(ctx.exception.reason, "HTTP 302")
            self.assertEqual([req[1] for req in server.requests], ["/metrics"])

    def test_api_key_absent_from_errors(self):
        with FakeLlamaServer() as server:
            server.set_route("/metrics", 401, "SENTINELKEY is invalid")
            client = LlamaClient(server.url, api_key=Secret("SENTINELKEY"))
            with self.assertRaises(FetchError) as ctx:
                client.fetch_metrics()
            err = ctx.exception
            self.assertEqual(err.status, 401)
            self.assertNotIn("SENTINELKEY", str(err))
            self.assertNotIn("SENTINELKEY", repr(err))

    def test_snapshot_collects_errors(self):
        with FakeLlamaServer() as server:
            server.set_route("/metrics", 200, load_fixture("metrics_minimal.txt"))
            client = LlamaClient(server.url, clock=lambda: 1234.5)
            snap = client.snapshot()
            self.assertIsNotNone(snap.metrics)
            self.assertIsNone(snap.props)
            self.assertIsNone(snap.models)
            self.assertEqual(set(snap.errors), {"/props", "/v1/models"})
            for err in snap.errors.values():
                self.assertEqual(err.status, 404)
            self.assertTrue(snap.alive)
            self.assertEqual(snap.taken_at, 1234.5)

    def test_snapshot_alive_without_metrics(self):
        with FakeLlamaServer() as server:
            server.set_route("/metrics", 501, "boom")
            server.set_route(
                "/v1/models",
                200,
                load_fixture("models_full.json"),
                content_type="application/json",
            )
            client = LlamaClient(server.url)
            snap = client.snapshot()
            self.assertIsNone(snap.metrics)
            self.assertIsNotNone(snap.models)
            self.assertTrue(snap.alive)
            self.assertEqual(snap.errors["/metrics"].status, 501)

    def test_never_requests_health(self):
        with FakeLlamaServer() as server:
            server.set_route("/metrics", 200, load_fixture("metrics_minimal.txt"))
            server.set_route(
                "/props",
                200,
                load_fixture("props_unsloth_minimal.json"),
                content_type="application/json",
            )
            server.set_route(
                "/v1/models",
                200,
                load_fixture("models_full.json"),
                content_type="application/json",
            )
            server.set_route("/health", 200, '{"status": "ok"}', content_type="application/json")
            client = LlamaClient(server.url)
            client.snapshot()
            paths = [req[1] for req in server.requests]
            self.assertNotIn("/health", paths)
            self.assertEqual(paths.count("/metrics"), 1)

    def test_complete_returns_timings_only(self):
        body = json.dumps(
            {
                "content": "SENTINEL",
                "timings": {
                    "prompt_n": 1200,
                    "prompt_ms": 3000.0,
                    "predicted_n": 64,
                    "predicted_ms": 8000.0,
                },
            }
        )
        with FakeLlamaServer() as server:
            server.set_route("/completion", 200, body, content_type="application/json")
            client = LlamaClient(server.url)
            timings = client.complete("hello", 64)
            self.assertEqual(timings.prompt_n, 1200)
            self.assertEqual(timings.prompt_ms, 3000.0)
            self.assertEqual(timings.predicted_n, 64)
            self.assertEqual(timings.predicted_ms, 8000.0)
            self.assertNotIn("SENTINEL", repr(timings))
            method, path, _headers, raw = server.requests[0]
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/completion")
            request_json = json.loads(raw)
            self.assertIs(request_json["cache_prompt"], False)
            self.assertEqual(request_json["n_predict"], 64)
            self.assertEqual(request_json["temperature"], 0)
            self.assertFalse(request_json["stream"])
            self.assertEqual(request_json["prompt"], "hello")

    def test_complete_without_timings(self):
        with FakeLlamaServer() as server:
            server.set_route(
                "/completion",
                200,
                json.dumps({"content": "hi"}),
                content_type="application/json",
            )
            client = LlamaClient(server.url)
            with self.assertRaises(FetchError) as ctx:
                client.complete("hi", 8)
            self.assertEqual(ctx.exception.reason, "no timings in response")
            self.assertEqual(ctx.exception.endpoint, "/completion")

    def test_complete_rejects_overflowing_timing(self):
        body = json.dumps(
            {
                "timings": {
                    "prompt_n": 1200,
                    "prompt_ms": 10**400,
                    "predicted_n": 64,
                    "predicted_ms": 8000.0,
                },
            }
        )
        with FakeLlamaServer() as server:
            server.set_route("/completion", 200, body, content_type="application/json")
            timings = LlamaClient(server.url).complete("hello", 64)
        self.assertIsNone(timings.prompt_ms)
        self.assertEqual(timings.predicted_ms, 8000.0)


class TestRoutedModel(unittest.TestCase):
    def test_model_defaults_to_none(self):
        self.assertIsNone(LlamaClient("http://h").model)
        self.assertEqual(LlamaClient("http://h", model="qwen3").model, "qwen3")

    def test_empty_model_rejected(self):
        for value in ("", " ", "\t"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as ctx:
                    LlamaClient("http://h", model=value)
                self.assertEqual(str(ctx.exception), "model must not be empty")

    def test_url_with_model_query_still_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_base_url("http://h:8080?model=qwen3")
        self.assertEqual(str(ctx.exception), "base URL must not contain '?' or '#'")

    def test_fetch_metrics_appends_model(self):
        with FakeLlamaServer() as server:
            server.set_route("/metrics?model=qwen3", 200, load_fixture("metrics_moe_offload.txt"))
            client = LlamaClient(server.url, clock=lambda: 1234.5, model="qwen3")
            snap = client.fetch_metrics()
            self.assertAlmostEqual(snap.prefill_tps, 40.0, places=6)
            self.assertEqual(snap.taken_at, 1234.5)
            self.assertEqual([req[1] for req in server.requests], ["/metrics?model=qwen3"])

    def test_model_is_percent_encoded(self):
        target = "/metrics?model=org%2FQwen3%2030B%3AQ4_K%26x%3D1"
        with FakeLlamaServer() as server:
            server.set_route(target, 200, load_fixture("metrics_minimal.txt"))
            client = LlamaClient(server.url, model="org/Qwen3 30B:Q4_K&x=1")
            client.fetch_metrics()
            self.assertEqual(server.requests[0][1], target)

    def test_model_with_surrounding_spaces_is_sent_verbatim(self):
        for model, target in (
            (" qwen3", "/metrics?model=%20qwen3"),
            ("qwen3 ", "/metrics?model=qwen3%20"),
        ):
            with self.subTest(model=model), FakeLlamaServer() as server:
                server.set_route(target, 200, load_fixture("metrics_minimal.txt"))
                client = LlamaClient(server.url, model=model)
                client.fetch_metrics()
                self.assertEqual(server.requests[0][1], target)

    def test_model_non_ascii_is_utf8_percent_encoded(self):
        target = "/metrics?model=qwen3-%C3%A9"
        with FakeLlamaServer() as server:
            server.set_route(target, 200, load_fixture("metrics_minimal.txt"))
            client = LlamaClient(server.url, model="qwen3-é")
            client.fetch_metrics()
            self.assertEqual(server.requests[0][1], target)

    def test_model_only_on_metrics(self):
        with FakeLlamaServer() as server:
            server.set_route("/metrics?model=m", 200, load_fixture("metrics_minimal.txt"))
            server.set_route(
                "/props", 200, load_fixture("props_full.json"), content_type="application/json"
            )
            server.set_route(
                "/v1/models", 200, load_fixture("models_full.json"), content_type="application/json"
            )
            server.set_route("/slots", 200, "[]", content_type="application/json")
            client = LlamaClient(server.url, model="m")
            snap = client.snapshot(include_slots=True)
            self.assertEqual(
                [req[1] for req in server.requests],
                ["/metrics?model=m", "/props", "/v1/models", "/slots"],
            )
            self.assertEqual(snap.errors, {})

    def test_metrics_error_endpoint_stays_bare(self):
        with FakeLlamaServer() as server:
            server.set_route(
                "/props", 200, load_fixture("props_full.json"), content_type="application/json"
            )
            server.set_route(
                "/v1/models", 200, load_fixture("models_full.json"), content_type="application/json"
            )
            client = LlamaClient(server.url, model="m")
            with self.assertRaises(FetchError) as ctx:
                client.fetch_metrics()
            err = ctx.exception
            self.assertEqual(err.endpoint, "/metrics")
            self.assertEqual(err.status, 404)
            self.assertEqual(err.reason, "HTTP 404")
            self.assertNotIn("model", str(err))
            snap = client.snapshot()
            self.assertEqual(set(snap.errors), {"/metrics"})
            self.assertEqual(snap.errors["/metrics"].endpoint, "/metrics")

    def test_model_with_api_key(self):
        with FakeLlamaServer() as server:
            server.set_route("/metrics?model=m", 200, load_fixture("metrics_minimal.txt"))
            client = LlamaClient(server.url, api_key=Secret("k3y"), model="m")
            client.fetch_metrics()
            self.assertEqual(server.requests[0][2]["Authorization"], "Bearer k3y")

    def test_complete_ignores_model(self):
        body = json.dumps(
            {
                "timings": {
                    "prompt_n": 1200,
                    "prompt_ms": 3000.0,
                    "predicted_n": 64,
                    "predicted_ms": 8000.0,
                },
            }
        )
        with FakeLlamaServer() as server:
            server.set_route("/completion", 200, body, content_type="application/json")
            client = LlamaClient(server.url, model="m")
            timings = client.complete("hello", 64)
            self.assertEqual(timings.prompt_n, 1200)
            self.assertEqual(server.requests[0][1], "/completion")
            self.assertNotIn("model", json.loads(server.requests[0][3]))


if __name__ == "__main__":
    unittest.main()
