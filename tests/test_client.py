import json
import math
import socket
import threading
import unittest
from unittest.mock import patch

from llama_doctor.client import (
    FetchError,
    LlamaClient,
    ProbeRouteError,
    _falls_through,
    normalize_base_url,
)
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
            request = next(req for req in server.requests if req[1] == "/completion")
            method, path, _headers, raw = request
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
            with self.assertRaises(ProbeRouteError) as ctx:
                client.complete("hi", 8)
            err = ctx.exception
            self.assertIsInstance(err, FetchError)
            self.assertEqual(err.reason, "no usable completion route")
            last = err.attempts[-1]
            self.assertEqual(
                (last.endpoint, last.status, last.reason),
                ("/completion", None, "no timings in response"),
            )


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
            request = next(req for req in server.requests if req[1] == "/completion")
            self.assertEqual(request[1], "/completion")
            self.assertNotIn("model", json.loads(request[3]))


def _timings_body(**overrides: object) -> str:
    timings = {
        "prompt_n": 1200,
        "prompt_ms": 1500.0,
        "predicted_n": 64,
        "predicted_ms": 3200.0,
    }
    timings.update(overrides)
    return json.dumps({"timings": timings})


class TestCompleteTimeout(unittest.TestCase):
    def test_explicit_timeout_is_passed_to_every_attempt(self):
        with FakeLlamaServer() as server:
            server.set_route("/v1/completions", 200, _timings_body(), "application/json")
            client = LlamaClient(server.url)
            with patch.object(client._opener, "open", wraps=client._opener.open) as open_:
                client.complete("x", 1, timeout=42.0)
                client.complete("x", 1, timeout=7.0)

        self.assertEqual(
            [call.kwargs["timeout"] for call in open_.call_args_list], [42.0, 7.0]
        )

    def test_default_timeout_when_not_given(self):
        with FakeLlamaServer() as server:
            server.set_route("/v1/completions", 200, _timings_body(), "application/json")
            client = LlamaClient(server.url, timeout=5.0)
            with patch.object(client._opener, "open", wraps=client._opener.open) as open_:
                client.complete("x", 1)

        self.assertEqual(open_.call_args.kwargs["timeout"], 5.0)

    def test_timeout_does_not_carry_over_to_later_calls(self):
        with FakeLlamaServer() as server:
            server.set_route("/metrics", 200, load_fixture("metrics_minimal.txt"))
            server.set_route("/v1/completions", 200, _timings_body(), "application/json")
            client = LlamaClient(server.url, timeout=5.0)
            with patch.object(client._opener, "open", wraps=client._opener.open) as open_:
                client.complete("x", 1, timeout=42.0)
                client.fetch_metrics()

        self.assertEqual(
            [call.kwargs["timeout"] for call in open_.call_args_list], [42.0, 5.0]
        )

    def test_invalid_timeout_rejected_before_any_request(self):
        with FakeLlamaServer() as server:
            server.set_route("/v1/completions", 200, _timings_body(), "application/json")
            client = LlamaClient(server.url)
            for timeout in (0, -1, math.inf, math.nan):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    client.complete("x", 1, timeout=timeout)

            self.assertEqual(server.requests, [])

    def test_discovery_attempts_receive_the_timeout(self):
        with FakeLlamaServer() as server:
            server.set_route("/v1/completions", 404, "")
            server.set_route("/completion", 200, _timings_body(), "application/json")
            client = LlamaClient(server.url)
            with patch.object(client._opener, "open", wraps=client._opener.open) as open_:
                client.complete("x", 1, timeout=9.0)

        self.assertGreaterEqual(open_.call_count, 2)
        self.assertTrue(
            all(call.kwargs["timeout"] == 9.0 for call in open_.call_args_list)
        )
        self.assertEqual(
            [req[1] for req in server.requests],
            ["/v1/completions", "/v1/chat/completions", "/completion"],
        )

    def test_default_timeout_still_applies_to_gets(self):
        with FakeLlamaServer() as server:
            server.set_route("/metrics", 200, load_fixture("metrics_minimal.txt"), delay=0.5)
            client = LlamaClient(server.url, timeout=0.1)
            with self.assertRaises(FetchError) as ctx:
                client.fetch_metrics()

        self.assertEqual(ctx.exception.reason, "timed out")


class TestCompletionRouteDiscovery(unittest.TestCase):
    def test_llamacpp_style_wins_on_v1_completions(self):
        body = _timings_body()
        with FakeLlamaServer() as server:
            for route in ("/v1/completions", "/v1/chat/completions", "/completion"):
                server.set_route(route, 200, body, content_type="application/json")
            client = LlamaClient(server.url)
            timings = client.complete("hello", 64)
            self.assertEqual([req[1] for req in server.requests], ["/v1/completions"])
            self.assertEqual(client.completion_route, "/v1/completions")
            self.assertEqual(
                (timings.prompt_n, timings.prompt_ms, timings.predicted_n, timings.predicted_ms),
                (1200, 1500.0, 64, 3200.0),
            )
            request_json = json.loads(server.requests[0][3])
            self.assertEqual(request_json["prompt"], "hello")
            self.assertEqual(request_json["max_tokens"], 64)
            self.assertNotIn("n_predict", request_json)
            self.assertNotIn("messages", request_json)
            self.assertNotIn("model", request_json)

    def test_studio_style_uses_v1_completions(self):
        body = _timings_body(draft_n=256, draft_n_accepted=200)
        with FakeLlamaServer() as server:
            server.set_route("/v1/completions", 200, body, content_type="application/json")
            server.set_route(
                "/v1/chat/completions",
                200,
                json.dumps({"choices": []}),
                content_type="application/json",
            )
            client = LlamaClient(server.url)
            timings = client.complete("hello", 64)
            self.assertEqual([req[1] for req in server.requests], ["/v1/completions"])
            self.assertEqual(
                (timings.prompt_n, timings.prompt_ms, timings.predicted_n, timings.predicted_ms),
                (1200, 1500.0, 64, 3200.0),
            )
            self.assertNotIn("draft", repr(timings))

    def test_strata_style_falls_to_chat(self):
        body = _timings_body()
        with FakeLlamaServer() as server:
            server.set_route("/v1/chat/completions", 200, body, content_type="application/json")
            client = LlamaClient(server.url)
            timings = client.complete("hello", 64)
            self.assertEqual(
                [req[1] for req in server.requests],
                ["/v1/completions", "/v1/chat/completions"],
            )
            request_json = json.loads(server.requests[1][3])
            self.assertEqual(request_json["messages"][0], {"role": "user", "content": "hello"})
            self.assertEqual(client.completion_route, "/v1/chat/completions")
            self.assertEqual(timings.prompt_n, 1200)

    def test_legacy_style_falls_to_completion(self):
        body = _timings_body()
        with FakeLlamaServer() as server:
            server.set_route("/completion", 200, body, content_type="application/json")
            client = LlamaClient(server.url)
            timings = client.complete("hello", 64)
            self.assertEqual(
                [req[1] for req in server.requests],
                ["/v1/completions", "/v1/chat/completions", "/completion"],
            )
            request = next(req for req in server.requests if req[1] == "/completion")
            request_json = json.loads(request[3])
            self.assertIs(request_json["cache_prompt"], False)
            self.assertEqual(request_json["n_predict"], 64)
            self.assertEqual(request_json["temperature"], 0)
            self.assertFalse(request_json["stream"])
            self.assertEqual(request_json["prompt"], "hello")
            self.assertNotIn("model", request_json)
            self.assertEqual(client.completion_route, "/completion")
            self.assertEqual(timings.prompt_n, 1200)

    def test_all_routes_404_raises_route_error(self):
        with FakeLlamaServer() as server:
            client = LlamaClient(server.url)
            with self.assertRaises(ProbeRouteError) as ctx:
                client.complete("hello", 64)
            err = ctx.exception
            self.assertIsInstance(err, FetchError)
            self.assertEqual(
                [(a.endpoint, a.status, a.reason) for a in err.attempts],
                [
                    ("/v1/completions", 404, "HTTP 404"),
                    ("/v1/chat/completions", 404, "HTTP 404"),
                    ("/completion", 404, "HTTP 404"),
                ],
            )
            self.assertIsNone(client.completion_route)

    def test_405_falls_through(self):
        with FakeLlamaServer() as server:
            server.set_route("/v1/completions", 405, "")
            server.set_route(
                "/v1/chat/completions", 200, _timings_body(), content_type="application/json"
            )
            client = LlamaClient(server.url)
            client.complete("hello", 64)
            self.assertEqual(client.completion_route, "/v1/chat/completions")

    def test_non_dict_json_falls_through(self):
        with FakeLlamaServer() as server:
            server.set_route("/v1/completions", 200, "[1, 2]", content_type="application/json")
            server.set_route(
                "/v1/chat/completions", 200, _timings_body(), content_type="application/json"
            )
            client = LlamaClient(server.url)
            client.complete("hello", 64)
            self.assertEqual(client.completion_route, "/v1/chat/completions")

    def test_html_body_falls_through(self):
        with FakeLlamaServer() as server:
            server.set_route("/v1/completions", 200, "<html>", content_type="text/html")
            server.set_route(
                "/v1/chat/completions", 200, _timings_body(), content_type="application/json"
            )
            client = LlamaClient(server.url)
            client.complete("hello", 64)
            self.assertEqual(client.completion_route, "/v1/chat/completions")

    def test_non_numeric_timings_falls_through(self):
        with FakeLlamaServer() as server:
            server.set_route(
                "/v1/completions",
                200,
                _timings_body(prompt_ms="fast"),
                content_type="application/json",
            )
            server.set_route("/completion", 200, _timings_body(), content_type="application/json")
            client = LlamaClient(server.url)
            client.complete("hello", 64)
            self.assertEqual(client.completion_route, "/completion")
            self.assertEqual(
                [req[1] for req in server.requests],
                ["/v1/completions", "/v1/chat/completions", "/completion"],
            )
            server.set_route("/completion", 404, "")
            with self.assertRaises(ProbeRouteError) as ctx:
                LlamaClient(server.url).complete("hello", 64)
            self.assertEqual(ctx.exception.attempts[0].reason, "unusable timings")

    def _assert_stops_discovery(self, status: int) -> None:
        with FakeLlamaServer() as server:
            server.set_route("/v1/completions", status, "")
            server.set_route(
                "/v1/chat/completions", 200, _timings_body(), content_type="application/json"
            )
            client = LlamaClient(server.url)
            with self.assertRaises(ProbeRouteError) as ctx:
                client.complete("hello", 64)
            err = ctx.exception
            self.assertEqual([req[1] for req in server.requests], ["/v1/completions"])
            self.assertEqual(len(err.attempts), 1)
            self.assertEqual(err.attempts[0].status, status)

    def test_server_error_stops_discovery(self):
        self._assert_stops_discovery(500)

    def test_auth_error_stops_discovery(self):
        self._assert_stops_discovery(401)

    def test_status_taxonomy_fall_through_and_stop(self):
        for status in (400, 422, 501):
            with self.subTest(status=status), FakeLlamaServer() as server:
                server.set_route("/v1/completions", status, "")
                server.set_route(
                    "/v1/chat/completions", 200, _timings_body(), content_type="application/json"
                )
                client = LlamaClient(server.url)
                client.complete("hello", 64)
                self.assertEqual(
                    [req[1] for req in server.requests],
                    ["/v1/completions", "/v1/chat/completions"],
                )
        for status in (403, 429):
            with self.subTest(status=status):
                self._assert_stops_discovery(status)

    def test_falls_through_classification(self):
        for reason in ("timed out", "connection refused", "response too large"):
            with self.subTest(reason=reason):
                self.assertFalse(_falls_through(FetchError("/x", reason)))
        self.assertTrue(_falls_through(FetchError("/x", "HTTP 404", status=404)))
        self.assertTrue(_falls_through(FetchError("/x", "invalid JSON")))

    def test_route_cached_across_calls(self):
        with FakeLlamaServer() as server:
            server.set_route(
                "/v1/chat/completions", 200, _timings_body(), content_type="application/json"
            )
            client = LlamaClient(server.url)
            client.complete("hello", 64)
            client.complete("again", 64)
            self.assertEqual(
                [req[1] for req in server.requests],
                ["/v1/completions", "/v1/chat/completions", "/v1/chat/completions"],
            )
            self.assertEqual(client.completion_route, "/v1/chat/completions")

    def test_cached_route_failure_does_not_rediscover(self):
        with FakeLlamaServer() as server:
            server.set_route(
                "/v1/completions", 200, _timings_body(), content_type="application/json"
            )
            client = LlamaClient(server.url)
            client.complete("hello", 64)
            server.set_route("/v1/completions", 404, "")
            with self.assertRaises(FetchError) as ctx:
                client.complete("again", 64)
            err = ctx.exception
            self.assertNotIsInstance(err, ProbeRouteError)
            self.assertEqual(err.endpoint, "/v1/completions")
            self.assertEqual(
                [req[1] for req in server.requests],
                ["/v1/completions", "/v1/completions"],
            )
            self.assertEqual(client.completion_route, "/v1/completions")

    def test_model_id_on_openai_routes_only(self):
        with FakeLlamaServer() as server:
            server.set_route("/completion", 200, _timings_body(), content_type="application/json")
            client = LlamaClient(server.url)
            client.complete("hello", 64, model_id="m")
            bodies = {req[1]: json.loads(req[3]) for req in server.requests}
            self.assertEqual(bodies["/v1/completions"]["model"], "m")
            self.assertEqual(bodies["/v1/chat/completions"]["model"], "m")
            self.assertNotIn("model", bodies["/completion"])

    def test_overflowing_timing_falls_through(self):
        with FakeLlamaServer() as server:
            server.set_route(
                "/v1/completions",
                200,
                _timings_body(prompt_ms=10**400),
                content_type="application/json",
            )
            server.set_route(
                "/v1/chat/completions", 200, _timings_body(), content_type="application/json"
            )
            client = LlamaClient(server.url)
            client.complete("hello", 64)
            self.assertEqual(client.completion_route, "/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
