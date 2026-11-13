import json
import unittest

from llama_doctor.client import FetchError, LlamaClient, ProbeRouteError
from llama_doctor.models import CompletionTimings, EndpointError, ProbeResult, ProbeSkip
from llama_doctor.probe import (
    WARMUP_N_PREDICT,
    build_prompt,
    describe_skip,
    run_probe,
    run_probe_detailed,
)

from .support import FakeLlamaServer


class _FakeClient:
    def __init__(
        self,
        timings: CompletionTimings | None = None,
        fail: bool = False,
        route_error: ProbeRouteError | None = None,
        measure_error: FetchError | None = None,
        route: str | None = "/completion",
    ) -> None:
        self.calls: list[tuple[str, int, str | None]] = []
        self.timings = timings or CompletionTimings(1200, 1500.0, 64, 3200.0)
        self.fail = fail
        self.route_error = route_error
        self.measure_error = measure_error
        self.completion_route = route

    def complete(
        self, prompt: str, n_predict: int, *, model_id: str | None = None
    ) -> CompletionTimings:
        self.calls.append((prompt, n_predict, model_id))
        if self.route_error is not None:
            raise self.route_error
        if self.fail:
            raise FetchError("/completion", "unreachable")
        if self.measure_error is not None and len(self.calls) > 1:
            raise self.measure_error
        return self.timings


def _route_error() -> ProbeRouteError:
    return ProbeRouteError(
        (
            EndpointError("/v1/completions", 404, "HTTP 404"),
            EndpointError("/v1/chat/completions", None, "no timings in response"),
            EndpointError("/completion", 404, "HTTP 404"),
        )
    )


class TestProbe(unittest.TestCase):
    def test_rates_from_timings(self):
        result = run_probe(_FakeClient())

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.prefill_tps, 800.0, places=6)
        self.assertAlmostEqual(result.decode_tps, 20.0, places=6)
        self.assertEqual(result.prompt_n, 1200)
        self.assertTrue(result.reliable)

    def test_warmup_runs_first(self):
        client = _FakeClient()

        run_probe(client)

        self.assertEqual(client.calls[0], ("Hello", WARMUP_N_PREDICT, None))
        self.assertEqual(client.calls[1][1], 64)

    def test_warmup_seconds_from_clock(self):
        values = iter((10.0, 10.5))

        result = run_probe(_FakeClient(), clock=lambda: next(values))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.warmup_s, 0.5, places=6)

    def test_returns_none_on_fetch_error(self):
        self.assertIsNone(run_probe(_FakeClient(fail=True)))

    def test_returns_none_on_zero_ms(self):
        client = _FakeClient(CompletionTimings(1200, 0.0, 64, 3200.0))

        self.assertIsNone(run_probe(client))

    def test_returns_none_on_missing_timing(self):
        client = _FakeClient(CompletionTimings(1200, None, 64, 3200.0))

        self.assertIsNone(run_probe(client))

    def test_prompt_size_and_nonce(self):
        first = build_prompt()
        second = build_prompt()

        self.assertGreaterEqual(len(first.split()), 1400)
        self.assertNotEqual(first, second)
        self.assertTrue(build_prompt(nonce="abc").startswith("[abc] "))


class TestProbeDetailed(unittest.TestCase):
    def test_discovery_failure_is_skip(self):
        error = _route_error()

        skip = run_probe_detailed(_FakeClient(route_error=error))

        self.assertIsInstance(skip, ProbeSkip)
        assert isinstance(skip, ProbeSkip)
        self.assertEqual(skip.stage, "discovery")
        self.assertEqual(skip.reason, "no usable completion route")
        self.assertIsNone(skip.route)
        self.assertEqual(skip.attempts, error.attempts)
        self.assertIsNone(run_probe(_FakeClient(route_error=_route_error())))

    def test_measure_failure_is_skip(self):
        client = _FakeClient(
            measure_error=FetchError("/v1/completions", "timed out"),
            route="/v1/completions",
        )

        skip = run_probe_detailed(client)

        self.assertIsInstance(skip, ProbeSkip)
        assert isinstance(skip, ProbeSkip)
        self.assertEqual(skip.stage, "measure")
        self.assertEqual(skip.route, "/v1/completions")
        self.assertEqual(skip.reason, "timed out")

    def test_zero_ms_is_timings_skip(self):
        skip = run_probe_detailed(_FakeClient(CompletionTimings(1200, 0.0, 64, 3200.0)))

        self.assertIsInstance(skip, ProbeSkip)
        assert isinstance(skip, ProbeSkip)
        self.assertEqual(skip.stage, "timings")
        self.assertEqual(skip.route, "/completion")

    def test_model_id_forwarded_to_both_calls(self):
        client = _FakeClient()

        outcome = run_probe_detailed(client, model_id="q")

        self.assertIsInstance(outcome, ProbeResult)
        self.assertEqual([call[2] for call in client.calls], ["q", "q"])


class TestDescribeSkip(unittest.TestCase):
    def test_all_routes_fell_through(self):
        skip = ProbeSkip(
            "discovery",
            "no usable completion route",
            None,
            (
                EndpointError("/v1/completions", 404, "HTTP 404"),
                EndpointError("/v1/chat/completions", None, "no timings in response"),
                EndpointError("/completion", 404, "HTTP 404"),
            ),
        )

        self.assertEqual(
            describe_skip(skip),
            "probe: no usable completion route (/v1/completions: HTTP 404; "
            "/v1/chat/completions: no timings in response; /completion: HTTP 404), "
            "using lifetime counters",
        )

    def test_stopped_discovery(self):
        skip = ProbeSkip(
            "discovery",
            "no usable completion route",
            None,
            (EndpointError("/v1/completions", None, "timed out"),),
        )

        self.assertEqual(
            describe_skip(skip),
            "probe: no usable completion route (/v1/completions: timed out), "
            "using lifetime counters",
        )

    def test_measure_failure(self):
        skip = ProbeSkip("measure", "timed out", "/v1/completions")

        self.assertEqual(
            describe_skip(skip),
            "probe: measured prompt failed on /v1/completions: timed out, using lifetime counters",
        )

    def test_timings_failure(self):
        skip = ProbeSkip("timings", "unusable timings", "/v1/completions")

        self.assertEqual(
            describe_skip(skip),
            "probe: unusable timings from /v1/completions, using lifetime counters",
        )


_COMPLETION_ROUTES = ("/v1/completions", "/v1/chat/completions", "/completion")


def _set_family_routes(server: FakeLlamaServer, family: str) -> str:
    timings = {
        "prompt_n": 1200,
        "prompt_ms": 1500.0,
        "predicted_n": 64,
        "predicted_ms": 3200.0,
    }
    body = json.dumps({"timings": timings})
    if family == "llamacpp":
        for route in _COMPLETION_ROUTES:
            server.set_route(route, 200, body, "application/json")
        return "/v1/completions"
    if family == "studio":
        studio = dict(timings, draft_n=300, draft_n_accepted=250)
        server.set_route(
            "/v1/completions", 200, json.dumps({"timings": studio}), "application/json"
        )
        server.set_route(
            "/v1/chat/completions", 200, json.dumps({"choices": []}), "application/json"
        )
        return "/v1/completions"
    if family == "strata":
        server.set_route("/v1/chat/completions", 200, body, "application/json")
        return "/v1/chat/completions"
    server.set_route("/completion", 200, body, "application/json")
    return "/completion"


class TestProbeEndToEnd(unittest.TestCase):
    def _assert_family_probe(self, family: str) -> None:
        with FakeLlamaServer() as server:
            route = _set_family_routes(server, family)
            client = LlamaClient(server.url)
            outcome = run_probe_detailed(client)
            requests = [req for req in server.requests if req[1] in _COMPLETION_ROUTES]
        self.assertIsInstance(outcome, ProbeResult)
        assert isinstance(outcome, ProbeResult)
        self.assertAlmostEqual(outcome.prefill_tps, 800.0, places=6)
        # Each earlier route is tried once during warm-up discovery (404 fall-through).
        expected = list(_COMPLETION_ROUTES[: _COMPLETION_ROUTES.index(route)]) + [route, route]
        self.assertEqual([req[1] for req in requests], expected)
        measured = json.loads(requests[-1][3])
        prompt = (
            measured["messages"][0]["content"]
            if route == "/v1/chat/completions"
            else measured["prompt"]
        )
        self.assertTrue(prompt.startswith("["))
        self.assertGreaterEqual(len(prompt.split()), 1400)

    def test_end_to_end_llamacpp(self):
        self._assert_family_probe("llamacpp")

    def test_end_to_end_studio(self):
        self._assert_family_probe("studio")

    def test_end_to_end_strata(self):
        self._assert_family_probe("strata")

    def test_end_to_end_legacy(self):
        self._assert_family_probe("legacy")

    def test_end_to_end_all_404(self):
        with FakeLlamaServer() as server:
            outcome = run_probe_detailed(LlamaClient(server.url))

        self.assertIsInstance(outcome, ProbeSkip)
        assert isinstance(outcome, ProbeSkip)
        self.assertEqual(
            describe_skip(outcome),
            "probe: no usable completion route (/v1/completions: HTTP 404; "
            "/v1/chat/completions: HTTP 404; /completion: HTTP 404), using lifetime counters",
        )


if __name__ == "__main__":
    unittest.main()
