import json
import math
import time
import unittest

from llama_doctor.client import FetchError, LlamaClient, ProbeRouteError
from llama_doctor.models import CompletionTimings, EndpointError, ProbeResult, ProbeSkip
from llama_doctor.probe import (
    PROBE_BUDGET_EXHAUSTED,
    PROBE_BUDGET_S,
    PROBE_REDUCED_PROMPT_WORDS,
    WARMUP_N_PREDICT,
    MeasurePlan,
    budget_hint,
    build_prompt,
    describe_skip,
    plan_measurement,
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
        warmup_timings: CompletionTimings | None = None,
    ) -> None:
        self.calls: list[tuple[str, int, str | None, float | None]] = []
        self.timings = timings or CompletionTimings(1200, 1500.0, 64, 3200.0)
        self.fail = fail
        self.route_error = route_error
        self.measure_error = measure_error
        self.completion_route = route
        self.warmup_timings = warmup_timings

    def complete(
        self, prompt: str, n_predict: int, *, model_id: str | None = None,
        timeout: float | None = None,
    ) -> CompletionTimings:
        self.calls.append((prompt, n_predict, model_id, timeout))
        if self.route_error is not None:
            raise self.route_error
        if self.fail:
            raise FetchError("/completion", "unreachable")
        if self.measure_error is not None and len(self.calls) > 1:
            raise self.measure_error
        if self.warmup_timings is not None and len(self.calls) == 1:
            return self.warmup_timings
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

        self.assertEqual(client.calls[0][:3], ("Hello", WARMUP_N_PREDICT, None))
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


class TestPlanMeasurement(unittest.TestCase):
    def test_full_plan(self):
        plan = plan_measurement(CompletionTimings(2, 10.0, 64, 3200.0), 200.0)

        self.assertEqual(plan, MeasurePlan(1400, 64, 200.0))

    def test_reduced_plan_on_slow_decode(self):
        plan = plan_measurement(CompletionTimings(2, 10.0, 8, 4000.0), 200.0)

        self.assertEqual(plan, MeasurePlan(1000, 16, 200.0))

    def test_estimate_boundary(self):
        warmup = CompletionTimings(2, 10.0, 20, 1000.0)
        estimate = (1400 * 1.25 + 64) / 20

        self.assertEqual(plan_measurement(warmup, estimate).words, 1400)
        self.assertEqual(plan_measurement(warmup, estimate - 0.01).words, 1000)

    def test_unusable_decode_rate_gives_reduced_plan(self):
        cases = (
            CompletionTimings(2, 10.0, 0, 4000.0),
            CompletionTimings(2, 10.0, None, 4000.0),
            CompletionTimings(2, 10.0, 8, 0.0),
            CompletionTimings(2, 10.0, 8, None),
            CompletionTimings(2, 10.0, 8, -1.0),
        )
        for warmup in cases:
            with self.subTest(warmup=warmup):
                plan = plan_measurement(warmup, 1000.0)

                self.assertEqual(plan.words, 1000)

    def test_small_n_predict_is_kept(self):
        reduced = plan_measurement(CompletionTimings(2, 10.0, 8, 4000.0), 200.0, n_predict=8)
        fast = CompletionTimings(2, 10.0, 64, 100.0)
        full = plan_measurement(fast, 200.0, n_predict=128)

        self.assertEqual(reduced.n_predict, 8)
        self.assertEqual(full.n_predict, 128)

    def test_minimum_remaining_time(self):
        warmup = CompletionTimings(2, 10.0, None, None)

        self.assertIsNone(plan_measurement(warmup, 0.99))
        self.assertIsNotNone(plan_measurement(warmup, 1.0))
        self.assertIsNone(plan_measurement(warmup, float("nan")))


class TestProbeBudget(unittest.TestCase):
    def test_timeouts_follow_the_budget(self):
        client = _FakeClient()
        clock = iter((10.0, 12.0)).__next__

        run_probe_detailed(client, clock=clock, budget_s=30.0)

        self.assertAlmostEqual(client.calls[0][3], 10.0)
        self.assertAlmostEqual(client.calls[1][3], 28.0)

    def test_default_budget_timeouts(self):
        client = _FakeClient()
        clock = iter((0.0, 5.0)).__next__

        run_probe_detailed(client, clock=clock)

        self.assertAlmostEqual(client.calls[0][3], 100.0)
        self.assertAlmostEqual(client.calls[1][3], 295.0)

    def test_budget_exhausted_after_warmup(self):
        client = _FakeClient()
        clock = iter((0.0, 299.5)).__next__
        plans: list[MeasurePlan] = []

        outcome = run_probe_detailed(
            client, clock=clock, on_plan=plans.append
        )

        self.assertEqual(
            outcome, ProbeSkip("measure", "probe budget exhausted", "/completion")
        )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(plans, [])

    def test_reduced_plan_end_to_end(self):
        client = _FakeClient(warmup_timings=CompletionTimings(2, 10.0, 8, 4000.0))

        outcome = run_probe_detailed(client)

        self.assertIsInstance(outcome, ProbeResult)
        self.assertEqual(client.calls[1][1], 16)
        self.assertEqual(
            len(client.calls[1][0].split()),
            len(build_prompt(1000, nonce="x").split()),
        )

    def test_on_plan_called_once_before_measured_request(self):
        client = _FakeClient()
        seen: list[tuple[MeasurePlan, int]] = []

        def spy(plan: MeasurePlan) -> None:
            seen.append((plan, len(client.calls)))

        outcome = run_probe_detailed(client, clock=iter((0.0, 0.0)).__next__, on_plan=spy)

        self.assertIsInstance(outcome, ProbeResult)
        self.assertEqual(len(seen), 1)
        self.assertEqual((seen[0][0].words, seen[0][0].n_predict), (1400, 64))
        self.assertEqual(seen[0][1], 1)

    def test_on_plan_not_called_on_discovery_failure(self):
        plans: list[MeasurePlan] = []

        outcome = run_probe_detailed(
            _FakeClient(route_error=_route_error()), on_plan=plans.append
        )

        self.assertIsInstance(outcome, ProbeSkip)
        self.assertEqual(plans, [])

    def test_invalid_budget_rejected_before_any_request(self):
        for budget in (0, -1, math.inf, math.nan):
            with self.subTest(budget=budget):
                client = _FakeClient()

                with self.assertRaises(ValueError):
                    run_probe_detailed(client, budget_s=budget)

                self.assertEqual(client.calls, [])

    def test_run_probe_forwards_budget(self):
        client = _FakeClient()

        run_probe(client, budget_s=9.0, clock=iter((0.0, 0.0)).__next__)

        self.assertAlmostEqual(client.calls[0][3], 3.0)

    def test_clock_called_exactly_twice_on_success(self):
        ticks = iter((0.0, 0.5, 99.0)).__next__
        calls = 0

        def counting_clock() -> float:
            nonlocal calls
            calls += 1
            return ticks()

        result = run_probe(_FakeClient(), clock=counting_clock)

        self.assertIsNotNone(result)
        self.assertEqual(calls, 2)


class TestBudgetHint(unittest.TestCase):
    def test_hint_conditions(self):
        timed_out = EndpointError("/v1/completions", None, "timed out")
        cases = (
            (ProbeSkip("discovery", "no usable completion route", None, (timed_out,)), True),
            (
                ProbeSkip(
                    "discovery",
                    "no usable completion route",
                    None,
                    (EndpointError("/v1/completions", 404, "HTTP 404"),),
                ),
                False,
            ),
            (ProbeSkip("measure", "timed out", "/v1/completions"), True),
            (ProbeSkip("measure", PROBE_BUDGET_EXHAUSTED, "/completion"), True),
            (ProbeSkip("measure", "HTTP 500", "/v1/completions"), False),
            (ProbeSkip("timings", "unusable timings", "/v1/completions"), False),
        )
        for skip, hinted in cases:
            with self.subTest(skip=skip):
                hint = budget_hint(skip, PROBE_BUDGET_S)
                if hinted:
                    self.assertIsNotNone(hint)
                else:
                    self.assertIsNone(hint)

    def test_hint_string(self):
        skip = ProbeSkip("measure", "timed out", "/v1/completions")

        self.assertEqual(
            budget_hint(skip, 300.0),
            "probe: the server did not answer within the 300 s probe budget; "
            "retry with a larger --probe-budget",
        )
        self.assertIn("within the 45.5 s probe budget", budget_hint(skip, 45.5))

    def test_exhausted_skip_string(self):
        self.assertEqual(
            describe_skip(ProbeSkip("measure", PROBE_BUDGET_EXHAUSTED, "/completion")),
            "probe: measured prompt failed on /completion: probe budget exhausted, "
            "using lifetime counters",
        )


class TestProbeServerBudget(unittest.TestCase):
    def _timings_body(self, **overrides: object) -> str:
        timings = {
            "prompt_n": 1200,
            "prompt_ms": 1500.0,
            "predicted_n": 64,
            "predicted_ms": 3200.0,
        }
        timings.update(overrides)
        return json.dumps({"timings": timings})

    def test_probe_ignores_client_timeout(self):
        with FakeLlamaServer() as server:
            server.set_route(
                "/v1/completions", 200, self._timings_body(), "application/json", delay=0.3
            )
            client = LlamaClient(server.url, timeout=0.1)

            outcome = run_probe_detailed(client, budget_s=6.0)

        self.assertIsInstance(outcome, ProbeResult)

    def test_warmup_timeout_stops_discovery_within_budget(self):
        with FakeLlamaServer() as server:
            server.set_route(
                "/v1/completions", 200, self._timings_body(), "application/json", delay=1.0
            )
            client = LlamaClient(server.url)
            started = time.monotonic()

            outcome = run_probe_detailed(client, budget_s=1.5)

            wall_s = time.monotonic() - started
        self.assertEqual(
            outcome,
            ProbeSkip(
                "discovery",
                "no usable completion route",
                None,
                (EndpointError("/v1/completions", None, "timed out"),),
            ),
        )
        completion_requests = [
            req for req in server.requests if req[1] == "/v1/completions"
        ]
        self.assertEqual(len(completion_requests), 1)
        self.assertLess(wall_s, 1.5 + 1.0)

    def test_reduced_prompt_stays_reliable(self):
        self.assertGreaterEqual(
            len(build_prompt(PROBE_REDUCED_PROMPT_WORDS).split()), 1000
        )


if __name__ == "__main__":
    unittest.main()
