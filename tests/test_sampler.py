import json
import time
import unittest

from llama_doctor.dashboard.sampler import POINT_KEYS, Sampler, _delta_rate, delta_point
from llama_doctor.models import EndpointError, MetricsSnapshot, ServerSnapshot
from llama_doctor.promparse import parse_prometheus

from .support import load_fixture

PREVIOUS = {
    "prompt_tokens_total": 1000,
    "prompt_seconds_total": 2,
    "tokens_predicted_total": 500,
    "tokens_predicted_seconds_total": 10,
    "prompt_tokens_cached_total": 0,
    "draft_tokens_total": 100,
    "drafts_total": 50,
    "draft_tokens_accepted_total": 80,
}
CURRENT = {
    "prompt_tokens_total": 3000,
    "prompt_seconds_total": 4,
    "tokens_predicted_total": 700,
    "tokens_predicted_seconds_total": 14,
    "prompt_tokens_cached_total": 2000,
    "draft_tokens_total": 160,
    "drafts_total": 70,
    "draft_tokens_accepted_total": 125,
}


def _metrics(samples: dict[str, float], taken_at: float = 0.0) -> MetricsSnapshot:
    return MetricsSnapshot.from_samples(samples, taken_at=taken_at)


def _snapshot(metrics: MetricsSnapshot | None, taken_at: float = 0.0) -> ServerSnapshot:
    return ServerSnapshot(taken_at, metrics, None, None, {})


class TestSampler(unittest.TestCase):
    def test_delta_rate_rejects_non_finite_inputs_and_results(self):
        self.assertIsNone(_delta_rate(float("nan"), 1.0, 0.0, 1.0))
        self.assertIsNone(_delta_rate(0.0, float("inf"), 0.0, 1.0))
        self.assertIsNone(_delta_rate(-1e308, 1e308, 0.0, 1.0))

    def test_delta_point_rates(self):
        point = delta_point(_metrics(PREVIOUS), _metrics(CURRENT), 2.0)

        self.assertAlmostEqual(point["prefill_tps"], 1000.0, places=6)
        self.assertAlmostEqual(point["decode_tps"], 50.0, places=6)
        self.assertAlmostEqual(point["cache_hit_ratio"], 0.5, places=6)
        self.assertAlmostEqual(point["avg_draft_len"], 3.0, places=6)
        self.assertAlmostEqual(point["draft_acceptance"], 0.75, places=6)

    def test_delta_point_idle_is_none(self):
        point = delta_point(_metrics(PREVIOUS), _metrics(PREVIOUS), 1.0)

        for key in POINT_KEYS[1:6]:
            self.assertIsNone(point[key])

    def test_delta_point_counter_reset_is_none(self):
        reset = {key: value - 1 for key, value in CURRENT.items()}
        point = delta_point(_metrics(CURRENT), _metrics(reset), 1.0)

        for key in POINT_KEYS[1:6]:
            self.assertIsNone(point[key])

    def test_delta_point_first_sample_none(self):
        point = delta_point(None, _metrics(CURRENT), 1.0)

        for key in POINT_KEYS[1:6]:
            self.assertIsNone(point[key])

    def test_delta_point_gauges_passthrough(self):
        metrics = MetricsSnapshot.from_samples(parse_prometheus(load_fixture("metrics_healthy.txt")))
        point = delta_point(None, metrics, 1.0)

        self.assertAlmostEqual(point["kv_usage_ratio"], 0.42, places=6)
        self.assertAlmostEqual(point["requests_processing"], 1.0, places=6)

    def test_point_keys_exact(self):
        self.assertEqual(
            POINT_KEYS,
            (
                "t",
                "decode_tps",
                "prefill_tps",
                "cache_hit_ratio",
                "avg_draft_len",
                "draft_acceptance",
                "kv_usage_ratio",
                "requests_processing",
            ),
        )

    def test_tick_appends_history(self):
        snapshots = iter([_snapshot(_metrics(PREVIOUS), 1.0), _snapshot(_metrics(CURRENT), 2.0)])
        sampler = Sampler(lambda: next(snapshots), target="http://server")

        sampler.tick()
        sampler.tick()

        history = sampler.state()["history"]
        self.assertEqual(len(history), 2)
        self.assertIsNone(history[0]["decode_tps"])
        self.assertAlmostEqual(history[1]["decode_tps"], 50.0, places=6)
        self.assertEqual(history[1]["t"], 2.0)

    def test_missing_metrics_preserves_baseline(self):
        snapshots = iter(
            [
                _snapshot(_metrics(PREVIOUS), 1.0),
                _snapshot(None, 2.0),
                _snapshot(_metrics(CURRENT), 3.0),
            ]
        )
        sampler = Sampler(lambda: next(snapshots), target="http://server")

        sampler.tick()
        sampler.tick()
        sampler.tick()

        history = sampler.state()["history"]
        self.assertEqual(len(history), 2)
        self.assertAlmostEqual(history[1]["decode_tps"], 50.0, places=6)

    def test_history_bounded(self):
        sampler = Sampler(
            lambda: _snapshot(_metrics(PREVIOUS), 0.0),
            target="http://server",
            history=5,
        )
        for taken_at in range(8):
            sampler._fetch = lambda taken_at=taken_at: _snapshot(
                _metrics(PREVIOUS, float(taken_at)), float(taken_at)
            )
            sampler.tick()

        history = sampler.state()["history"]
        self.assertEqual(len(history), 5)
        self.assertEqual(history[0]["t"], 3.0)

    def test_state_keys_exact(self):
        sampler = Sampler(lambda: _snapshot(_metrics(PREVIOUS)), target="http://server")
        sampler.tick()

        self.assertEqual(
            tuple(sampler.state()),
            (
                "version",
                "target",
                "interval_s",
                "updated_at",
                "server",
                "lifetime",
                "history",
                "report",
                "errors",
            ),
        )

    def test_state_moe_report(self):
        metrics = MetricsSnapshot.from_samples(
            parse_prometheus(load_fixture("metrics_moe_offload.txt"))
        )
        from llama_doctor.models import PropsSnapshot

        props = PropsSnapshot.from_payload(json.loads(load_fixture("props_full.json")))
        sampler = Sampler(
            lambda: ServerSnapshot(1.0, metrics, props, None, {}),
            target="http://server",
        )
        sampler.tick()

        state = sampler.state()
        self.assertEqual(state["report"]["overall"], "critical")
        self.assertEqual(state["server"]["quant"], "UD-Q4_K_XL")

    def test_state_has_no_sentinel(self):
        for fixture in ("props_full.json", "props_unsloth_minimal.json"):
            from llama_doctor.models import PropsSnapshot

            props = PropsSnapshot.from_payload(json.loads(load_fixture(fixture)))
            sampler = Sampler(
                lambda props=props: ServerSnapshot(1.0, None, props, None, {}),
                target="http://server",
            )
            sampler.tick()
            self.assertNotIn("SENTINEL", json.dumps(sampler.state()))

    def test_tick_survives_fetch_exception(self):
        def failing_fetch() -> ServerSnapshot:
            raise RuntimeError("SENTINEL")

        sampler = Sampler(failing_fetch, target="http://server")
        sampler.tick()

        state = sampler.state()
        self.assertEqual(state["errors"], {"sampler": "RuntimeError"})
        self.assertNotIn("SENTINEL", json.dumps(state))

    def test_unreachable_snapshot_keeps_history_empty(self):
        snapshot = ServerSnapshot(
            1.0,
            None,
            None,
            None,
            {"/metrics": EndpointError("/metrics", None, "unreachable")},
        )
        sampler = Sampler(lambda: snapshot, target="http://server")
        sampler.tick()

        state = sampler.state()
        self.assertEqual(state["history"], [])
        self.assertIn(
            "connectivity.unreachable",
            [finding["code"] for finding in state["report"]["findings"]],
        )

    def test_start_stop(self):
        calls = 0

        def fetch() -> ServerSnapshot:
            nonlocal calls
            calls += 1
            return _snapshot(_metrics(PREVIOUS), float(calls))

        sampler = Sampler(fetch, target="http://server", interval=0.01)
        sampler.start()
        sampler.start()
        deadline = time.monotonic() + 2
        while calls < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        sampler.stop()

        self.assertGreaterEqual(calls, 2)
        self.assertFalse(sampler._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
