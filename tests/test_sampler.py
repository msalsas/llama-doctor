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

    def _run_ticks(self, sampler, samples_at):
        for taken_at in samples_at:
            metrics = _metrics(samples_at[taken_at], taken_at)
            sampler._fetch = lambda metrics=metrics, taken_at=taken_at: _snapshot(metrics, taken_at)
            sampler.tick()
        return sampler.state()["history"]

    def test_recent_maxlen_derived_from_rate_window(self):
        self.assertEqual(Sampler(lambda: None, target="t", interval=2.0)._recent.maxlen, 17)
        self.assertEqual(
            Sampler(lambda: None, target="t", interval=2.0, rate_window=1.0)._recent.maxlen, 4
        )
        self.assertEqual(
            Sampler(lambda: None, target="t", interval=1.0, rate_window=4.0)._recent.maxlen, 6
        )

    def test_state_reports_rate_window(self):
        sampler = Sampler(lambda: _snapshot(_metrics(PREVIOUS)), target="t", rate_window=12.5)

        self.assertEqual(sampler.state()["rate_window_s"], 12.5)

    def test_sparse_counter_jumps_stay_visible_inside_window(self):
        sampler = Sampler(lambda: None, target="t", interval=2.0, rate_window=30.0)
        samples_at = {}
        for taken_at in range(0, 62, 2):
            jumps = taken_at // 12
            samples_at[float(taken_at)] = {
                "tokens_predicted_total": 600.0 * jumps,
                "tokens_predicted_seconds_total": 12.0 * jumps,
            }

        history = self._run_ticks(sampler, samples_at)

        for point in history:
            if point["t"] < 12:
                self.assertIsNone(point["decode_tps"], point["t"])
            else:
                self.assertAlmostEqual(point["decode_tps"], 50.0, places=6, msg=point["t"])

    def test_idle_counters_inside_window_are_none(self):
        sampler = Sampler(lambda: None, target="t", interval=2.0, rate_window=30.0)
        history = self._run_ticks(
            sampler, {float(taken_at): PREVIOUS for taken_at in range(0, 62, 2)}
        )

        self.assertEqual(len(history), 31)
        for point in history:
            for key in POINT_KEYS[1:6]:
                self.assertIsNone(point[key])

    def test_baseline_is_newest_entry_at_or_before_window_start(self):
        # decode_tps = (cur^2 - base^2) / (cur - base) = cur + base, so it reveals the baseline.
        sampler = Sampler(lambda: None, target="t", interval=1.0, rate_window=4.0)
        history = self._run_ticks(
            sampler,
            {
                float(taken_at): {
                    "tokens_predicted_total": float(taken_at * taken_at),
                    "tokens_predicted_seconds_total": float(taken_at),
                }
                for taken_at in range(10)
            },
        )

        bases = [point["decode_tps"] - point["t"] for point in history[1:]]
        self.assertEqual(bases, [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    def test_counter_reset_clears_window(self):
        sampler = Sampler(lambda: None, target="t", interval=1.0, rate_window=30.0)
        samples_at = {}
        for taken_at in range(4):
            samples_at[float(taken_at)] = {
                "tokens_predicted_total": 100.0 * (taken_at + 1),
                "tokens_predicted_seconds_total": float(taken_at + 1),
            }
        for taken_at in range(4, 7):
            samples_at[float(taken_at)] = {
                "tokens_predicted_total": 20.0 * (taken_at - 3),
                "tokens_predicted_seconds_total": float(taken_at - 3),
            }

        history = self._run_ticks(sampler, samples_at)

        self.assertEqual(len(sampler._recent), 3)
        self.assertEqual([point["decode_tps"] for point in history[4:]], [None, 20.0, 20.0])

    def test_single_sample_history_is_none(self):
        sampler = Sampler(lambda: _snapshot(_metrics(CURRENT), 5.0), target="t", rate_window=30.0)
        sampler.tick()

        history = sampler.state()["history"]
        self.assertEqual(len(history), 1)
        for key in POINT_KEYS[1:6]:
            self.assertIsNone(history[0][key])

    def test_state_keys_exact(self):
        sampler = Sampler(lambda: _snapshot(_metrics(PREVIOUS)), target="http://server")
        sampler.tick()

        self.assertEqual(
            tuple(sampler.state()),
            (
                "version",
                "target",
                "interval_s",
                "rate_window_s",
                "updated_at",
                "server",
                "lifetime",
                "history",
                "report",
                "errors",
                "notes",
            ),
        )

    def _notes(self, samples_at):
        sampler = Sampler(lambda: None, target="t", interval=2.0, rate_window=30.0)
        self._run_ticks(sampler, samples_at)
        return sampler.state()["notes"]

    def test_notes_keys_exact(self):
        self.assertEqual(
            Sampler(lambda: None, target="t").state()["notes"],
            {
                "kv_unexported": False,
                "cache_hit_unexported": False,
                "decode_idle": False,
                "prefill_idle": False,
            },
        )

    def test_notes_flag_kv_usage_missing_from_server(self):
        notes = self._notes({0.0: CURRENT, 2.0: CURRENT})

        self.assertTrue(notes["kv_unexported"])
        self.assertFalse(notes["cache_hit_unexported"])

    def test_notes_flag_prefix_cache_counter_missing_from_server(self):
        samples = dict(CURRENT)
        del samples["prompt_tokens_cached_total"]

        self.assertTrue(self._notes({0.0: samples})["cache_hit_unexported"])

    def test_notes_exported_metrics_not_flagged_when_window_rates_idle(self):
        samples = dict(CURRENT, kv_cache_usage_ratio=0.0)
        notes = self._notes({float(taken_at): samples for taken_at in range(0, 20, 2)})

        self.assertFalse(notes["kv_unexported"])
        self.assertFalse(notes["cache_hit_unexported"])

    def test_notes_idle_window_with_lifetime_traffic(self):
        notes = self._notes({float(taken_at): CURRENT for taken_at in range(0, 20, 2)})

        self.assertTrue(notes["decode_idle"])
        self.assertTrue(notes["prefill_idle"])

    def test_notes_single_point_is_never_idle(self):
        notes = self._notes({0.0: CURRENT})

        self.assertFalse(notes["decode_idle"])
        self.assertFalse(notes["prefill_idle"])

    def test_notes_null_lifetime_is_never_idle(self):
        samples = {"kv_cache_usage_ratio": 0.0, "prompt_tokens_cached_total": 0.0}
        notes = self._notes({float(taken_at): samples for taken_at in range(0, 20, 2)})

        self.assertFalse(notes["decode_idle"])
        self.assertFalse(notes["prefill_idle"])

    def test_notes_active_window_is_not_idle(self):
        notes = self._notes({0.0: PREVIOUS, 2.0: CURRENT})

        self.assertFalse(notes["decode_idle"])
        self.assertFalse(notes["prefill_idle"])

    def test_notes_without_metrics_snapshot_are_false(self):
        sampler = Sampler(lambda: _snapshot(None), target="t")
        sampler.tick()

        self.assertFalse(any(sampler.state()["notes"].values()))

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

    def test_state_server_quant_from_models_meta(self):
        from llama_doctor.models import ModelsMeta, PropsSnapshot

        props = PropsSnapshot.from_payload(json.loads(load_fixture("props_unsloth_minimal.json")))
        models = ModelsMeta.from_payload(json.loads(load_fixture("models_full.json")))
        sampler = Sampler(
            lambda: ServerSnapshot(1.0, None, props, models, {}),
            target="http://server",
        )
        sampler.tick()

        self.assertEqual(sampler.state()["server"]["quant"], "UD-Q4_K_XL")

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
