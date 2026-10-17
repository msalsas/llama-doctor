import unittest

from llama_doctor.client import FetchError
from llama_doctor.models import CompletionTimings
from llama_doctor.probe import WARMUP_N_PREDICT, build_prompt, run_probe


class _FakeClient:
    def __init__(self, timings: CompletionTimings | None = None, fail: bool = False) -> None:
        self.calls: list[tuple[str, int]] = []
        self.timings = timings or CompletionTimings(1200, 1500.0, 64, 3200.0)
        self.fail = fail

    def complete(self, prompt: str, n_predict: int) -> CompletionTimings:
        self.calls.append((prompt, n_predict))
        if self.fail:
            raise FetchError("/completion", "unreachable")
        return self.timings


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

        self.assertEqual(client.calls[0], ("Hello", WARMUP_N_PREDICT))
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


if __name__ == "__main__":
    unittest.main()
