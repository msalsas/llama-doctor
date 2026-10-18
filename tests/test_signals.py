import math
import unittest
from dataclasses import replace

from llama_doctor.analysis.signals import (
    context_used_ratio,
    decode_rate,
    is_cpu_bound,
    mean_prompt_tokens,
    prefill_rate,
)
from llama_doctor.models import ProbeResult

from .support import make_context


class TestSignals(unittest.TestCase):
    def test_prefill_rate_lifetime(self):
        rate = prefill_rate(make_context())
        self.assertIsNotNone(rate)
        self.assertAlmostEqual(rate.value, 40.0, places=6)
        self.assertEqual(rate.source, "lifetime")

    def test_prefill_rate_insufficient_below_1000_tokens(self):
        self.assertIsNone(prefill_rate(make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None)))

    def test_probe_preferred_when_reliable(self):
        probe = ProbeResult(850.0, 20.0, 1200, 8, 0.0)
        rate = prefill_rate(make_context(probe=probe))
        self.assertEqual((rate.value, rate.source), (850.0, "probe"))

    def test_unreliable_probe_ignored(self):
        rate = prefill_rate(make_context(probe=ProbeResult(850.0, 20.0, 500, 8, 0.0)))
        self.assertEqual((rate.value, rate.source), (40.0, "lifetime"))

    def _window(self, totals):
        metrics = make_context().metrics
        return tuple(replace(metrics, prompt_tokens_total=total, taken_at=float(index))
                     for index, total in enumerate(totals))

    def test_mean_prompt_tokens_averages_steps_where_prompts_completed(self):
        context = replace(make_context(), window=self._window([1000, 1000, 1300, 1300, 1800]))
        self.assertEqual(mean_prompt_tokens(context), 400)

    def test_mean_prompt_tokens_skips_missing_and_backwards_samples(self):
        context = replace(make_context(), window=self._window([1000, None, 1600, 900, 1100]))
        self.assertEqual(mean_prompt_tokens(context), 200)

    def test_mean_prompt_tokens_none_without_window_or_completions(self):
        self.assertIsNone(mean_prompt_tokens(make_context()))
        self.assertIsNone(mean_prompt_tokens(replace(make_context(), window=self._window([1000, 1000]))))
        self.assertIsNone(mean_prompt_tokens(replace(make_context(), window=self._window([1000]))))

    def test_context_used_ratio(self):
        self.assertAlmostEqual(context_used_ratio(78501, 80128), 0.97969499, places=8)
        self.assertEqual(context_used_ratio(90000, 100000), 0.9)

    def test_context_used_ratio_missing_or_invalid_is_none(self):
        for n_tokens_max, n_ctx in ((None, 80128), (78501, None), (78501, 0), (math.nan, 80128),
                                    (78501, math.inf), (-1, 80128)):
            with self.subTest(n_tokens_max=n_tokens_max, n_ctx=n_ctx):
                self.assertIsNone(context_used_ratio(n_tokens_max, n_ctx))

    def test_is_cpu_bound(self):
        self.assertTrue(is_cpu_bound(make_context()))
        self.assertFalse(is_cpu_bound(make_context("metrics_healthy.txt")))
        self.assertIsNone(is_cpu_bound(make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None)))

    def test_cpu_bound_by_ratio(self):
        context = make_context(probe=ProbeResult(150.0, 100.0, 1200, 10, 0.0))
        self.assertTrue(is_cpu_bound(context))
        self.assertEqual(decode_rate(context).source, "probe")

    def test_cpu_bound_ignores_non_positive_or_non_finite_decode_rate(self):
        for value in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(decode_tps=value):
                context = make_context(probe=ProbeResult(150.0, value, 1200, 10, 0.0))
                self.assertFalse(is_cpu_bound(context))


if __name__ == "__main__":
    unittest.main()
