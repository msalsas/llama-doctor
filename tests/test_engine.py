import math
import unittest
from dataclasses import replace

from llama_doctor.analysis import BUILTIN_ANALYZERS, diagnose
from llama_doctor.analysis.base import AnalyzerSpec, Finding, Impact, Recommendation, Severity
from llama_doctor.analysis.engine import run_analyzers

from .support import make_context


class TestEngine(unittest.TestCase):
    def test_moe_report_overall_critical_exit_2(self):
        report = diagnose(make_context())
        self.assertEqual((report.overall, report.exit_code), (Severity.CRITICAL, 2))

    def test_healthy_report_exit_0(self):
        report = diagnose(make_context("metrics_healthy.txt"))
        self.assertEqual(report.exit_code, 0)
        self.assertEqual(report.overall, Severity.INFO)

    def test_minimal_report_exit_0(self):
        report = diagnose(make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None))
        self.assertEqual(report.exit_code, 0)

    def test_findings_sorted(self):
        report = run_analyzers(make_context(), (
            AnalyzerSpec("lower-impact", "Lower impact", lambda _: [Finding(
                "moe", "moe.lower_impact", Severity.WARN, "Lower impact", "Lower impact",
                recommendations=(Recommendation("Lower-impact action.", "lower", Impact.LOW),),
            )]),
            AnalyzerSpec("higher-impact", "Higher impact", lambda _: [Finding(
                "moe", "moe.higher_impact", Severity.WARN, "Higher impact", "Higher impact",
                recommendations=(Recommendation("Higher-impact action.", "higher", Impact.HIGH),),
            )]),
        ))
        self.assertEqual([finding.code for finding in report.findings],
                         ["moe.higher_impact", "moe.lower_impact"])

    def test_recommendations_deduped_by_action_and_flag(self):
        unflagged_low = Recommendation("Use a smaller quantization.", "low", Impact.LOW)
        flagged_low = Recommendation("Reduce context conservatively.", "low", Impact.LOW, "-c")
        medium = Recommendation("Tune batch size.", "medium", Impact.MEDIUM, "--batch-size")
        unflagged_high = Recommendation("Use a smaller quantization.", "high", Impact.HIGH)
        flagged_high = Recommendation("Reduce context aggressively.", "high", Impact.HIGH, "-c")
        report = run_analyzers(make_context(), (
            AnalyzerSpec("one", "One", lambda _: [Finding("one", "one.warn", Severity.WARN, "One", "One",
                                                           recommendations=(unflagged_low, flagged_low, medium))]),
            AnalyzerSpec("two", "Two", lambda _: [Finding("two", "two.warn", Severity.WARN, "Two", "Two",
                                                           recommendations=(unflagged_high, flagged_high))]),
        ))
        recommendations = report.recommendations()
        self.assertEqual(recommendations, [unflagged_high, flagged_high, medium, flagged_low])

    def test_opposite_recommendations_for_the_same_flag_both_survive(self):
        shrink = Recommendation("Set context to 12288.", "A smaller context frees memory.", Impact.MEDIUM, "-c")
        grow = Recommendation("Raise the context size or shorten prompts.", "Avoid truncation.", Impact.MEDIUM, "-c")
        duplicate = Recommendation("Set context to 12288.", "Same action again.", Impact.LOW, "-c")
        report = run_analyzers(make_context(), (
            AnalyzerSpec("one", "One", lambda _: [Finding("one", "one.warn", Severity.WARN, "One", "One",
                                                           recommendations=(shrink, duplicate))]),
            AnalyzerSpec("two", "Two", lambda _: [Finding("two", "two.warn", Severity.WARN, "Two", "Two",
                                                           recommendations=(grow,))]),
        ))
        self.assertEqual(report.recommendations(), [shrink, grow])

    def test_recommendations_sorted_by_impact_within_finding(self):
        low = Recommendation("a-action", "a-flag", Impact.LOW)
        high = Recommendation("b-action", "b-flag", Impact.HIGH)
        medium = Recommendation("c-action", "c-flag", Impact.MEDIUM)
        report = run_analyzers(make_context(), (
            AnalyzerSpec("higher-impact", "Higher impact", lambda _: [Finding(
                "one", "one.warn", Severity.WARN, "One", "One",
                recommendations=(low, high),
            )]),
            AnalyzerSpec("lower-impact", "Lower impact", lambda _: [Finding(
                "two", "two.warn", Severity.WARN, "Two", "Two",
                recommendations=(medium,),
            )]),
        ))

        self.assertEqual(report.recommendations(), [high, medium, low])

    def test_analyzer_exception_isolated(self):
        def raises(_):
            raise ValueError("SENTINEL")

        healthy = AnalyzerSpec("healthy", "Healthy", lambda _: [Finding(
            "healthy", "healthy.ok", Severity.OK, "Healthy", "Healthy",
        )])
        report = run_analyzers(make_context(), (AnalyzerSpec("bad", "Bad", raises), healthy))
        finding = report.by_code("bad.error")
        self.assertEqual((finding.analyzer_id, finding.severity, finding.title, finding.detail),
                         ("bad", Severity.INSUFFICIENT_DATA, "Bad: analyzer failed", "ValueError"))
        self.assertNotIn("SENTINEL", finding.detail)
        self.assertIsNotNone(report.by_code("healthy.ok"))

    def test_non_iterable_analyzer_return_is_isolated(self):
        report = run_analyzers(make_context(), (AnalyzerSpec("none", "None", lambda _: None),))
        finding = report.by_code("none.error")
        self.assertEqual((finding.analyzer_id, finding.title, finding.detail),
                         ("none", "None: analyzer failed", "TypeError"))

    def _near_limit_context(self, kv_ratio):
        context = make_context("metrics_context_near_limit.txt", "props_context_80k.json")
        return replace(context, snapshot=replace(context.snapshot,
                       metrics=replace(context.metrics, kv_cache_usage_ratio=kv_ratio)))

    def test_kv_ok_never_coexists_with_context_near_limit_when_kv_ratio_high_or_missing(self):
        for kv_ratio in (None, math.nan, math.inf, 0.9, 0.91, 0.95, 0.98, 0.99, 1.0):
            with self.subTest(kv_ratio=kv_ratio):
                report = diagnose(self._near_limit_context(kv_ratio))
                self.assertIsNotNone(report.by_code("context.near_limit"))
                self.assertIsNone(report.by_code("kv.ok"))

    def test_kv_ok_beside_context_near_limit_only_speaks_of_the_kv_gauge(self):
        for kv_ratio in (0.0, 0.1, 0.42, 0.5, 0.75, 0.89):
            with self.subTest(kv_ratio=kv_ratio):
                report = diagnose(self._near_limit_context(kv_ratio))
                self.assertIsNotNone(report.by_code("context.near_limit"))
                kv_ok = report.by_code("kv.ok")
                self.assertIsNotNone(kv_ok)
                text = f"{kv_ok.title} {kv_ok.detail}".lower()
                for word in ("sizing", "context", "peak"):
                    self.assertNotIn(word, text)

    def test_custom_registry(self):
        registry = (BUILTIN_ANALYZERS[0],)
        report = diagnose(make_context(), registry)
        self.assertEqual([finding.code for finding in report.findings], ["connectivity.ok"])


if __name__ == "__main__":
    unittest.main()
