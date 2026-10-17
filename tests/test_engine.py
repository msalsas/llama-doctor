import unittest

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
        def moe_warn(code, impact):
            return lambda _: [Finding("moe", code, Severity.WARN, code, code,
                                     recommendations=(Recommendation(code, code, impact),) if impact else ())]

        report = run_analyzers(make_context(), (
            AnalyzerSpec("first", "First", moe_warn("warn.first", Impact.HIGH)),
            AnalyzerSpec("second", "Second", moe_warn("warn.second", Impact.HIGH)),
            AnalyzerSpec("third", "Third", moe_warn("warn.third", Impact.MEDIUM)),
            AnalyzerSpec("fourth", "Fourth", moe_warn("warn.fourth", 0)),
        ))
        self.assertEqual([finding.code for finding in report.findings],
                         ["warn.first", "warn.second", "warn.third", "warn.fourth"])

    def test_recommendations_deduped_by_flag(self):
        first = Recommendation("Use a smaller quantization.", "first", Impact.HIGH)
        duplicate = Recommendation("Use a smaller quantization.", "second", Impact.HIGH)
        flagged = Recommendation("Reduce context.", "flagged", Impact.MEDIUM, "-c")
        report = run_analyzers(make_context(), (
            AnalyzerSpec("one", "One", lambda _: [Finding("one", "one.warn", Severity.WARN, "One", "One",
                                                           recommendations=(first, flagged))]),
            AnalyzerSpec("two", "Two", lambda _: [Finding("two", "two.warn", Severity.WARN, "Two", "Two",
                                                           recommendations=(duplicate,))]),
        ))
        recommendations = report.recommendations()
        self.assertEqual(recommendations, [first, flagged])

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

    def test_custom_registry(self):
        registry = (BUILTIN_ANALYZERS[0],)
        report = diagnose(make_context(), registry)
        self.assertEqual([finding.code for finding in report.findings], ["connectivity.ok"])


if __name__ == "__main__":
    unittest.main()
