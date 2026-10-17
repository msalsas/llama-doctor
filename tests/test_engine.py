import unittest

from llama_doctor.analysis import BUILTIN_ANALYZERS, diagnose
from llama_doctor.analysis.base import AnalyzerSpec, Severity
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
        report = diagnose(make_context())
        self.assertEqual(report.findings[0].code, "prefill.moe_cpu_experts")
        self.assertEqual(
            [item.severity for item in report.findings],
            sorted((item.severity for item in report.findings), reverse=True),
        )

    def test_recommendations_deduped_by_flag(self):
        recommendations = diagnose(make_context()).recommendations()
        flags = [item.flag for item in recommendations]
        self.assertEqual(flags.count("-c"), 1)
        self.assertEqual(flags.count("--draft-max"), 1)
        self.assertEqual(next(item.impact for item in recommendations if item.flag == "-c"), 3)
        self.assertEqual([item.impact for item in recommendations], sorted((item.impact for item in recommendations), reverse=True))

    def test_analyzer_exception_isolated(self):
        def raises(_):
            raise ValueError("SENTINEL")

        report = run_analyzers(make_context(), (AnalyzerSpec("bad", "Bad", raises),))
        finding = report.by_code("bad.error")
        self.assertEqual((finding.severity, finding.detail), (Severity.INSUFFICIENT_DATA, "ValueError"))
        self.assertNotIn("SENTINEL", finding.detail)

    def test_custom_registry(self):
        registry = (BUILTIN_ANALYZERS[0],)
        self.assertEqual(len(diagnose(make_context(), registry).findings), 1)


if __name__ == "__main__":
    unittest.main()
