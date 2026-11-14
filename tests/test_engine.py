import math
import unittest
from dataclasses import replace

from llama_doctor.analysis import BUILTIN_ANALYZERS, diagnose
from llama_doctor.analysis.base import (
    AnalyzerSpec,
    Finding,
    HostHints,
    Impact,
    Recommendation,
    Severity,
    requires_gpu,
)
from llama_doctor.analysis.engine import guard_gpu, run_analyzers
from llama_doctor.analysis.signals import GpuEvidence
from llama_doctor.models import ProbeResult, PropsSnapshot

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


class TestGpuGuard(unittest.TestCase):
    def _dense_context(self):
        props = PropsSnapshot.from_payload({"model_path": "Llama-3.1-8B-Q4_K_M.gguf"})
        context = make_context()
        return replace(context, snapshot=replace(context.snapshot, props=props))

    def test_guard_keeps_gpu_advice_with_server_evidence(self):
        finding = diagnose(make_context()).by_code("prefill.moe_cpu_experts")
        self.assertEqual([item.flag for item in finding.recommendations],
                         [None, "--n-cpu-moe", "--draft-max", "-c"])
        self.assertEqual([item.impact for item in finding.recommendations], [3, 3, 2, 1])
        self.assertEqual(finding.detail, "Slow prefill indicates CPU-resident experts.")
        self.assertEqual(dict(finding.evidence), {
            "prefill_tps": 40.0, "decode_tps": 7.0, "source": "lifetime",
            "gpu_status": "gpu", "gpu_source": "props.n_gpu_layers", "gpu_recs_withheld": 0,
        })

    def test_guard_withholds_without_evidence(self):
        report = diagnose(make_context(props="props_unsloth_minimal.json"))
        finding = report.by_code("prefill.moe_cpu_experts")
        self.assertEqual(finding.severity, Severity.CRITICAL)
        self.assertEqual([item.flag for item in finding.recommendations], [None, "--draft-max"])
        self.assertEqual([item.impact for item in finding.recommendations], [3, 2])
        self.assertEqual(finding.evidence["gpu_status"], "unknown")
        self.assertIsNone(finding.evidence["gpu_source"])
        self.assertEqual(finding.evidence["gpu_recs_withheld"], 2)
        self.assertEqual(
            finding.detail,
            "Slow prefill indicates CPU-resident experts. GPU advice withheld: no evidence a GPU "
            "serves this model; pass --vram-gib to include it.",
        )
        self.assertEqual(report.exit_code, 2)

    def test_guard_withholds_on_zero_vram_hint(self):
        report = diagnose(make_context(props="props_unsloth_minimal.json", hints=HostHints(vram_gib=0.0)))
        finding = report.by_code("prefill.moe_cpu_experts")
        self.assertEqual([item.flag for item in finding.recommendations], [None, "--draft-max"])
        self.assertEqual(finding.evidence["gpu_status"], "cpu")
        self.assertEqual(finding.evidence["gpu_source"], "hint.vram_gib")
        self.assertEqual(finding.evidence["gpu_recs_withheld"], 2)
        self.assertTrue(finding.detail.endswith(
            "GPU advice withheld: --vram-gib 0 declares no GPU."))

    def test_guard_keeps_with_vram_hint(self):
        report = diagnose(make_context(props="props_unsloth_minimal.json", hints=HostHints(vram_gib=24.0)))
        finding = report.by_code("prefill.moe_cpu_experts")
        self.assertEqual([item.flag for item in finding.recommendations],
                         [None, "--n-cpu-moe", "--draft-max", "-c"])
        self.assertEqual(finding.evidence["gpu_status"], "gpu")
        self.assertEqual(finding.evidence["gpu_source"], "hint.vram_gib")
        self.assertEqual(finding.evidence["gpu_recs_withheld"], 0)
        self.assertEqual(finding.detail, "Slow prefill indicates CPU-resident experts.")

    def test_guard_dense_cpu_bound(self):
        finding = diagnose(self._dense_context()).by_code("prefill.cpu_bound")
        self.assertEqual(finding.severity, Severity.WARN)
        self.assertEqual([(item.impact, item.flag) for item in finding.recommendations], [(3, None)])
        self.assertEqual(finding.evidence["gpu_status"], "unknown")
        self.assertIsNone(finding.evidence["gpu_source"])
        self.assertEqual(finding.evidence["gpu_recs_withheld"], 2)

    def test_guard_empties_decode_slow(self):
        context = make_context(props="props_unsloth_minimal.json", probe=ProbeResult(500, 7, 1200, 4, 0))
        finding = diagnose(context).by_code("decode.slow")
        self.assertEqual(finding.severity, Severity.WARN)
        self.assertEqual(finding.recommendations, ())
        self.assertEqual(finding.evidence["gpu_recs_withheld"], 1)
        self.assertIs(finding.evidence["cpu_bound"], False)
        self.assertEqual(
            finding.detail,
            "Decode throughput is below 10 tok/s. GPU advice withheld: no evidence a GPU serves "
            "this model; pass --vram-gib to include it.",
        )

    def test_guard_empties_prefill_slow(self):
        context = make_context(props="props_unsloth_minimal.json", probe=ProbeResult(250, 20, 1200, 4, 0))
        finding = diagnose(context).by_code("prefill.slow")
        self.assertEqual(finding.severity, Severity.WARN)
        self.assertEqual(finding.recommendations, ())
        self.assertEqual(finding.evidence["gpu_recs_withheld"], 2)

    def test_report_recommendations_respect_guard(self):
        report = diagnose(make_context(props="props_unsloth_minimal.json"))
        for recommendation in report.recommendations():
            self.assertFalse(requires_gpu(recommendation))
        report = diagnose(make_context())
        flags = {item.flag for item in report.recommendations()}
        self.assertIn("--n-cpu-moe", flags)
        self.assertIn("-c", flags)
        actions = {item.action for item in report.recommendations() if item.flag == "-c"}
        self.assertIn("Reduce context to free VRAM for experts.", actions)

    def test_guard_leaves_neutral_findings_untouched(self):
        report = diagnose(make_context(props="props_unsloth_minimal.json"))
        low = report.by_code("prefix_cache.low_hit_ratio")
        self.assertEqual(set(low.evidence), {"hit_ratio"})
        self.assertIs(guard_gpu(low, GpuEvidence("unknown", None)), low)
        oversized = diagnose(make_context()).by_code("context.oversized")
        self.assertEqual(set(oversized.evidence),
                         {"used_ratio", "n_tokens_max", "n_ctx", "suggested_ctx"})
        self.assertIs(guard_gpu(oversized, GpuEvidence("unknown", None)), oversized)

    def test_sort_uses_guarded_impact(self):
        specs = (
            AnalyzerSpec("a", "A", lambda _: [Finding(
                "a", "a.warn", Severity.WARN, "A", "A",
                recommendations=(Recommendation("A action.", "A.", Impact.HIGH, "-ngl"),))]),
            AnalyzerSpec("b", "B", lambda _: [Finding(
                "b", "b.warn", Severity.WARN, "B", "B",
                recommendations=(Recommendation("B action.", "B.", Impact.MEDIUM),))]),
        )
        context = make_context(props="props_unsloth_minimal.json")
        report = run_analyzers(context, specs)
        self.assertEqual([item.code for item in report.findings], ["b.warn", "a.warn"])
        self.assertEqual(report.by_code("a.warn").max_impact, 0)
        report = run_analyzers(replace(context, hints=HostHints(vram_gib=8.0)), specs)
        self.assertEqual([item.code for item in report.findings], ["a.warn", "b.warn"])

    def test_guard_skips_error_findings(self):
        def raises(_):
            raise ValueError("SENTINEL")

        report = run_analyzers(make_context(props="props_unsloth_minimal.json"),
                               (AnalyzerSpec("bad", "Bad", raises),))
        finding = report.by_code("bad.error")
        self.assertEqual(finding.severity, Severity.INSUFFICIENT_DATA)
        self.assertEqual(finding.detail, "ValueError")
        self.assertEqual(set(finding.evidence) & {"gpu_status", "gpu_source", "gpu_recs_withheld"}, set())


if __name__ == "__main__":
    unittest.main()
