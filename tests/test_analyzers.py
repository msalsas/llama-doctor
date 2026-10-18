import math
import unittest
from dataclasses import replace

from llama_doctor.analysis import BUILTIN_ANALYZERS
from llama_doctor.analysis.base import DiagnosisContext, HostHints, Severity
from llama_doctor.models import EndpointError, ProbeResult, PropsSnapshot, ServerSnapshot

from .support import make_context


def findings(context: DiagnosisContext, analyzer_id: str):
    return next(spec.func(context) for spec in BUILTIN_ANALYZERS if spec.id == analyzer_id)


class TestAnalyzers(unittest.TestCase):
    def assert_finding(self, finding, code, severity, evidence):
        self.assertEqual((finding.code, finding.severity), (code, severity))
        self.assertEqual(finding.evidence, evidence)

    def test_connectivity_ok(self):
        self.assert_finding(findings(make_context(), "connectivity")[0], "connectivity.ok", Severity.OK, {})

    def test_connectivity_unreachable(self):
        errors = {path: EndpointError(path, None, "down") for path in ("/metrics", "/props", "/v1/models")}
        finding = findings(make_context(None, None, None, errors=errors), "connectivity")[0]
        self.assert_finding(finding, "connectivity.unreachable", Severity.CRITICAL, {})
        self.assertEqual(finding.recommendations[0].impact, 3)

    def test_connectivity_metrics_disabled_501(self):
        finding = findings(make_context(None, "props_full.json", "models_full.json",
                                        errors={"/metrics": EndpointError("/metrics", 501, "unsupported")}),
                           "connectivity")[0]
        self.assert_finding(finding, "connectivity.metrics_disabled", Severity.WARN, {})
        self.assertEqual((finding.recommendations[0].impact, finding.recommendations[0].flag), (3, "--metrics"))

    def test_context_props_are_backfilled_server_props(self):
        context = make_context("metrics_minimal.txt", "props_unsloth_minimal.json", "models_full.json")
        self.assertEqual(context.props, context.snapshot.server_props)
        self.assertEqual(context.props.quant, "UD-Q4_K_XL")
        self.assertIsNone(context.snapshot.props.quant)

    def test_connectivity_props_unavailable(self):
        finding = findings(make_context(props=None), "connectivity")[0]
        self.assert_finding(finding, "connectivity.props_unavailable", Severity.INFO, {})

    def test_prefill_moe_cpu_experts(self):
        finding = findings(make_context(), "prefill")[0]
        self.assert_finding(finding, "prefill.moe_cpu_experts", Severity.CRITICAL,
                            {"prefill_tps": 40.0, "decode_tps": 7.0, "source": "lifetime"})
        self.assertEqual([item.impact for item in finding.recommendations], [3, 3, 2, 1])
        self.assertEqual([item.flag for item in finding.recommendations],
                         [None, "--ple-on-cpu", "--draft-max", "-c"])

    def test_prefill_cpu_bound_dense(self):
        props = PropsSnapshot.from_payload({"model_path": "Llama-3.1-8B-Q4_K_M.gguf"})
        context = make_context()
        finding = findings(replace(context, snapshot=replace(context.snapshot, props=props)), "prefill")[0]
        self.assert_finding(finding, "prefill.cpu_bound", Severity.WARN,
                            {"prefill_tps": 40.0, "decode_tps": 7.0, "source": "lifetime"})
        self.assertEqual([(item.impact, item.flag) for item in finding.recommendations],
                         [(3, "-ngl"), (3, None), (1, "-c")])

    def test_prefill_ok_healthy(self):
        finding = findings(make_context("metrics_healthy.txt"), "prefill")[0]
        self.assert_finding(finding, "prefill.ok", Severity.OK,
                            {"prefill_tps": 2000.0, "decode_tps": 50.0, "source": "lifetime"})

    def test_prefill_insufficient_minimal(self):
        finding = findings(make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None), "prefill")[0]
        self.assert_finding(finding, "prefill.insufficient_data", Severity.INSUFFICIENT_DATA,
                            {"prompt_tokens_total": 600.0})

    def test_prefill_slow(self):
        finding = findings(replace(make_context(), probe=ProbeResult(250, 20, 1200, 4, 0)), "prefill")[0]
        self.assert_finding(finding, "prefill.slow", Severity.INFO,
                            {"prefill_tps": 250, "decode_tps": 20, "source": "probe"})
        self.assertEqual((finding.recommendations[0].impact, finding.recommendations[0].flag), (2, "--ubatch-size"))

    def test_decode_slow_is_symptom(self):
        finding = findings(make_context(), "decode")[0]
        self.assert_finding(finding, "decode.slow", Severity.WARN,
                            {"decode_tps": 7.0, "source": "lifetime", "cpu_bound": True})
        self.assertEqual(finding.recommendations, ())

    def test_decode_slow_offload_recommendation(self):
        finding = findings(replace(make_context(), probe=ProbeResult(500, 7, 1200, 4, 0)), "decode")[0]
        self.assert_finding(finding, "decode.slow", Severity.WARN,
                            {"decode_tps": 7, "source": "probe", "cpu_bound": False})
        self.assertEqual((finding.recommendations[0].impact, finding.recommendations[0].flag), (2, "-ngl"))

    def test_decode_ok_minimal(self):
        finding = findings(make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None), "decode")[0]
        self.assert_finding(finding, "decode.ok", Severity.OK,
                            {"decode_tps": 30.0, "source": "lifetime", "cpu_bound": None})

    def test_prefix_cache_low_moe(self):
        finding = findings(make_context(), "prefix_cache")[0]
        self.assertEqual((finding.code, finding.severity), ("prefix_cache.low_hit_ratio", Severity.WARN))
        self.assertAlmostEqual(finding.evidence["hit_ratio"], 0.1111111, places=6)
        self.assertEqual([(item.impact, item.flag) for item in finding.recommendations], [(3, None), (2, "--cache-reuse")])

    def test_prefix_cache_ok_healthy(self):
        finding = findings(make_context("metrics_healthy.txt"), "prefix_cache")[0]
        self.assertEqual((finding.code, finding.severity), ("prefix_cache.ok", Severity.OK))
        self.assertAlmostEqual(finding.evidence["hit_ratio"], 0.75, places=6)

    def test_prefix_cache_insufficient_minimal(self):
        finding = findings(make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None), "prefix_cache")[0]
        self.assert_finding(finding, "prefix_cache.insufficient_data", Severity.INSUFFICIENT_DATA, {})

    def test_kv_context_oversized_moe(self):
        finding = findings(make_context(), "kv")[0]
        self.assertEqual((finding.code, finding.severity), ("kv.context_oversized", Severity.WARN))
        self.assertAlmostEqual(finding.evidence["peak_ratio"], 0.1875, places=6)
        self.assertEqual((finding.evidence["suggested_ctx"], finding.recommendations[0].impact,
                          finding.recommendations[0].flag), (12288, 3, "-c"))

    def test_kv_ok_healthy(self):
        self.assert_finding(findings(make_context("metrics_healthy.txt"), "kv")[0], "kv.ok", Severity.OK, {})

    def test_kv_insufficient_minimal(self):
        self.assert_finding(findings(make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None), "kv")[0],
                            "kv.insufficient_data", Severity.INSUFFICIENT_DATA, {})

    def test_kv_saturated(self):
        context = make_context()
        finding = findings(replace(context, snapshot=replace(context.snapshot,
                            metrics=replace(context.metrics, requests_deferred=3))), "kv")[0]
        self.assertEqual((finding.code, finding.severity), ("kv.saturated", Severity.WARN))
        self.assertEqual((finding.recommendations[0].impact, finding.recommendations[0].flag), (2, "--parallel"))

    def test_kv_non_finite_peak_is_insufficient(self):
        context = make_context()
        finding = findings(replace(context, snapshot=replace(context.snapshot,
                            metrics=replace(context.metrics, n_tokens_max=math.inf))), "kv")[0]
        self.assert_finding(finding, "kv.insufficient_data", Severity.INSUFFICIENT_DATA, {})

    def test_spec_drafts_too_long_offloaded_moe(self):
        finding = findings(make_context(), "spec")[0]
        self.assertEqual((finding.code, finding.severity), ("spec.drafts_too_long_offloaded", Severity.WARN))
        self.assertAlmostEqual(finding.evidence["avg_draft_len"], 3.0, places=6)
        self.assertAlmostEqual(finding.evidence["acceptance"], 0.4, places=6)
        self.assertEqual((finding.recommendations[0].impact, finding.recommendations[0].flag), (2, "--draft-max"))

    def test_spec_raise_draft_healthy(self):
        finding = findings(make_context("metrics_healthy.txt"), "spec")[0]
        self.assertEqual((finding.code, finding.severity), ("spec.raise_draft", Severity.INFO))
        self.assertAlmostEqual(finding.evidence["avg_draft_len"], 2.0, places=6)
        self.assertAlmostEqual(finding.evidence["acceptance"], 0.8, places=6)
        self.assertEqual((finding.recommendations[0].impact, finding.recommendations[0].flag), (2, "--draft-max"))

    def test_spec_low_acceptance(self):
        context = make_context()
        finding = findings(replace(context, snapshot=replace(context.snapshot, metrics=replace(
            context.metrics, draft_tokens_accepted_total=2000, draft_tokens_total=9000))), "spec")[0]
        self.assertEqual((finding.code, finding.severity), ("spec.low_acceptance", Severity.WARN))
        self.assertAlmostEqual(finding.evidence["acceptance"], 2000 / 9000, places=6)
        self.assertEqual((finding.recommendations[0].impact, finding.recommendations[0].flag), (2, "--draft-max"))

    def test_spec_insufficient_minimal(self):
        self.assert_finding(findings(make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None), "spec")[0],
                            "spec.insufficient_data", Severity.INSUFFICIENT_DATA, {})

    def test_spec_inactive(self):
        context = make_context()
        finding = findings(replace(context, snapshot=replace(context.snapshot,
                            metrics=replace(context.metrics, drafts_total=0))), "spec")[0]
        self.assert_finding(finding, "spec.inactive", Severity.INFO, {})

    def test_memory_partial_offload(self):
        finding = findings(replace(make_context(), hints=HostHints(8, 32)), "memory")[0]
        self.assertEqual((finding.code, finding.severity), ("memory.partial_offload", Severity.INFO))
        self.assertAlmostEqual(finding.evidence["offloaded_gib"], 8.5, places=6)

    def test_memory_exceeds_total(self):
        finding = findings(replace(make_context(), hints=HostHints(8, 4)), "memory")[0]
        self.assert_finding(finding, "memory.exceeds_total", Severity.CRITICAL, {})
        self.assertEqual(finding.recommendations[0].impact, 3)

    def test_memory_fits(self):
        self.assert_finding(findings(replace(make_context(), hints=HostHints(24)), "memory")[0],
                            "memory.fits_vram", Severity.OK, {})

    def test_memory_insufficient_without_hints(self):
        self.assert_finding(findings(make_context(), "memory")[0], "memory.insufficient_data",
                            Severity.INSUFFICIENT_DATA, {})

    def test_memory_rejects_invalid_hints(self):
        for hints in (HostHints(-1, 32), HostHints(math.nan, 32), HostHints(8, -1), HostHints(8, math.nan)):
            self.assertEqual(findings(replace(make_context(), hints=hints), "memory")[0].code,
                             "memory.insufficient_data" if hints.vram_gib != 8 else "memory.partial_offload")

    def test_props_minimal_unsloth(self):
        finding = findings(make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None), "props")[0]
        self.assert_finding(finding, "props.minimal", Severity.INSUFFICIENT_DATA,
                            {"missing": "n_ctx,n_gpu_layers,n_threads,cache_type_k,cache_type_v,modalities"})

    def test_props_collects_minimal_and_ctx_exceeds_training(self):
        props = PropsSnapshot(n_ctx=32768, missing=("modalities",))
        context = make_context()
        context = replace(context, snapshot=replace(context.snapshot, props=props,
                                                     models=replace(context.models, n_ctx_train=16384)))
        results = findings(context, "props")
        self.assertEqual([item.code for item in results], ["props.minimal", "props.ctx_exceeds_training"])
        self.assertEqual(results[1].severity, Severity.WARN)

    def test_props_ctx_exceeds_training(self):
        context = make_context()
        finding = findings(replace(context, snapshot=replace(context.snapshot,
                            models=replace(context.models, n_ctx_train=16384))), "props")[0]
        self.assert_finding(finding, "props.ctx_exceeds_training", Severity.WARN, {})
        self.assertEqual((finding.recommendations[0].impact, finding.recommendations[0].flag), (2, "-c"))

    def test_props_ok_full(self):
        self.assert_finding(findings(make_context(), "props")[0], "props.ok", Severity.OK, {})

    def _burst_context(self, *stages: str):
        """Context whose window holds real scrapes of a 4-request burst on one slot; last is current."""
        window = tuple(replace(make_context(f"queue_burst/{stage}.txt").metrics, taken_at=float(index))
                       for index, stage in enumerate(stages))
        context = make_context(f"queue_burst/{stages[-1]}.txt")
        return replace(context, snapshot=replace(context.snapshot, metrics=window[-1]), window=window)

    def test_queue_one_shot_uses_current_gauge(self):
        finding = findings(make_context("queue_burst/1_during.txt"), "queue")[0]
        self.assert_finding(finding, "queue.deferred", Severity.INFO,
                            {"deferred_peak": 3.0, "requests_deferred": 3.0, "requests_processing": 1.0})
        self.assertEqual(finding.title, "Requests are queued; all slots busy")
        self.assert_finding(findings(make_context("queue_burst/3_after.txt"), "queue")[0],
                            "queue.ok", Severity.OK, {"deferred_peak": 0.0})

    def test_queue_fires_on_window_peak_after_queue_drains(self):
        finding = findings(self._burst_context("0_base", "1_during", "2_during", "3_after"), "queue")[0]
        self.assert_finding(finding, "queue.deferred", Severity.INFO,
                            {"deferred_peak": 3.0, "requests_deferred": 0.0,
                             "requests_processing": 0.0, "window_s": 3.0})

    def test_queue_follows_gauge_down_inside_window(self):
        finding = findings(self._burst_context("1_during", "2_during"), "queue")[0]
        self.assertEqual((finding.evidence["deferred_peak"], finding.evidence["requests_deferred"]), (3.0, 2.0))
        finding = findings(self._burst_context("2_during", "3_after"), "queue")[0]
        self.assertEqual(finding.evidence["deferred_peak"], 2.0)

    def test_queue_stops_firing_once_burst_leaves_window(self):
        self.assert_finding(findings(self._burst_context("3_after", "0_base", "3_after"), "queue")[0],
                            "queue.ok", Severity.OK, {"deferred_peak": 0.0})

    def test_queue_missing_gauge_is_insufficient(self):
        self.assert_finding(findings(make_context("metrics_minimal.txt"), "queue")[0],
                            "queue.insufficient_data", Severity.INSUFFICIENT_DATA, {})
        self.assert_finding(findings(make_context(None), "queue")[0],
                            "queue.insufficient_data", Severity.INSUFFICIENT_DATA, {})

    def test_kv_saturated_tracks_queue_peak(self):
        for context in (make_context("queue_burst/1_during.txt"),
                        self._burst_context("1_during", "2_during", "3_after")):
            self.assertEqual(findings(context, "kv")[0].code, "kv.saturated")
        for context in (make_context("queue_burst/3_after.txt"),
                        self._burst_context("0_base", "3_after")):
            self.assertNotIn("kv.saturated", [item.code for item in findings(context, "kv")])

    def test_no_analyzer_raises_on_empty_context(self):
        empty = DiagnosisContext(ServerSnapshot(0, None, None, None, {}))
        for spec in BUILTIN_ANALYZERS:
            with self.subTest(spec=spec.id):
                self.assertTrue(spec.func(empty))


if __name__ == "__main__":
    unittest.main()
