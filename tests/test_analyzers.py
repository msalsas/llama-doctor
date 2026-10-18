import math
import unittest
from dataclasses import replace
from pathlib import Path

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
                         [None, "--n-cpu-moe", "--draft-max", "-c"])
        experts = finding.recommendations[1]
        for phrase in ("-ngl 999", "--cpu-moe", "--n-cpu-moe N"):
            self.assertIn(phrase, experts.action)

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

    def test_no_recommendation_names_a_nonexistent_ple_flag(self):
        root = Path(__file__).resolve().parent.parent
        paths = [*(root / "src").rglob("*.py"), root / "README.md", *(root / "docs").rglob("*.md")]
        for path in paths:
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("--ple-on-cpu", text)
                self.assertNotIn("PLE/N-gram", text)

    def test_prefill_slow(self):
        finding = findings(replace(make_context(), probe=ProbeResult(250, 20, 1200, 4, 0)), "prefill")[0]
        self.assert_finding(finding, "prefill.slow", Severity.WARN,
                            {"prefill_tps": 250, "decode_tps": 20, "source": "probe", "cache_hit_ratio": None})
        for phrase in ("300 tok/s", "decode", "12.5x"):
            self.assertIn(phrase, finding.detail)
        self.assertEqual([item.flag for item in finding.recommendations], ["-ngl", "--n-cpu-moe"])
        self.assertIn("-ngl 999", finding.recommendations[0].action)
        for phrase in ("--cpu-moe", "--n-cpu-moe N"):
            self.assertIn(phrase, finding.recommendations[1].action)

    def test_prefill_slow_dense_only_checks_gpu_layers(self):
        props = PropsSnapshot.from_payload({"model_path": "Llama-3.1-8B-Q4_K_M.gguf"})
        context = replace(make_context(), probe=ProbeResult(250, 20, 1200, 4, 0))
        finding = findings(replace(context, snapshot=replace(context.snapshot, props=props)), "prefill")[0]
        self.assertEqual(finding.code, "prefill.slow")
        self.assertEqual([item.flag for item in finding.recommendations], ["-ngl"])

    def _lifetime_prefill(self, prompt_tps, cached_total):
        context = make_context("metrics_healthy.txt")
        metrics = replace(context.metrics, prompt_tokens_total=10000, prompt_seconds_total=10000 / prompt_tps,
                          tokens_predicted_total=2000, tokens_predicted_seconds_total=100,
                          prompt_tokens_cached_total=cached_total)
        return replace(context, snapshot=replace(context.snapshot, metrics=metrics))

    def test_prefill_slow_with_low_cache_hit_warns(self):
        finding = findings(self._lifetime_prefill(135.0, 10000 * 0.49 / 0.51), "prefill")[0]
        self.assertEqual((finding.code, finding.severity), ("prefill.slow", Severity.WARN))
        self.assertAlmostEqual(finding.evidence["cache_hit_ratio"], 0.49, places=6)

    def test_prefill_slow_without_cache_hit_data_warns(self):
        finding = findings(self._lifetime_prefill(135.0, None), "prefill")[0]
        self.assertEqual((finding.code, finding.severity), ("prefill.slow", Severity.WARN))
        self.assertIsNone(finding.evidence["cache_hit_ratio"])

    def _with_window(self, context, prompt_steps):
        """Window whose prompt_tokens_total climbs by ``prompt_steps``; the last sample is current."""
        current = context.metrics
        totals = [current.prompt_tokens_total - sum(prompt_steps)]
        for step in prompt_steps:
            totals.append(totals[-1] + step)
        window = tuple(replace(current, prompt_tokens_total=total, taken_at=float(index))
                       for index, total in enumerate(totals))
        return replace(context, window=window)

    def test_prefill_high_cache_hit_without_window_is_insufficient(self):
        finding = findings(self._lifetime_prefill(135.0, 290000), "prefill")[0]
        self.assertEqual((finding.code, finding.severity), ("prefill.insufficient_data", Severity.INSUFFICIENT_DATA))
        self.assertAlmostEqual(finding.evidence["cache_hit_ratio"], 0.966666, places=5)
        self.assertIsNone(finding.evidence["mean_prompt_tokens"])
        self.assertEqual(finding.recommendations, ())
        for phrase in ("prefix cache", "--probe", "serve"):
            self.assertIn(phrase, finding.detail)

    def test_prefill_high_cache_hit_with_short_window_prompts_is_ok(self):
        context = self._with_window(self._lifetime_prefill(135.0, 290000), [0, 300, 0, 500])
        finding = findings(context, "prefill")[0]
        self.assertEqual((finding.code, finding.severity), ("prefill.ok", Severity.OK))
        self.assertEqual(finding.evidence["mean_prompt_tokens"], 400)
        self.assertEqual(finding.recommendations, ())
        for phrase in ("300 tok/s", "prefix cache", "decode", "6.8x", "400"):
            self.assertIn(phrase, finding.detail)

    def test_prefill_high_cache_hit_with_large_window_prompts_is_unreliable(self):
        context = self._with_window(self._lifetime_prefill(135.0, 290000), [200, 6000, 100])
        finding = findings(context, "prefill")[0]
        self.assertEqual((finding.code, finding.severity), ("prefill.rate_unreliable", Severity.INSUFFICIENT_DATA))
        self.assertEqual(finding.evidence["mean_prompt_tokens"], 2100)
        self.assertIn("large prompts", finding.detail)
        self.assertIn("2048", finding.detail)

    def test_prefill_large_prompt_threshold_is_a_module_constant(self):
        from llama_doctor.analysis import prefill
        self.assertEqual(prefill.LARGE_PROMPT_TOKENS, 2048)
        at_limit = self._with_window(self._lifetime_prefill(135.0, 290000), [2048])
        self.assertEqual(findings(at_limit, "prefill")[0].code, "prefill.ok")
        above = self._with_window(self._lifetime_prefill(135.0, 290000), [2049])
        self.assertEqual(findings(above, "prefill")[0].code, "prefill.rate_unreliable")

    def test_prefill_high_cache_hit_with_idle_window_is_insufficient(self):
        context = self._with_window(self._lifetime_prefill(135.0, 290000), [0, 0])
        self.assertEqual(findings(context, "prefill")[0].code, "prefill.insufficient_data")

    def test_prefill_slow_low_cache_hit_ignores_prompt_size(self):
        context = self._with_window(self._lifetime_prefill(135.0, 10000 * 0.2 / 0.8), [6000])
        finding = findings(context, "prefill")[0]
        self.assertEqual((finding.code, finding.severity), ("prefill.slow", Severity.WARN))
        self.assertTrue(finding.recommendations)

    def test_prefill_probe_ignores_cache_hit(self):
        context = replace(self._lifetime_prefill(135.0, 290000), probe=ProbeResult(250, 20, 1200, 4, 0))
        self.assertEqual(findings(context, "prefill")[0].code, "prefill.slow")

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

    def test_context_oversized_moe(self):
        finding = findings(make_context(), "context")[0]
        self.assertEqual((finding.code, finding.severity), ("context.oversized", Severity.WARN))
        self.assertAlmostEqual(finding.evidence["used_ratio"], 0.1875, places=6)
        self.assertEqual((finding.evidence["suggested_ctx"], finding.recommendations[0].impact,
                          finding.recommendations[0].flag), (12288, 3, "-c"))

    def test_context_oversized_needs_large_window(self):
        self.assertEqual(findings(self._context_at(2000, n_ctx=16383), "context")[0].code, "context.ok")
        self.assertEqual(findings(self._context_at(4095, n_ctx=16384), "context")[0].code, "context.oversized")
        self.assertEqual(findings(self._context_at(4096, n_ctx=16384), "context")[0].code, "context.ok")

    def test_kv_only_reports_usage_ratio(self):
        self.assertEqual([item.code for item in findings(make_context(), "kv")], ["kv.insufficient_data"])

    def test_kv_ok_healthy(self):
        finding = findings(make_context("metrics_healthy.txt"), "kv")[0]
        self.assert_finding(finding, "kv.ok", Severity.OK, {})
        self.assertEqual(finding.title, "KV cache usage is well below capacity")
        self.assertNotIn("sizing", f"{finding.title} {finding.detail}".lower())

    def _kv_context(self, kv_ratio):
        context = make_context("metrics_healthy.txt")
        return replace(context, snapshot=replace(context.snapshot,
                       metrics=replace(context.metrics, kv_cache_usage_ratio=kv_ratio)))

    def test_kv_without_usage_ratio_is_insufficient(self):
        self.assert_finding(findings(self._kv_context(None), "kv")[0],
                            "kv.insufficient_data", Severity.INSUFFICIENT_DATA, {})

    def test_kv_non_finite_usage_ratio_is_insufficient(self):
        self.assert_finding(findings(self._kv_context(math.nan), "kv")[0],
                            "kv.insufficient_data", Severity.INSUFFICIENT_DATA, {})

    def test_kv_just_below_saturation_is_ok(self):
        self.assertEqual([item.code for item in findings(self._kv_context(0.89), "kv")], ["kv.ok"])

    def test_kv_saturated_does_not_emit_ok(self):
        self.assertEqual([item.code for item in findings(self._kv_context(0.9), "kv")], ["kv.saturated"])

    def test_kv_insufficient_minimal(self):
        self.assert_finding(findings(make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None), "kv")[0],
                            "kv.insufficient_data", Severity.INSUFFICIENT_DATA, {})

    def test_kv_saturated(self):
        context = make_context()
        finding = findings(replace(context, snapshot=replace(context.snapshot,
                            metrics=replace(context.metrics, kv_cache_usage_ratio=0.95))), "kv")[0]
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

    def _spec_context(self, spec_types, drafts_total=0, tokens_predicted_total=5000):
        context = make_context()
        return replace(context, snapshot=replace(
            context.snapshot,
            metrics=replace(context.metrics, drafts_total=drafts_total,
                            tokens_predicted_total=tokens_predicted_total),
            props=replace(context.snapshot.props, spec_types=spec_types)))

    def test_spec_not_configured_is_neutral_ok(self):
        finding = findings(self._spec_context(()), "spec")[0]
        self.assert_finding(finding, "spec.not_configured", Severity.OK, {})
        self.assertEqual(finding.recommendations, ())

    def test_spec_unknown_types_with_zero_counters_is_insufficient(self):
        finding = findings(self._spec_context(None), "spec")[0]
        self.assert_finding(finding, "spec.insufficient_data", Severity.INSUFFICIENT_DATA, {})
        self.assertIn("speculative.types", finding.detail)

    def test_spec_inactive_warns_only_when_types_configured(self):
        finding = findings(self._spec_context(("draft",)), "spec")[0]
        self.assert_finding(finding, "spec.inactive", Severity.WARN, {"spec_types": "draft"})
        self.assertIn("draft", finding.detail)

    def test_spec_configured_without_generation_is_insufficient(self):
        finding = findings(self._spec_context(("draft",), tokens_predicted_total=0), "spec")[0]
        self.assertEqual(finding.code, "spec.insufficient_data")

    def test_spec_counters_moving_win_over_types_none(self):
        finding = findings(self._spec_context((), drafts_total=1000), "spec")[0]
        self.assertNotIn(finding.code, ("spec.not_configured", "spec.inactive", "spec.insufficient_data"))

    def test_memory_partial_offload(self):
        finding = findings(replace(make_context(), hints=HostHints(8, 32)), "memory")[0]
        self.assertEqual((finding.code, finding.severity), ("memory.partial_offload", Severity.INFO))
        self.assertAlmostEqual(finding.evidence["offloaded_gib"], 8.5, places=6)
        self.assertEqual(finding.evidence["weights_source"], "server")

    def test_memory_exceeds_total(self):
        finding = findings(replace(make_context(), hints=HostHints(8, 4)), "memory")[0]
        self.assert_finding(finding, "memory.exceeds_total", Severity.CRITICAL, {"weights_source": "server"})
        self.assertEqual(finding.recommendations[0].impact, 3)

    def test_memory_fits(self):
        self.assert_finding(findings(replace(make_context(), hints=HostHints(24)), "memory")[0],
                            "memory.fits_vram", Severity.OK, {"weights_source": "server"})

    def test_memory_server_size_wins_over_weights_hint(self):
        finding = findings(replace(make_context(), hints=HostHints(8, 32, weights_gib=4)), "memory")[0]
        self.assertEqual(finding.code, "memory.partial_offload")
        self.assertAlmostEqual(finding.evidence["offloaded_gib"], 8.5, places=6)
        self.assertEqual(finding.evidence["weights_source"], "server")
        self.assertNotIn("--weights-gib", finding.detail)

    def test_memory_uses_weights_hint_without_server_size(self):
        context = make_context(models=None)
        cases = (
            (HostHints(24, weights_gib=16.5), "memory.fits_vram", {"weights_source": "hint"}),
            (HostHints(8, 32, weights_gib=16.5), "memory.partial_offload",
             {"offloaded_gib": 8.5, "weights_source": "hint"}),
            (HostHints(8, 4, weights_gib=16.5), "memory.exceeds_total", {"weights_source": "hint"}),
            (HostHints(8, weights_gib=16.5), "memory.partial_offload",
             {"offloaded_gib": 8.5, "weights_source": "hint"}),
        )
        for hints, code, evidence in cases:
            with self.subTest(hints=hints):
                finding = findings(replace(context, hints=hints), "memory")[0]
                self.assertEqual(finding.code, code)
                self.assertEqual(dict(finding.evidence), evidence)
                self.assertIn("--weights-gib", finding.detail)

    def test_memory_insufficient_when_weights_or_vram_missing(self):
        cases = (
            (make_context(models=None), HostHints(24), "--weights-gib"),
            (make_context(models=None), HostHints(ram_gib=32), "--weights-gib"),
            (make_context(models=None), HostHints(weights_gib=16.5), "--vram-gib"),
            (make_context(), HostHints(weights_gib=16.5), "--vram-gib"),
        )
        for context, hints, flag in cases:
            with self.subTest(hints=hints):
                finding = findings(replace(context, hints=hints), "memory")[0]
                self.assert_finding(finding, "memory.insufficient_data", Severity.INSUFFICIENT_DATA, {})
                self.assertIn(flag, finding.detail)

    def test_memory_rejects_invalid_weights_hint(self):
        for weights in (0, -1, math.nan, math.inf):
            with self.subTest(weights=weights):
                hints = HostHints(24, weights_gib=weights)
                self.assertEqual(findings(replace(make_context(models=None), hints=hints), "memory")[0].code,
                                 "memory.insufficient_data")

    def test_memory_insufficient_without_hints(self):
        self.assert_finding(findings(make_context(), "memory")[0], "memory.insufficient_data",
                            Severity.INSUFFICIENT_DATA, {})

    def test_memory_rejects_invalid_hints(self):
        for hints in (HostHints(-1, 32), HostHints(math.nan, 32), HostHints(8, -1), HostHints(8, math.nan)):
            self.assertEqual(findings(replace(make_context(), hints=hints), "memory")[0].code,
                             "memory.insufficient_data" if hints.vram_gib != 8 else "memory.partial_offload")

    def test_props_missing_fields_blame_nobody(self):
        finding = findings(make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None), "props")[0]
        self.assert_finding(finding, "props.ok", Severity.OK, {})
        self.assertNotIn("unsloth", f"{finding.title} {finding.detail}".lower())

    def test_props_startup_only_fields_are_not_required(self):
        payload = {"default_generation_settings": {"n_ctx": 4096}, "total_slots": 1,
                   "model_path": "/m/Llama-3.1-8B-Q4_K_M.gguf", "build_info": "b11712", "modalities": {}}
        props = PropsSnapshot.from_payload(payload)
        self.assertEqual(props.missing, ())
        context = make_context()
        context = replace(context, snapshot=replace(context.snapshot, props=props))
        self.assertEqual([item.code for item in findings(context, "props")], ["props.ok"])

    def test_props_missing_fields_do_not_mask_ctx_exceeds_training(self):
        props = PropsSnapshot(n_ctx=32768, missing=("modalities",))
        context = make_context()
        context = replace(context, snapshot=replace(context.snapshot, props=props,
                                                     models=replace(context.models, n_ctx_train=16384)))
        results = findings(context, "props")
        self.assertEqual([item.code for item in results], ["props.ctx_exceeds_training"])
        self.assertEqual(results[0].severity, Severity.WARN)

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
        self.assertEqual([item.flag for item in finding.recommendations], [None, "--parallel"])
        self.assertIn("divides", finding.recommendations[1].rationale)
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

    def test_kv_saturated_ignores_queue_peak(self):
        for context in (make_context("queue_burst/1_during.txt"),
                        self._burst_context("1_during", "2_during", "3_after"),
                        make_context("queue_burst/3_after.txt"),
                        self._burst_context("0_base", "3_after")):
            self.assertNotIn("kv.saturated", [item.code for item in findings(context, "kv")])
        context = self._burst_context("0_base", "3_after")
        context = replace(context, snapshot=replace(context.snapshot,
                          metrics=replace(context.metrics, kv_cache_usage_ratio=0.9)))
        self.assertEqual(findings(context, "kv")[0].code, "kv.saturated")

    def test_full_queue_with_low_kv_is_only_queue_deferred(self):
        context = make_context("queue_burst/1_during.txt")
        context = replace(context, snapshot=replace(context.snapshot,
                          metrics=replace(context.metrics, kv_cache_usage_ratio=0.2)))
        codes = [item.code for item in findings(context, "queue") + findings(context, "kv")]
        self.assertIn("queue.deferred", codes)
        self.assertNotIn("kv.saturated", codes)

    def _context_at(self, n_tokens_max, n_ctx=100000):
        context = make_context("metrics_healthy.txt")
        return replace(context, snapshot=replace(
            context.snapshot,
            metrics=replace(context.metrics, n_tokens_max=n_tokens_max),
            props=replace(context.snapshot.props, n_ctx=n_ctx)))

    def test_context_89_percent_is_ok(self):
        self.assert_finding(findings(self._context_at(89000), "context")[0], "context.ok", Severity.OK,
                            {"used_ratio": 0.89, "n_tokens_max": 89000, "n_ctx": 100000})

    def test_context_90_percent_is_near_limit(self):
        finding = findings(self._context_at(90000), "context")[0]
        self.assert_finding(finding, "context.near_limit", Severity.WARN,
                            {"used_ratio": 0.9, "n_tokens_max": 90000, "n_ctx": 100000})
        self.assertEqual(finding.recommendations[0].flag, "-c")

    def test_context_91_percent_is_near_limit(self):
        self.assert_finding(findings(self._context_at(91000), "context")[0], "context.near_limit", Severity.WARN,
                            {"used_ratio": 0.91, "n_tokens_max": 91000, "n_ctx": 100000})

    def test_context_near_limit_fixture_80k(self):
        finding = findings(make_context("metrics_context_near_limit.txt", "props_context_80k.json"), "context")[0]
        self.assertEqual((finding.code, finding.severity), ("context.near_limit", Severity.WARN))
        self.assertEqual((finding.evidence["n_tokens_max"], finding.evidence["n_ctx"]), (78501, 80128))
        self.assertAlmostEqual(finding.evidence["used_ratio"], 78501 / 80128, places=9)

    def test_context_without_n_ctx_does_not_fire(self):
        self.assert_finding(findings(self._context_at(91000, n_ctx=None), "context")[0],
                            "context.insufficient_data", Severity.INSUFFICIENT_DATA, {})

    def test_context_unsloth_minimal_is_insufficient(self):
        context = make_context("metrics_context_near_limit.txt", "props_unsloth_minimal.json", None)
        self.assertEqual(findings(context, "context")[0].code, "context.insufficient_data")

    def test_no_analyzer_raises_on_empty_context(self):
        empty = DiagnosisContext(ServerSnapshot(0, None, None, None, {}))
        for spec in BUILTIN_ANALYZERS:
            with self.subTest(spec=spec.id):
                self.assertTrue(spec.func(empty))


if __name__ == "__main__":
    unittest.main()
