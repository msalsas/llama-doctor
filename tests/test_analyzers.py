import unittest
from dataclasses import replace

from llama_doctor.analysis import BUILTIN_ANALYZERS
from llama_doctor.analysis.base import DiagnosisContext, HostHints, Severity
from llama_doctor.models import EndpointError, ProbeResult, PropsSnapshot, ServerSnapshot

from .support import make_context


def findings(context: DiagnosisContext, analyzer_id: str):
    return next(spec.func(context) for spec in BUILTIN_ANALYZERS if spec.id == analyzer_id)


class TestAnalyzers(unittest.TestCase):
    def test_connectivity_paths(self):
        self.assertEqual(findings(make_context(), "connectivity")[0].code, "connectivity.ok")
        errors = {
            "/metrics": EndpointError("/metrics", None, "down"),
            "/props": EndpointError("/props", None, "down"),
            "/v1/models": EndpointError("/v1/models", None, "down"),
        }
        unreachable = findings(make_context(None, None, None, errors=errors), "connectivity")[0]
        self.assertEqual((unreachable.code, unreachable.severity), ("connectivity.unreachable", Severity.CRITICAL))
        disabled = findings(
            make_context(None, "props_full.json", "models_full.json",
                         errors={"/metrics": EndpointError("/metrics", 501, "unsupported")}),
            "connectivity",
        )[0]
        self.assertEqual((disabled.code, disabled.recommendations[0].flag),
                         ("connectivity.metrics_disabled", "--metrics"))
        props = findings(make_context(props=None), "connectivity")
        self.assertEqual(props[0].code, "connectivity.props_unavailable")

    def test_prefill_paths(self):
        moe = findings(make_context(), "prefill")[0]
        self.assertEqual((moe.code, moe.severity), ("prefill.moe_cpu_experts", Severity.CRITICAL))
        self.assertAlmostEqual(moe.evidence["prefill_tps"], 40.0, places=6)
        self.assertAlmostEqual(moe.evidence["decode_tps"], 7.0, places=6)
        self.assertEqual([item.flag for item in moe.recommendations],
                         [None, "--ple-on-cpu", "--draft-max", "-c"])
        dense_props = PropsSnapshot.from_payload({"model_path": "Llama-3.1-8B-Q4_K_M.gguf"})
        dense = replace(make_context(), snapshot=replace(make_context().snapshot, props=dense_props))
        self.assertEqual(findings(dense, "prefill")[0].code, "prefill.cpu_bound")
        self.assertEqual(findings(make_context("metrics_healthy.txt"), "prefill")[0].code, "prefill.ok")
        minimal = make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None)
        self.assertEqual(findings(minimal, "prefill")[0].code, "prefill.insufficient_data")
        probe = replace(make_context(), probe=ProbeResult(250, 20, 1200, 4, 0))
        self.assertEqual(findings(probe, "prefill")[0].code, "prefill.slow")

    def test_decode_and_prefix_cache_paths(self):
        slow = findings(make_context(), "decode")[0]
        self.assertEqual((slow.code, slow.recommendations), ("decode.slow", ()))
        minimal = make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None)
        self.assertEqual(findings(minimal, "decode")[0].code, "decode.ok")
        self.assertEqual(findings(make_context(), "prefix_cache")[0].code, "prefix_cache.low_hit_ratio")
        healthy = findings(make_context("metrics_healthy.txt"), "prefix_cache")[0]
        self.assertEqual((healthy.code, healthy.evidence["hit_ratio"]), ("prefix_cache.ok", 0.75))
        self.assertEqual(findings(minimal, "prefix_cache")[0].code, "prefix_cache.insufficient_data")

    def test_kv_paths(self):
        context = findings(make_context(), "kv")[0]
        self.assertEqual((context.code, context.severity), ("kv.context_oversized", Severity.WARN))
        self.assertAlmostEqual(context.evidence["peak_ratio"], 0.1875, places=6)
        self.assertEqual((context.evidence["suggested_ctx"], context.recommendations[0].impact), (12288, 3))
        self.assertEqual(findings(make_context("metrics_healthy.txt"), "kv")[0].code, "kv.ok")
        minimal = make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None)
        self.assertEqual(findings(minimal, "kv")[0].code, "kv.insufficient_data")
        saturated = replace(make_context(), snapshot=replace(make_context().snapshot,
                            metrics=replace(make_context().metrics, requests_deferred=3)))
        self.assertEqual(findings(saturated, "kv")[0].code, "kv.saturated")

    def test_spec_paths(self):
        moe = findings(make_context(), "spec")[0]
        self.assertEqual((moe.code, moe.evidence["avg_draft_len"], moe.evidence["acceptance"]),
                         ("spec.drafts_too_long_offloaded", 3.0, 0.4))
        healthy = findings(make_context("metrics_healthy.txt"), "spec")[0]
        self.assertEqual((healthy.code, healthy.evidence["avg_draft_len"]), ("spec.raise_draft", 2.0))
        low = replace(make_context(), snapshot=replace(make_context().snapshot,
                      metrics=replace(make_context().metrics, draft_tokens_accepted_total=2000,
                                      draft_tokens_total=9000)))
        self.assertEqual(findings(low, "spec")[0].code, "spec.low_acceptance")
        minimal = make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None)
        self.assertEqual(findings(minimal, "spec")[0].code, "spec.insufficient_data")
        inactive = replace(make_context(), snapshot=replace(make_context().snapshot,
                           metrics=replace(make_context().metrics, drafts_total=0)))
        self.assertEqual(findings(inactive, "spec")[0].code, "spec.inactive")

    def test_memory_props_and_empty_context(self):
        partial = replace(make_context(), hints=HostHints(8, 32))
        finding = findings(partial, "memory")[0]
        self.assertEqual(finding.code, "memory.partial_offload")
        self.assertAlmostEqual(finding.evidence["offloaded_gib"], 8.5, places=6)
        self.assertEqual(findings(replace(make_context(), hints=HostHints(8, 4)), "memory")[0].code,
                         "memory.exceeds_total")
        self.assertEqual(findings(replace(make_context(), hints=HostHints(24)), "memory")[0].code,
                         "memory.fits_vram")
        self.assertEqual(findings(make_context(), "memory")[0].code, "memory.insufficient_data")
        minimal = make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None)
        props = findings(minimal, "props")[0]
        self.assertEqual((props.code, props.evidence["missing"]), ("props.minimal", ",".join(minimal.props.missing)))
        oversized = replace(make_context(), snapshot=replace(make_context().snapshot,
                            models=replace(make_context().models, n_ctx_train=16384)))
        self.assertEqual(findings(oversized, "props")[0].code, "props.ctx_exceeds_training")
        self.assertEqual(findings(make_context(), "props")[0].code, "props.ok")
        empty = DiagnosisContext(ServerSnapshot(0, None, None, None, {}))
        for spec in BUILTIN_ANALYZERS:
            with self.subTest(spec=spec.id):
                self.assertTrue(spec.func(empty))


if __name__ == "__main__":
    unittest.main()
