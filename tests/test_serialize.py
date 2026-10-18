import json
import unittest

from llama_doctor.analysis import diagnose
from llama_doctor.analysis.base import Finding, Report, Severity
from llama_doctor.models import EndpointError, MetricsSnapshot
from llama_doctor.serialize import (
    context_to_dict,
    errors_to_dict,
    finding_to_dict,
    format_tokens,
    props_to_dict,
    rates_to_dict,
    report_to_dict,
    severity_name,
)

from .support import make_context


class TestSerialize(unittest.TestCase):
    def test_format_tokens_k_and_m(self):
        for value, expected in ((0, "0"), (999, "999"), (1000, "1.0k"), (78501, "78.5k"),
                                (80128, "80.1k"), (999949, "999.9k"), (999950, "1.0M"),
                                (1048576, "1.0M"), (12_345_678, "12.3M")):
            with self.subTest(value=value):
                self.assertEqual(format_tokens(value), expected)

    def test_format_tokens_missing_is_none(self):
        self.assertIsNone(format_tokens(None))
        self.assertIsNone(format_tokens(float("nan")))

    def test_context_to_dict_fixture_80k(self):
        context = make_context("metrics_context_near_limit.txt", "props_context_80k.json")
        result = context_to_dict(context.metrics, context.props)
        self.assertEqual(set(result), {"used_ratio", "n_tokens_max", "n_ctx", "label"})
        self.assertAlmostEqual(result["used_ratio"], 78501 / 80128, places=9)
        self.assertEqual(round(result["used_ratio"] * 100, 1), 98.0)
        self.assertEqual((result["n_tokens_max"], result["n_ctx"], result["label"]),
                         (78501, 80128, "78.5k / 80.1k"))

    def test_context_to_dict_without_n_ctx(self):
        context = make_context("metrics_context_near_limit.txt", "props_unsloth_minimal.json")
        self.assertEqual(context_to_dict(context.metrics, context.props),
                         {"used_ratio": None, "n_tokens_max": 78501, "n_ctx": None, "label": None})
        self.assertEqual(context_to_dict(None, None),
                         {"used_ratio": None, "n_tokens_max": None, "n_ctx": None, "label": None})

    def test_report_json_roundtrip(self):
        for context in (
            make_context(),
            make_context("metrics_healthy.txt"),
            make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None),
        ):
            json.dumps(report_to_dict(diagnose(context)), allow_nan=False)

    def test_non_finite_values_are_sanitized_for_json(self):
        finding = Finding("test", "test.value", Severity.OK, "Test", "Test",
                          {"infinite": float("inf"), "nan": float("nan")})
        self.assertEqual(finding_to_dict(finding)["evidence"], {"infinite": None, "nan": None})
        rates = rates_to_dict(MetricsSnapshot(prompt_tokens_total=float("inf"), prompt_seconds_total=1))
        self.assertIsNone(rates["prefill_tps"])
        json.dumps({"finding": finding_to_dict(finding), "rates": rates}, allow_nan=False)

    def test_evidence_floats_keep_full_precision(self):
        finding = Finding("test", "test.value", Severity.OK, "Test", "Test",
                          {"prefill_tps": 132.60928071193192, "decode_tps": 24.591443296157937,
                           "hit_ratio": 0.1111111, "cpu_bound": True, "suggested_ctx": 8192,
                           "source": "lifetime", "missing": None})
        self.assertEqual(finding_to_dict(finding)["evidence"], {
            "prefill_tps": 132.60928071193192, "decode_tps": 24.591443296157937, "hit_ratio": 0.1111111,
            "cpu_bound": True, "suggested_ctx": 8192, "source": "lifetime", "missing": None,
        })

    def test_nested_non_finite_evidence_is_sanitized(self):
        finding = Finding("test", "test.value", Severity.OK, "Test", "Test",
                          {"samples": [1.5, float("nan")], "window": {"peak": float("-inf"), "mean": 2.25}})
        self.assertEqual(finding_to_dict(finding)["evidence"],
                         {"samples": [1.5, None], "window": {"peak": None, "mean": 2.25}})

    def test_report_keeps_full_precision_and_sanitizes(self):
        finding = Finding("test", "test.value", Severity.WARN, "Test", "Test",
                          {"prefill_tps": 132.60928071193192, "nan": float("nan")})
        evidence = report_to_dict(Report((finding,)))["findings"][0]["evidence"]
        self.assertEqual(evidence, {"prefill_tps": 132.60928071193192, "nan": None})

    def test_rates_keep_full_precision(self):
        rates = rates_to_dict(MetricsSnapshot(prompt_tokens_total=1000, prompt_seconds_total=3))
        self.assertAlmostEqual(rates["prefill_tps"], 1000 / 3, places=9)

    def test_severity_names(self):
        self.assertEqual(severity_name(Severity.INSUFFICIENT_DATA), "insufficient_data")
        self.assertEqual(severity_name(Severity.OK), "ok")
        self.assertEqual(severity_name(Severity.INFO), "info")
        self.assertEqual(severity_name(Severity.WARN), "warn")
        self.assertEqual(severity_name(Severity.CRITICAL), "critical")

    def test_props_dict_has_no_sentinel(self):
        for name in ("props_full.json", "props_unsloth_minimal.json"):
            self.assertNotIn("SENTINEL", repr(props_to_dict(make_context(props=name).props)))

    def test_serializer_key_sets_and_tuple_conversion(self):
        report = report_to_dict(diagnose(make_context()))
        self.assertEqual(set(report), {"overall", "exit_code", "findings", "recommendations"})
        finding = report["findings"][0]
        self.assertEqual(set(finding), {"analyzer", "code", "severity", "title", "detail", "evidence", "recommendations"})
        recommendation = finding["recommendations"][0]
        self.assertEqual(set(recommendation), {"action", "rationale", "impact", "flag"})
        self.assertIn(recommendation["impact"], {"low", "medium", "high"})
        props = props_to_dict(make_context().props)
        self.assertIsInstance(props["missing"], list)
        self.assertEqual(set(rates_to_dict(make_context().metrics)), {
            "prefill_tps", "decode_tps", "cache_hit_ratio", "avg_draft_len", "draft_acceptance",
            "kv_usage_ratio", "n_tokens_max",
        })

    def test_rates_none_when_no_metrics(self):
        self.assertTrue(all(value is None for value in rates_to_dict(None).values()))

    def test_errors_to_dict(self):
        self.assertEqual(errors_to_dict({"x": EndpointError("x", 500, "failed")}), {"x": "failed"})


if __name__ == "__main__":
    unittest.main()
