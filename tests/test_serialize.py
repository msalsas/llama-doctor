import json
import unittest

from llama_doctor.analysis import diagnose
from llama_doctor.analysis.base import Finding, Severity
from llama_doctor.models import EndpointError, MetricsSnapshot
from llama_doctor.serialize import (
    errors_to_dict,
    finding_to_dict,
    props_to_dict,
    rates_to_dict,
    report_to_dict,
    severity_name,
)

from .support import make_context


class TestSerialize(unittest.TestCase):
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
