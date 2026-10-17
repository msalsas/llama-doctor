import json
import unittest

from llama_doctor.analysis import diagnose
from llama_doctor.analysis.base import Severity
from llama_doctor.models import EndpointError
from llama_doctor.serialize import (
    errors_to_dict,
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

    def test_severity_names(self):
        self.assertEqual(severity_name(Severity.INSUFFICIENT_DATA), "insufficient_data")
        self.assertEqual(severity_name(Severity.OK), "ok")
        self.assertEqual(severity_name(Severity.INFO), "info")
        self.assertEqual(severity_name(Severity.WARN), "warn")
        self.assertEqual(severity_name(Severity.CRITICAL), "critical")

    def test_props_dict_has_no_sentinel(self):
        for name in ("props_full.json", "props_unsloth_minimal.json"):
            self.assertNotIn("SENTINEL", repr(props_to_dict(make_context(props=name).props)))

    def test_rates_none_when_no_metrics(self):
        self.assertTrue(all(value is None for value in rates_to_dict(None).values()))

    def test_errors_to_dict(self):
        self.assertEqual(errors_to_dict({"x": EndpointError("x", 500, "failed")}), {"x": "failed"})


if __name__ == "__main__":
    unittest.main()
