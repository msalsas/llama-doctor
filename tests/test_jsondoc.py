import json
import unittest

from llama_doctor.analysis import diagnose
from llama_doctor.analysis.base import Finding, Report, Severity
from llama_doctor.render.jsondoc import diagnosis_document

from .support import make_context


class TestJsonDocument(unittest.TestCase):
    def test_server_quant_from_models_meta_when_props_lack_it(self):
        ctx = make_context("metrics_minimal.txt", "props_unsloth_minimal.json", "models_full.json")
        document = diagnosis_document(diagnose(ctx), ctx.snapshot, target="http://server")
        self.assertEqual(document["server"]["quant"], "UD-Q4_K_XL")
        self.assertEqual(document["server"]["model_name"], "Qwen3-30B-A3B-GGUF")
        self.assertEqual(document["model"]["model_id"], "Qwen3-30B-A3B-UD-Q4_K_XL.gguf")

    def test_findings_keep_full_precision_in_cli_json(self):
        ctx = make_context()
        finding = Finding("test", "test.value", Severity.WARN, "Test", "Test",
                          {"prefill_tps": 132.60928071193192, "hit_ratio": 0.1111111, "nan": float("nan")})
        document = diagnosis_document(Report((finding,)), ctx.snapshot, target="http://server")
        decoded = json.loads(json.dumps(document, allow_nan=False))
        self.assertEqual(decoded["findings"][0]["evidence"],
                         {"prefill_tps": 132.60928071193192, "hit_ratio": 0.1111111, "nan": None})


if __name__ == "__main__":
    unittest.main()
