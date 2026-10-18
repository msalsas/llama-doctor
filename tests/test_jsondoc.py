import unittest

from llama_doctor.analysis import diagnose
from llama_doctor.render.jsondoc import diagnosis_document

from .support import make_context


class TestJsonDocument(unittest.TestCase):
    def test_server_quant_from_models_meta_when_props_lack_it(self):
        ctx = make_context("metrics_minimal.txt", "props_unsloth_minimal.json", "models_full.json")
        document = diagnosis_document(diagnose(ctx), ctx.snapshot, target="http://server")
        self.assertEqual(document["server"]["quant"], "UD-Q4_K_XL")
        self.assertEqual(document["server"]["model_name"], "Qwen3-30B-A3B-GGUF")
        self.assertEqual(document["model"]["model_id"], "Qwen3-30B-A3B-UD-Q4_K_XL.gguf")


if __name__ == "__main__":
    unittest.main()
