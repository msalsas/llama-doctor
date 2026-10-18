import re
import unittest

from llama_doctor.catalog import METRICS
from llama_doctor.dashboard.server import default_static_dir

REQUIRED_IDS = {
    "decode_tps",
    "prefill_tps",
    "prefix_cache_hit",
    "kv_usage",
    "draft_length",
    "acceptance",
    "queue_processing",
    "queue_deferred",
    "quant",
    "context_tokens",
    "slots",
    "context_used",
}


def dashboard_metric_ids(app_js: str) -> set[str]:
    block = re.search(r"const METRIC_INFO = \{(.*?)\n  \};", app_js, re.DOTALL)
    assert block is not None, "METRIC_INFO block not found in app.js"
    lists = re.findall(r"\[([^\]]*)\]", block.group(1))
    return {name for group in lists for name in re.findall(r'"([a-z_]+)"', group)}


class TestCatalog(unittest.TestCase):
    def setUp(self):
        self.app_js = (default_static_dir() / "app.js").read_text(encoding="utf-8")

    def test_covers_required_ids(self):
        self.assertEqual(set(METRICS), REQUIRED_IDS)

    def test_entries_have_ascii_label_description_interpretation(self):
        for metric_id, entry in METRICS.items():
            with self.subTest(metric_id=metric_id):
                self.assertEqual(set(entry), {"label", "description", "interpretation"})
                for value in entry.values():
                    self.assertIsInstance(value, str)
                    self.assertTrue(value.strip())
                    self.assertTrue(value.isascii())
                self.assertTrue(entry["description"].endswith("."))
                self.assertTrue(entry["interpretation"].endswith("."))

    def test_every_dashboard_metric_id_has_catalog_entry(self):
        used = dashboard_metric_ids(self.app_js)
        self.assertTrue(used)
        self.assertEqual(used - set(METRICS), set())

    def test_dashboard_explains_every_catalog_entry(self):
        self.assertEqual(set(METRICS) - dashboard_metric_ids(self.app_js), set())

    def test_context_used_explains_peak_and_unsloth(self):
        text = " ".join(METRICS["context_used"].values())
        for phrase in ("n_tokens_max", "peak", "instantaneous", "Unsloth"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_speculative_texts_cover_draft_model_and_mtp(self):
        for metric_id in ("draft_length", "acceptance"):
            with self.subTest(metric_id=metric_id):
                text = " ".join(METRICS[metric_id].values())
                self.assertIn("the speculative mechanism (separate draft model or MTP head)", text)
                self.assertNotIn("the draft model", text.replace("separate draft model", ""))

    def test_kv_usage_says_no_current_server_exports_it(self):
        text = METRICS["kv_usage"]["interpretation"]
        self.assertIn("No current llama-server build exports", text)
        self.assertNotIn("Unsloth", text)
        self.assertNotIn("Some servers", text)

    def test_draft_length_explains_when_inactive_drafts_warn(self):
        text = METRICS["draft_length"]["interpretation"]
        for phrase in ("speculative.types", "at 0", "only warns"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_slots_explain_shared_context_budget(self):
        text = METRICS["slots"]["interpretation"]
        for phrase in ("same -c", "--parallel", "less context"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_prefill_explains_prefix_cache_effect(self):
        self.assertIn("prefix-cache hit", METRICS["prefill_tps"]["interpretation"])

    def test_texts_are_model_agnostic(self):
        pattern = re.compile(r"\d+\s*GB|tok/s|\b\d+(-\d+)?B\b|\bQ\d\b|\d+\s*-\s*\d+")
        for metric_id, entry in METRICS.items():
            with self.subTest(metric_id=metric_id):
                self.assertIsNone(pattern.search(entry["interpretation"]))

    def test_context_used_explained_on_tile_and_kv_chart(self):
        block = re.search(r"const METRIC_INFO = \{(.*?)\n  \};", self.app_js, re.DOTALL).group(1)
        self.assertRegex(block, r'"tile-context": \["context_used"\]')
        self.assertRegex(block, r'"chart-kv-cache": \[[^\]]*"context_used"')

    def test_app_js_does_not_hardcode_catalog_text(self):
        for metric_id, entry in METRICS.items():
            for key in ("description", "interpretation"):
                with self.subTest(metric_id=metric_id, key=key):
                    self.assertNotIn(entry[key][:40], self.app_js)


if __name__ == "__main__":
    unittest.main()
