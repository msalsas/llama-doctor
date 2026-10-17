import math
import unittest

from llama_doctor.promparse import parse_prometheus

from .support import load_fixture


class TestParsePrometheus(unittest.TestCase):
    def test_parses_colon_prefixed_names(self):
        samples = parse_prometheus(load_fixture("metrics_healthy.txt"))
        self.assertEqual(samples["llamacpp:prompt_tokens_total"], 200000.0)

    def test_skips_comments_and_blank(self):
        text = "# a comment line\n\n" + load_fixture("metrics_healthy.txt")
        samples = parse_prometheus(text)
        self.assertTrue(samples)
        self.assertFalse(any(key.startswith("#") for key in samples))

    def test_keeps_label_block_in_key(self):
        samples = parse_prometheus(load_fixture("metrics_healthy.txt"))
        self.assertEqual(samples['llamacpp:slot_state{slot="0",state="idle"}'], 1.0)

    def test_ignores_trailing_timestamp(self):
        samples = parse_prometheus(load_fixture("metrics_healthy.txt"))
        self.assertEqual(samples["llamacpp:n_decode_total"], 50000.0)

    def test_special_floats(self):
        samples = parse_prometheus("m_nan NaN\nm_inf +Inf\nm_ninf -Inf\n")
        self.assertTrue(math.isnan(samples["m_nan"]))
        self.assertTrue(math.isinf(samples["m_inf"]))
        self.assertGreater(samples["m_inf"], 0)
        self.assertLess(samples["m_ninf"], 0)

    def test_skips_malformed_lines(self):
        samples = parse_prometheus("garbage line here\n{bad} 1\nfoo 1\n")
        self.assertEqual(samples, {"foo": 1.0})


if __name__ == "__main__":
    unittest.main()
