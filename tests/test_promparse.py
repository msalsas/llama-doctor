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

    def test_rejects_python_only_numeric_spellings(self):
        samples = parse_prometheus("m_us 1_000\nm_lc_nan nan\nm_lc_inf infinity\nm_inf +Inf\n")
        self.assertEqual(set(samples), {"m_inf"})
        self.assertTrue(math.isinf(samples["m_inf"]))

    def test_escaped_label_value_and_multiline_help(self):
        text = (
            "# HELP some_metric A help text that spans\n"
            "# multiple lines via a backslash continuation\n"
            "# TYPE some_metric gauge\n"
            'some_metric{a="escaped\\\"value",b="x"} 42\n'
            "\n"
        )
        samples = parse_prometheus(text)
        self.assertEqual(samples['some_metric{a="escaped\\\"value",b="x"}'], 42.0)

    def test_skips_malformed_lines(self):
        samples = parse_prometheus("garbage line here\n{bad} 1\nfoo 1\n")
        self.assertEqual(samples, {"foo": 1.0})


if __name__ == "__main__":
    unittest.main()
