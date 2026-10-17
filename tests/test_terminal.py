import unittest
from io import StringIO

from llama_doctor.analysis import diagnose
from llama_doctor.models import ProbeResult
from llama_doctor.render.terminal import render_report, use_color

from .support import make_context


class _Stream(StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class TestTerminal(unittest.TestCase):
    def test_moe_report_contents(self):
        context = make_context()
        output = render_report(diagnose(context), context.snapshot, target="http://server")

        for text in (
            "Qwen3-30B-A3B-UD-Q4_K_XL",
            "quant UD-Q4_K_XL",
            "ctx 32768",
            "40.0 tok/s",
            "7.0 tok/s",
            "11.1 %",
            "len 3.00",
            "prefill.moe_cpu_experts",
            "overall: CRITICAL (exit 2)",
        ):
            self.assertIn(text, output)

    def test_no_ansi_without_color(self):
        context = make_context()
        output = render_report(diagnose(context), context.snapshot, target="http://server")

        self.assertNotIn("\x1b", output)

    def test_color_wraps_labels(self):
        context = make_context()
        output = render_report(diagnose(context), context.snapshot, target="http://server", color=True)

        self.assertIn("\x1b[1;31mCRIT\x1b[0m", output)

    def test_crit_before_warn(self):
        context = make_context()
        output = render_report(diagnose(context), context.snapshot, target="http://server")

        self.assertLess(output.index("CRIT prefill.moe_cpu_experts"), output.index("WARN decode.slow"))

    def test_recommendations_numbered_by_impact(self):
        context = make_context()
        output = render_report(diagnose(context), context.snapshot, target="http://server")

        line = next(line for line in output.splitlines() if line.startswith("  1."))
        self.assertTrue(line.startswith("  1. high"))

    def test_minimal_report_degrades(self):
        context = make_context("metrics_minimal.txt", "props_unsloth_minimal.json", None)
        output = render_report(diagnose(context), context.snapshot, target="http://server")

        self.assertIn("quant -", output)
        self.assertIn("ctx -", output)
        self.assertIn("prefix cache  n/a", output)
        self.assertIn("N/A", output)
        self.assertIn("overall: OK (exit 0)", output)

    def test_probe_line_shown(self):
        context = make_context()
        probe = ProbeResult(800.0, 20.0, 1200, 64, 0.5)

        output = render_report(diagnose(context), context.snapshot, target="http://server", probe=probe)

        self.assertIn("probe         prefill 800.0 tok/s  decode 20.0 tok/s  (1200-token prompt)", output)

    def test_no_sentinel_in_output(self):
        for props in ("props_full.json", "props_unsloth_minimal.json"):
            context = make_context(props=props)
            output = render_report(diagnose(context), context.snapshot, target="http://server")
            self.assertNotIn("SENTINEL", output)

    def test_use_color_rules(self):
        self.assertTrue(use_color(_Stream(True), {}))
        self.assertFalse(use_color(_Stream(True), {"NO_COLOR": "1"}))
        self.assertFalse(use_color(_Stream(True), {"TERM": "dumb"}))
        self.assertFalse(use_color(_Stream(False), {}))


if __name__ == "__main__":
    unittest.main()
