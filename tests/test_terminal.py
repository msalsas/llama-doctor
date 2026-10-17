import unittest
from io import StringIO

from llama_doctor.analysis import diagnose
from llama_doctor.analysis.base import Finding, Impact, Recommendation, Report, Severity
from llama_doctor.models import MetricsSnapshot, ProbeResult, PropsSnapshot, ServerSnapshot
from llama_doctor.render.terminal import render_report, use_color

from .support import make_context


class _Stream(StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class TestTerminal(unittest.TestCase):
    def test_render_report_golden_string(self):
        report = Report((
            Finding(
                "test",
                "test.ok",
                Severity.OK,
                "Everything works",
                "This must not render.",
            ),
            Finding(
                "test",
                "test.warn",
                Severity.WARN,
                "Needs tuning",
                "Tune this setting.",
                recommendations=(
                    Recommendation("Set a value.", "It helps.", Impact.HIGH, "--value 4"),
                ),
            ),
        ))
        snapshot = ServerSnapshot(
            0.0,
            MetricsSnapshot(
                prompt_tokens_total=100.0,
                prompt_seconds_total=2.0,
                tokens_predicted_total=50.0,
                tokens_predicted_seconds_total=2.0,
                prompt_tokens_cached_total=100.0,
                draft_tokens_total=6.0,
                drafts_total=2.0,
                draft_tokens_accepted_total=3.0,
            ),
            PropsSnapshot("model", "Q4_0", 4096, 1, "build"),
            None,
            {},
        )

        output = render_report(report, snapshot, target="http://server", color=False)

        self.assertEqual(
            output,
            """llama-doctor 0.1.0  http://server
model  model  quant Q4_0  ctx 4096  slots 1  build build

rates (lifetime averages)
  prefill       50.0 tok/s
  decode        25.0 tok/s
  prefix cache  50.0 %
  drafts        len 3.00  acceptance 50.0 %

findings
  OK   test.ok  Everything works
  WARN test.warn  Needs tuning
        Tune this setting.
recommendations (by impact)
  1. high   Set a value.  [--value 4]
overall: WARN (exit 1)""",
        )

    def test_render_report_golden_string_without_recommendations(self):
        report = Report((
            Finding("test", "test.ok", Severity.OK, "Everything works", "Not rendered."),
        ))
        snapshot = ServerSnapshot(0.0, None, None, None, {})

        output = render_report(report, snapshot, target="http://server", color=False)

        self.assertEqual(
            output,
            """llama-doctor 0.1.0  http://server
model  -  quant -  ctx -  slots -  build -

rates (lifetime averages)
  prefill       n/a
  decode        n/a
  prefix cache  n/a
  drafts        len n/a  acceptance n/a

findings
  OK   test.ok  Everything works
recommendations (by impact)
  none
overall: OK (exit 0)""",
        )

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
        lines = output.splitlines()

        self.assertIn("quant -", output)
        self.assertIn("ctx -", output)
        self.assertIn("  prefix cache  n/a", lines)
        self.assertIn("  drafts        len n/a  acceptance n/a", lines)
        self.assertIn("N/A", output)
        self.assertIn("overall: OK (exit 0)", output)

    def test_na_rates_have_no_trailing_units(self):
        output = render_report(
            Report(()),
            ServerSnapshot(0.0, None, None, None, {}),
            target="http://server",
        )
        lines = output.splitlines()

        self.assertIn("  prefill       n/a", lines)
        self.assertIn("  decode        n/a", lines)
        self.assertIn("  prefix cache  n/a", lines)
        self.assertIn("  drafts        len n/a  acceptance n/a", lines)

    def test_probe_line_shown(self):
        context = make_context()
        probe = ProbeResult(800.0, 20.0, 1200, 64, 0.5)

        output = render_report(diagnose(context), context.snapshot, target="http://server", probe=probe)
        lines = output.splitlines()

        self.assertEqual(lines[3], "rates (lifetime averages)")
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
