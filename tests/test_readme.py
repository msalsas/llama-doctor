import ast
import re
import unittest
from pathlib import Path

from llama_doctor.analysis import BUILTIN_ANALYZERS, diagnose
from llama_doctor.cli import build_parser
from llama_doctor.dashboard.server import default_static_dir
from llama_doctor.render.terminal import render_report

from .support import make_context

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT / "src" / "llama_doctor" / "analysis"


def static_finding_codes() -> set[str]:
    """Literal codes passed to Finding(...); the engine's f-string `<id>.error` is excluded."""
    codes: set[str] = set()
    for path in ANALYSIS_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Finding"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
            ):
                codes.add(node.args[1].value)
    return codes


class TestReadme(unittest.TestCase):
    def setUp(self):
        self.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_analyzer_and_finding_counts_match_code(self):
        match = re.search(r"Runs (\d+) analyzers that emit (\d+) distinct findings", self.readme)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), len(BUILTIN_ANALYZERS))
        self.assertEqual(int(match.group(2)), len(static_finding_codes()))

    def test_cited_finding_codes_exist(self):
        codes = static_finding_codes()
        ids = "|".join(spec.id for spec in BUILTIN_ANALYZERS)
        cited = set(re.findall(rf"`((?:{ids})\.[a-z_]+)`", self.readme))
        self.assertTrue(cited)
        self.assertEqual(cited - codes, set())

    def test_tile_list_matches_dashboard_order(self):
        app_js = (default_static_dir() / "app.js").read_text(encoding="utf-8")
        block = re.search(r"const TILE_LABELS = \{(.*?)\n  \};", app_js, re.DOTALL).group(1)
        labels = re.findall(r'"tile-[a-z]+": "([^"]+)"', block)
        self.assertEqual(len(labels), 6)
        self.assertIn(", ".join(labels), self.readme)

    def test_install_is_editable_and_pythonpath_runs_from_repo_root(self):
        install = self.readme.split("## Install", 1)[1].split("\n## ", 1)[0]
        self.assertIn("pip install -e .", install)
        self.assertIn("from the repository root", install)

    def test_findings_screenshot_is_embedded(self):
        self.assertIn("](docs/assets/findings.png)", self.readme)

    def test_serve_documents_vram_gib(self):
        serve = self.readme.split("`llama-doctor serve URL`", 1)[1].split("\n## ", 1)[0]
        self.assertIn("`--vram-gib N`", serve)

    def test_serve_documents_every_flag(self):
        serve = self.readme.split("`llama-doctor serve URL`", 1)[1].split("\n## ", 1)[0]
        parser = build_parser()
        serve_parser = parser._subparsers._group_actions[0].choices["serve"]
        flags = {option for action in serve_parser._actions for option in action.option_strings
                 if option.startswith("--") and option != "--help"}
        self.assertIn("--ram-gib", flags)
        for flag in flags:
            with self.subTest(flag=flag):
                self.assertIn(f"`{flag}", serve)

    def test_quick_start_example_matches_rendered_fixture(self):
        """The example block is real output: analyzers run on a fixture with the same counters."""
        quick_start = self.readme.split("## Quick start", 1)[1].split("\n## ", 1)[0]
        block = re.search(r"```text\n(.*?)\n```", quick_start, re.DOTALL).group(1)
        context = make_context("readme_example/metrics.txt", "readme_example/props.json",
                               "readme_example/models.json")
        rendered = render_report(diagnose(context), context.snapshot, target="http://laptop-msi:8888")
        self.assertEqual(block, rendered)
        self.assertNotIn("kv.ok", block)
        self.assertIn("N/A  kv.insufficient_data", block)
        findings = block.split("findings\n", 1)[1]
        self.assertTrue(findings.startswith("  WARN context.near_limit"))

    def test_diagnose_and_serve_document_weights_gib(self):
        reference = self.readme.split("`llama-doctor diagnose URL`", 1)[1]
        diagnose, serve = reference.split("`llama-doctor serve URL`", 1)
        self.assertIn("`--weights-gib N`", diagnose)
        self.assertIn("`--weights-gib N`", serve.split("\n## ", 1)[0])

    def test_linked_repo_paths_exist(self):
        paths = re.findall(r"\]\(((?:docs|src)/[^)#]+|LICENSE)\)", self.readme)
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue((ROOT / path).exists())


if __name__ == "__main__":
    unittest.main()
