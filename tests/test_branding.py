import re
import unittest
import xml.etree.ElementTree as ET
from fnmatch import fnmatch
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

from llama_doctor.dashboard.server import default_static_dir

ROOT = Path(__file__).resolve().parent.parent
SVG_NS = "{http://www.w3.org/2000/svg}"
LOGO = ROOT / "docs" / "assets" / "logo-mark.svg"
DOCS_FAVICON = ROOT / "docs" / "assets" / "favicon.svg"


def css_token(css: str, name: str) -> str:
    return re.search(rf"{re.escape(name)}\s*:\s*(#[0-9a-fA-F]{{6}})", css).group(1).lower()


class TestBranding(unittest.TestCase):
    def setUp(self):
        self.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.static = default_static_dir()
        self.index = (self.static / "index.html").read_text(encoding="utf-8")
        self.app_css = (self.static / "app.css").read_text(encoding="utf-8")
        self.svgs = (LOGO, DOCS_FAVICON, Path(str(self.static / "favicon.svg")))

    def test_readme_image_sources_exist(self):
        paths = re.findall(r'<img[^>]*\ssrc="((?:docs|src)/[^"]+)"', self.readme)
        self.assertIn("docs/assets/logo-mark.svg", paths)
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue((ROOT / path).is_file())

    def test_readme_header_layout(self):
        self.assertTrue(
            self.readme.startswith(
                '<p align="center"><img src="docs/assets/logo-mark.svg"'
                ' alt="llama-doctor logo" width="96" height="96"></p>\n\n'
                '<h1 align="center">llama-doctor</h1>\n'
            )
        )

    def test_index_favicon_link_resolves_to_shipped_file(self):
        self.assertIn(
            '<link rel="icon" href="/static/favicon.svg" type="image/svg+xml">', self.index
        )
        for value in re.findall(r'(?:src|href)="(/static/[^"]+)"', self.index):
            with self.subTest(value=value):
                self.assertTrue((self.static / value.removeprefix("/static/")).is_file())

    def test_svgs_parse_and_are_self_contained(self):
        for path in self.svgs:
            with self.subTest(path=path):
                raw = path.read_bytes()
                root = ET.fromstring(raw)
                self.assertEqual(root.tag, f"{SVG_NS}svg")
                self.assertEqual(root.get("viewBox"), "0 0 96 96")
                self.assertEqual(root.get("aria-label"), "llama-doctor")
                text = raw.decode("utf-8")
                self.assertNotIn("<script", text)
                self.assertNotIn("href=", text)
                self.assertNotIn("@import", text)
                self.assertTrue(all(byte < 128 for byte in raw))

    def test_svgs_use_dashboard_palette(self):
        for path in self.svgs:
            with self.subTest(path=path):
                text = path.read_text(encoding="utf-8")
                for token in ("--bg", "--text", "--warn", "--crit", "--grid"):
                    with self.subTest(token=token):
                        self.assertIn(css_token(self.app_css, token), text.lower())
                self.assertIn("@media (prefers-color-scheme: light)", text)

    def test_favicon_copies_are_identical(self):
        self.assertEqual(
            DOCS_FAVICON.read_bytes(), (self.static / "favicon.svg").read_bytes()
        )

    @unittest.skipIf(tomllib is None, "tomllib requires Python 3.11")
    def test_favicon_is_declared_package_data(self):
        """The wheel/sdist includes the favicon without a build that is slow and needs network."""
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        patterns = data["tool"]["setuptools"]["package-data"]["llama_doctor.dashboard"]
        self.assertTrue(any(fnmatch("static/favicon.svg", p) for p in patterns))


if __name__ == "__main__":
    unittest.main()
