import http.client
import re
import threading
import unittest

from llama_doctor.dashboard.server import default_static_dir, make_server


class TestStaticFiles(unittest.TestCase):
    def setUp(self):
        self.root = default_static_dir()
        self.index = (self.root / "index.html").read_text(encoding="utf-8")
        self.app_js = (self.root / "app.js").read_text(encoding="utf-8")
        self.app_css = (self.root / "app.css").read_text(encoding="utf-8")

    def test_static_files_exist(self):
        for filename in ("index.html", "app.css", "app.js", "vendor/chart.umd.js"):
            with self.subTest(filename=filename):
                self.assertTrue((self.root / filename).is_file())
                self.assertGreater(len((self.root / filename).read_bytes()), 0)

    def test_index_references_local_assets_only(self):
        for asset in (
            'href="/static/app.css"',
            'src="/static/vendor/chart.umd.js"',
            'src="/static/app.js"',
        ):
            self.assertIn(asset, self.index)
        for value in re.findall(r'(?:src|href)="([^"]+)"', self.index):
            self.assertTrue(value.startswith("/static/"), value)

    def test_index_has_no_inline_code(self):
        self.assertNotIn("<style", self.index)
        self.assertNotIn(" style=", self.index)
        self.assertIsNone(re.search(r"<script(?![^>]*\ssrc=)", self.index))
        self.assertIsNone(re.search(r"\son[a-z]+=", self.index))

    def test_required_ids_once(self):
        identifiers = (
            "hdr-target",
            "hdr-model",
            "hdr-quant",
            "hdr-ctx",
            "hdr-slots",
            "hdr-build",
            "overall-badge",
            "updated-at",
            "conn-banner",
            "tile-decode",
            "tile-prefill",
            "tile-cache",
            "tile-kv",
            "chart-decode",
            "chart-prefill",
            "chart-kv-cache",
            "chart-draft-len",
            "chart-acceptance",
            "findings-list",
            "recs-list",
            "skipped-list",
        )
        for identifier in identifiers:
            with self.subTest(identifier=identifier):
                self.assertEqual(self.index.count(f'id="{identifier}"'), 1)

    def test_vendor_script_order(self):
        self.assertLess(self.index.index("chart.umd.js"), self.index.index("app.js"))

    def test_app_js_avoids_html_sinks(self):
        forbidden = ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function")
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, self.app_js)

    def test_no_remote_urls_in_app_code(self):
        for source in (self.index, self.app_css, self.app_js):
            self.assertNotIn("http://", source)
            self.assertNotIn("https://", source)

    def test_app_js_polls_state_endpoint(self):
        self.assertIn('"/api/state"', self.app_js)
        self.assertIn("no-store", self.app_js)

    def test_css_severity_classes(self):
        for severity in ("critical", "warn", "info", "ok", "insufficient_data"):
            self.assertIn(f".sev-{severity}", self.app_css)

    def test_css_tokens_present(self):
        tokens = (
            "--bg",
            "--surface",
            "--surface-2",
            "--border",
            "--grid",
            "--text",
            "--text-2",
            "--text-3",
            "--s-decode",
            "--s-prefill",
            "--s-kv",
            "--s-cache",
            "--s-draft",
            "--s-accept",
            "--ok",
            "--warn",
            "--crit",
            "--info",
        )
        for token in tokens:
            self.assertRegex(self.app_css, rf"{re.escape(token)}\s*:")

    def test_vendor_is_chartjs_v4_umd(self):
        vendor = (self.root / "vendor/chart.umd.js").read_bytes()
        first_kib = vendor[:1024].decode("utf-8")
        self.assertRegex(first_kib, r"Chart\.js v4\.")
        self.assertIn("MIT License", first_kib)
        self.assertGreater(len(vendor), 100000)

    def test_served_by_dashboard_server(self):
        state = {}
        server = make_server(lambda: state, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
            connection.request("GET", "/")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn(b'id="findings-list"', response.read())
            connection.close()
            connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
            connection.request("GET", "/static/vendor/chart.umd.js")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Content-Type"), "text/javascript; charset=utf-8")
            response.read()
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
