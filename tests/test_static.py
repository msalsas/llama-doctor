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

    def test_app_assets_contain_only_ascii_characters(self):
        for filename, source in (
            ("index.html", self.index),
            ("app.js", self.app_js),
            ("app.css", self.app_css),
        ):
            with self.subTest(filename=filename):
                self.assertTrue(all(ord(character) < 128 for character in source))

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
            "tile-context",
            "tile-queue",
            "chart-decode",
            "chart-prefill",
            "chart-kv-cache",
            "chart-draft-len",
            "chart-acceptance",
            "findings-list",
            "recs-list",
            "skipped-list",
            "note-decode",
            "note-prefill",
            "note-kv",
            "note-cache-hit",
            "note-slots",
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

    def test_chart_axes_have_unit_captions(self):
        for caption in (
            "Decode (tokens/s)",
            "Prefill (tokens/s)",
            "Usage (%)",
            "Draft length (tokens)",
            "Acceptance (%)",
        ):
            with self.subTest(caption=caption):
                self.assertIn(f'"{caption}"', self.app_js)

    def test_axis_titles_break_before_unit(self):
        self.assertIn("text: caption.split(/ (?=\\()/)", self.app_js)

    def test_evidence_numbers_are_rounded_only_for_display(self):
        self.assertIn("function displayNumber(value)", self.app_js)
        self.assertIn("Math.abs(value) >= 1 ? 1 : 3", self.app_js)
        self.assertIn("`${key} ${displayNumber(value)}`", self.app_js)

    def test_kv_tile_says_not_exported_without_lifetime_line(self):
        self.assertIn('"not exported"', self.app_js)
        self.assertIn("notes.kv_unexported === true", self.app_js)

    def test_kv_tile_is_a_gauge_without_lifetime_average(self):
        self.assertIn('percent(latest(history, "kv_usage_ratio"))', self.app_js)
        self.assertNotIn("lifetime.kv_usage_ratio", self.app_js)

    def test_metric_notes_start_hidden_and_are_styled(self):
        for note in ("note-decode", "note-prefill", "note-kv", "note-cache-hit", "note-slots"):
            with self.subTest(note=note):
                self.assertRegex(self.index, rf'id="{note}" class="chart-note hidden"')
                self.assertIn(f'"{note}"', self.app_js)
        self.assertIn(".chart-note", self.app_css)

    def test_metric_notes_follow_sampler_flags(self):
        for flag in ("decode_idle", "prefill_idle", "kv_unexported", "cache_hit_unexported", "slots_unavailable"):
            with self.subTest(flag=flag):
                self.assertIn(flag, self.app_js)

    def test_kv_and_cache_notes_explain_unexported_metrics(self):
        self.assertIn(">KV usage not exported by this server<", self.index)
        self.assertIn(">Prefix cache hits not exported by this server<", self.index)

    def test_kv_and_cache_notes_sit_under_kv_chart(self):
        canvas = self.index.index('id="chart-kv-cache"')
        self.assertLess(canvas, self.index.index('id="note-kv"'))
        self.assertLess(self.index.index('id="note-kv"'), self.index.index('id="note-cache-hit"'))
        self.assertLess(self.index.index('id="note-cache-hit"'), self.index.index('id="note-slots"'))
        self.assertLess(self.index.index('id="note-slots"'), self.index.index('data-table="kv-cache"'))

    def test_slots_note_text_depends_on_slot_availability(self):
        self.assertIn("Live slot data unavailable; context use is estimated from metrics", self.app_js)
        self.assertIn("KV usage is estimated from /slots", self.app_js)
        self.assertIn("notes.slots_unavailable !== true", self.app_js)
        self.assertIn("kv.estimated === true", self.app_js)
        self.assertIn("updateSlotsNote(state, notes)", self.app_js)

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

    def test_six_tiles_share_one_row_with_equal_height_and_flexible_width(self):
        self.assertEqual(self.index.count('class="tile"'), 6)
        self.assertIn(".tiles { grid-template-columns: repeat(6, minmax(0, 1fr)); align-items: stretch;",
                      self.app_css)
        self.assertIn(".tile { padding: 16px; min-height: 102px; }", self.app_css)

    def test_context_tile_sits_between_kv_and_queue(self):
        kv, context, queue = (self.index.index(f'id="tile-{name}"') for name in ("kv", "context", "queue"))
        self.assertLess(kv, context)
        self.assertLess(context, queue)
        self.assertIn('"tile-context": "Context"', self.app_js)

    def test_context_tile_shows_peak_ratio_and_label_from_state(self):
        self.assertIn("percent(number(context.used_ratio))", self.app_js)
        self.assertIn('text(context.label, "n/a")', self.app_js)

    def test_context_tile_live_hero_with_peak_secondary_like_rate_tiles(self):
        self.assertIn('append(body, "div", percent(used / capacity), "hero");', self.app_js)
        self.assertIn('`peak ${percent(number(context.used_ratio))}`, "secondary"', self.app_js)
        self.assertNotIn("live ${percent", self.app_js)

    def test_kv_chart_has_context_used_series(self):
        self.assertIn('label: "Context used"', self.app_js)
        self.assertIn('["kv_usage_ratio", "cache_hit_ratio", "context_used_ratio"]', self.app_js)
        self.assertIn('{ label: "Context used", key: "context_used_ratio", format: percent }', self.app_js)
        self.assertRegex(self.app_css, r"--s-ctx\s*:")

    def test_legend_swatches_are_lines_that_keep_dataset_dash(self):
        self.assertIn('usePointStyle: true, pointStyle: "line", pointStyleWidth: 28', self.app_js)
        self.assertIn("borderDash: [6, 4]", self.app_js)

    def test_checks_summary_counts_run_and_skipped(self):
        self.assertIn(
            ".textContent = `${findings.length - skipped.length} run, ${skipped.length} skipped`;",
            self.app_js,
        )
        self.assertIn("<summary>0 run, 0 skipped</summary>", self.index)
        self.assertNotIn("checks skipped", self.app_js + self.index)

    def test_queue_tile_follows_kv_tile(self):
        self.assertLess(self.index.index('id="tile-kv"'), self.index.index('id="tile-queue"'))
        self.assertIn('"Queue"', self.app_js)
        self.assertIn('"requests_processing"', self.app_js)
        self.assertIn('"requests_deferred"', self.app_js)

    def test_queue_tile_secondary_shows_only_waiting_count(self):
        self.assertIn('`${count(latest(history, "requests_deferred"))} waiting`', self.app_js)
        self.assertNotIn("processing, ", self.app_js)

    def test_info_tooltips_text_comes_from_catalog_endpoint(self):
        self.assertIn('fetch("/api/catalog"', self.app_js)
        self.assertIn('setAttribute("aria-describedby"', self.app_js)
        self.assertIn('setAttribute("role", "tooltip")', self.app_js)
        self.assertIn("entry.description", self.app_js)
        self.assertIn("entry.interpretation", self.app_js)

    def test_tiles_get_one_info_button_with_prefixed_sections(self):
        self.assertIn('if (element.classList.contains("tile")) {', self.app_js)
        self.assertIn("infoButton(host, items);", self.app_js)
        self.assertIn("items.forEach((item) => infoButton(host, [item]));", self.app_js)
        self.assertRegex(self.app_js, r'queue_processing: "Processing",\s*queue_deferred: "Deferred"')
        self.assertIn('description.prepend(`${INFO_PREFIX[metricId] || labels[index]}: `);', self.app_js)
        self.assertIn('button.setAttribute("aria-label", `About ${labels.join(" and ")}`);', self.app_js)
        self.assertRegex(self.app_css, r"\.info-tip \.info-interp\s*\{")

    def test_info_button_is_a_real_button(self):
        self.assertIn('document.createElement("button")', self.app_js)
        self.assertIn('button.type = "button"', self.app_js)

    def test_info_tooltip_opens_on_hover_and_focus_in_pure_css(self):
        self.assertRegex(self.app_css, r"\.info:hover \+ \.info-tip")
        self.assertRegex(self.app_css, r"\.info:focus-visible \+ \.info-tip")
        self.assertRegex(self.app_css, r"\.info-tip:hover")
        self.assertIn('"chart-note info-tip"', self.app_js)

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
