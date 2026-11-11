=== CONTRACT (PART 1) ===

I've checked the repo, and several of the brief's "repo facts" don't match it. One of them blocks scope item 4. The contract below uses what the repo actually contains.

# Implementation contract: llama-doctor branding assets

Worktree `~/llama-doctor-wt-logo`, branch `feature/branding`, based on `2058683`. Work only in this worktree.

## 0. Corrections to the brief

I read each of these directly in the worktree. Where the brief and the repo disagree, follow the repo.

| Brief says | Repo actually has | Effect |
|---|---|---|
| README head: 5 badges (CI, Release, MIT, Python, Downloads), tagline "Local LLM doctor." | `# llama-doctor`, a blank line, the line *"A command-line tool that reads a running llama.cpp or Unsloth Studio server's metrics and tells you what is misconfigured."*, a blank line, then **3** badges (CI, Python 3.10+, License MIT) | README edit is based on the real header (§2.3) |
| `index.html` has `.brand` / `.mark` / tagline; stylesheet `href="app.css"` | `<header class="header">`, `<strong class="product">`, `<link rel="stylesheet" href="/static/app.css">`, 2-space indent, no self-closing slashes | `<link>` must use `/static/...` (enforced by `tests/test_static.py::test_index_references_local_assets_only`) |
| `pyproject.toml`: `dynamic = ["version"]`; package-data `llama_doctor = [...]` | static `version = "0.2.0"`; `[tool.setuptools.package-data] "llama_doctor.dashboard" = ["static/*", "static/vendor/*"]`; no `docs/assets` entry, no MANIFEST.in, no `readme =` field | Favicon ships through `static/*` with no pyproject change. Nothing under `docs/` ships. README is not used on PyPI. |
| `test_linked_repo_paths_exist` regex `\]\(((?:docs\|src)/[^)#]+)(?:#...)?\)` | regex `\]\(((?:docs\|src)/[^)#]+\|LICENSE)\)` | This regex only checks Markdown links, not `<img src="...">`, so a new test is needed |
| `docs/assets/` has 7 PNGs; `docs/` has PLAN-*.md | `docs/assets/` = `dashboard.png`, `dashboard-findings.png`, `findings.png`; `docs/` = `assets/`, `contracts/`, `DESIGN.md` | No naming conflicts |
| `__init__.py` `0.1.0` / `test_scaffold.py` pins `0.1.0` | Confirmed | Out of scope, don't touch |
| Palette | `app.css`: `--bg #121211`, `--grid #2a2a28`, `--text #f0efec`, `--text-2 #c3c2b7`, `--ok #0ca30c`, `--warn #fab219`, `--crit #d03b3b` | Confirmed. The SVG's `#2a2a28` = `--grid` and `#c3c2b7` = `--text-2`. Only the light-theme `#8b6003` and `#c73030` are new hexes. |
| CI | `.github/workflows/ci.yml`: Python 3.11 and 3.13, `pip install -e . ruff`, `ruff check src tests`, `PYTHONPATH=src python -m unittest discover -v` | Must stay green |

### BLOCKER: scope item 4 can't work without touching `server.py`

`src/llama_doctor/dashboard/server.py` serves static files from a fixed allowlist:

```python
STATIC_ROUTES: dict[str, str] = {
    "/": "index.html",
    "/static/app.css": "app.css",
    "/static/app.js": "app.js",
    "/static/vendor/chart.umd.js": "vendor/chart.umd.js",
}
...
            content_type = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
            }[Path(filename).suffix]
```

If only `static/favicon.svg` and the `<link>` are added, the browser's request to `/static/favicon.svg` returns **404** and the dashboard never shows the icon. A `data:` URI won't work either: `test_index_references_local_assets_only` requires every `href` to start with `/static/`.

**Decision (needs orchestrator sign-off before Kimi starts step 4):** make the smallest possible change to `server.py`: one route entry and one MIME entry (§2.5). It's the only Python change. If sign-off is refused, drop scope item 4 completely: no `static/favicon.svg`, no `<link>`, no static/server tests. Do steps 1–3 and the README tests only. Don't ship a link that 404s.

## 1. File list

| Action | Path |
|---|---|
| create | `docs/assets/logo-mark.svg` |
| create | `docs/assets/favicon.svg` |
| create | `src/llama_doctor/dashboard/static/favicon.svg` (byte-identical to `docs/assets/favicon.svg`) |
| modify | `README.md` (header only) |
| modify | `src/llama_doctor/dashboard/static/index.html` (one line) |
| modify | `src/llama_doctor/dashboard/server.py` (two lines, **gated**, see §0) |
| create | `tests/test_branding.py` |
| modify | `tests/test_server.py` (one test, gated with server.py) |

## 2. Content

### 2.1 `docs/assets/logo-mark.svg`

Use the **A-bold** candidate source from the brief, byte-for-byte, with no changes. The file ends with one trailing `\n` and contains ASCII only. Reason: it's the measured geometry and its hexes already match `app.css`. Changing anything would invalidate the measurements.

### 2.2 `docs/assets/favicon.svg` and `src/llama_doctor/dashboard/static/favicon.svg`

Use the **A-favicon** candidate source from the brief, byte-for-byte, with no changes. That includes the unused `.grid { ... display: none; }` rules: they're harmless, and keeping the file verbatim keeps it identical to what was measured. One trailing `\n`, ASCII only. Write it once, then `cp docs/assets/favicon.svg src/llama_doctor/dashboard/static/favicon.svg` so the two files are byte-identical.

### 2.3 `README.md`: change lines 1–2 only

Current first line is `# llama-doctor`. New top of file, verbatim:

```markdown
<img src="docs/assets/logo-mark.svg" alt="llama-doctor logo" width="96" height="96">

# llama-doctor

A command-line tool that reads a running llama.cpp or Unsloth Studio server's metrics and tells you what is misconfigured.

[![CI](https://github.com/msalsas/llama-doctor/actions/workflows/ci.yml/badge.svg)](https://github.com/msalsas/llama-doctor/actions/workflows/ci.yml)
```

So you only insert two lines (the `<img>` line and a blank line) before `# llama-doctor`. Leave everything from `# llama-doctor` down unchanged: description line, 3 badges, everything after.

- **Ambiguous, decided:** no centring. The `<img>` sits alone, left-aligned. GitHub ignores CSS, so centring would need a `<p align="center">`/`<div>` wrapper, and the scope requires the image tag alone on its own line.
- The blank line after `<img>` is required. A raw-HTML line ends at a blank line in CommonMark/GFM, so without it the heading would be swallowed into the HTML block.

### 2.4 `src/llama_doctor/dashboard/static/index.html`: insert one line

Put it directly after line 7, `  <link rel="stylesheet" href="/static/app.css">`:

```html
  <link rel="icon" href="/static/favicon.svg" type="image/svg+xml">
```

Two-space indent, no closing `/`, same as the surrounding lines. Change nothing else.

### 2.5 `src/llama_doctor/dashboard/server.py` (GATED, see §0)

Make exactly two additions:

```python
STATIC_ROUTES: dict[str, str] = {
    "/": "index.html",
    "/static/app.css": "app.css",
    "/static/app.js": "app.js",
    "/static/favicon.svg": "favicon.svg",
    "/static/vendor/chart.umd.js": "vendor/chart.umd.js",
}
```

```python
            content_type = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".svg": "image/svg+xml",
            }[Path(filename).suffix]
```

Leave CSP, headers and everything else alone. The existing CSP `img-src 'self' data:` already allows a same-origin icon.

## 3. Tests

### 3.1 New `tests/test_branding.py`

Use stdlib `unittest` only, same style as `test_readme.py` and `test_static.py`: module-level `ROOT`, `setUp` reading files, `subTest` for loops. CI runs Python 3.11+, so `tomllib` is fine; locally it's Python 3.13.13.

```python
import re
import tomllib
import unittest
import xml.etree.ElementTree as ET
from fnmatch import fnmatch
from pathlib import Path

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
```

Tests, with the reason each one is worth having:

1. **`test_readme_image_sources_exist`**: `paths = re.findall(r'<img[^>]*\ssrc="((?:docs|src)/[^"]+)"', self.readme)`, then `assertIn("docs/assets/logo-mark.svg", paths)` and, for each path, `assertTrue((ROOT / path).is_file())` inside `subTest`. *Why:* `test_linked_repo_paths_exist` only checks Markdown `](...)` links, so a moved or renamed logo would otherwise go unnoticed.
2. **`test_readme_logo_sits_alone_above_title`**: `assertTrue(self.readme.startswith('<img src="docs/assets/logo-mark.svg" alt="llama-doctor logo" width="96" height="96">\n\n# llama-doctor\n'))`. *Why:* this locks the stacked layout and the blank line that keeps `#` working as a heading. It also protects the order the badge and link regexes rely on.
3. **`test_index_favicon_link_resolves_to_shipped_file`**: `assertIn('<link rel="icon" href="/static/favicon.svg" type="image/svg+xml">', self.index)`; then, for every `href`/`src` in `index.html` that starts with `/static/`, `assertTrue((self.static / value.removeprefix("/static/")).is_file())`. *Why:* the existing static test checks that the prefix is `/static/` but never checks that the file exists.
4. **`test_svgs_parse_and_are_self_contained`**: for each of `self.svgs`, `root = ET.fromstring(path.read_bytes())`; `assertEqual(root.tag, f"{SVG_NS}svg")`; `assertEqual(root.get("viewBox"), "0 0 96 96")`; `assertEqual(root.get("aria-label"), "llama-doctor")`; the raw text has no `<script`, no `href=`, no `@import`, and every byte is under 128. *Why:* malformed SVG renders as a broken image on GitHub. An external reference or script would break the dashboard CSP and offline use. The ASCII check matches the static assets' existing convention.
5. **`test_svgs_use_dashboard_palette`**: for each SVG, check that each of `css_token(self.app_css, t)` for `t in ("--bg", "--text", "--warn", "--crit", "--grid")` appears (case-insensitive) in the SVG text, and that `"@media (prefers-color-scheme: light)"` is present. *Why:* this is the brand-consistency contract. If `app.css` changes a hex, the logo has to follow. (`--grid #2a2a28` is the light-theme ink and dark-theme grid, so it's in both files.)
6. **`test_favicon_copies_are_identical`**: `assertEqual(DOCS_FAVICON.read_bytes(), (self.static / "favicon.svg").read_bytes())`. *Why:* stops the docs copy and the shipped copy from drifting apart.
7. **`test_favicon_is_declared_package_data`**: `data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))`; `patterns = data["tool"]["setuptools"]["package-data"]["llama_doctor.dashboard"]`; `assertTrue(any(fnmatch("static/favicon.svg", p) for p in patterns))`. *Why:* it proves the wheel/sdist will include the file without running a build, which would be slow and need network access. Add a docstring stating that.

### 3.2 `tests/test_server.py` (gated with §2.5): add after `test_static_allowlist`

```python
    def test_favicon_served_as_svg(self):
        Path(self.temp_dir.name, "favicon.svg").write_text("<svg/>", encoding="utf-8")
        status, headers, body = self._request("GET", "/static/favicon.svg")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/svg+xml")
        self.assertEqual(body, b"<svg/>")
```

*Why:* the browser gets the icon at runtime only if both the route and the MIME entry exist. Without the MIME entry, `.svg` would raise `KeyError` in `content_type[...]`.

Don't modify `tests/test_readme.py` or `tests/test_static.py`. Existing assertions there (e.g. `/static/` prefix, `findings.png` embed) must keep passing unchanged.

## 4. Verify (from the worktree root, non-interactive)

```bash
PYTHONPATH=src python -m unittest -v tests.test_branding
PYTHONPATH=src python -m unittest -v tests.test_readme tests.test_static tests.test_server
python -c "import xml.etree.ElementTree as E; [E.parse(p) for p in ('docs/assets/logo-mark.svg','docs/assets/favicon.svg','src/llama_doctor/dashboard/static/favicon.svg')]"
cmp docs/assets/favicon.svg src/llama_doctor/dashboard/static/favicon.svg
ruff check src tests
git diff --stat main -- pyproject.toml src/llama_doctor/__init__.py tests/test_scaffold.py .github docs/DESIGN.md   # must print nothing
PYTHONPATH=src python -m unittest discover -v   # full run, once, last
```

All must exit 0. If `ruff` isn't installed, say so in the PR body; don't install it globally.

## 5. Commits and PR

Commit order. Each commit must pass the tests on its own. Every message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

1. `docs: add llama-doctor logo mark and favicon SVGs`: the two `docs/assets/*.svg` files.
2. `README: show logo mark above the title`: `README.md`, plus the README and SVG tests (1, 2, 4, 5) in a new `tests/test_branding.py`.
3. `dashboard: serve and link an SVG favicon`: `static/favicon.svg`, the `index.html` line, the `server.py` route and MIME entries, tests 3, 6, 7 added to `test_branding.py`, and the `test_server.py` test. *(Gated. If sign-off is refused, skip this commit and remove tests 3, 6 and 7 from the contract.)*

Push `feature/branding` and open a PR against `main` only after the orchestrator says so.

- **Title:** `Add logo mark, README header logo and dashboard favicon`
- **Body (English):**
  - Summary: what was added (theme-aware SVG mark and favicon using the `app.css` palette, with a light-theme override that darkens warn/crit for contrast).
  - Changes per file.
  - Note that `server.py` got one route and one MIME entry because static files are served from an allowlist.
  - Tests added, and the verify commands with their results.
  - Not verified: GitHub's light/dark rendering of the SVG.
  - Out of scope: the 0.1.0/0.2.0 version drift.
  - End the body with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

## 6. Do not touch

- The `__version__ = "0.1.0"` / `version = "0.2.0"` drift: `src/llama_doctor/__init__.py`, `tests/test_scaffold.py`.
- `pyproject.toml`, all of it, including the authors string and package-data.
- `.github/workflows/ci.yml` and any CI config.
- `docs/DESIGN.md` and `docs/contracts/`.
- Every Python module under `src/` **except** the two gated lines in `dashboard/server.py`.
- In the dashboard: `app.css`, `app.js`, `vendor/`, and everything in `index.html` except the one `<link>` line.
- README content below the inserted lines, including the badges and description line.
- Existing tests in `test_readme.py`, `test_static.py`, `test_server.py`. You may only add the one server test.
- Branches `feature/agentcheck` and `main`, and PRs #7 and #8. Don't rebase onto or merge them.
- Shared git stash: don't use bare `git stash`.

## 7. Risks

- **CI breakage:**
  - `tomllib` is missing on Python 3.10. CI only runs 3.11 and 3.13, and the test suite already assumes ≥3.10. If 3.10 is ever added to CI, test 7 will fail at import. *Mitigation if that happens: wrap the import with `skipUnless`.* Not needed today.
  - Ruff on the new test: default rules (E4/E7/E9/F) and line-length 100. Keep lines ≤100.
  - `test_readme_logo_sits_alone_above_title` breaks if anyone edits the logo line. That's on purpose.
- **Not verified:**
  - That GitHub's README renderer honours `prefers-color-scheme` inside an SVG loaded with `<img>`. Browsers generally do, but GitHub may proxy or sanitise it. Worst case, the dark tile shows on light themes, which is still legible.
  - Safari's support for SVG favicons (older Safari ignores them, so no icon, but nothing breaks).
  - The contrast and coverage numbers in the brief. I used them as given and didn't re-render.
  - That `gh` PR numbers #7 and #8 exist. My `gh pr list` call wasn't approved; I saw only local branches `feature/agentcheck`, `feature/branding`, `main`.
- **Scope deviation:** the two `server.py` lines (§0). Without them, scope item 4 can't work.

---

ORCHESTRATOR AUDIT AND AMENDMENTS (PART 2)


Audited live in this worktree on 2026-11-11, against `feature/branding` at `2058683`. Part 1 is
Opus's contract; Part 2 is the orchestrator's fact-check and the decisions only the orchestrator
can make. Implement Part 1 as amended by Part 2, then archive both.

## A1 — Gate signed off

Section 0's blocker is real and the decision is approved: `src/llama_doctor/dashboard/server.py`
gets exactly the two additions in §2.5 (one `STATIC_ROUTES` entry, one MIME entry), plus the one
test in §3.2. Commit 3 proceeds. This is the only Python change allowed in the branch.

## A2 — Facts confirmed by the orchestrator (independent read, same worktree)

- `README.md` first lines: `# llama-doctor`, blank, the sentence "A command-line tool that reads a
  running llama.cpp or Unsloth Studio server's metrics and tells you what is misconfigured.",
  blank, then exactly 3 badges (CI, Python 3.10+, License: MIT). Part 1 is right; the brief's
  README excerpt was wrong (it was reconstructed, not read — do not trust the brief's excerpts).
- `index.html` head: `<title>llama-doctor</title>`, `<link rel="stylesheet" href="/static/app.css">`,
  2-space indent, no self-closing slashes. Part 1 is right.
- `pyproject.toml`: static `version = "0.2.0"` (NOT dynamic), `authors = [{name = "Manolo Alsas"}]`
  (typo for Salsas, deliberately out of scope), `[tool.setuptools.package-data]
  "llama_doctor.dashboard" = ["static/*", "static/vendor/*"]`, `line-length = 100`,
  `target-version = "py310"`, `requires-python = ">=3.10"`.
- `src/llama_doctor/dashboard/server.py`: `STATIC_ROUTES` at line 16 and the `content_type` dict at
  line 137 exactly as quoted in Part 1; `default_static_dir()` is defined at line 38.
- `tests/test_readme.py:100` regex is `r"\]\(((?:docs|src)/[^)#]+|LICENSE)\)"` — Markdown links and
  `LICENSE` only, so an `<img src>` reference is indeed unchecked today and §3.1 test 1 earns its
  keep.
- `tests/test_server.py`: `setUp` builds a temp static root (`index.html`, `app.css`, `app.js`,
  `vendor/chart.umd.js`), `_request(method, path, ...)` returns `(status, headers, body)`, and
  `test_static_allowlist` is at line 99. §3.2's new test fits as written; put it right after that
  test.
- `tests/test_readme.py` imports via `from .support import make_context`, so `tests/` is a package;
  the new test module lives in `tests/` and needs no relative imports.

## A3 — Attribution overrides (repo convention, non-negotiable)

Delete the `Co-Authored-By: Claude ...` requirement from §5. Commit messages carry no AI
attribution of any kind: plain imperative subject, body only when the change needs one (see
`git log`). The PR body must not end with the "Generated with Claude Code" line either.

## A4 — Python-version guard in test 7

`requires-python = ">=3.10"` but `tomllib` is 3.11+. In `test_favicon_is_declared_package_data`
use:

```python
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None
```

and decorate the test with `@unittest.skipIf(tomllib is None, "tomllib requires Python 3.11")`.
Everything else stays stdlib-only and 3.10-safe.

## A5 — Toolchain inside the sandbox

`python` may not exist here; use `python3` in every §4 command. `ruff` is NOT installed in the
sandbox: skip `ruff check`, the orchestrator runs it outside the sandbox after your commits.
Keep every line at or under 100 characters (`target-version = "py310"`, ruff line-length 100).
Zero comments in new code unless one carries information the code cannot express.

## A6 — Archive location

Repo convention is `docs/contracts/T<n>-<topic>.md` (T1-core .. T5-dashboard). Archive as
`docs/contracts/T6-branding.md`: copy this transport file verbatim from the standalone line
`=== CONTRACT (PART 1) ===` onward, including Part 2 and Part 3, then delete the transport file. It
is commit 1's companion file, in the same commit as the two `docs/assets/*.svg` files.

## A7 — Boundaries

Commit on `feature/branding` only. No `git push`, no `gh pr create`, no PR comments, no branch
switching, no `git stash`, no touching `feature/agentcheck` or `main` or PRs #7/#8. Report back
instead; the orchestrator handles review, remote and PR.

## A8 — Byte-exact assets

The only authoritative SVG sources are Part 3 of this file. Copy between the BEGIN/END markers
character-for-character: no reflow, no added comments, no attribute reordering, exactly one
trailing newline, ASCII only. Then `cp docs/assets/favicon.svg
src/llama_doctor/dashboard/static/favicon.svg` so the two are byte-identical (`cmp` must pass).
Do not regenerate or "improve" the art: the geometry is what was measured.

## A9 — Header layout replaces §2.3 (verified pattern, do not revert)

Use the stacked-centred header that has been measured to survive GitHub's sanitizer (inline
`<img>` beside an `h1` cannot be centred because GitHub forces `vertical-align:
-webkit-baseline-middle`; tables render visible borders). Replace §2.3's block, the README's first
two lines becoming:

```markdown
<p align="center"><img src="docs/assets/logo-mark.svg" alt="llama-doctor logo" width="96" height="96"></p>

<h1 align="center">llama-doctor</h1>
```

Everything from the blank line after that heading down stays byte-identical (description sentence,
3 badges, the rest of the file). §3.1 test 2 therefore asserts instead:

```python
    def test_readme_header_layout(self):
        self.assertTrue(
            self.readme.startswith(
                '<p align="center"><img src="docs/assets/logo-mark.svg"'
                ' alt="llama-doctor logo" width="96" height="96"></p>\n\n'
                '<h1 align="center">llama-doctor</h1>\n'
            )
        )
```

Keep §3.1 test 1 unchanged; its `<img ... src>` regex still matches the wrapped tag. Rename the
test as shown and keep the reason comment in the contract record.


---

ASSET SOURCES (PART 3) — these are the measured candidate sources the contract refers to as "from the brief". Copy between the markers character-for-character; each file ends with exactly one newline.

### docs/assets/logo-mark.svg (source id A-bold)

<<<BEGIN A-bold>>>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" role="img" aria-label="llama-doctor">
  <style>
    .bg { fill: #121211; }
    .ink { fill: none; stroke: #f0efec; stroke-width: 7.5; stroke-linecap: round; stroke-linejoin: round; }
    .grid { fill: none; stroke: #2a2a28; stroke-width: 3.5; stroke-linecap: round; }
    .warn { fill: none; stroke: #fab219; stroke-width: 7.5; stroke-linecap: round; stroke-linejoin: round; }
    .crit { fill: #d03b3b; }
    @media (prefers-color-scheme: light) {
      .bg { fill: #f0efec; }
      .ink { fill: none; stroke: #2a2a28; stroke-width: 7.5; stroke-linecap: round; stroke-linejoin: round; }
      .grid { fill: none; stroke: #c3c2b7; stroke-width: 3.5; stroke-linecap: round; }
      .warn { fill: none; stroke: #8b6003; stroke-width: 7.5; stroke-linecap: round; stroke-linejoin: round; }
      .crit { fill: #c73030; }
    }
  </style>
  <rect x="4" y="4" width="88" height="88" rx="22" class="bg"/>
  <path class="grid" d="M22 34 H74 M22 62 H74"/>
  <path class="ink" d="M16 62 H29 L35 49 L44 73 L53 24"/>
  <path class="warn" d="M53 24 L61 45 L67 62 H80"/>
  <circle class="crit" cx="80" cy="62" r="6.0"/>
</svg>
<<<END A-bold>>>

### docs/assets/favicon.svg and src/llama_doctor/dashboard/static/favicon.svg (source id A-favicon)

<<<BEGIN A-favicon>>>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" role="img" aria-label="llama-doctor">
  <style>
    .bg { fill: #121211; }
    .ink { fill: none; stroke: #f0efec; stroke-width: 11; stroke-linecap: round; stroke-linejoin: round; }
    .grid { fill: none; stroke: #2a2a28; stroke-width: 3.5; stroke-linecap: round; display: none; }
    .warn { fill: none; stroke: #fab219; stroke-width: 11; stroke-linecap: round; stroke-linejoin: round; }
    .crit { fill: #d03b3b; }
    @media (prefers-color-scheme: light) {
      .bg { fill: #f0efec; }
      .ink { fill: none; stroke: #2a2a28; stroke-width: 11; stroke-linecap: round; stroke-linejoin: round; }
      .grid { fill: none; stroke: #c3c2b7; stroke-width: 3.5; stroke-linecap: round; display: none; }
      .warn { fill: none; stroke: #8b6003; stroke-width: 11; stroke-linecap: round; stroke-linejoin: round; }
      .crit { fill: #c73030; }
    }
  </style>
  <rect x="4" y="4" width="88" height="88" rx="20" class="bg"/>
  <path class="ink" d="M18 62 H31 L38 47 L47 75 L57 21"/>
  <path class="warn" d="M57 21 L65 45 L71 62 H78"/>
  <circle class="crit" cx="78" cy="62" r="7.0"/>
</svg>
<<<END A-favicon>>>

=== RECORD 2 — version single-source ===

# TRANSPORT FILE — version single-source fix (second task on branch feature/branding)

Part 1 below (after `=== SPEC (PART 1) ===`) is the architect's spec. Part 2 is the orchestrator's
audit and overrides, and WINS where they contradict. Part 3 is repo ground truth the orchestrator
read live from this same worktree.

## ORCHESTRATOR AUDIT AND OVERRIDES (PART 2)

- **O1 — Scope cut.** The author-metadata change (§5 of the spec) is HELD: the repo consistently
  says "Manolo Alsas" (pyproject.toml, LICENSE, git config) and the orchestrator's assumption that
  it should be "Salsas" was an inference from the GitHub handle, not a fact. The user has not
  confirmed a spelling yet. So: do NOT touch the `authors` line in `pyproject.toml`, do NOT touch
  `LICENSE`, do NOT change any git identity. One commit only, the version fix.
- **O2 — No attribution.** Drop both `Co-Authored-By: Claude ...` trailers and the
  `Generated with Claude Code` footer from §6. Repo convention: plain imperative subject, body only
  when the change needs one. Keep the subject and body text the spec proposes.
- **O3 — Confirmed by the orchestrator, live in this worktree.** `[build-system]
  requires = ["setuptools>=77"]` (so the dynamic-attr path is supported); `tests/test_scaffold.py`
  currently contains exactly one test (`test_version` pinning 0.1.0), so replacing the whole file
  with the spec's version loses nothing; the two golden strings in `tests/test_terminal.py` (lines
  63 and 91) contain no `{` or `}` characters, so f-string conversion is inert apart from the
  version token; `llama_doctor/render/terminal.py:66` builds the same first line from
  `__version__`, which is why the goldens break if you bump the version without touching them;
  `jsondoc.py:27` and `dashboard/sampler.py:250` also report `__version__`, so the stale 0.1.0
  leaks into `--json` and the dashboard `/state`, not just `--version`.
- **O4 — Tag fact now confirmed by the orchestrator** (the spec could not check it): the remote tag
  `v0.2.0` exists and points at `2058683` ("Version 0.2.0"), the current `main`. So the published
  release does report 0.1.0 from the CLI, `--json` and `/state`. Do not move, delete or re-push any
  tag; just record the fact. It belongs in the PR description.
- **O5 — Toolchain.** `python3` everywhere (no `python` alias here). `ruff` is not installed in the
  sandbox: skip `ruff check`, the orchestrator runs it outside. Keep lines ≤ 100 characters. No
  comments in code unless one carries information the code cannot express.
- **O6 — Venv check adjustment.** §7's editable-install check asserts `Author == "Manolo Salsas"`:
  drop that assertion (metadata unchanged per O1) and keep the version equality plus the CLI probe.
  It needs PyPI for build isolation; if the network blocks it, report that the step did not run —
  do not substitute a weaker claim.
- **O7 — Boundaries.** Work on `feature/branding` in this worktree, on top of the three existing
  branding commits. No `git push`, no `gh`, no PR, no branch switch, no `git stash`, no `git
  config`, no tag operations. `README.md` line 48 stays as printed (it is a captured 0.1.0
  transcript; changing one line of it would mix runs). Delete this transport file at the end and
  leave `git status --porcelain` empty.
- **O8 — Archive.** Append the spec (Part 1) plus this Part 2 and Part 3 to
  `docs/contracts/T6-branding.md` as a clearly delimited second record ("Record 2 — version
  single-source"), in the same commit as the fix. The existing 416-line record stays byte-identical.

Work order: apply the §2 edits except the author line; rewrite `tests/test_scaffold.py` with the
guard; run §7's verify list as amended by O5 and O6 (skip the Alsas grep for `Salsas`; the
`0\.1\.0` grep must come back empty except `./README.md:48:`); commit once with the spec's message
minus trailers; report `git show --stat`, each command with exit code and test counts, and anything
you could not run. If the full suite fails in a way the spec did not anticipate, stop and report.

=== SPEC (PART 1) ===

# Spec: version drift and author metadata (llama-doctor @ 2058683)

**Fact check.** I re-read the repo. Every stated fact holds, with these corrections and gaps:
- **Remote tags not checked.** `git tag -l` shows only `v0.1.0`. I was not permitted to run `git ls-remote`, so I could not confirm the remote `v0.2.0` tag or which commit it points to.
- **Three more places hardcode `0.1.0`:**
  - `tests/test_cli.py:169`: `self.assertIn("llama-doctor 0.1.0", stdout.getvalue())`
  - `tests/test_terminal.py:63` and `:91`: the golden strings start with `llama-doctor 0.1.0  http://server`
  - `README.md:48`: sample output, `llama-doctor 0.1.0  http://laptop-msi:8888`
- **The version is also used by `--json` and the dashboard.** `src/llama_doctor/render/jsondoc.py:27` and `src/llama_doctor/dashboard/sampler.py:250` read `__version__`. So the 0.1.0 also shows up in `--json` output and the dashboard `/state`, not only in `--version`.
- **No publish step exists.** `.github/workflows/` contains only `ci.yml`, and there is no PyPI or wheel-build workflow, so nothing builds a release wheel from `pyproject.toml`. CI does run `pip install -e . ruff`, which means CI builds the package metadata on every run.
- **The author name has a second hit:** `LICENSE:3`. Also, local `git config user.name` is "Manolo Alsas", so new commits will keep the old spelling until the orchestrator changes it (see §5).

## 1. Which fix

**Decision: make `pyproject.toml` read the version from `llama_doctor.__version__`, and make `src/llama_doctor/__init__.py` say `0.2.0`.**

The trade-off:
- **Just bumping `__init__.py`** fixes today's mismatch, but every future release still means editing two files that can drift apart again.
- **Reading the version from the code** leaves `__init__.py` as the only place the version is written, and the CLI, JSON output and dashboard already read from there. The cost is a three-line setuptools table. setuptools ≥77 (already required) reads the literal `__version__ = "..."` without importing the package. There is no PyPI artifact or release pipeline to adapt, and CI's `pip install -e .` already tests this build path on every push.

## 2. Version edits

**`src/llama_doctor/__init__.py`**
```python
# before
__version__ = "0.1.0"
# after
__version__ = "0.2.0"
```

**`pyproject.toml`**: remove the static version and add `dynamic`.
```toml
# before
[project]
name = "llama-doctor"
version = "0.2.0"
description = "Non-invasive health diagnosis for local LLM inference servers via /metrics and /props."
# after
[project]
name = "llama-doctor"
dynamic = ["version"]
description = "Non-invasive health diagnosis for local LLM inference servers via /metrics and /props."
```
Then add a new table right after `[tool.setuptools.packages.find]`:
```toml
# before
[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
# after
[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.dynamic]
version = {attr = "llama_doctor.__version__"}

[tool.setuptools.package-data]
```

**`tests/test_scaffold.py`**: this stays the one deliberate place where a test spells out the version; the guard from §3 goes in the same file. Replace the whole file with:
```python
import re
import unittest
from pathlib import Path

import llama_doctor

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


class TestScaffold(unittest.TestCase):
    def test_version(self):
        self.assertEqual(llama_doctor.__version__, "0.2.0")

    def test_pyproject_takes_version_from_package(self):
        text = PYPROJECT.read_text(encoding="utf-8")
        self.assertRegex(text, r'(?m)^dynamic = \["version"\]$')
        self.assertEqual(
            re.findall(r"(?m)^version\s*=.*$", text),
            ['version = {attr = "llama_doctor.__version__"}'],
        )


if __name__ == "__main__":
    unittest.main()
```

**`tests/test_cli.py`**: stop pinning the number.
```python
# before (imports, line 10-11)
from llama_doctor.analysis.base import HostHints
from llama_doctor.cli import EXIT_TOOL_ERROR, main
# after
from llama_doctor import __version__
from llama_doctor.analysis.base import HostHints
from llama_doctor.cli import EXIT_TOOL_ERROR, main
```
```python
# before (line 169)
        self.assertIn("llama-doctor 0.1.0", stdout.getvalue())
# after
        self.assertIn(f"llama-doctor {__version__}", stdout.getvalue())
```

**`tests/test_terminal.py`**: the golden strings read the version from the package. Neither string contains `{` or `}`, so turning them into f-strings changes nothing else.
```python
# before (imports, line 4)
from llama_doctor.analysis import diagnose
# after
from llama_doctor import __version__
from llama_doctor.analysis import diagnose
```
```python
# before (line 63)
            """llama-doctor 0.1.0  http://server
# after
            f"""llama-doctor {__version__}  http://server
```
```python
# before (line 91)
            """llama-doctor 0.1.0  http://server
# after
            f"""llama-doctor {__version__}  http://server
```

**`README.md:48`: leave it.** It is a captured transcript from a real 0.1.0 run, next to screenshots from that same run. Changing only the version line would make the sample a mix of two runs.

## 3. Guard against drift coming back

After §2, the version can't be set in two places, because `pyproject.toml` no longer has its own. The real risk is someone adding a static `version = "x.y.z"` back. `test_pyproject_takes_version_from_package` (in §2) catches that:
- It fails if any `^version =` line other than the `attr` line appears, or if `dynamic = ["version"]` is removed.
- It uses only `re` and `pathlib`, with no `tomllib` (that is 3.11+), so it runs on 3.10.

I rejected a check based on `importlib.metadata.version()`. It fails when the package isn't installed (the suite runs with `PYTHONPATH=src`), and an editable install keeps the version from install time, so it would give false failures after a local bump.

## 4. Tags

- **The fix does not depend on tags.**
- **Orchestrator step, before merge:**
  1. Run `git fetch --tags origin`.
  2. Run `git rev-parse 'v0.2.0^{commit}'` and record the result.
- **If the tag points to 2058683** (likely, given the commit message "Version 0.2.0"), the published v0.2.0 source reports `0.1.0` from `--version`, `--json` and `/state`. Do not move or re-push the tag. Rewriting a published release tag is worse than the cosmetic bug.
- Cutting a v0.2.1 release is out of scope for this spec and needs explicit user approval. Mention the issue in the PR description.

## 5. Author metadata

Hits for `Alsas|Salsas|Manolo`, skipping `.git` and `__pycache__`:

| file:line | content | action |
|---|---|---|
| `pyproject.toml:9` | `authors = [{name = "Manolo Alsas"}]` | change |
| `LICENSE:3` | `Copyright (c) 2026 Manolo Alsas` | change |

There are no hits for `Salsas`, and no other hits in `README.md`, `docs/`, `src/`, `tests/` or `.github/`.

```toml
# pyproject.toml, before
authors = [{name = "Manolo Alsas"}]
# after
authors = [{name = "Manolo Salsas"}]
```
```text
# LICENSE line 3, before
Copyright (c) 2026 Manolo Alsas
# after
Copyright (c) 2026 Manolo Salsas
```
Do not rewrite history or touch GitHub profiles.

**Orchestrator note (not for the implementer):**
- `git config user.name` is currently "Manolo Alsas", so these fix commits will themselves carry the misspelling.
- The orchestrator should commit with `-c user.name="Manolo Salsas"`, or the user should fix their git config. That change is outside the repo, so ask the user first.
- The surname spelling comes from the request, not from the repo. Every repo source says "Alsas".

## 6. Commit plan

The user asked for one PR. Unrelated changes in one PR make a selective revert harder, but these two are small and self-contained, so comply. Use separate commits, ordered after (or before) the branding commits, never mixed into them.

1. Commit 1:
```
fix: single-source version from llama_doctor.__version__

pyproject.toml declared 0.2.0 while __init__.py said 0.1.0, so
--version, --json and the dashboard reported 0.1.0. pyproject now
reads the version via [tool.setuptools.dynamic]; tests read
__version__ instead of pinning it, and test_scaffold guards against
a static version returning.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```
   Files: `pyproject.toml` (version lines only), `src/llama_doctor/__init__.py`, `tests/test_scaffold.py`, `tests/test_cli.py`, `tests/test_terminal.py`.

2. Commit 2:
```
chore: correct author surname to Salsas

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```
   Files: `pyproject.toml` (authors line), `LICENSE`.

**PR description:** mention the v0.2.0 tag finding from §4 and end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

## 7. Verify commands

Run from the worktree root:
```bash
grep -rn --exclude-dir=.git --exclude-dir=__pycache__ -E 'Alsas|0\.1\.0' . ; test "$(grep -rn --exclude-dir=.git --exclude-dir=__pycache__ -E 'Alsas|0\.1\.0' . | grep -v '^./README.md:48:' | wc -l)" -eq 0
PYTHONPATH=src python3 -m llama_doctor --version            # expect: llama-doctor 0.2.0
PYTHONPATH=src python3 -m unittest tests.test_scaffold -v
python3 -m venv /tmp/ld-venv && /tmp/ld-venv/bin/pip install -q -e . && /tmp/ld-venv/bin/python -c "import importlib.metadata as m, llama_doctor; v=m.version('llama-doctor'); a=m.metadata('llama-doctor')['Author']; assert v == llama_doctor.__version__ == '0.2.0', v; assert a == 'Manolo Salsas', a; print('ok', v, a)"
/tmp/ld-venv/bin/llama-doctor --version                     # expect: llama-doctor 0.2.0
ruff check src tests
PYTHONPATH=src python3 -m unittest discover -v
```
If `pip install` in the venv can't reach PyPI to fetch `setuptools>=77` for the isolated build, rerun with network access. Do not drop the check: it is the only local test that the version really comes from the code.

## Risks
- **The released tag is wrong:** if v0.2.0 points to 2058683, its source reports 0.1.0, and this fix only corrects `main` going forward.
- **Older pip:** `{attr = ...}` needs build isolation to pick up setuptools ≥77. A pip that is too old to honour `[build-system]` would fail to build. CI's pinned `setup-python@v5` is fine.
- **Python 3.10 isn't tested in CI:** the matrix runs 3.11 and 3.13 although `requires-python = ">=3.10"`. The new test avoids `tomllib`, but nothing in CI proves 3.10 works.
- **The guard is a text match:** reformatting the `attr` line (spacing, quote style) fails `test_pyproject_takes_version_from_package`. That is a false alarm, not missed drift.
- **New commits keep the old spelling:** they will still show "Alsas" unless the orchestrator overrides `user.name` (§5).

---

REPO GROUND TRUTH (PART 3) — read live in this worktree at HEAD of feature/branding (three branding commits on top of main = 2058683)

```
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "llama-doctor"
version = "0.2.0"
description = "Non-invasive health diagnosis for local LLM inference servers via /metrics and /props."
authors = [{name = "Manolo Alsas"}]
license = "MIT"
--- tests/test_scaffold.py (whole file, 12 lines)
import unittest

import llama_doctor


class TestScaffold(unittest.TestCase):
    def test_version(self):
        self.assertEqual(llama_doctor.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
--- tests/test_cli.py:166-170
            main(["--version"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("llama-doctor 0.1.0", stdout.getvalue())

--- tests/test_terminal.py:63 and :91
            """llama-doctor 0.1.0  http://server
            """llama-doctor 0.1.0  http://server
--- version consumers
src/llama_doctor/__init__.py:1:__version__ = "0.1.0"
src/llama_doctor/cli.py:7:from . import __version__
src/llama_doctor/cli.py:45:    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
src/llama_doctor/dashboard/sampler.py:7:from llama_doctor import __version__
src/llama_doctor/dashboard/sampler.py:250:                "version": __version__,
src/llama_doctor/render/jsondoc.py:3:from llama_doctor import __version__
src/llama_doctor/render/jsondoc.py:27:        "version": __version__,
src/llama_doctor/render/terminal.py:4:from llama_doctor import __version__
src/llama_doctor/render/terminal.py:66:        f"llama-doctor {__version__}  {target}",
```

<<<END Record 2 — version single-source>>>
### Record 2 addendum (orchestrator override O9, 2026-11-11)  Spec 2 said to leave the README sample version at 0.1.0 on the grounds that it is a captured transcript. That is wrong for this repo: tests/test_readme.py's test_quick_start_example_matches_rendered_fixture renders the readme_example fixture through render_report and asserts byte equality with the README fenced block, and render_report reads __version__ for its first line. Bumping the version therefore forces the sample version token to move to 0.2.0 in the same commit.
### Record 2 addendum (architect review R2, 2026-11-11)  The regex guard specified in Record 2 was replaced by the TOML-parsing guard proposed in the architect review: the regex matched any version = line anywhere in the file and failed on harmless reformatting. The replacement skips on Python 3.10, where tomllib is absent, and both CI versions (3.11 and 3.13) run it. Note that CI does not run 3.10 at all despite requires-python. The review also added assertions that the --json diagnosis document and the dashboard state report the package version, the two reporting paths that stayed silent during the drift.
