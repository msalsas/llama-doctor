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
