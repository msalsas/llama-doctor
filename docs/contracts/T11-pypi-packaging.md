# T11: publish-ready PyPI packaging

Context: read `pyproject.toml`, `.github/workflows/ci.yml`, `README.md`, `src/llama_doctor/__init__.py`, `src/llama_doctor/dashboard/server.py`, `LICENSE`, `.gitignore`. Stdlib only, no em dashes. Name `llama-doctor`, version `0.2.0` (no bump), `dependencies = []`. No changes under `src/llama_doctor/` and none to `tests/`. Ruff config and the `test` job in `ci.yml` stay byte-identical.

Checked against the repo: the brief's facts all hold. `cli.py:68` wires `--version` to `__version__`. `[tool.setuptools.package-data]` declares `static/*` and `static/vendor/*`. The dashboard already loads its assets with `importlib.resources.files("llama_doctor.dashboard") / "static"` (`server.py:40`). `.gitignore` already covers `dist/`, `build/` and `*.egg-info/`. The brief left out one point: `license = "MIT"` is a PEP 639 SPDX expression, so adding a `License ::` classifier would make the build fail. Do not add one.

## 1. pyproject metadata

Add to `[project]`, leaving existing keys and order as they are:

```toml
readme = "README.md"
keywords = ["llama.cpp", "llm", "inference", "diagnostics", "metrics", "prometheus", "cli", "gguf"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Environment :: Web Environment",
    "Intended Audience :: Developers",
    "Intended Audience :: System Administrators",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: System :: Monitoring",
    "Typing :: Typed",
]

[project.urls]
Homepage = "https://github.com/msalsas/llama-doctor"
Repository = "https://github.com/msalsas/llama-doctor"
Issues = "https://github.com/msalsas/llama-doctor/issues"
```

Why Alpha: this is the first public release. It has only been tested against llama.cpp and Unsloth Studio. Finding ids, thresholds and advice still change between minor versions (T10 just withheld GPU advice). The JSON output is not frozen. Beta would claim a stability that 0.2.x does not offer. Move to Beta once the finding ids and the `--json` schema are declared stable.

`Typing :: Typed` only counts if the package ships `py.typed`. If it does not, drop that classifier. Do not add the file in this task.

`license`, `dynamic`, `[tool.setuptools.dynamic]` and `package-data` stay unchanged. setuptools>=77 includes `LICENSE` through the default `license-files` globs.

## 2. Packaging job: `.github/workflows/package.yml` (new)

This is a separate workflow rather than a job inside `ci.yml`, so `ci.yml` stays untouched and `release.yml` can reuse it via `workflow_call`.

```yaml
name: Package

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_call:

permissions:
  contents: read

jobs:
  package:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: |
            3.10
            3.13
      - name: Build sdist and wheel
        run: |
          python -m pip install "build>=1.2" "twine>=6.1"
          python -m build
      - name: Twine check
        run: python -m twine check --strict dist/*
      - name: Smoke test on 3.10 and 3.13
        run: |
          python3.10 -I scripts/check_dist.py dist
          python -I scripts/check_dist.py dist
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
          if-no-files-found: error
```

`twine>=6.1` is required because setuptools>=77 writes Metadata-Version 2.4 (`License-Expression`), which older twine rejects.

`scripts/check_dist.py` (new, stdlib only, CI tooling rather than a test) takes the dist directory as `argv[1]` and exits non-zero on the first failure:

1. Exactly one `llama_doctor-*-py3-none-any.whl` and one `llama_doctor-*.tar.gz` exist. Take `VERSION` from the wheel filename.
2. Wheel listing (`zipfile`): it contains `llama_doctor/dashboard/static/{index.html,app.css,app.js,favicon.svg}` and `llama_doctor/dashboard/static/vendor/chart.umd.js`. Its `*.dist-info/METADATA` has no `Requires-Dist:` line and has `License-Expression: MIT`.
3. Sdist listing (`tarfile`): it contains `LICENSE`, `README.md`, `pyproject.toml` and the same five static files under `src/`.
4. Create a fresh venv in a `tempfile.TemporaryDirectory()` using `sys.executable -m venv`, then run `<venv>/bin/python -m pip install --no-index <wheel>`. `--no-index` proves the wheel has no dependencies.
5. With `cwd` set to the temp directory (so `src/` cannot leak in), run `<venv>/bin/llama-doctor --version`. Stdout stripped must equal `f"llama-doctor {VERSION}"`.
6. With the same `cwd`, run `<venv>/bin/python -I -c` with a snippet that does `from importlib.resources import files`. It asserts that each of the five paths, relative to `files("llama_doctor.dashboard") / "static"`, `.is_file()` is true and is non-empty. It also asserts that `llama_doctor.__file__` resolves inside the venv.

## 3. Release workflow: `.github/workflows/release.yml` (new)

```yaml
name: Release

on:
  push:
    tags: ["v*"]

permissions:
  contents: read

jobs:
  package:
    uses: ./.github/workflows/package.yml

  publish:
    needs: package
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/project/llama-doctor/
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - name: Tag matches packaged version
        env:
          TAG: ${{ github.ref_name }}
        run: |
          test -f "dist/llama_doctor-${TAG#v}-py3-none-any.whl"
          test -f "dist/llama_doctor-${TAG#v}.tar.gz"
      - uses: pypa/gh-action-pypi-publish@release/v1
```

- The workflow builds once. `publish` uploads the exact files that `package` built, checked with twine and smoke-tested in the same run.
- The workflow stores no secrets or API tokens. Only `publish` gets `id-token: write`. The publish step must stay in `release.yml`, because PyPI trusted publishing does not accept a publish step inside a reusable workflow.
- A tag such as `v0.2.1` pushed while `__version__` is `0.2.0` fails the tag check, and nothing is uploaded.

## 4. One-time manual setup (owner)

1. On PyPI, go to Account → Publishing → "Add a new pending publisher" (GitHub). Enter PyPI project `llama-doctor`, owner `msalsas`, repository `llama-doctor`, workflow `release.yml` and environment `pypi`. A pending publisher does not reserve the name; the project is created on the first successful upload.
2. On GitHub, go to repo Settings → Environments → New environment `pypi`. Under deployment branches and tags, choose "Selected", then add the tag rule `v*`. Optionally add yourself as a required reviewer as a manual gate.
3. Release: `git tag v0.2.0 && git push origin v0.2.0` on the merged `main` commit.
4. After the first upload, PyPI turns the pending publisher into a normal one. Optionally enable 2FA-only and remove any API tokens.

## 5. README changes

Replace lines 25-39 (from `## Install` through the `PYTHONPATH=src` code block) with:

````markdown
## Install

Requires Python 3.10 or newer. There are no runtime dependencies; it uses the standard library only.

The recommended way is [pipx](https://pipx.pypa.io/), which installs the command in its own environment:

```sh
pipx install llama-doctor
```

Or with pip, ideally inside a virtual environment:

```sh
pip install llama-doctor
```

Check the install with `llama-doctor --version`.

### From source

For development, or to run unreleased changes:

```sh
git clone https://github.com/msalsas/llama-doctor.git
cd llama-doctor
pip install -e .
```

To run it without installing anything, from the repository root (the `PYTHONPATH=src` commands only resolve the package from there):

```sh
PYTHONPATH=src python3 -m llama_doctor diagnose http://127.0.0.1:8080
```
````

In `## Development`, add this after the existing code block and before "Pull requests are welcome.":

````markdown
To check the package the way CI does:

```sh
pip install build "twine>=6.1"
python3 -m build
python3 -m twine check --strict dist/*
python3 -I scripts/check_dist.py dist
```
````

PyPI renders the README with relative paths broken. Rewrite the four image sources (`docs/assets/logo-mark.svg`, `dashboard.png`, `dashboard-findings.png`, `findings.png`) as `https://raw.githubusercontent.com/msalsas/llama-doctor/main/docs/assets/<file>`. Rewrite the `LICENSE` links and the `src/llama_doctor/analysis/` link as `https://github.com/msalsas/llama-doctor/blob/main/<path>`. No other README text changes.

## Requirements

- **R1** `pyproject.toml` has exactly the section 1 additions. `name`, `version` source, `license`, `requires-python`, `dependencies = []`, `package-data` and `[tool.ruff]` are unchanged. It has no `License ::` classifier.
- **R2** `python -m build` from a clean checkout produces `dist/llama_doctor-0.2.0-py3-none-any.whl` and `dist/llama_doctor-0.2.0.tar.gz` with no warnings about license or readme.
- **R3** `python -m twine check --strict dist/*` passes with twine>=6.1.
- **R4** `python -I scripts/check_dist.py dist` exits 0 on Python 3.10 and 3.13, performing checks 1-6 of section 2.
- **R5** Each failure mode makes `check_dist.py` exit non-zero: a static file removed from `package-data`, a `Requires-Dist` added, or `--version` output that does not match the wheel version. Verify each once locally by temporarily editing a copy; do not commit those edits.
- **R6** `package.yml` and `release.yml` match section 2 and section 3. Both parse as YAML (`python -c "import sys,yaml; yaml.safe_load(open(sys.argv[1]))" <file>`, or `actionlint` if available). Top-level permissions are `contents: read`, and only `publish` has `id-token: write`. Neither file contains `secrets.` or `password:`.
- **R7** `ci.yml`, `src/llama_doctor/` and `tests/` show no diff (`git diff --stat main -- .github/workflows/ci.yml src tests` is empty).
- **R8** README Install and Development match section 5 verbatim. The image and link sources are absolute, and no other README text changes.
- **R9** `python -m unittest discover -v` (with `PYTHONPATH=src`) and `ruff check src tests scripts` pass.
- **R10** Section 4 appears in the PR description so the owner can do the setup. No tag is pushed as part of this task.

## Open questions

- OQ1: Add `py.typed` (a one-line data-file change) in a follow-up so that `Typing :: Typed` is honest. Until then, the implementer drops that classifier if the file is absent.
- OQ2: Publish to TestPyPI first through a second environment, `testpypi`? This is out of scope here; the owner decides.
