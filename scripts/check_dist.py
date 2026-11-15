"""Smoke checks for the built distribution, run in CI after packaging.

Takes the dist directory as argv[1] and exits non-zero on the first failure.
"""

import glob
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

DIST_NAME = "llama_doctor"
WHEEL_SUFFIX = "-py3-none-any.whl"
STATIC_PREFIX = "llama_doctor/dashboard/static/"
STATIC_FILES = ["index.html", "app.css", "app.js", "favicon.svg", "vendor/chart.umd.js"]
SDIST_ROOT_FILES = ["LICENSE", "README.md", "pyproject.toml"]


def fail(message: str) -> None:
    print(f"check_dist: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=600, check=False)
    if proc.returncode != 0:
        fail(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}{proc.stderr}")
    return proc


def resource_snippet(venv: Path) -> str:
    return (
        "import llama_doctor\n"
        "from importlib.resources import files\n"
        "from pathlib import Path\n"
        "static = files('llama_doctor.dashboard') / 'static'\n"
        f"for rel in {STATIC_FILES!r}:\n"
        "    item = static / rel\n"
        "    assert item.is_file(), rel\n"
        "    assert item.read_bytes(), rel\n"
        f"venv = Path({str(venv)!r}).resolve()\n"
        "assert Path(llama_doctor.__file__).resolve().is_relative_to(venv)\n"
    )


def main() -> int:
    if len(sys.argv) != 2:
        fail("usage: python -I scripts/check_dist.py DIST_DIR")
    dist = Path(sys.argv[1])

    wheels = sorted(glob.glob(str(dist / f"{DIST_NAME}-*{WHEEL_SUFFIX}")))
    sdists = sorted(glob.glob(str(dist / f"{DIST_NAME}-*.tar.gz")))
    if len(wheels) != 1 or len(sdists) != 1:
        fail(
            f"expected exactly one wheel and one sdist in {dist}, "
            f"got {len(wheels)} wheels and {len(sdists)} sdists"
        )
    wheel = Path(wheels[0])
    version = wheel.name.removeprefix(f"{DIST_NAME}-").removesuffix(WHEEL_SUFFIX)

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for rel in STATIC_FILES:
            wanted = STATIC_PREFIX + rel
            if wanted not in names:
                fail(f"wheel {wheel.name} is missing {wanted}")
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            fail(f"wheel {wheel.name} has {len(metadata_paths)} METADATA files")
        metadata = archive.read(metadata_paths[0]).decode("utf-8", errors="replace").splitlines()
    if any(line.startswith("Requires-Dist:") for line in metadata):
        fail(f"wheel {wheel.name} METADATA declares Requires-Dist")
    if "License-Expression: MIT" not in metadata:
        fail(f"wheel {wheel.name} METADATA lacks 'License-Expression: MIT'")

    sdist_name = Path(sdists[0]).name
    with tarfile.open(sdists[0]) as archive:
        names = archive.getnames()
    for rel in SDIST_ROOT_FILES:
        if not any(name == rel or name.endswith("/" + rel) for name in names):
            fail(f"sdist {sdist_name} is missing {rel}")
    for rel in STATIC_FILES:
        wanted = "src/" + STATIC_PREFIX + rel
        if not any(name == wanted or name.endswith("/" + wanted) for name in names):
            fail(f"sdist {sdist_name} is missing {wanted}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        venv = tmpdir / "venv"
        venv_python = venv / "bin" / "python"
        run([sys.executable, "-m", "venv", str(venv)])
        run([str(venv_python), "-m", "pip", "install", "--no-index", str(wheel.resolve())])

        proc = run([str(venv / "bin" / "llama-doctor"), "--version"], cwd=tmpdir)
        expected = f"llama-doctor {version}"
        if proc.stdout.strip() != expected:
            fail(f"--version printed {proc.stdout.strip()!r}, expected {expected!r}")

        run([str(venv_python), "-I", "-c", resource_snippet(venv)], cwd=tmpdir)

    print(f"check_dist: OK ({wheel.name}, {sdist_name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
