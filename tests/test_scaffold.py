import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

import llama_doctor

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


class TestScaffold(unittest.TestCase):
    def test_version(self):
        self.assertEqual(llama_doctor.__version__, "0.2.0")

    @unittest.skipIf(tomllib is None, "tomllib requires Python 3.11+")
    def test_pyproject_takes_version_from_package(self):
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        self.assertIn("version", data["project"]["dynamic"])
        self.assertNotIn("version", data["project"])
        self.assertEqual(
            data["tool"]["setuptools"]["dynamic"]["version"],
            {"attr": "llama_doctor.__version__"},
        )


if __name__ == "__main__":
    unittest.main()
