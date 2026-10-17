import unittest

import llama_doctor


class TestScaffold(unittest.TestCase):
    def test_version(self):
        self.assertEqual(llama_doctor.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
