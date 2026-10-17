import os
import unittest
from unittest import mock

from llama_doctor.redact import Secret, safe_float, safe_int, safe_str


class TestSecret(unittest.TestCase):
    def test_secret_repr_masks(self):
        secret = Secret("s3cret-value")
        self.assertEqual(repr(secret), "Secret('***')")
        self.assertEqual(str(secret), "Secret('***')")
        self.assertEqual(secret.reveal(), "s3cret-value")

    def test_secret_from_env(self):
        with mock.patch.dict(os.environ, {"LLAMA_DOCTOR_KEY": "k3y"}, clear=True):
            secret = Secret.from_env("LLAMA_DOCTOR_KEY")
            self.assertIsNotNone(secret)
            self.assertEqual(secret.reveal(), "k3y")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(Secret.from_env("LLAMA_DOCTOR_KEY"))
        with mock.patch.dict(os.environ, {"LLAMA_DOCTOR_KEY": ""}, clear=True):
            self.assertIsNone(Secret.from_env("LLAMA_DOCTOR_KEY"))


class TestSafeStr(unittest.TestCase):
    def test_safe_str_strips_control_and_truncates(self):
        self.assertEqual(safe_str("a\x1b[31mb\n"), "a[31mb")
        long_value = "x" * 200
        self.assertEqual(safe_str(long_value), "x" * 128)
        self.assertEqual(len(safe_str(long_value)), 128)

    def test_safe_str_forbidden(self):
        self.assertIsNone(safe_str("value with SENTINEL inside", forbidden=("SENTINEL",)))
        self.assertEqual(safe_str("clean value", forbidden=("SENTINEL",)), "clean value")
        self.assertEqual(safe_str("anything", forbidden=("", "SENTINEL")), "anything")
        padded = "x" * 120 + "<__media_SENTINEL__>"
        self.assertIsNone(safe_str(padded, forbidden=("<__media_SENTINEL__>",)))
        self.assertIsNone(safe_str(123, forbidden=()))
        self.assertIsNone(safe_str(""))


class TestSafeNumbers(unittest.TestCase):
    def test_safe_int_rules(self):
        self.assertIsNone(safe_int(True))
        self.assertIsNone(safe_int(3.5))
        self.assertEqual(safe_int(4.0), 4)
        self.assertIsNone(safe_int("4"))
        self.assertEqual(safe_int(-7), -7)

    def test_safe_float_rejects_nonfinite(self):
        self.assertIsNone(safe_float(float("nan")))
        self.assertIsNone(safe_float(float("inf")))
        self.assertIsNone(safe_float(float("-inf")))
        self.assertIsNone(safe_float(10**400))
        self.assertIsNone(safe_float(True))
        self.assertEqual(safe_float(1.5), 1.5)
        self.assertEqual(safe_float(2), 2.0)


if __name__ == "__main__":
    unittest.main()
