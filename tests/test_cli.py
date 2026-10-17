import contextlib
import io
import json
import os
import unittest
from unittest.mock import patch

from llama_doctor.cli import EXIT_TOOL_ERROR, main

from .support import FakeLlamaServer, load_fixture


def _set_routes(
    server: FakeLlamaServer,
    *,
    metrics: str = "metrics_moe_offload.txt",
    props: str = "props_full.json",
    models: str = "models_full.json",
) -> None:
    server.set_route("/metrics", 200, load_fixture(metrics))
    server.set_route("/props", 200, load_fixture(props), "application/json")
    server.set_route("/v1/models", 200, load_fixture(models), "application/json")


class TestCli(unittest.TestCase):
    def _run(self, args: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = main(args)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_diagnose_moe_exit_2(self):
        with FakeLlamaServer() as server:
            _set_routes(server)
            result, stdout, _ = self._run(["diagnose", server.url])

        self.assertEqual(result, 2)
        self.assertIn("overall: CRITICAL (exit 2)", stdout)

    def test_diagnose_healthy_exit_0(self):
        with FakeLlamaServer() as server:
            _set_routes(server, metrics="metrics_healthy.txt")
            result, stdout, _ = self._run(["diagnose", server.url])

        self.assertEqual(result, 0)
        self.assertIn("overall:", stdout)

    def test_diagnose_exit_1_on_warn(self):
        props = json.dumps({
            "default_generation_settings": {"n_ctx": 65536},
            "model_path": "m-Q8_0.gguf",
        })
        with FakeLlamaServer() as server:
            server.set_route("/metrics", 200, load_fixture("metrics_healthy.txt"))
            server.set_route("/props", 200, props, "application/json")
            server.set_route("/v1/models", 200, load_fixture("models_full.json"), "application/json")
            result, stdout, _ = self._run(["diagnose", server.url])

        self.assertEqual(result, 1)
        self.assertIn("kv.context_oversized", stdout)

    def test_json_output(self):
        with FakeLlamaServer() as server:
            _set_routes(server)
            result, stdout, _ = self._run(["diagnose", server.url, "--json"])

        document = json.loads(stdout)
        self.assertIn(result, (0, 1, 2))
        self.assertEqual(document["exit_code"], 2)
        self.assertEqual(document["rates"]["prefill_tps"], 40.0)
        self.assertEqual(document["server"]["quant"], "UD-Q4_K_XL")
        self.assertNotIn("SENTINEL", stdout)

    def test_unreachable_exit_2(self):
        result, stdout, _ = self._run(["diagnose", "http://127.0.0.1:1"])

        self.assertEqual(result, 2)
        self.assertIn("connectivity.unreachable", stdout)

    def test_api_key_from_env(self):
        with FakeLlamaServer() as server, patch.dict(os.environ, {"MY_KEY": "k3y"}):
            _set_routes(server)
            result, stdout, stderr = self._run(
                ["diagnose", server.url, "--api-key-env", "MY_KEY"]
            )

        self.assertEqual(result, 2)
        self.assertEqual(server.requests[0][2]["Authorization"], "Bearer k3y")
        self.assertNotIn("k3y", stdout)
        self.assertNotIn("k3y", stderr)

    def test_missing_key_env_exit_3(self):
        with patch.dict(os.environ, {}, clear=True):
            result, _, stderr = self._run(
                ["diagnose", "http://127.0.0.1:1", "--api-key-env", "MISSING_KEY"]
            )

        self.assertEqual(result, EXIT_TOOL_ERROR)
        self.assertIn("environment variable MISSING_KEY is not set or empty", stderr)

    def test_invalid_url_exit_3(self):
        result, _, stderr = self._run(["diagnose", "localhost:8080"])

        self.assertEqual(result, EXIT_TOOL_ERROR)
        self.assertIn("base URL scheme must be http or https", stderr)

    def test_usage_error_exit_3(self):
        with self.assertRaises(SystemExit) as raised:
            main(["diagnose"])

        self.assertEqual(raised.exception.code, EXIT_TOOL_ERROR)

    def test_version(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            main(["--version"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("llama-doctor 0.1.0", stdout.getvalue())

    def test_probe_overrides_lifetime(self):
        timings = json.dumps({
            "timings": {
                "prompt_n": 1200,
                "prompt_ms": 1500.0,
                "predicted_n": 64,
                "predicted_ms": 3200.0,
            }
        })
        with FakeLlamaServer() as server:
            _set_routes(server)
            server.set_route("/completion", 200, timings, "application/json")
            result, stdout, _ = self._run(["diagnose", server.url, "--probe", "--json"])

        document = json.loads(stdout)
        self.assertIn(result, (0, 1, 2))
        self.assertEqual(document["probe"]["prefill_tps"], 800.0)
        self.assertNotIn(
            "prefill.moe_cpu_experts",
            [finding["code"] for finding in document["findings"]],
        )
        self.assertEqual(sum(request[1] == "/completion" for request in server.requests), 2)

    def test_probe_failure_falls_back(self):
        with FakeLlamaServer() as server:
            _set_routes(server)
            result, _, stderr = self._run(["diagnose", server.url, "--probe"])

        self.assertEqual(result, 2)
        self.assertIn("probe: failed", stderr)


if __name__ == "__main__":
    unittest.main()
