import contextlib
import io
import json
import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

from llama_doctor.analysis.base import HostHints
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
        self.assertEqual(stdout.splitlines()[-1], "overall: INFO (exit 0)")

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
        self.assertIn("context.oversized", stdout)

    def test_serve_passes_vram_gib_to_dashboard(self):
        with patch("llama_doctor.dashboard.server.run", return_value=0) as run:
            result, _, _ = self._run(["serve", "http://127.0.0.1:1", "--vram-gib", "24"])

        self.assertEqual(result, 0)
        self.assertEqual(run.call_args.kwargs["hints"], HostHints(vram_gib=24.0))

    def test_serve_passes_weights_gib_to_dashboard(self):
        with patch("llama_doctor.dashboard.server.run", return_value=0) as run:
            self._run(["serve", "http://127.0.0.1:1", "--vram-gib", "24", "--weights-gib", "16.5"])

        self.assertEqual(run.call_args.kwargs["hints"], HostHints(vram_gib=24.0, weights_gib=16.5))

    def test_diagnose_uses_weights_gib_when_server_omits_size(self):
        models = json.dumps({"data": [{"id": "m"}]})
        with FakeLlamaServer() as server:
            _set_routes(server, metrics="metrics_healthy.txt")
            server.set_route("/v1/models", 200, models, "application/json")
            _, stdout, _ = self._run(
                ["diagnose", server.url, "--json", "--vram-gib", "24", "--weights-gib", "16.5"]
            )

        memory = [item for item in json.loads(stdout)["findings"] if item["analyzer"] == "memory"]
        self.assertEqual([item["code"] for item in memory], ["memory.fits_vram"])
        self.assertEqual(memory[0]["evidence"], {"weights_source": "hint"})

    def test_serve_without_vram_gib_passes_empty_hints(self):
        with patch("llama_doctor.dashboard.server.run", return_value=0) as run:
            self._run(["serve", "http://127.0.0.1:1"])

        self.assertEqual(run.call_args.kwargs["hints"], HostHints())

    def test_json_output(self):
        with FakeLlamaServer() as server:
            _set_routes(server)
            result, stdout, _ = self._run(["diagnose", server.url, "--json"])

        document = json.loads(stdout)
        self.assertEqual(result, 2)
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
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            main(["diagnose"])

        self.assertEqual(raised.exception.code, EXIT_TOOL_ERROR)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("the following arguments are required: URL", stderr.getvalue())

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
        self.assertEqual(result, 1)
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
        self.assertIn(
            "probe: warm-up plus one synthetic prompt (prefix cache disabled)",
            stderr,
        )
        self.assertIn("probe: failed", stderr)

    def test_probe_is_skipped_when_server_is_dead(self):
        with FakeLlamaServer() as server:
            result, _, stderr = self._run(["diagnose", server.url, "--probe"])

        self.assertEqual(result, 2)
        self.assertNotIn("probe: warm-up plus one synthetic prompt", stderr)
        self.assertNotIn("probe: failed", stderr)
        self.assertFalse(any(request[1] == "/completion" for request in server.requests))

    def test_no_color_flag_disables_color_output(self):
        with (
            FakeLlamaServer() as server,
            patch("llama_doctor.cli.use_color", return_value=True) as use_color,
            patch("llama_doctor.cli.render_report", return_value="report") as render_report,
        ):
            _set_routes(server)
            result, stdout, _ = self._run(["diagnose", server.url, "--no-color"])

        self.assertEqual(result, 2)
        self.assertEqual(stdout, "report\n")
        use_color.assert_called_once()
        self.assertFalse(render_report.call_args.kwargs["color"])

    def test_serve_lazily_imports_and_calls_dashboard_runner(self):
        dashboard_server = types.ModuleType("llama_doctor.dashboard.server")
        dashboard_server.run = Mock(return_value=17)
        with patch.dict(sys.modules, {"llama_doctor.dashboard.server": dashboard_server}):
            result, stdout, _ = self._run(
                ["serve", "http://server", "--host", "0.0.0.0", "--port", "9000",
                 "--interval", "3.5", "--history", "20", "--rate-window", "45", "--vram-gib", "12"]
            )

        self.assertEqual(result, 17)
        self.assertEqual(stdout, "")
        dashboard_server.run.assert_called_once()
        args, kwargs = dashboard_server.run.call_args
        self.assertEqual(args[0].base_url, "http://server")
        self.assertEqual(
            {key: value for key, value in kwargs.items() if key != "out"},
            {"host": "0.0.0.0", "port": 9000, "interval": 3.5, "history": 20, "rate_window": 45.0,
             "hints": HostHints(vram_gib=12.0)},
        )

    def test_serve_rejects_non_positive_rate_window(self):
        dashboard_server = types.ModuleType("llama_doctor.dashboard.server")
        dashboard_server.run = Mock(return_value=0)
        for value in ("0", "-5", "nan"):
            with self.subTest(value=value), patch.dict(
                sys.modules, {"llama_doctor.dashboard.server": dashboard_server}
            ):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    main(["serve", "http://server", "--rate-window", value])

                self.assertEqual(raised.exception.code, EXIT_TOOL_ERROR)
                self.assertIn("argument --rate-window: must be > 0", stderr.getvalue())
        dashboard_server.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
