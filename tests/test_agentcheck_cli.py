import contextlib
import io
import json
import os
import unittest
from unittest import mock

from llama_doctor import agentcheck
from llama_doctor.cli import main

from .agentcheck_fakes import ChatResponder, agent_server, call, reply

CHAT_PATH = "/v1/chat/completions"


def run_cli(args):
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(args)
    return code, stdout.getvalue(), stderr.getvalue()


class TestAgentcheckCli(unittest.TestCase):
    def test_perfect_server_exits_zero(self):
        with agent_server() as server:
            code, stdout, stderr = run_cli(["agentcheck", server.url, "--replays", "2"])
        self.assertEqual(code, 0)
        self.assertTrue(stdout.rstrip().endswith("overall: PASS (exit 0)"))
        self.assertIn("agentcheck: server family llama.cpp", stderr)

    def test_one_wrong_case_exits_one(self):
        overrides = {
            "select_read_file": reply(tool_calls=[call("search_files", {"pattern": "x"})]),
        }
        with agent_server(responder=ChatResponder(overrides=overrides)) as server:
            code, stdout, _ = run_cli(["agentcheck", server.url, "--replays", "2"])
        self.assertEqual(code, 1)
        self.assertTrue(stdout.rstrip().endswith("overall: WARN (exit 1)"))

    def test_one_empty_replay_exits_two(self):
        agentic = [reply(content=None)] + [reply(content="working on it")] * 5
        with agent_server(responder=ChatResponder(agentic=agentic)) as server:
            code, stdout, _ = run_cli(["agentcheck", server.url, "--replays", "2"])
        self.assertEqual(code, 2)
        self.assertTrue(stdout.rstrip().endswith("overall: FAIL (exit 2)"))

    def test_chat_http_401_exits_three(self):
        class UnauthorizedResponder:
            def __call__(self, body):
                return 401, "unauthorized", "text/plain"

        with agent_server(responder=UnauthorizedResponder()) as server:
            code, _, stderr = run_cli(["agentcheck", server.url])
        self.assertEqual(code, 3)
        self.assertIn("HTTP 401", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_chat_invalid_json_exits_three(self):
        class TextResponder:
            def __call__(self, body):
                return 200, "this is not json", "text/plain"

        with agent_server(responder=TextResponder()) as server:
            code, _, stderr = run_cli(["agentcheck", server.url])
        self.assertEqual(code, 3)
        self.assertIn("invalid JSON", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_bad_arguments_exit_three(self):
        cases = [
            ["agentcheck", "http://127.0.0.1:9", "--timeout", "0"],
            ["agentcheck", "http://127.0.0.1:9", "--timeout", "-1"],
            ["agentcheck", "http://127.0.0.1:9", "--replays", "0"],
            ["agentcheck", "http://127.0.0.1:9", "--max-tokens", "abc"],
        ]
        for args in cases:
            with self.subTest(args=args[2:]):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    main(args)
                self.assertEqual(raised.exception.code, 3)

    def test_unexpected_error_exits_three_without_traceback(self):
        def boom(*args, **kwargs):
            raise RuntimeError("synthetic crash")

        with agent_server() as server:
            with mock.patch.object(agentcheck, "run_agentcheck", boom):
                code, _, stderr = run_cli(["agentcheck", server.url])
        self.assertEqual(code, 3)
        self.assertIn("llama-doctor: unexpected error: synthetic crash", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_unset_api_key_env_exits_three(self):
        os.environ.pop("LLAMA_DOCTOR_AGENTCHECK_TEST_KEY", None)
        code, _, stderr = run_cli(
            ["agentcheck", "http://127.0.0.1:9", "--api-key-env",
             "LLAMA_DOCTOR_AGENTCHECK_TEST_KEY"])
        self.assertEqual(code, 3)
        self.assertIn("LLAMA_DOCTOR_AGENTCHECK_TEST_KEY", stderr)

    def test_api_key_env_sends_bearer_header(self):
        os.environ["LLAMA_DOCTOR_AGENTCHECK_TEST_KEY"] = "secret-key"
        self.addCleanup(os.environ.pop, "LLAMA_DOCTOR_AGENTCHECK_TEST_KEY", None)
        with agent_server() as server:
            code, _, _ = run_cli([
                "agentcheck", server.url, "--replays", "1",
                "--api-key-env", "LLAMA_DOCTOR_AGENTCHECK_TEST_KEY",
            ])
        self.assertEqual(code, 0)
        chat_headers = [headers for _, path, headers, _ in server.requests if path == CHAT_PATH]
        self.assertTrue(chat_headers)
        for headers in chat_headers:
            self.assertEqual(headers["Authorization"], "Bearer secret-key")

    def test_json_output_and_request_counts(self):
        with agent_server() as server:
            code, stdout, _ = run_cli(
                ["agentcheck", server.url, "--json", "--replays", "2"])
        document = json.loads(stdout)
        self.assertEqual(document["exit_code"], code)
        chat_requests = [body for _, path, _, body in server.requests if path == CHAT_PATH]
        self.assertEqual(len(chat_requests), 10 + 3 * 2)
        phase_b = 0
        for body in chat_requests:
            payload = json.loads(body)
            if payload["messages"][-1]["role"] == "tool":
                phase_b += 1
                self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(phase_b, 3 * 2)


if __name__ == "__main__":
    unittest.main()
