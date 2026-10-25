import io
import json
import time
import unittest

from llama_doctor.agentcheck.cases import CONFORMANCE_CASES
from llama_doctor.agentcheck.chat import ChatClient, ChatReply, ToolCall
from llama_doctor.agentcheck.checks import (
    MIN_REPLAY_CTX,
    AgentCheckResult,
    CaseResult,
    ConformanceResult,
    EmptyTurnResult,
    Outcome,
    ReplayResult,
    TurnResult,
    Verdict,
    classify,
    conformance_verdict,
    empty_turn_verdict,
    evaluate_case,
    has_tool_markup,
    run_conformance,
    run_empty_turn,
    split_think,
)
from llama_doctor.agentcheck.family import Family, ServerFamily
from llama_doctor.agentcheck.remediation import Symptom
from llama_doctor.client import LlamaClient

from .agentcheck_fakes import agent_server, perfect_reply, reply

CASES_BY_ID = {case.id: case for case in CONFORMANCE_CASES}


def make_reply(content=None, tool_calls=(), reasoning=None, finish="stop"):
    return ChatReply(content, reasoning, tuple(tool_calls), finish, 9000, 10)


class TestSplitThink(unittest.TestCase):
    def test_none_and_plain_text(self):
        self.assertEqual(split_think(None), ("", ""))
        self.assertEqual(split_think("hello"), ("hello", ""))

    def test_closed_think(self):
        self.assertEqual(split_think("<think>deep</think>answer"), ("answer", "deep"))

    def test_unclosed_think(self):
        self.assertEqual(split_think("<think>deep"), ("", "deep"))


class TestHasToolMarkup(unittest.TestCase):
    def test_markup_variants(self):
        self.assertTrue(has_tool_markup('<tool_call>{"name": "read_file"}</tool_call>'))
        self.assertTrue(has_tool_markup("[TOOL_CALLS]"))
        self.assertTrue(has_tool_markup("<|tool_call|>"))
        self.assertTrue(has_tool_markup("<|python_tag|>"))
        self.assertTrue(has_tool_markup("<function=read_file>"))
        self.assertTrue(has_tool_markup('{"name": "web_search"}'))
        self.assertFalse(has_tool_markup("just some text"))
        self.assertFalse(has_tool_markup('{"name": "other_thing"}'))


class TestClassify(unittest.TestCase):
    def test_text_is_actionable(self):
        self.assertEqual(classify(make_reply(content="done")), Outcome.ACTIONABLE)

    def test_tool_calls_are_actionable_with_empty_content(self):
        tool_call = ToolCall("c1", "read_file", {"path": "a"}, None)
        self.assertEqual(classify(make_reply(content="", tool_calls=(tool_call,))),
                         Outcome.ACTIONABLE)

    def test_empty_for_stop_without_content(self):
        self.assertEqual(classify(make_reply(content=None, finish="stop")), Outcome.EMPTY)

    def test_empty_for_length_with_empty_content(self):
        self.assertEqual(classify(make_reply(content="", finish="length")), Outcome.EMPTY)

    def test_reasoning_only_is_thought_runaway(self):
        self.assertEqual(classify(make_reply(content=None, reasoning="hmm")),
                         Outcome.THOUGHT_RUNAWAY)

    def test_unclosed_inline_think_is_thought_runaway(self):
        self.assertEqual(classify(make_reply(content="<think>hmm")), Outcome.THOUGHT_RUNAWAY)

    def test_closed_think_with_answer_is_actionable(self):
        self.assertEqual(classify(make_reply(content="<think>hmm</think>answer")),
                         Outcome.ACTIONABLE)

    def test_closed_think_without_answer_is_thought_runaway(self):
        self.assertEqual(classify(make_reply(content="<think>hmm</think>")),
                         Outcome.THOUGHT_RUNAWAY)


class TestEvaluateCase(unittest.TestCase):
    def test_parallel_case_accepts_either_order(self):
        case = CASES_BY_ID["parallel_calls"]
        calls = (
            ToolCall("c1", "read_file", {"path": "setup.cfg"}, None),
            ToolCall("c2", "read_file", {"path": "README.md"}, None),
        )
        self.assertTrue(evaluate_case(case, make_reply(tool_calls=calls)).passed)
        single = (ToolCall("c1", "read_file", {"path": "README.md"}, None),)
        result = evaluate_case(case, make_reply(tool_calls=single))
        self.assertIn(Symptom.NO_PARALLEL_CALLS, result.symptoms)
        self.assertFalse(result.passed)

    def test_unknown_tool_is_hallucinated(self):
        case = CASES_BY_ID["select_read_file"]
        calls = (ToolCall("c1", "deploy_to_kubernetes", {}, None),)
        result = evaluate_case(case, make_reply(tool_calls=calls))
        self.assertIn(Symptom.HALLUCINATED_TOOL, result.symptoms)

    def test_long_tool_name_is_briefed_in_detail_and_called(self):
        case = CASES_BY_ID["tool_choice_auto_no_tool"]
        name = "bad\x00tool\n" + "x" * 5000
        calls = (ToolCall("c1", name, None, None),)
        result = evaluate_case(case, make_reply(tool_calls=calls))
        self.assertIn(Symptom.HALLUCINATED_TOOL, result.symptoms)
        self.assertLess(len(result.detail), 200)
        self.assertEqual(result.called, ("bad tool " + "x" * 71 + "…",))
        self.assertTrue(all(ord(char) >= 32 and ord(char) != 127 for char in result.detail))
        for entry in result.called:
            self.assertTrue(all(ord(char) >= 32 and ord(char) != 127 for char in entry))

    def test_tool_choice_none(self):
        case = CASES_BY_ID["tool_choice_none"]
        calls = (ToolCall("c1", "read_file", {"path": "notes.txt"}, None),)
        result = evaluate_case(case, make_reply(tool_calls=calls))
        self.assertIn(Symptom.TOOL_CHOICE_IGNORED, result.symptoms)
        leaked = evaluate_case(case, make_reply(content='<tool_call>{"name": "read_file"}'
                                                "</tool_call>"))
        self.assertIn(Symptom.TEMPLATE_LEAK, leaked.symptoms)

    def test_markup_inside_think_block_is_not_template_leak(self):
        case = CASES_BY_ID["tool_choice_none"]
        content = "<think>planning a <tool_call>{\"name\": \"read_file\"}</tool_call></think>"
        result = evaluate_case(case, make_reply(content=content))
        self.assertNotIn(Symptom.TEMPLATE_LEAK, result.symptoms)
        self.assertIn(Symptom.THOUGHT_RUNAWAY, result.symptoms)

    def test_wrong_argument_type_names_path(self):
        case = CASES_BY_ID["args_integer"]
        calls = (ToolCall("c1", "web_search", {"query": "reasoning budget",
                                               "max_results": "5"}, None),)
        result = evaluate_case(case, make_reply(tool_calls=calls))
        self.assertIn(Symptom.MALFORMED_TOOL_ARGS, result.symptoms)
        self.assertIn("$.max_results", result.detail)

    def test_unknown_tool_refusal(self):
        case = CASES_BY_ID["unknown_tool_refusal"]
        polite = evaluate_case(case, make_reply(content="I cannot deploy; the tool does not "
                                                      "exist, but I can prepare manifests."))
        self.assertTrue(polite.passed)
        clarify = (ToolCall("c1", "clarify", {"question": "Which cluster?"}, None),)
        self.assertTrue(evaluate_case(case, make_reply(tool_calls=clarify)).passed)
        run = (ToolCall("c1", "run_command", {"command": "deploy"}, None),)
        self.assertIn(Symptom.WRONG_TOOL,
                      evaluate_case(case, make_reply(tool_calls=run)).symptoms)

    def test_expected_call_with_no_reply_is_empty_turn(self):
        case = CASES_BY_ID["select_read_file"]
        result = evaluate_case(case, make_reply(content=None))
        self.assertIn(Symptom.EMPTY_TURN, result.symptoms)

    def test_wrong_tool_reported(self):
        case = CASES_BY_ID["select_read_file"]
        calls = (ToolCall("c1", "search_files", {"pattern": "x"}, None),)
        result = evaluate_case(case, make_reply(tool_calls=calls))
        self.assertIn(Symptom.WRONG_TOOL, result.symptoms)
        self.assertEqual(result.detail, "expected read_file, got search_files")

    def test_wrong_arguments_reported_as_hint_mismatch(self):
        case = CASES_BY_ID["select_read_file"]
        calls = (ToolCall("c1", "read_file", {"path": "somewhere/else.py"}, None),)
        result = evaluate_case(case, make_reply(tool_calls=calls))
        self.assertIn(Symptom.WRONG_TOOL, result.symptoms)
        self.assertEqual(
            result.detail,
            "expected read_file with arguments path~'src/app/main.py', got read_file",
        )


class TestVerdicts(unittest.TestCase):
    def test_conformance_verdict_map(self):
        for score, expected in ((10, Verdict.PASS), (9, Verdict.WARN), (8, Verdict.WARN),
                                (7, Verdict.FAIL)):
            with self.subTest(score=score):
                self.assertEqual(conformance_verdict(score, 10), expected)

    def test_empty_turn_verdict_cases(self):
        actionable = {outcome: 0 for outcome in Outcome}
        actionable[Outcome.ACTIONABLE] = 6
        self.assertEqual(empty_turn_verdict(actionable), Verdict.PASS)
        one_empty = {**actionable, Outcome.EMPTY: 1, Outcome.ACTIONABLE: 5}
        self.assertEqual(empty_turn_verdict(one_empty), Verdict.FAIL)
        runaway = {**actionable, Outcome.THOUGHT_RUNAWAY: 1, Outcome.ACTIONABLE: 5}
        self.assertEqual(empty_turn_verdict(runaway), Verdict.WARN)
        all_error = {**actionable, Outcome.ACTIONABLE: 0, Outcome.ERROR: 6}
        self.assertEqual(empty_turn_verdict(all_error), Verdict.FAIL)
        one_error = {**actionable, Outcome.ACTIONABLE: 5, Outcome.ERROR: 1}
        self.assertEqual(empty_turn_verdict(one_error), Verdict.WARN)


class TestRunEmptyTurn(unittest.TestCase):
    def test_skipped_when_context_too_small(self):
        with agent_server() as server:
            chat = ChatClient(LlamaClient(server.url), None)
            result = run_empty_turn(chat, replays=2, max_tokens=512, n_ctx=8192)
        self.assertEqual(server.requests, [])
        self.assertEqual(result.skipped, f"context 8192 tokens is below the {MIN_REPLAY_CTX} "
                         "needed for the replay")
        self.assertEqual(result.verdict, Verdict.WARN)
        self.assertEqual(result.symptoms, (Symptom.CONTEXT_TOO_SMALL,))

    def test_timeout_on_replay_gives_error_outcome(self):
        def slow_responder(body):
            payload = json.loads(body)
            if payload["messages"][-1]["role"] == "tool":
                time.sleep(0.5)
            return 200, reply(content="still working on it"), "application/json"

        with agent_server(responder=slow_responder) as server:
            chat = ChatClient(LlamaClient(server.url, timeout=0.2), None)
            result = run_empty_turn(chat, replays=1, max_tokens=512, n_ctx=32768)
        for turn in result.turns:
            for replay in turn.replays:
                self.assertEqual(replay.outcome, Outcome.ERROR)
                self.assertEqual(replay.error, "timed out")


class TestRunConformance(unittest.TestCase):
    def test_http_500_on_one_case_is_request_error(self):
        class ServerErrorResponder:
            def __init__(self):
                self.payloads = []

            def __call__(self, body):
                payload = json.loads(body)
                self.payloads.append(payload)
                prompt = payload["messages"][-1]["content"]
                case = next(c for c in CONFORMANCE_CASES if c.prompt == prompt)
                if case.id == "args_enum":
                    return 500, "boom", "text/plain"
                return 200, perfect_reply(case), "application/json"

        with agent_server(responder=ServerErrorResponder()) as server:
            chat = ChatClient(LlamaClient(server.url), None)
            result = run_conformance(chat)
        by_id = {case.case_id: case for case in result.cases}
        self.assertEqual(len(result.cases), 10)
        self.assertEqual(result.score, 9)
        self.assertFalse(by_id["args_enum"].passed)
        self.assertEqual(by_id["args_enum"].symptoms, (Symptom.REQUEST_ERROR,))
        self.assertEqual(by_id["args_enum"].detail, "HTTP 500")
        for case_id in ("select_read_file", "parallel_calls"):
            self.assertTrue(by_id[case_id].passed, case_id)

    def test_progress_lines_written(self):
        with agent_server() as server:
            chat = ChatClient(LlamaClient(server.url), None)
            progress = io.StringIO()
            run_conformance(chat, progress)
        self.assertIn("agentcheck: conformance 1/10 select_read_file", progress.getvalue())


class TestExitMatrix(unittest.TestCase):
    def test_all_verdict_pairs(self):
        family = ServerFamily(Family.LLAMA_CPP, "evidence", "model", 32768)
        expected = {
            (Verdict.PASS, Verdict.PASS): 0,
            (Verdict.PASS, Verdict.WARN): 1,
            (Verdict.PASS, Verdict.FAIL): 2,
            (Verdict.WARN, Verdict.PASS): 1,
            (Verdict.WARN, Verdict.WARN): 1,
            (Verdict.WARN, Verdict.FAIL): 2,
            (Verdict.FAIL, Verdict.PASS): 2,
            (Verdict.FAIL, Verdict.WARN): 2,
            (Verdict.FAIL, Verdict.FAIL): 2,
        }
        for (conformance_verdict_value, empty_verdict_value), code in expected.items():
            with self.subTest(conformance=conformance_verdict_value, empty=empty_verdict_value):
                conformance = self._conformance_with_verdict(conformance_verdict_value)
                empty = self._empty_with_verdict(empty_verdict_value)
                result = AgentCheckResult(family, conformance, empty, ())
                self.assertEqual(result.overall, Verdict(code))
                self.assertEqual(result.exit_code, code)

    @staticmethod
    def _conformance_with_verdict(verdict: Verdict) -> ConformanceResult:
        passed = {Verdict.PASS: 10, Verdict.WARN: 9, Verdict.FAIL: 7}[verdict]
        cases = tuple(
            CaseResult(f"case{i}", f"case {i}", i < passed, (), "ok", ()) for i in range(10)
        )
        return ConformanceResult(cases)

    @staticmethod
    def _empty_with_verdict(verdict: Verdict) -> EmptyTurnResult:
        outcomes = {
            Verdict.PASS: (Outcome.ACTIONABLE, Outcome.ACTIONABLE),
            Verdict.WARN: (Outcome.ACTIONABLE, Outcome.THOUGHT_RUNAWAY),
            Verdict.FAIL: (Outcome.EMPTY, Outcome.ACTIONABLE),
        }[verdict]
        replays = tuple(ReplayResult(outcome, "stop", 9000, 10) for outcome in outcomes)
        return EmptyTurnResult((TurnResult("turn", "title", replays),))


if __name__ == "__main__":
    unittest.main()
