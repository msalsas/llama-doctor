import json
import unittest

from llama_doctor import __version__
from llama_doctor.agentcheck.cases import CONFORMANCE_CASES
from llama_doctor.agentcheck.checks import (
    AgentCheckResult,
    CaseResult,
    ConformanceResult,
    EmptyTurnResult,
    Outcome,
    ReplayResult,
    TurnResult,
)
from llama_doctor.agentcheck.family import Family, ServerFamily
from llama_doctor.agentcheck.remediation import Symptom, remedies_for
from llama_doctor.agentcheck.report import agentcheck_document, render_text

PARALLEL_DETAIL = "expected 2 tool calls in one turn, got 1"


def build_result(family: ServerFamily) -> AgentCheckResult:
    cases = []
    for case in CONFORMANCE_CASES:
        if case.id == "parallel_calls":
            cases.append(CaseResult(case.id, case.title, False, (Symptom.NO_PARALLEL_CALLS,),
                                    PARALLEL_DETAIL, ("read_file",)))
        else:
            cases.append(CaseResult(case.id, case.title, True, (), "ok",
                                    tuple(call.name for call in case.expected)))
    conformance = ConformanceResult(tuple(cases))
    turns = (
        TurnResult(
            "fix_failing_test", "Fix a failing fee test",
            tuple(
                [ReplayResult(Outcome.ACTIONABLE, "tool_calls", 9104, 12)] * 5
                + [ReplayResult(Outcome.THOUGHT_RUNAWAY, "length", 9104, 512)]
            ),
        ),
        TurnResult("parallel_investigation", "Investigate missing settlement dates",
                   tuple([ReplayResult(Outcome.ACTIONABLE, "tool_calls", 8870, 9)] * 6)),
        TurnResult("after_command", "Fix whatever the test run broke",
                   tuple([ReplayResult(Outcome.ACTIONABLE, "tool_calls", 8712, 9)] * 6)),
    )
    empty_turn = EmptyTurnResult(turns)
    remedies = remedies_for(
        (*conformance.symptoms, *empty_turn.symptoms), family.name
    )
    return AgentCheckResult(family, conformance, empty_turn, remedies)


LLAMA_FAMILY = ServerFamily(Family.LLAMA_CPP, "GET /props returned 200", "Qwen3-30B-A3B", 32768)


class TestRenderText(unittest.TestCase):
    def test_warn_report_layout(self):
        text = render_text(build_result(LLAMA_FAMILY), target="http://127.0.0.1:8080")
        self.assertTrue(text.startswith(f"llama-doctor {__version__} agentcheck"))
        self.assertIn("target: http://127.0.0.1:8080", text)
        self.assertIn("family: llama.cpp (GET /props returned 200)", text)
        self.assertIn("model: Qwen3-30B-A3B", text)
        self.assertIn("tool-call conformance: 9/10 WARN", text)
        self.assertIn("  FAIL parallel_calls: " + PARALLEL_DETAIL, text)
        self.assertIn("  PASS select_read_file", text)
        self.assertIn("empty-turn detector: 17/18 actionable WARN", text)
        self.assertIn("  fix_failing_test: 5 actionable, 1 thought_runaway "
                      "(prompt 9104 tokens)", text)
        self.assertIn("  parallel_investigation: 6 actionable (prompt 8870 tokens)", text)
        self.assertTrue(text.endswith("overall: WARN (exit 1)"))

    def test_unknown_family_shows_generic_advice_only(self):
        unknown = ServerFamily(Family.UNKNOWN, "no /props and no Strata markers in "
                             "/v1/models", None, None)
        text = render_text(build_result(unknown), target="http://127.0.0.1:8080")
        self.assertIn("remediation (unknown server family: generic advice only)", text)
        self.assertNotIn("fix:", text)
        self.assertIn("model: -", text)

    def test_no_remediations_line(self):
        result = build_result(LLAMA_FAMILY)
        clean = AgentCheckResult(result.family, result.conformance, result.empty_turn, ())
        text = render_text(clean, target="http://x")
        self.assertIn("remediation: none needed", text)


class TestAgentcheckDocument(unittest.TestCase):
    def test_document_structure(self):
        result = build_result(LLAMA_FAMILY)
        document = agentcheck_document(result, target="http://127.0.0.1:8080")
        self.assertEqual(document["tool"], "llama-doctor")
        self.assertEqual(document["command"], "agentcheck")
        self.assertEqual(document["version"], __version__)
        self.assertEqual(document["target"], "http://127.0.0.1:8080")
        self.assertEqual(document["overall"], "warn")
        self.assertEqual(document["exit_code"], 1)
        self.assertEqual(
            set(document["family"]),
            {"name", "evidence", "model", "n_ctx"},
        )
        self.assertEqual(document["family"]["name"], "llama.cpp")
        self.assertEqual(document["family"]["n_ctx"], 32768)
        self.assertEqual(document["conformance"]["score"], 9)
        self.assertEqual(document["conformance"]["total"], 10)
        self.assertEqual(document["conformance"]["verdict"], "warn")
        self.assertEqual(len(document["conformance"]["cases"]), 10)
        self.assertEqual(
            set(document["empty_turn"]),
            {"verdict", "skipped", "counts", "turns"},
        )
        self.assertEqual(
            document["empty_turn"]["counts"],
            {"actionable": 17, "empty": 0, "thought_runaway": 1, "error": 0},
        )
        self.assertIsNone(document["empty_turn"]["skipped"])
        self.assertEqual(len(document["empty_turn"]["turns"]), 3)
        replay = document["empty_turn"]["turns"][0]["replays"][0]
        self.assertEqual(
            set(replay),
            {"outcome", "finish_reason", "prompt_tokens", "completion_tokens", "error"},
        )
        self.assertEqual(replay["outcome"], "actionable")
        self.assertEqual(document["remediation"][0]["symptom"], "thought_runaway")
        self.assertEqual(document["remediation"][1]["symptom"], "no_parallel_calls")
        dumped = json.dumps(document, allow_nan=False)
        self.assertEqual(document["exit_code"], result.exit_code)
        self.assertNotIn("ledger", dumped)
        self.assertNotIn("I found the issue", dumped)


if __name__ == "__main__":
    unittest.main()
