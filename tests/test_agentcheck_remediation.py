import unittest

from llama_doctor.agentcheck.family import Family
from llama_doctor.agentcheck.remediation import REMEDIES, Symptom, remedies_for


class TestRemedies(unittest.TestCase):
    def test_llama_cpp_empty_turn_fix(self):
        remedy = REMEDIES[Symptom.EMPTY_TURN]
        self.assertIn("--reasoning-budget 2048", remedy.llama_cpp)
        self.assertNotIn("thinking_budget_tokens", remedy.llama_cpp)

    def test_strata_empty_turn_fix(self):
        remedy = REMEDIES[Symptom.EMPTY_TURN]
        self.assertIn("thinking_budget_tokens", remedy.strata)

    def test_no_strata_cell_uses_cli_flags(self):
        for symptom, remedy in REMEDIES.items():
            with self.subTest(symptom=symptom.value):
                if remedy.strata is not None:
                    self.assertNotIn("--", remedy.strata)

    def test_strata_cells_name_the_config_keys(self):
        joined = " ".join(
            remedy.strata for remedy in REMEDIES.values() if remedy.strata is not None
        )
        for key in (
            "reasoning_close_retry",
            "reasoning_budget_tokens",
            "reasoning_loop_recovery",
            "reasoning_effort",
        ):
            self.assertIn(key, joined)

    def test_unknown_family_has_no_fix(self):
        symptoms = [Symptom.EMPTY_TURN, Symptom.MALFORMED_TOOL_ARGS, Symptom.CONTEXT_TOO_SMALL]
        applied = remedies_for(symptoms, Family.UNKNOWN)
        self.assertEqual(len(applied), 3)
        for remedy in applied:
            self.assertIsNone(remedy.fix)

    def test_dedupes_and_orders_by_declaration(self):
        symptoms = [
            Symptom.CONTEXT_TOO_SMALL,
            Symptom.EMPTY_TURN,
            Symptom.CONTEXT_TOO_SMALL,
            Symptom.WRONG_TOOL,
        ]
        applied = remedies_for(symptoms, Family.LLAMA_CPP)
        self.assertEqual(
            [remedy.symptom for remedy in applied],
            [Symptom.EMPTY_TURN, Symptom.WRONG_TOOL, Symptom.CONTEXT_TOO_SMALL],
        )

    def test_every_symptom_has_a_remedy(self):
        for symptom in Symptom:
            with self.subTest(symptom=symptom.value):
                self.assertIn(symptom, REMEDIES)
                remedy = REMEDIES[symptom]
                self.assertEqual(remedy.symptom, symptom)
                self.assertTrue(remedy.title)
                self.assertTrue(remedy.generic)


if __name__ == "__main__":
    unittest.main()
