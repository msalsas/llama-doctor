import hashlib
import json
import os
import re
import subprocess
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

from llama_doctor.agentcheck.cases import (
    CONFORMANCE_CASES,
    SENTENCE_POOL,
    TOOL_CATALOG,
    TOOL_NAMES,
    TOOL_SCHEMAS,
    agentic_turns,
    filler,
)
from llama_doctor.agentcheck.schema import validate

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_NAMES = (
    "run_command",
    "read_file",
    "patch_file",
    "search_files",
    "write_file",
    "delegate_task",
    "memory",
    "web_search",
    "browser_exec",
    "clarify",
    "skill_view",
    "terminal",
    "cronjob",
    "todo_list",
    "process_manage",
)

BANNED_CHARS = set("<>{}[]|`\\")
BANNED_WORDS = re.compile(r"\b(tool|call|think|reasoning|ignore|instruction|you|assistant)\b")


def corpus_payload() -> dict:
    return {
        "catalog": list(TOOL_CATALOG),
        "names": list(TOOL_NAMES),
        "cases": [asdict(case) for case in CONFORMANCE_CASES],
        "pool": list(SENTENCE_POOL),
        "turns": [asdict(turn) for turn in agentic_turns()],
    }


class TestToolCatalog(unittest.TestCase):
    def test_catalog_names_in_order(self):
        self.assertEqual(TOOL_NAMES, EXPECTED_NAMES)
        self.assertEqual(len(set(TOOL_NAMES)), len(TOOL_NAMES))

    def test_catalog_schemas_well_formed(self):
        for name, schema in TOOL_SCHEMAS.items():
            with self.subTest(tool=name):
                self.assertEqual(schema.get("type"), "object")
                properties = schema.get("properties")
                required = schema.get("required")
                self.assertIsInstance(properties, dict)
                self.assertIsInstance(required, list)
                self.assertTrue(set(required) <= set(properties))
                for prop in properties.values():
                    enum = prop.get("enum")
                    if enum is not None:
                        self.assertTrue(all(isinstance(value, str) for value in enum))

    def test_conformance_cases_reference_valid_tools(self):
        self.assertEqual(len(CONFORMANCE_CASES), 10)
        ids = [case.id for case in CONFORMANCE_CASES]
        self.assertEqual(len(set(ids)), len(ids))
        for case in CONFORMANCE_CASES:
            for expected in case.expected:
                with self.subTest(case=case.id, tool=expected.name):
                    self.assertIn(expected.name, TOOL_SCHEMAS)
                    self.assertEqual(validate(TOOL_SCHEMAS[expected.name], dict(expected.args)), [])


class TestSentencePool(unittest.TestCase):
    def test_pool_rules(self):
        self.assertEqual(len(SENTENCE_POOL), 48)
        for index, sentence in enumerate(SENTENCE_POOL):
            with self.subTest(index=index):
                words = sentence.split()
                self.assertTrue(10 <= len(words) <= 18, f"{len(words)} words")
                self.assertTrue(sentence.isascii())
                self.assertFalse(any(char.isdigit() for char in sentence))
                self.assertFalse(set(sentence) & BANNED_CHARS)
                lowered = sentence.lower()
                for name in TOOL_NAMES:
                    self.assertNotIn(name, lowered)
                self.assertIsNone(BANNED_WORDS.search(lowered))


class TestFiller(unittest.TestCase):
    def test_filler_is_deterministic(self):
        self.assertEqual(filler(7, 100), filler(7, 100))

    def test_filler_is_seed_sensitive(self):
        self.assertNotEqual(filler(1, 100), filler(2, 100))

    def test_filler_word_count(self):
        for seed in (1, 42, 1000):
            for words in (50, 200, 2000):
                count = len(filler(seed, words).split())
                self.assertGreaterEqual(count, words)
                self.assertLess(count, words + 18)


class TestAgenticTurns(unittest.TestCase):
    def test_turn_sizes_in_range(self):
        catalog_size = len(json.dumps(TOOL_CATALOG))
        for turn in agentic_turns():
            with self.subTest(turn=turn.id):
                size = len(json.dumps(turn.messages)) + catalog_size
                self.assertGreaterEqual(size, 32000)
                self.assertLessEqual(size, 40000)

    def test_turns_end_on_tool_message_with_matching_call_ids(self):
        turn_ids = [turn.id for turn in agentic_turns()]
        self.assertEqual(len(set(turn_ids)), len(turn_ids))
        for turn in agentic_turns():
            with self.subTest(turn=turn.id):
                messages = turn.messages
                self.assertEqual(messages[-1]["role"], "tool")
                seen_call_ids = set()
                for message in messages:
                    for call in message.get("tool_calls") or []:
                        seen_call_ids.add(call["id"])
                for message in messages:
                    if message["role"] == "tool":
                        self.assertIn(message["tool_call_id"], seen_call_ids)


class TestCorpusDeterminism(unittest.TestCase):
    def test_corpus_identical_across_processes(self):
        text = json.dumps(corpus_payload(), sort_keys=True)
        local = hashlib.sha256(text.encode("utf-8")).hexdigest()
        script = (
            "import hashlib, json\n"
            "from dataclasses import asdict\n"
            "from llama_doctor.agentcheck.cases import (\n"
            "    CONFORMANCE_CASES, SENTENCE_POOL, TOOL_CATALOG, TOOL_NAMES, agentic_turns)\n"
            "payload = {\n"
            "    'catalog': list(TOOL_CATALOG),\n"
            "    'names': list(TOOL_NAMES),\n"
            "    'cases': [asdict(case) for case in CONFORMANCE_CASES],\n"
            "    'pool': list(SENTENCE_POOL),\n"
            "    'turns': [asdict(turn) for turn in agentic_turns()],\n"
            "}\n"
            "print(hashlib.sha256(json.dumps(payload, sort_keys=True)"
            ".encode('utf-8')).hexdigest())\n"
        )
        env = dict(os.environ, PYTHONHASHSEED="1", PYTHONPATH=str(ROOT / "src"))
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), local)


if __name__ == "__main__":
    unittest.main()
