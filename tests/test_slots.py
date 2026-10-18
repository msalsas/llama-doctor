import contextlib
import io
import json
import unittest

from llama_doctor.cli import main
from llama_doctor.client import LlamaClient
from llama_doctor.dashboard.sampler import Sampler
from llama_doctor.redact import Secret
from llama_doctor.slots import MAX_SLOTS, parse_slots

from .support import FakeLlamaServer, load_fixture

PROMPT = "SECRET-PROMPT-TEXT-do-not-leak"
REQUEST_ID = "REQID-9f8e7d6c-leak"

FULL_SLOTS = [
    {
        "id": REQUEST_ID,
        "id_task": REQUEST_ID,
        "n_ctx": 1000,
        "is_processing": True,
        "prompt": PROMPT,
        "n_prompt_tokens": 100,
        "next_token": [{"n_decoded": 20, "has_next_token": True}],
    },
    {
        "id": 1,
        "n_ctx": 1000,
        "is_processing": False,
        "prompt": PROMPT,
        "n_prompt_tokens": 50,
        "next_token": {"n_decoded": 5},
    },
]


def _routes(server: FakeLlamaServer, metrics: str = "metrics_moe_offload.txt") -> None:
    server.set_route("/metrics", 200, load_fixture(metrics))
    server.set_route("/props", 200, load_fixture("props_full.json"), "application/json")
    server.set_route("/v1/models", 200, load_fixture("models_full.json"), "application/json")


def _state(server: FakeLlamaServer, **kwargs) -> dict:
    client = LlamaClient(server.url, api_key=Secret("k3y"))
    sampler = Sampler(
        lambda: client.snapshot(include_slots=kwargs.get("slots", True)), target=server.url
    )
    sampler.tick()
    return sampler.state()


def _slot_requests(server: FakeLlamaServer) -> list:
    return [request for request in server.requests if request[1] == "/slots"]


class TestSlots(unittest.TestCase):
    def test_full_slots_live_context_and_estimated_kv(self):
        with FakeLlamaServer() as server:
            _routes(server)
            server.set_route("/slots", 200, json.dumps(FULL_SLOTS), "application/json")
            state = _state(server)
            (request,) = _slot_requests(server)

        self.assertEqual(request[2].get("Authorization"), "Bearer k3y")
        self.assertEqual(state["context"]["live_used"], 175)
        self.assertEqual(state["context"]["live_capacity"], 2000)
        self.assertFalse(state["notes"]["slots_unavailable"])
        self.assertTrue(state["kv"]["estimated"])
        self.assertAlmostEqual(state["kv"]["usage_ratio"], 175 / 2000)
        self.assertEqual(state["kv"]["evidence"]["source"], "/slots")
        kv_findings = [f for f in state["report"]["findings"] if f["analyzer"] == "kv"]
        self.assertEqual([f["code"] for f in kv_findings], ["kv.estimated"])
        self.assertTrue(kv_findings[0]["evidence"]["estimated"])

    def test_prompt_and_ids_never_exported(self):
        with FakeLlamaServer() as server:
            _routes(server)
            server.set_route("/slots", 200, json.dumps(FULL_SLOTS), "application/json")
            dumped = [json.dumps(_state(server))]
            with contextlib.redirect_stdout(io.StringIO()) as out:
                main(["diagnose", server.url, "--json"])
            dumped.append(out.getvalue())
            with contextlib.redirect_stdout(io.StringIO()) as out:
                main(["diagnose", server.url, "--no-color"])
            dumped.append(out.getvalue())

        for text in dumped:
            self.assertNotIn(PROMPT, text)
            self.assertNotIn(REQUEST_ID, text)
            self.assertNotIn("SECRET-PROMPT", text)

    def test_partial_fields(self):
        slots = [{"n_ctx": 500, "n_prompt_tokens": 10}, {"n_ctx": 500, "is_processing": True}]
        with FakeLlamaServer() as server:
            _routes(server)
            server.set_route("/slots", 200, json.dumps(slots), "application/json")
            state = _state(server)

        self.assertEqual(state["context"]["live_used"], 10)
        self.assertEqual(state["context"]["live_capacity"], 1000)

    def test_slots_truncated_flag(self):
        full = [{"n_ctx": 10}] * MAX_SLOTS
        self.assertFalse(parse_slots(full).slots_truncated)
        snapshot = parse_slots(full + [{"n_ctx": 10}])
        self.assertTrue(snapshot.slots_truncated)
        self.assertEqual(snapshot.n_slots, MAX_SLOTS)

    def test_weird_items_are_ignored(self):
        slots = [
            "junk",
            None,
            {"n_ctx": True, "n_prompt_tokens": 5},
            {"n_ctx": 100, "n_prompt_tokens": "7", "next_token": "x"},
            {"n_ctx": 100, "n_prompt_tokens": -3, "next_token": [{"n_decoded": 4}]},
        ]
        with FakeLlamaServer() as server:
            _routes(server)
            server.set_route("/slots", 200, json.dumps(slots), "application/json")
            state = _state(server)

        self.assertEqual(state["context"]["live_used"], 4)
        self.assertEqual(state["context"]["live_capacity"], 200)

    def test_non_list_body_means_absent_with_note(self):
        for body in ('{"error": "nope"}', "not json", "[]", '["x"]'):
            with self.subTest(body=body), FakeLlamaServer() as server:
                _routes(server)
                server.set_route("/slots", 200, body, "application/json")
                state = _state(server)

            self.assertNotIn("live_used", state["context"])
            self.assertTrue(state["notes"]["slots_unavailable"])
            self.assertFalse(state["kv"]["estimated"])
            self.assertEqual(state["errors"], {})

    def test_404_means_absent_with_note(self):
        with FakeLlamaServer() as server:
            _routes(server)
            state = _state(server)

        self.assertNotIn("live_used", state["context"])
        self.assertTrue(state["notes"]["slots_unavailable"])
        self.assertIsNone(state["kv"]["usage_ratio"])
        self.assertIn("n_tokens_max", state["context"])

    def test_note_is_sticky_after_recovery(self):
        with FakeLlamaServer() as server:
            _routes(server)
            client = LlamaClient(server.url)
            sampler = Sampler(lambda: client.snapshot(include_slots=True), target=server.url)
            sampler.tick()
            sampler.tick()
            self.assertTrue(sampler.state()["notes"]["slots_unavailable"])
            server.set_route("/slots", 200, json.dumps(FULL_SLOTS), "application/json")
            sampler.tick()
            self.assertTrue(sampler.state()["notes"]["slots_unavailable"])
            self.assertEqual(sampler.state()["context"]["live_used"], 175)

    def test_no_slots_skips_request(self):
        with FakeLlamaServer() as server:
            _routes(server)
            server.set_route("/slots", 200, json.dumps(FULL_SLOTS), "application/json")
            state = _state(server, slots=False)
            with contextlib.redirect_stdout(io.StringIO()):
                main(["diagnose", server.url, "--no-slots", "--json"])
            self.assertEqual(_slot_requests(server), [])
            with contextlib.redirect_stdout(io.StringIO()):
                main(["diagnose", server.url, "--json"])
            self.assertEqual(len(_slot_requests(server)), 1)

        self.assertFalse(state["notes"]["slots_unavailable"])
        self.assertNotIn("live_used", state["context"])

    def test_real_gauge_wins_and_saturation_ignores_estimate(self):
        full = [{"n_ctx": 100, "n_prompt_tokens": 100}]
        with FakeLlamaServer() as server:
            _routes(server, "metrics_healthy.txt")
            server.set_route("/slots", 200, json.dumps(full), "application/json")
            state = _state(server)
        self.assertFalse(state["kv"]["estimated"])
        self.assertAlmostEqual(state["kv"]["usage_ratio"], 0.42)

        with FakeLlamaServer() as server:
            _routes(server)
            server.set_route("/slots", 200, json.dumps(full), "application/json")
            state = _state(server)
        codes = [f["code"] for f in state["report"]["findings"]]
        self.assertNotIn("kv.saturated", codes)
        self.assertAlmostEqual(state["kv"]["usage_ratio"], 1.0)
        self.assertTrue(state["kv"]["estimated"])


if __name__ == "__main__":
    unittest.main()
