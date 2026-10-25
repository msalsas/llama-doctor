import json
import unittest

from llama_doctor.agentcheck.cases import CONFORMANCE_CASES
from llama_doctor.agentcheck.chat import ChatClient
from llama_doctor.agentcheck.errors import AgentCheckError
from llama_doctor.agentcheck.family import Family, detect_family
from llama_doctor.client import LlamaClient

from .agentcheck_fakes import agent_server
from .support import FakeLlamaServer


def make_client(url: str) -> LlamaClient:
    return LlamaClient(url, timeout=5.0)


class TestDetectFamily(unittest.TestCase):
    def test_props_200_gives_llama_cpp(self):
        with FakeLlamaServer() as server:
            server.set_route(
                "/props", 200,
                json.dumps({"default_generation_settings": {"n_ctx": 32768}}),
                "application/json")
            server.set_route(
                "/v1/models", 200,
                json.dumps({"data": [{"id": "Qwen3-30B-A3B"}]}),
                "application/json")
            family = detect_family(make_client(server.url))
        self.assertEqual(family.name, Family.LLAMA_CPP)
        self.assertEqual(family.evidence, "GET /props returned 200")
        self.assertEqual(family.model_id, "Qwen3-30B-A3B")
        self.assertEqual(family.n_ctx, 32768)

    def test_instance_id_marker_gives_strata(self):
        with FakeLlamaServer() as server:
            server.set_route("/props", 404, "not found")
            server.set_route(
                "/v1/models", 200,
                json.dumps({"data": [{"id": "m", "meta": {"instanceId": "i-1"}}]}),
                "application/json")
            family = detect_family(make_client(server.url))
        self.assertEqual(family.name, Family.STRATA)
        self.assertIn("instanceId", family.evidence)
        self.assertIsNone(family.n_ctx)

    def test_owned_by_strata_gives_strata(self):
        with FakeLlamaServer() as server:
            server.set_route("/props", 404, "not found")
            server.set_route(
                "/v1/models", 200,
                json.dumps({"data": [{"id": "m", "owned_by": "Strata"}]}),
                "application/json")
            family = detect_family(make_client(server.url))
        self.assertEqual(family.name, Family.STRATA)
        self.assertIn("owned_by", family.evidence)

    def test_no_props_and_no_markers_gives_unknown(self):
        with FakeLlamaServer() as server:
            server.set_route("/props", 404, "not found")
            server.set_route(
                "/v1/models", 200,
                json.dumps({"data": [{"id": "plain-model"}]}),
                "application/json")
            family = detect_family(make_client(server.url))
        self.assertEqual(family.name, Family.UNKNOWN)
        self.assertEqual(family.model_id, "plain-model")
        self.assertIsNone(family.n_ctx)

    def test_html_props_body_falls_through(self):
        with FakeLlamaServer() as server:
            server.set_route("/props", 200, "<html><body>proxy</body></html>", "text/html")
            server.set_route(
                "/v1/models", 200,
                json.dumps({"data": [{"id": "m", "owned_by": "strata"}]}),
                "application/json")
            family = detect_family(make_client(server.url))
        self.assertEqual(family.name, Family.STRATA)

    def test_model_id_with_200_characters_sent_intact_in_chat_payload(self):
        long_id = "m" * 200
        with agent_server(models={"data": [{"id": long_id}]}) as server:
            family = detect_family(make_client(server.url))
            chat = ChatClient(make_client(server.url), family.model_id)
            chat.chat([{"role": "user", "content": CONFORMANCE_CASES[0].prompt}],
                      [], max_tokens=32)
        self.assertEqual(family.model_id, long_id)
        payload = json.loads(server.requests[-1][3])
        self.assertEqual(payload["model"], long_id)

    def test_model_id_strips_controls_and_is_capped_in_chat_payload(self):
        control_id = "model\x1b[31m\nname"
        with agent_server(models={"data": [{"id": control_id}]}) as server:
            family = detect_family(make_client(server.url))
            chat = ChatClient(make_client(server.url), family.model_id)
            chat.chat(
                [{"role": "user", "content": CONFORMANCE_CASES[0].prompt}],
                [],
                max_tokens=32,
            )
        self.assertEqual(family.model_id, "model[31mname")
        payload = json.loads(server.requests[-1][3])
        self.assertEqual(payload["model"], "model[31mname")

        long_id = "m" * 1000
        with agent_server(models={"data": [{"id": long_id}]}) as server:
            family = detect_family(make_client(server.url))
            chat = ChatClient(make_client(server.url), family.model_id)
            chat.chat(
                [{"role": "user", "content": CONFORMANCE_CASES[0].prompt}],
                [],
                max_tokens=32,
            )
        self.assertEqual(family.model_id, "m" * 512)
        payload = json.loads(server.requests[-1][3])
        self.assertEqual(payload["model"], "m" * 512)

    def test_auth_error_raises(self):
        with FakeLlamaServer() as server:
            server.set_route("/props", 401, "unauthorized")
            server.set_route(
                "/v1/models", 200,
                json.dumps({"data": [{"id": "m"}]}),
                "application/json")
            with self.assertRaises(AgentCheckError) as ctx:
                detect_family(make_client(server.url))
        message = str(ctx.exception)
        self.assertIn("HTTP 401", message)
        self.assertIn("--api-key-env", message)

    def test_unreachable_server_raises(self):
        server = FakeLlamaServer()
        url = server.url
        server.__exit__()
        with self.assertRaises(AgentCheckError) as ctx:
            detect_family(make_client(url))
        message = str(ctx.exception)
        self.assertIn("server unreachable", message)
        self.assertIn("/props: connection refused", message)
        self.assertIn("/v1/models: connection refused", message)


if __name__ == "__main__":
    unittest.main()
