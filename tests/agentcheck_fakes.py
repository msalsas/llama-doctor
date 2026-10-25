import json

from llama_doctor.agentcheck.cases import CONFORMANCE_CASES

ACTIONABLE_TEXT = "I found the issue and will apply a minimal fix now."


def reply(content=None, tool_calls=(), reasoning=None, finish="stop", prompt_tokens=9000):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = list(tool_calls)
    payload = {
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 10},
    }
    return json.dumps(payload)


def call(name, args, id="c1"):
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def perfect_reply(case):
    if not case.expected:
        return reply(content=ACTIONABLE_TEXT)
    calls = [call(expected.name, dict(expected.args), id=f"c{i + 1}")
             for i, expected in enumerate(case.expected)]
    return reply(tool_calls=calls)


class ChatResponder:
    def __init__(self, overrides=None, agentic=()):
        self.overrides = dict(overrides or {})
        self.agentic = list(agentic)
        self.payloads = []

    def __call__(self, body):
        payload = json.loads(body)
        self.payloads.append(payload)
        messages = payload["messages"]
        if messages[-1]["role"] == "tool":
            if self.agentic:
                return 200, self.agentic.pop(0), "application/json"
            return 200, reply(content=ACTIONABLE_TEXT), "application/json"
        prompt = messages[-1]["content"]
        case = next(c for c in CONFORMANCE_CASES if c.prompt == prompt)
        canned = self.overrides.get(case.id)
        if canned is None:
            canned = perfect_reply(case)
        return 200, canned, "application/json"


class agent_server:
    def __init__(self, props=True, models=None, responder=None):
        self._props = props
        self._models = models
        self._responder = responder or ChatResponder()

    def __enter__(self):
        from .support import FakeLlamaServer

        self._server = FakeLlamaServer()
        if self._props:
            self._server.set_route(
                "/props", 200, json.dumps({"default_generation_settings": {"n_ctx": 32768}}),
                "application/json")
        else:
            self._server.set_route("/props", 404, "not found")
        if self._models is None:
            self._models = {"data": [{"id": "Qwen3-30B-A3B"}]}
        self._server.set_route("/v1/models", 200, json.dumps(self._models), "application/json")
        self._server.set_responder("/v1/chat/completions", self._responder)
        return self._server

    def __exit__(self, *exc):
        self._server.__exit__(*exc)
