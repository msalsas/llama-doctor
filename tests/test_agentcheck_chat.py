import json
import unittest

from llama_doctor.agentcheck.cases import CONFORMANCE_CASES
from llama_doctor.agentcheck.chat import CHAT_PATH, ChatClient, parse_reply
from llama_doctor.client import FetchError, LlamaClient
from llama_doctor.redact import Secret

from .agentcheck_fakes import agent_server, call


def make_reply(tool_calls=(), content="done", reasoning=None, finish="stop", usage=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = list(tool_calls)
    payload = {"choices": [{"index": 0, "message": message, "finish_reason": finish}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


class TestParseReply(unittest.TestCase):
    def test_decodes_string_arguments(self):
        payload = make_reply(tool_calls=[call("read_file", {"path": "a.py"})])
        result = parse_reply(payload)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "read_file")
        self.assertEqual(result.tool_calls[0].arguments, {"path": "a.py"})
        self.assertIsNone(result.tool_calls[0].error)
        self.assertEqual(result.tool_calls[0].id, "c1")

    def test_accepts_dict_arguments(self):
        raw_call = {"id": "c9", "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "b.py"}}}
        result = parse_reply(make_reply(tool_calls=[raw_call]))
        self.assertEqual(result.tool_calls[0].arguments, {"path": "b.py"})
        self.assertIsNone(result.tool_calls[0].error)

    def test_invalid_json_arguments_set_error(self):
        raw_call = {"id": "c1", "type": "function",
                    "function": {"name": "read_file", "arguments": "{not json"}}
        result = parse_reply(make_reply(tool_calls=[raw_call]))
        self.assertEqual(result.tool_calls[0].error, "arguments are not valid JSON")

    def test_deeply_nested_arguments_set_error(self):
        raw_call = {"id": "c1", "type": "function",
                    "function": {"name": "read_file",
                                 "arguments": "[" * 4000 + "]" * 4000}}
        result = parse_reply(make_reply(tool_calls=[raw_call]))
        self.assertIsNone(result.tool_calls[0].arguments)
        self.assertTrue(result.tool_calls[0].error)

    def test_non_object_arguments_set_error(self):
        raw_call = {"id": "c1", "type": "function",
                    "function": {"name": "read_file", "arguments": "[1, 2]"}}
        result = parse_reply(make_reply(tool_calls=[raw_call]))
        self.assertEqual(result.tool_calls[0].error, "arguments are not a JSON object")

    def test_missing_arguments_set_error(self):
        raw_call = {"id": "c1", "type": "function", "function": {"name": "read_file"}}
        result = parse_reply(make_reply(tool_calls=[raw_call]))
        self.assertEqual(result.tool_calls[0].error, "arguments missing")

    def test_reads_reasoning_variants(self):
        for key in ("reasoning_content", "reasoning"):
            with self.subTest(key=key):
                payload = make_reply(content="answer")
                payload["choices"][0]["message"][key] = "let me think"
                self.assertEqual(parse_reply(payload).reasoning, "let me think")

    def test_reasoning_content_wins_over_reasoning(self):
        payload = make_reply(content="answer")
        message = payload["choices"][0]["message"]
        message["reasoning_content"] = "primary"
        message["reasoning"] = "secondary"
        self.assertEqual(parse_reply(payload).reasoning, "primary")

    def test_joins_list_content_parts(self):
        content = [{"type": "text", "text": "hello "}, {"type": "text"}, "skip",
                   {"type": "text", "text": "world"}, {"type": "text", "text": 3}]
        payload = make_reply(content=content)
        self.assertEqual(parse_reply(payload).content, "hello world")

    def test_long_content_not_truncated(self):
        long_text = "x" * 500
        payload = make_reply(content=long_text)
        self.assertEqual(parse_reply(payload).content, long_text)

    def test_reads_usage_tokens(self):
        payload = make_reply(usage={"prompt_tokens": 9104, "completion_tokens": 0})
        result = parse_reply(payload)
        self.assertEqual(result.prompt_tokens, 9104)
        self.assertEqual(result.completion_tokens, 0)

    def test_no_choices_raises(self):
        for payload in ({}, {"choices": []}, {"choices": [{"message": None}]},
                        {"choices": ["x"]}, "not a dict"):
            with self.subTest(payload=payload):
                with self.assertRaises(FetchError) as ctx:
                    parse_reply(payload)
                self.assertEqual(ctx.exception.reason, "no choices in response")
                self.assertEqual(ctx.exception.endpoint, CHAT_PATH)

    def test_non_list_tool_calls_ignored(self):
        payload = make_reply(content="ok")
        payload["choices"][0]["message"]["tool_calls"] = {"id": "c1"}
        self.assertEqual(parse_reply(payload).tool_calls, ())


class TestChatClient(unittest.TestCase):
    def test_request_payload_and_headers(self):
        prompt = CONFORMANCE_CASES[0].prompt
        with agent_server() as server:
            client = LlamaClient(server.url, api_key=Secret("test-key"))
            chat = ChatClient(client, None)
            result = chat.chat([{"role": "user", "content": prompt}],
                               [{"type": "function"}], max_tokens=512, tool_choice="none")
            self.assertEqual(result.tool_calls[0].name, "read_file")
        method, path, headers, body = server.requests[-1]
        self.assertEqual(method, "POST")
        self.assertEqual(path, CHAT_PATH)
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        payload = json.loads(body)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 0)
        self.assertEqual(payload["stream"], False)
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(payload["tool_choice"], "none")
        self.assertNotIn("model", payload)
        self.assertNotIn("parallel_tool_calls", payload)

    def test_model_and_parallel_tool_calls_sent_when_set(self):
        prompt = CONFORMANCE_CASES[0].prompt
        with agent_server() as server:
            client = LlamaClient(server.url)
            chat = ChatClient(client, "Qwen3-30B-A3B")
            chat.chat([{"role": "user", "content": prompt}], [], max_tokens=128,
                      parallel_tool_calls=True)
        payload = json.loads(server.requests[-1][3])
        self.assertEqual(payload["model"], "Qwen3-30B-A3B")
        self.assertEqual(payload["parallel_tool_calls"], True)


if __name__ == "__main__":
    unittest.main()
