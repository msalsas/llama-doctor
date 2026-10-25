import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llama_doctor.client import FetchError, LlamaClient
from llama_doctor.redact import safe_int

CHAT_PATH = "/v1/chat/completions"
GREEDY_SEED = 0


@dataclass(frozen=True)
class ToolCall:
    id: str | None
    name: str | None
    arguments: dict[str, Any] | None
    error: str | None


@dataclass(frozen=True)
class ChatReply:
    content: str | None
    reasoning: str | None
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None


def _text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            part["text"]
            for part in value
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return "".join(parts)
    return None


def _parse_call(raw: Mapping[str, Any]) -> ToolCall:
    call_id = raw.get("id") if isinstance(raw.get("id"), str) else None
    function = raw.get("function")
    if not isinstance(function, dict):
        function = {}
    name = function.get("name") if isinstance(function.get("name"), str) else None
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        return ToolCall(call_id, name, arguments, None)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except ValueError:
            return ToolCall(call_id, name, None, "arguments are not valid JSON")
        if not isinstance(parsed, dict):
            return ToolCall(call_id, name, None, "arguments are not a JSON object")
        return ToolCall(call_id, name, parsed, None)
    return ToolCall(call_id, name, None, "arguments missing")


def parse_reply(payload: Any) -> ChatReply:
    if not isinstance(payload, dict):
        raise FetchError(CHAT_PATH, "no choices in response")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise FetchError(CHAT_PATH, "no choices in response")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise FetchError(CHAT_PATH, "no choices in response")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise FetchError(CHAT_PATH, "no choices in response")
    content = _text(message.get("content"))
    reasoning = _text(message.get("reasoning_content")) or _text(message.get("reasoning"))
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        raw_calls = []
    calls = tuple(_parse_call(call) for call in raw_calls if isinstance(call, dict))
    finish = choice.get("finish_reason")
    finish_reason = finish if isinstance(finish, str) else None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return ChatReply(
        content=content,
        reasoning=reasoning,
        tool_calls=calls,
        finish_reason=finish_reason,
        prompt_tokens=safe_int(usage.get("prompt_tokens")),
        completion_tokens=safe_int(usage.get("completion_tokens")),
    )


class ChatClient:
    def __init__(self, client: LlamaClient, model: str | None) -> None:
        self._client = client
        self._model = model

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        tool_choice: str = "auto",
        parallel_tool_calls: bool | None = None,
    ) -> ChatReply:
        payload = {
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": tool_choice,
            "temperature": 0,
            "seed": GREEDY_SEED,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self._model is not None:
            payload["model"] = self._model
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = parallel_tool_calls
        return parse_reply(self._client.post_json(CHAT_PATH, payload))
