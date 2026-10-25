import json
from typing import TextIO

from llama_doctor.client import LlamaClient

from .cases import CONFORMANCE_CASES, agentic_turns
from .chat import ChatClient
from .checks import AgentCheckResult, run_conformance, run_empty_turn
from .errors import AgentCheckError
from .family import detect_family
from .remediation import remedies_for
from .report import agentcheck_document, render_text


def run_agentcheck(
    client: LlamaClient,
    *,
    replays: int = 6,
    max_tokens: int = 512,
    as_json: bool = False,
    out: TextIO,
    err: TextIO,
) -> int:
    """Detect the server family, run both phases, print the report and return the exit code."""
    family = detect_family(client)
    print(f"agentcheck: server family {family.name.value} ({family.evidence})", file=err)
    chat = ChatClient(client, family.model_id)
    print(f"agentcheck: tool-call conformance, {len(CONFORMANCE_CASES)} cases", file=err)
    conformance = run_conformance(chat, err)
    print(
        f"agentcheck: empty-turn detector, {len(agentic_turns())} turns x {replays} replays",
        file=err,
    )
    empty = run_empty_turn(
        chat, replays=replays, max_tokens=max_tokens, n_ctx=family.n_ctx, progress=err
    )
    remedies = remedies_for((*conformance.symptoms, *empty.symptoms), family.name)
    result = AgentCheckResult(family, conformance, empty, remedies)
    if as_json:
        text = json.dumps(
            agentcheck_document(result, target=client.base_url), indent=2, allow_nan=False
        )
    else:
        text = render_text(result, target=client.base_url)
    print(text, file=out)
    return result.exit_code


__all__ = ["AgentCheckError", "run_agentcheck"]
