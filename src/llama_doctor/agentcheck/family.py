from dataclasses import dataclass
from enum import Enum
from typing import Any

from llama_doctor.client import FetchError, LlamaClient
from llama_doctor.models import ModelsMeta, PropsSnapshot

from .errors import AgentCheckError


class Family(Enum):
    LLAMA_CPP = "llama.cpp"
    STRATA = "strata"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ServerFamily:
    name: Family
    evidence: str
    model_id: str | None
    n_ctx: int | None


AUTH_STATUSES = frozenset({401, 403})
UNREACHABLE_REASONS = frozenset({"connection refused", "unreachable", "timed out"})


def detect_family(client: LlamaClient) -> ServerFamily:
    """Classify the server from GET /props and GET /v1/models without assuming a family."""
    props, props_err = _try_get(client, "/props")
    models, models_err = _try_get(client, "/v1/models")
    for exc in (props_err, models_err):
        if exc is not None and exc.status in AUTH_STATUSES:
            raise AgentCheckError(f"{exc.endpoint}: {exc.reason}; check --api-key-env")
    if (
        props_err is not None
        and models_err is not None
        and props_err.status is None
        and models_err.status is None
        and props_err.reason in UNREACHABLE_REASONS
        and models_err.reason in UNREACHABLE_REASONS
    ):
        raise AgentCheckError(
            "server unreachable "
            f"(/props: {props_err.reason}, /v1/models: {models_err.reason})"
        )
    model_id = ModelsMeta.from_payload(models).model_id if models is not None else None
    marker = _strata_marker(models)
    if marker is not None:
        return ServerFamily(Family.STRATA, marker, model_id, None)
    if isinstance(props, dict):
        return ServerFamily(
            Family.LLAMA_CPP,
            "GET /props returned 200",
            model_id,
            PropsSnapshot.from_payload(props).n_ctx,
        )
    return ServerFamily(
        Family.UNKNOWN, "no /props and no Strata markers in /v1/models", model_id, None
    )


def _try_get(client: LlamaClient, path: str) -> tuple[Any, FetchError | None]:
    try:
        return client.get_json(path), None
    except FetchError as exc:
        return None, exc


def _strata_marker(models: Any) -> str | None:
    if not isinstance(models, dict):
        return None
    data = models.get("data")
    if not isinstance(data, list):
        return None
    entries = [entry for entry in data if isinstance(entry, dict)]
    if any(
        isinstance(entry.get("meta"), dict) and "instanceId" in entry["meta"]
        for entry in entries
    ):
        return "GET /v1/models lists meta.instanceId"
    if any(
        isinstance(entry.get("owned_by"), str)
        and entry["owned_by"].strip().lower().startswith("strata")
        for entry in entries
    ):
        return "GET /v1/models owned_by strata"
    return None
