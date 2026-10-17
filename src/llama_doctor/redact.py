import math
import os

_STRIP_CONTROL = {code: None for code in range(0x20)}
_STRIP_CONTROL[0x7F] = None


class Secret:
    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def __repr__(self) -> str:
        return "Secret('***')"

    def __str__(self) -> str:
        return "Secret('***')"

    def reveal(self) -> str:
        return self._value

    @classmethod
    def from_env(cls, var: str) -> "Secret | None":
        value = os.environ.get(var)
        if not value:
            return None
        return cls(value)


def safe_str(
    value: object, *, forbidden: tuple[str, ...] = (), max_len: int = 128
) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.translate(_STRIP_CONTROL)
    for token in forbidden:
        if token and token in stripped:
            return None
    cleaned = stripped[:max_len]
    if not cleaned:
        return None
    return cleaned


def safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def safe_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isfinite(result):
            return result
    return None
