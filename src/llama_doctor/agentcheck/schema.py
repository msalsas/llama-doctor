from collections.abc import Mapping
from typing import Any


def validate(schema: Mapping[str, Any], value: Any, path: str = "$") -> list[str]:
    """Return validation errors for the type/properties/required/enum subset of JSON Schema."""
    if not isinstance(schema, Mapping):
        return []
    errors: list[str] = []
    expected = schema.get("type")
    if isinstance(expected, str):
        type_names = (expected,)
    elif isinstance(expected, list):
        type_names = tuple(item for item in expected if isinstance(item, str))
    else:
        type_names = ()
    if type_names and not any(_matches(name, value) for name in type_names):
        errors.append(f"{path}: expected {'/'.join(type_names)}, got {type(value).__name__}")
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"{path}: {value!r} not in enum")
    if not isinstance(value, Mapping):
        return errors
    required = schema.get("required")
    if isinstance(required, list):
        for name in required:
            if isinstance(name, str) and name not in value:
                errors.append(f"{path}: missing required property '{name}'")
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for name, subschema in properties.items():
            if name in value:
                errors.extend(validate(subschema, value[name], f"{path}.{name}"))
    return errors


def _matches(type_name: str, value: Any) -> bool:
    if type_name == "object":
        return isinstance(value, Mapping)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    return True
