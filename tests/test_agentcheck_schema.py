import unittest

from llama_doctor.agentcheck.schema import validate

TODO_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["add", "update", "remove", "list"]},
        "id": {"type": "integer"},
        "text": {"type": "string"},
        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
    },
    "required": ["action"],
}

READ_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "offset": {"type": "integer"},
        "limit": {"type": "integer"},
    },
    "required": ["path"],
}

WEB_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "max_results": {"type": "integer"},
    },
    "required": ["query"],
}


class TestValidate(unittest.TestCase):
    def test_valid_object_passes(self):
        value = {"action": "update", "id": 3, "status": "completed"}
        self.assertEqual(validate(TODO_LIST_SCHEMA, value), [])

    def test_missing_required_property(self):
        errors = validate(READ_FILE_SCHEMA, {"offset": 0})
        self.assertIn("$: missing required property 'path'", errors)

    def test_wrong_enum_value(self):
        value = {"action": "update", "status": "done"}
        errors = validate(TODO_LIST_SCHEMA, value)
        self.assertIn("$.status: 'done' not in enum", errors)

    def test_type_errors(self):
        cases = [
            (WEB_SEARCH_SCHEMA, {"query": "x", "max_results": "5"}),
            (WEB_SEARCH_SCHEMA, {"query": "x", "max_results": True}),
            (READ_FILE_SCHEMA, {"path": 3}),
            (READ_FILE_SCHEMA, ["not", "an", "object"]),
        ]
        for schema, value in cases:
            with self.subTest(value=value):
                self.assertTrue(validate(schema, value))

    def test_type_list_nullable_string(self):
        schema = {"type": ["string", "null"]}
        self.assertEqual(validate(schema, "text"), [])
        self.assertEqual(validate(schema, None), [])
        self.assertTrue(validate(schema, 3))

    def test_nested_properties_checked_recursively(self):
        schema = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {"inner": {"type": "integer"}},
                    "required": ["inner"],
                }
            },
        }
        self.assertEqual(validate(schema, {"outer": {"inner": 1}}), [])
        errors = validate(schema, {"outer": {}})
        self.assertIn("$.outer: missing required property 'inner'", errors)

    def test_unknown_keywords_ignored(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "description": "ignored",
            "additionalProperties": False,
            "$comment": "ignored",
        }
        self.assertEqual(validate(schema, {"name": "x", "extra": 1}), [])


if __name__ == "__main__":
    unittest.main()
