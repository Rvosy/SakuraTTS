"""Compare the public wire contract with a pinned, offline upstream snapshot.

The fixture was extracted with ast from the recorded Git sources, without
importing upstream TTS. HTTP execution and feature rejection live in test_server.
"""

import importlib.util
import json
from pathlib import Path
import unittest


CONTRACT = json.loads((Path(__file__).parent / "fixtures/gpt_sovits_api_v2.json").read_text(encoding="utf-8"))
WIRE_TYPES = {"str": {"string"}, "int": {"integer"}, "float": {"number"},
              "bool": {"boolean"}, "list": {"array"}, "Union[bool, int]": {"boolean", "integer"}}


def wire_types(schema):
    # Upstream annotates None defaults as str/list; the local model makes that
    # nullability explicit and adds string item validation to reference paths.
    alternatives = schema.get("anyOf", [schema])
    return {alternative["type"] for alternative in alternatives if alternative["type"] != "null"}


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "Install server extras")
class UpstreamApiContractTests(unittest.TestCase):
    def test_request_fields_types_and_defaults_match_upstream(self):
        from sakuratts.server import SpeechRequest

        expected = CONTRACT["request_fields"]
        self.assertEqual(set(SpeechRequest.model_fields), set(expected))
        self.assertEqual(SpeechRequest().model_dump(), {name: field["default"] for name, field in expected.items()})
        properties = SpeechRequest.model_json_schema()["properties"]
        for name, field in expected.items():
            with self.subTest(field=name):
                self.assertEqual(wire_types(properties[name]), WIRE_TYPES[field["annotation"]])
                self.assertFalse(SpeechRequest.model_fields[name].is_required())

    def test_openapi_routes_and_query_parameters_match_upstream(self):
        from sakuratts.server import create_app

        # Building OpenAPI does not enter lifespan or construct an inference engine.
        schema = create_app().openapi()
        for route in CONTRACT["routes"]:
            with self.subTest(method=route["method"], path=route["path"]):
                operation = schema["paths"][route["path"]][route["method"].lower()]
                if route["method"] == "POST":
                    self.assertEqual(route["parameters"], {"request": {"annotation": "TTS_Request"}})
                    body = operation["requestBody"]["content"]["application/json"]["schema"]
                    self.assertEqual(body["$ref"], "#/components/schemas/SpeechRequest")
                    continue
                parameters = {field["name"]: field for field in operation["parameters"]}
                self.assertEqual(set(parameters), set(route["parameters"]))
                for name, expected in route["parameters"].items():
                    actual = parameters[name]
                    self.assertEqual(actual["in"], "query")
                    self.assertFalse(actual["required"])
                    self.assertEqual(wire_types(actual["schema"]), WIRE_TYPES[expected["annotation"]])
                    self.assertEqual(actual["schema"].get("default"), expected["default"])

    def test_implemented_language_modes_are_an_upstream_subset(self):
        from sakuratts.frontend.profiles import SUPPORTED_LANGUAGE_MODES

        for version in ("v2Pro", "v2ProPlus"):
            upstream = next(profile["modes"] for profile in CONTRACT["language_profiles"]
                            if version in profile["model_versions"])
            with self.subTest(model_version=version):
                self.assertTrue(set(SUPPORTED_LANGUAGE_MODES).issubset(upstream))


if __name__ == "__main__":
    unittest.main()
