"""Tests for ThreatModel schema coercion and record_threat_model messages.

The threat_modeler prompt asks for structured threat entries, but the
schema had no `threats` field and rejected dict items with a raw pydantic
error wall the agent could not act on. Richer input is now flattened
instead of refused, an empty model is an explicit error instead of a
hollow success, and validation failures name the offending fields.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_REF_ROOT = str(Path(__file__).resolve().parent.parent)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)

from core.context import RunContext, current_run_context
from core.schemas import ThreatModel
from tools.research_tools import record_threat_model


class ThreatModelCoercionTest(unittest.TestCase):
    def test_dict_entries_flatten_to_text(self):
        tm = ThreatModel.model_validate({
            "threats": [{
                "threat_id": "T-01",
                "title": "SQL injection",
                "attack_vector": "login form",
            }],
            "entry_points": [{"name": "/api/login", "method": "POST"}],
        })
        self.assertEqual(len(tm.threats), 1)
        self.assertIn("threat_id: T-01", tm.threats[0])
        self.assertIn("title: SQL injection", tm.threats[0])
        self.assertIn("name: /api/login", tm.entry_points[0])

    def test_bare_string_becomes_singleton_list(self):
        tm = ThreatModel.model_validate({"key_risks": "account takeover"})
        self.assertEqual(tm.key_risks, ["account takeover"])

    def test_plain_string_lists_pass_through(self):
        tm = ThreatModel.model_validate({
            "threat_actors": ["Remote attacker"],
            "trust_boundaries": ["HTTP gateway"],
        })
        self.assertEqual(tm.threat_actors, ["Remote attacker"])
        self.assertEqual(tm.trust_boundaries, ["HTTP gateway"])

    def test_non_string_scalars_are_stringified(self):
        tm = ThreatModel.model_validate({"key_risks": [42]})
        self.assertEqual(tm.key_risks, ["42"])


class RecordThreatModelTest(unittest.TestCase):
    def setUp(self):
        tok = current_run_context.set(
            RunContext(jail_dir="/tmp", db_path="", run_id="r-test")
        )
        self.addCleanup(current_run_context.reset, tok)

    def test_empty_model_is_an_error_not_a_success(self):
        res = record_threat_model({})
        self.assertIn("ERROR SAVING THREAT MODEL", res)
        self.assertIn("empty", res)

    def test_validation_failure_names_the_fields(self):
        # An int is not coercible to a list of strings even by the
        # flattening validator; the error must name the field.
        res = record_threat_model({"threat_actors": 42})
        self.assertIn("ERROR SAVING THREAT MODEL", res)
        self.assertIn("threat_actors", res)
        self.assertNotIn("Traceback", res)

    def test_success_reports_threats_count(self):
        with patch("tools.research_tools._persist_artifact") as persist:
            res = record_threat_model({
                "threats": [{"threat_id": "T-01", "title": "SSRF"}],
                "entry_points": ["/fetch"],
            })
        self.assertIn("SUCCESS", res)
        self.assertIn("1 threat(s)", res)
        persist.assert_called_once()


if __name__ == "__main__":
    unittest.main()
