"""Tests for cross-campaign reuse of repo-scope prefix stage artifacts.

In cross-functional mode every campaign shares the repository jail, so the
architect and threat modeler re-derive the same repo-wide artifacts on every
campaign of a run. The reuse callbacks let campaigns 2..N of one process skip
those stages once the artifact row exists. These tests pin the fail-safe
gates: any missing precondition must run the stage (callback returns None).
No LLM calls.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_REF_ROOT = str(Path(__file__).resolve().parent.parent)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)

from core.context import RunContext, current_run_context
from core.database import init_db, record_artifact
from core.graph_loader import (
    _PREFIX_REUSE_ARTIFACTS,
    _SCAN_MODE_CROSS_FUNCTIONAL,
    _make_prefix_reuse_callbacks,
)


class _CallbackTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="mantis_prefix_reuse_")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = os.path.realpath(self._tmp.name)
        self.db = os.path.join(self.tmp, "knowledge.db")
        init_db(self.db)

    def install_ctx(self, scan_mode=_SCAN_MODE_CROSS_FUNCTIONAL, run_id="r1"):
        ctx = RunContext(
            jail_dir=self.tmp,
            db_path=self.db,
            run_id=run_id,
            scan_mode=scan_mode,
        )
        token = current_run_context.set(ctx)
        self.addCleanup(current_run_context.reset, token)
        return ctx

    def record_summary(self, run_id="r1"):
        record_artifact(
            self.db, run_id, "summary",
            "workspace/.structured/summary.json", json.dumps({"ok": True}),
        )


class PrefixReuseCallbackTest(_CallbackTest):
    def test_skips_only_after_completion_and_artifact(self):
        before, after = _make_prefix_reuse_callbacks("architect")
        self.install_ctx()
        self.record_summary()
        # Campaign #1: nothing completed yet, the stage must run.
        self.assertIsNone(before())
        after()
        # Campaign #2: completed and the artifact row exists -> skip.
        content = before()
        self.assertIsNotNone(content)
        self.assertIn("Reused the summary", content.parts[0].text)

    def test_file_by_file_never_skips_or_marks(self):
        before, after = _make_prefix_reuse_callbacks("architect")
        self.install_ctx(scan_mode="file-by-file")
        self.record_summary()
        after()
        self.assertIsNone(before())
        # The file-by-file completion must not seed a later cross-functional
        # skip either: per-file campaigns never prove the repo-wide artifact.
        self.install_ctx(scan_mode=_SCAN_MODE_CROSS_FUNCTIONAL)
        self.assertIsNone(before())

    def test_unknown_mode_never_skips(self):
        before, after = _make_prefix_reuse_callbacks("architect")
        self.install_ctx(scan_mode="")
        self.record_summary()
        after()
        self.assertIsNone(before())

    def test_missing_artifact_runs_the_stage(self):
        before, after = _make_prefix_reuse_callbacks("architect")
        self.install_ctx()
        after()
        self.assertIsNone(before())

    def test_no_context_runs_the_stage(self):
        before, _ = _make_prefix_reuse_callbacks("architect")
        self.assertIsNone(current_run_context.get())
        self.assertIsNone(before())

    def test_fresh_process_reruns(self):
        # Completion is process-local by design: a resumed run re-derives
        # against the resynced checkout instead of trusting an old artifact.
        before, after = _make_prefix_reuse_callbacks("architect")
        self.install_ctx()
        self.record_summary()
        after()
        self.assertIsNotNone(before())
        fresh_before, _ = _make_prefix_reuse_callbacks("architect")
        self.assertIsNone(fresh_before())

    def test_completion_is_per_run(self):
        before, after = _make_prefix_reuse_callbacks("architect")
        self.install_ctx(run_id="r1")
        self.record_summary(run_id="r1")
        after()
        self.record_summary(run_id="r2")
        self.install_ctx(run_id="r2")
        self.assertIsNone(before())

    def test_threat_modeler_requires_its_own_artifact(self):
        before, after = _make_prefix_reuse_callbacks("threat_modeler")
        self.install_ctx()
        self.record_summary()  # wrong artifact type for this node
        after()
        self.assertIsNone(before())
        record_artifact(
            self.db, "r1", "threat_model",
            "workspace/.structured/threat_model.json", json.dumps({"ok": True}),
        )
        content = before()
        self.assertIsNotNone(content)
        self.assertIn("Reused the threat_model", content.parts[0].text)


class VocabularyPinTest(unittest.TestCase):
    def test_loader_literal_matches_main(self):
        # graph_loader mirrors the literal because importing main would be
        # circular; this pin is what keeps the two from drifting.
        import main as main_module

        self.assertEqual(
            main_module.SCAN_MODE_CROSS_FUNCTIONAL, _SCAN_MODE_CROSS_FUNCTIONAL
        )


class WiringPinTest(unittest.TestCase):
    def test_loader_attaches_reuse_callbacks(self):
        import core.graph_loader as gl

        src = inspect.getsource(gl.load_workflow_from_json)
        self.assertIn(
            "node_id in _PREFIX_REUSE_ARTIFACTS and not node_cfg.on_enter_status",
            src,
        )
        self.assertIn("_make_prefix_reuse_callbacks(node_id)", src)

    def test_reuse_covers_exactly_the_repo_scope_stages(self):
        self.assertEqual(
            _PREFIX_REUSE_ARTIFACTS,
            {"architect": "summary", "threat_modeler": "threat_model"},
        )

    def test_shipped_workflow_keeps_reuse_active(self):
        # The attach guard defers to a completion stamp; the shipped workflow
        # must not declare one on the reuse nodes, or reuse silently dies.
        workflow = json.loads(
            (Path(_REF_ROOT) / "workflow.json").read_text(encoding="utf-8")
        )
        for node in workflow["nodes"]:
            if node["id"] in _PREFIX_REUSE_ARTIFACTS:
                self.assertNotIn("on_enter_status", node)


if __name__ == "__main__":
    unittest.main()
