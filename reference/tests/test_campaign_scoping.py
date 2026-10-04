"""Tests for campaign-scoped findings access and per-finding status stamps.

Multi-target runs share one run_id across campaigns. These tests pin the
three deterministic fixes for that: get_findings defaults to campaign
scope, dismissal stamps address the finding id (so a later filepath-wide
promotion cannot launder a dismissed sibling), and the zero-findings gate
only skips triage stages when it can prove the campaign is clean. No LLM
calls.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_REF_ROOT = str(Path(__file__).resolve().parent.parent)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)

from core.context import RunContext, current_run_context
from core.database import (
    init_db,
    read_findings,
    update_finding_status_by_id,
    update_status,
    write_findings,
)


def _finding(title, filepath, description="d", code_paths=None):
    f = {"title": title, "filepath": filepath, "description": description}
    if code_paths:
        f["code_paths"] = code_paths
    return f


def _statuses_by_title(db_path, run_id="r1"):
    return {
        f["title"]: str(f["status"]).lower()
        for f in read_findings(db_path, run_id=run_id)
    }


class _DbTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="mantis_scope_test_")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = os.path.realpath(self._tmp.name)
        self.db = os.path.join(self.tmp, "knowledge.db")
        init_db(self.db)

    def install_ctx(self, target_file="", active_node="", run_id="r1"):
        ctx = RunContext(
            jail_dir=self.tmp,
            db_path=self.db,
            target_file=target_file,
            run_id=run_id,
            active_node=active_node,
        )
        token = current_run_context.set(ctx)
        self.addCleanup(current_run_context.reset, token)
        return ctx


class UpdateFindingStatusByIdTest(_DbTest):
    def setUp(self):
        super().setUp()
        write_findings(
            self.db, "",
            [_finding("real", "app/login.py"), _finding("bogus", "app/login.py")],
            run_id="r1",
        )
        self.ids = {f["title"]: f["id"] for f in read_findings(self.db, run_id="r1")}

    def test_dismissal_survives_later_filepath_promotion(self):
        # The laundering case: dismiss one sibling by id, then promote the
        # whole file the way the reproducer's completion stamp does.
        update_finding_status_by_id(self.db, self.ids["bogus"], "r1", "false_positive")
        update_status(self.db, "app/login.py", "r1", "static_confirmed")
        statuses = _statuses_by_title(self.db)
        self.assertEqual(statuses["real"], "static_confirmed")
        self.assertEqual(statuses["bogus"], "false_positive")

    def test_dismissal_never_overwrites_dynamic_proof(self):
        update_status(self.db, "app/login.py", "r1", "dynamic_confirmed")
        update_finding_status_by_id(self.db, self.ids["real"], "r1", "false_positive")
        self.assertEqual(_statuses_by_title(self.db)["real"], "dynamic_confirmed")

    def test_wrong_run_id_updates_nothing(self):
        update_finding_status_by_id(self.db, self.ids["real"], "other_run", "false_positive")
        self.assertEqual(_statuses_by_title(self.db)["real"], "reported")

    def test_non_integer_id_is_ignored(self):
        update_finding_status_by_id(self.db, "not-an-id", "r1", "false_positive")
        self.assertEqual(set(_statuses_by_title(self.db).values()), {"reported"})


class ReadFindingsScopePathTest(_DbTest):
    def setUp(self):
        super().setUp()
        write_findings(
            self.db, "",
            [
                _finding("in-file", "app/login.py"),
                _finding("in-subdir", "app/sub/x.py"),
                _finding("outside", "lib/util.py"),
                _finding("prefix-collision", "application/evil.py"),
            ],
            run_id="r1",
        )

    def _titles(self, scope):
        return {f["title"] for f in read_findings(self.db, run_id="r1", scope_path=scope)}

    def test_directory_scope_is_component_aligned(self):
        # "app" must match app/ but never application/.
        self.assertEqual(self._titles("app"), {"in-file", "in-subdir"})

    def test_file_scope_matches_exactly(self):
        self.assertEqual(self._titles("app/login.py"), {"in-file"})

    def test_no_scope_returns_everything(self):
        self.assertEqual(len(self._titles(None)), 4)

    def test_root_scope_is_no_filter(self):
        self.assertEqual(len(self._titles(".")), 4)

    def test_cross_directory_finding_is_kept_by_its_code_paths(self):
        # A slice campaign on app/ can trace a flow into server/lib and
        # report the finding at the sink file. The in-scope citation in
        # code_paths must keep it visible to the campaign that found it.
        write_findings(
            self.db, "",
            [_finding("cross-dir", "server/lib/db.js",
                      code_paths=["app/login.js:42", "server/lib/db.js:7"])],
            run_id="r1",
        )
        self.assertIn("cross-dir", self._titles("app"))

    def test_code_paths_matching_is_quote_anchored(self):
        # Scope "app" must not match a citation of application/.
        write_findings(
            self.db, "",
            [_finding("decoy-citation", "lib/z.py",
                      code_paths=["application/evil.py:3"])],
            run_id="r1",
        )
        self.assertNotIn("decoy-citation", self._titles("app"))


class UpdateStatusScopeSymmetryTest(_DbTest):
    """update_status must promote exactly what read_findings(scope_path=...)
    lets the campaign see -- the two share _campaign_scope_clause."""

    def setUp(self):
        super().setUp()
        write_findings(
            self.db, "",
            [
                _finding("cross-dir", "server/lib/db.js",
                         code_paths=["app/login.js:42", "server/lib/db.js:7"]),
                _finding("decoy-citation", "lib/z.py",
                         code_paths=["application/evil.py:3"]),
                _finding("outside", "lib/util.py"),
            ],
            run_id="r1",
        )

    def test_campaign_stamp_promotes_cross_directory_finding(self):
        # The app campaign reviewed and reproduced this finding (read_findings
        # keeps it in scope via code_paths); the completion stamp must promote
        # it too, or it stays "reported" and is suppressed at export.
        update_status(self.db, "app", "r1", "static_confirmed")
        self.assertEqual(_statuses_by_title(self.db)["cross-dir"], "static_confirmed")

    def test_campaign_stamp_is_quote_anchored(self):
        update_status(self.db, "app", "r1", "static_confirmed")
        statuses = _statuses_by_title(self.db)
        self.assertEqual(statuses["decoy-citation"], "reported")
        self.assertEqual(statuses["outside"], "reported")

    def test_stamp_scope_equals_read_scope(self):
        update_status(self.db, "app", "r1", "static_confirmed")
        promoted = {t for t, s in _statuses_by_title(self.db).items()
                    if s == "static_confirmed"}
        visible = {f["title"]
                   for f in read_findings(self.db, run_id="r1", scope_path="app")}
        self.assertEqual(promoted, visible)


class GetFindingsCampaignScopeTest(_DbTest):
    def setUp(self):
        super().setUp()
        write_findings(
            self.db, "",
            [_finding("mine", "app/a.py"), _finding("other-campaign", "other/b.py")],
            run_id="r1",
        )

    def _titles(self, raw):
        import json
        if raw.startswith("NO_DATA") or raw.startswith("Error") or raw.startswith("ERROR"):
            return raw
        return {f["title"] for f in json.loads(raw)}

    def test_triage_stages_see_only_their_campaign(self):
        from tools.research_tools import get_findings
        self.install_ctx(target_file="app/a.py", active_node="reviewer")
        self.assertEqual(self._titles(get_findings()), {"mine"})

    def test_reporter_and_chainer_see_the_whole_run(self):
        from tools.research_tools import get_findings
        for stage in ("reporter", "chainer"):
            self.install_ctx(target_file="app/a.py", active_node=stage)
            self.assertEqual(
                self._titles(get_findings()), {"mine", "other-campaign"}, stage
            )

    def test_whole_repo_campaign_stays_run_wide(self):
        from tools.research_tools import get_findings
        self.install_ctx(target_file="", active_node="reviewer")
        self.assertEqual(self._titles(get_findings()), {"mine", "other-campaign"})

    def test_explicit_filepath_argument_still_works(self):
        from tools.research_tools import get_findings
        self.install_ctx(target_file="app/a.py", active_node="reviewer")
        self.assertEqual(self._titles(get_findings("other/b.py")), {"other-campaign"})


class PersistFindingDismissalsTest(_DbTest):
    def setUp(self):
        super().setUp()
        write_findings(
            self.db, "",
            [_finding("real", "app/login.py"), _finding("bogus", "app/login.py")],
            run_id="r1",
        )
        self.ids = {f["title"]: f["id"] for f in read_findings(self.db, run_id="r1")}

    def test_mixed_verdicts_on_one_file_stamp_only_the_dismissed_id(self):
        from core.graph_loader import _persist_finding_dismissals
        self.install_ctx(target_file="app/login.py", active_node="reviewer_classifier")
        _persist_finding_dismissals("reviewer_classifier", [
            {"finding_id": self.ids["real"], "route": "confirmed", "reason": "solid"},
            {"finding_id": self.ids["bogus"], "route": "false_positive", "reason": "parameterized"},
        ])
        statuses = _statuses_by_title(self.db)
        self.assertEqual(statuses["real"], "reported")
        self.assertEqual(statuses["bogus"], "false_positive")
        # And the dismissal must survive the later campaign-level promotion.
        update_status(self.db, "app/login.py", "r1", "static_confirmed")
        statuses = _statuses_by_title(self.db)
        self.assertEqual(statuses["real"], "static_confirmed")
        self.assertEqual(statuses["bogus"], "false_positive")

    def test_fallback_verdicts_are_routed_but_never_persisted(self):
        from core.graph_loader import _persist_finding_dismissals
        self.install_ctx(target_file="app/login.py", active_node="reviewer_classifier")
        _persist_finding_dismissals("reviewer_classifier", [
            {"finding_id": self.ids["bogus"], "route": "false_positive",
             "reason": "Fallback: schema parse failed"},
        ])
        self.assertEqual(_statuses_by_title(self.db)["bogus"], "reported")

    def test_missing_finding_id_is_skipped(self):
        from core.graph_loader import _persist_finding_dismissals
        self.install_ctx(target_file="app/login.py", active_node="reviewer_classifier")
        _persist_finding_dismissals("reviewer_classifier", [
            {"finding_id": None, "route": "false_positive", "reason": "x"},
        ])
        self.assertEqual(set(_statuses_by_title(self.db).values()), {"reported"})


class FindingsGateTest(_DbTest):
    def test_missing_db_cannot_answer(self):
        from core.graph_loader import campaign_has_findings_to_triage
        self.assertIsNone(
            campaign_has_findings_to_triage(os.path.join(self.tmp, "nope.db"), "r1", "")
        )

    def test_unreadable_db_cannot_answer(self):
        from core.graph_loader import campaign_has_findings_to_triage
        garbage = os.path.join(self.tmp, "garbage.db")
        with open(garbage, "w", encoding="utf-8") as f:
            f.write("this is not sqlite")
        self.assertIsNone(campaign_has_findings_to_triage(garbage, "r1", ""))

    def test_clean_campaign_is_clean(self):
        from core.graph_loader import campaign_has_findings_to_triage
        self.assertIs(campaign_has_findings_to_triage(self.db, "r1", ""), False)

    def test_live_finding_in_scope_requires_triage(self):
        from core.graph_loader import campaign_has_findings_to_triage
        write_findings(self.db, "", [_finding("real", "app/a.py")], run_id="r1")
        self.assertIs(campaign_has_findings_to_triage(self.db, "r1", "app/a.py"), True)

    def test_other_campaigns_findings_do_not_hold_the_gate_open(self):
        from core.graph_loader import campaign_has_findings_to_triage
        write_findings(self.db, "", [_finding("elsewhere", "other/b.py")], run_id="r1")
        self.assertIs(campaign_has_findings_to_triage(self.db, "r1", "app/a.py"), False)

    def test_cross_directory_finding_holds_the_gate_open(self):
        # If the campaign's only finding sits at a sink outside the slice,
        # gating it "clean" would leave a real finding unreviewed forever.
        from core.graph_loader import campaign_has_findings_to_triage
        write_findings(
            self.db, "",
            [_finding("cross-dir", "server/lib/db.js", code_paths=["app/a.py:10"])],
            run_id="r1",
        )
        self.assertIs(campaign_has_findings_to_triage(self.db, "r1", "app/a.py"), True)

    def test_fully_dismissed_campaign_is_clean(self):
        from core.graph_loader import campaign_has_findings_to_triage
        write_findings(self.db, "", [_finding("fp", "app/a.py")], run_id="r1")
        ids = {f["title"]: f["id"] for f in read_findings(self.db, run_id="r1")}
        update_finding_status_by_id(self.db, ids["fp"], "r1", "false_positive")
        self.assertIs(campaign_has_findings_to_triage(self.db, "r1", "app/a.py"), False)


class WorkflowGateTopologyTest(unittest.TestCase):
    """The shipped workflow must route clean campaigns around triage."""

    @classmethod
    def setUpClass(cls):
        import json
        wf_path = Path(_REF_ROOT) / "workflow.json"
        with open(wf_path, "r", encoding="utf-8") as f:
            cls.wf = json.load(f)
        cls.edges = {(e["from"], e.get("on", "")): e["to"] for e in cls.wf["edges"]}

    def test_gate_sits_between_researcher_and_deduplicator(self):
        self.assertEqual(self.edges[("researcher", "")], "findings_gate")
        self.assertEqual(self.edges[("findings_gate", "__DEFAULT__")], "deduplicator")

    def test_clean_route_short_circuits_to_reporter(self):
        self.assertEqual(self.edges[("findings_gate", "clean")], "reporter")

    def test_default_route_is_the_fail_safe(self):
        # Only an explicit "clean" verdict may skip stages; anything else,
        # including gate errors, must continue the full pipeline.
        gate = next(n for n in self.wf["nodes"] if n["id"] == "findings_gate")
        self.assertEqual(gate["type"], "classifier")
        self.assertEqual(gate["routes"], ["clean"])


if __name__ == "__main__":
    unittest.main()
