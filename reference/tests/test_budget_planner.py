"""Regression tests for R2 (per-campaign budget guard resets) and H-3 (planner
deterministic gates).

R2 -- core/budget.py:
  The 500-step / 50-visit / per-visit-tool ceilings are CAMPAIGN-scoped runaway
  loop guards, reset by BudgetController.begin_campaign() at each campaign
  boundary. The cumulative counters (graph_steps, llm_calls, accumulated_tokens,
  start_time) feed the spend ledger and the real run budget and must NEVER be
  reset. If the reset semantics regress in either direction, a multi-campaign
  sweep either pauses spuriously every ~31 campaigns (guards became cumulative
  again) or a wedged campaign is never stopped (guards stopped firing).

H-3 -- core/planner.py:
  The LLM planning pass is gated structurally: CP-3 path re-validation and
  root containment, TIER_INTENT claim filtering, an affordability cap that trims
  but never refuses, and a safe-empty-plan (`available: False`) on EVERY failure
  path except budget/auth exhaustion, which must propagate (INV-6). These tests
  exercise the non-LLM layers directly and script the model for the async entry
  point; no network and no real LLM calls are made.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import types as _pytypes
import unittest
from pathlib import Path
from unittest.mock import patch

_REF_ROOT = str(Path(__file__).resolve().parent.parent)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)

import core.planner as planner
from core.budget import BudgetConfig, BudgetController, BudgetExceededError
from core.llm_gateway import UNTRUSTED_DATA_END, UNTRUSTED_DATA_START

# Every control character except tab/newline, including ESC (\x1b) and the 8-bit
# C1 range: the same fail-closed class core.llm_gateway.strip_terminal_control
# removes. The pause banner is copy-paste executable, so one surviving escape
# sequence can repaint the command the operator is about to run.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _quiet_config(**overrides) -> BudgetConfig:
    """A config with generous run-level ceilings so ONLY the dimension under
    test can fire. Wall-clock and token limits are set far above anything a
    unit test can reach, so an assertion failure always names the guard that
    actually regressed instead of a bystander."""
    values = dict(
        max_wall_clock_seconds=3600.0,
        max_tokens=1_000_000_000,
        max_graph_steps=500,
        max_node_visits=50,
        max_llm_calls=2000,
        max_node_tool_calls=200,
    )
    values.update(overrides)
    return BudgetConfig(**values)


# =====================================================================================
# R2: per-campaign budget guard resets
# =====================================================================================


class TestBeginCampaignResetsOnlyCampaignScope(unittest.TestCase):
    """Test 1: begin_campaign() zeroes exactly the four per-campaign structures."""

    def test_begin_campaign_zeroes_campaign_counters_because_they_are_loop_guards(self):
        """If begin_campaign() stops zeroing campaign_graph_steps,
        campaign_llm_calls, node_visit_counts or node_tool_counts, the loop
        guards silently become cumulative run ceilings again and a healthy
        multi-campaign sweep pauses every ~31 campaigns."""
        ctrl = BudgetController(config=_quiet_config(), run_id="r_reset")
        ctrl.record_tokens(1000, cached_count=200)
        ctrl.record_step("nodeA")
        ctrl.record_step("nodeA")
        ctrl.record_tool_call("nodeA", "read_file")

        self.assertGreater(ctrl.campaign_graph_steps, 0)
        self.assertGreater(ctrl.campaign_llm_calls, 0)
        self.assertTrue(ctrl.node_visit_counts)
        self.assertTrue(ctrl.node_tool_counts)

        ctrl.begin_campaign()

        self.assertEqual(ctrl.campaign_graph_steps, 0)
        self.assertEqual(ctrl.campaign_llm_calls, 0)
        self.assertEqual(ctrl.node_visit_counts, {})
        self.assertEqual(ctrl.node_tool_counts, {})

    def test_begin_campaign_preserves_cumulative_state_because_ledger_deltas_depend_on_it(self):
        """main.py's spend ledger computes each campaign's cost as the delta of
        the CUMULATIVE counters either side of the campaign. If begin_campaign()
        ever resets accumulated_tokens, fresh/cached splits, llm_calls,
        graph_steps or start_time, those deltas go negative and the run budget
        (wall-clock/token ceilings) silently restarts at every boundary."""
        ctrl = BudgetController(config=_quiet_config(), run_id="r_preserve")
        ctrl.record_tokens(5000, cached_count=1000)
        ctrl.record_tokens(300)
        for _ in range(7):
            ctrl.record_step()

        before = dict(
            accumulated_tokens=ctrl.accumulated_tokens,
            fresh_tokens=ctrl.fresh_tokens,
            cached_tokens=ctrl.cached_tokens,
            llm_calls=ctrl.llm_calls,
            graph_steps=ctrl.graph_steps,
            start_time=ctrl.start_time,
        )
        self.assertGreater(before["accumulated_tokens"], 0)
        self.assertGreater(before["cached_tokens"], 0)

        ctrl.begin_campaign()

        self.assertEqual(ctrl.accumulated_tokens, before["accumulated_tokens"])
        self.assertEqual(ctrl.fresh_tokens, before["fresh_tokens"])
        self.assertEqual(ctrl.cached_tokens, before["cached_tokens"])
        self.assertEqual(ctrl.llm_calls, before["llm_calls"])
        self.assertEqual(ctrl.graph_steps, before["graph_steps"])
        self.assertEqual(ctrl.start_time, before["start_time"])


class TestMultiCampaignStepCeiling(unittest.TestCase):
    """Tests 2, 3, 6: the step ceiling is per-campaign, still catches runaways,
    and degrades to the old cumulative behaviour when begin_campaign is omitted."""

    def test_forty_campaigns_of_sixteen_steps_never_pause_because_guard_is_per_campaign(self):
        """Test 2. Measured live: a campaign costs exactly 16 graph steps, so a
        cumulative 500-step ceiling paused a sweep every ~31 campaigns for no
        protective benefit. 40 campaigns x 16 steps must complete without a
        pause while the cumulative counter honestly reports 640."""
        ctrl = BudgetController(config=_quiet_config(max_graph_steps=500), run_id="r_sweep")
        for _campaign in range(40):
            ctrl.begin_campaign()
            for _step in range(16):
                ctrl.record_step()  # raises BudgetExceededError on regression
        self.assertEqual(ctrl.graph_steps, 640)
        self.assertFalse(ctrl.is_paused)

    def test_single_runaway_campaign_still_pauses_with_graph_steps_trigger(self):
        """Test 3. The reset must not have destroyed the protection: one
        campaign that records 500 steps within its own boundary is a genuine
        runaway loop and must raise with the HISTORIC trigger name
        'graph_steps', because existing pause-handling and resume tooling key
        on that exact string."""
        ctrl = BudgetController(config=_quiet_config(max_graph_steps=500), run_id="r_runaway")
        for _campaign in range(5):
            ctrl.begin_campaign()
            for _step in range(16):
                ctrl.record_step()

        ctrl.begin_campaign()
        with self.assertRaises(BudgetExceededError) as ctx:
            for _step in range(500):
                ctrl.record_step()
        self.assertEqual(ctx.exception.trigger, "graph_steps")
        self.assertEqual(ctx.exception.current_value, 500)
        self.assertEqual(ctx.exception.limit_value, 500)
        self.assertTrue(ctrl.is_paused)
        # Cumulative counter is untouched by the pause: 5*16 prior + 500 now.
        self.assertEqual(ctrl.graph_steps, 580)

    def test_omitting_begin_campaign_fails_closed_to_cumulative_pause(self):
        """Test 6. A caller that forgets begin_campaign() must get the OLD
        cumulative behaviour -- a spurious pause at 500 total steps -- never a
        runaway. If this test fails, the degraded mode became 'no guard at
        all', which is the fail-open direction."""
        ctrl = BudgetController(config=_quiet_config(max_graph_steps=500), run_id="r_noreset")
        with self.assertRaises(BudgetExceededError) as ctx:
            for _campaign in range(4):
                # No begin_campaign() between these simulated campaigns.
                for _step in range(125):
                    ctrl.record_step()
        self.assertEqual(ctx.exception.trigger, "graph_steps")
        self.assertEqual(ctx.exception.current_value, 500)


class TestNodeVisitGuardResets(unittest.TestCase):
    """Test 4: the per-node visit guard is scoped to a single campaign."""

    def test_node_visits_reset_at_campaign_boundary_because_guard_detects_loops(self):
        """49 visits of the same node, a campaign boundary, then 49 more must
        NOT pause: the guard answers 'is THIS campaign looping on this node',
        and a node legitimately visited once per campaign would otherwise trip
        it after max_node_visits campaigns."""
        ctrl = BudgetController(config=_quiet_config(max_node_visits=50), run_id="r_visits")
        ctrl.begin_campaign()
        for _ in range(49):
            ctrl.record_step("hot_node")
        ctrl.begin_campaign()
        for _ in range(49):
            ctrl.record_step("hot_node")  # raises on regression
        self.assertEqual(ctrl.node_visit_counts["hot_node"], 49)

    def test_node_visit_runaway_within_one_campaign_still_raises_node_visits_ceiling(self):
        """51 visits of one node inside a single campaign is a wedged loop and
        must raise with trigger 'node_visits_ceiling' (the actual trigger name
        in budget.py record_step). Losing this raise means a cycling classifier
        burns the whole token budget on one node."""
        ctrl = BudgetController(config=_quiet_config(max_node_visits=50), run_id="r_visits2")
        ctrl.begin_campaign()
        with self.assertRaises(BudgetExceededError) as ctx:
            for _ in range(51):
                ctrl.record_step("hot_node")
        self.assertEqual(ctx.exception.trigger, "node_visits_ceiling")
        self.assertEqual(ctx.exception.current_value, 51)
        self.assertEqual(ctx.exception.limit_value, 50)


class TestOldCheckpointResumeCompat(unittest.TestCase):
    """Test 5 (INV-6): resuming from a checkpoint written before per-campaign
    counters existed must not instantly re-pause."""

    def test_resume_with_large_initial_steps_does_not_repause_on_steps(self):
        """--resume reconstructs the controller with the checkpoint's CUMULATIVE
        step count as initial_steps. The step ceiling is enforced against
        campaign_graph_steps (which starts at zero), so a checkpoint whose
        cumulative count already exceeds max_graph_steps must load and pass
        check_budget. If this raises, every resumed long run pauses again on
        its very first budget check -- an unrecoverable pause loop."""
        ctrl = BudgetController(
            config=_quiet_config(max_graph_steps=500),
            run_id="r_resume",
            initial_tokens=1_000_000,
            initial_steps=10_000,  # >> max_graph_steps
            start_time=time.time(),
        )
        # Must not raise: the per-campaign counter is fresh even though the
        # cumulative counter is far beyond the ceiling.
        ctrl.check_budget()
        self.assertEqual(ctrl.graph_steps, 10_000)
        self.assertEqual(ctrl.campaign_graph_steps, 0)
        # And the first campaign of the resumed run proceeds normally.
        ctrl.begin_campaign()
        ctrl.record_step()
        self.assertEqual(ctrl.graph_steps, 10_001)
        self.assertEqual(ctrl.campaign_graph_steps, 1)


class TestPauseBannerScopes(unittest.TestCase):
    """Test 7: the pause banner states BOTH the campaign scope and the run scope."""

    def test_banner_shows_campaign_and_run_step_scopes_without_control_characters(self):
        """The step limit is per campaign; printing only the run total against a
        per-campaign limit misstates how close the run is to pausing by
        whatever the sweep already accumulated. The banner must render
        'campaign / limit this campaign (total this run)'. It is also a
        copy-paste executable, so any surviving control character is a
        terminal-repaint primitive."""
        ctrl = BudgetController(config=_quiet_config(max_graph_steps=500), run_id="r_banner")
        ctrl.campaign_graph_steps = 17
        ctrl.graph_steps = 633

        banner = ctrl.format_pause_banner(
            trigger="graph\x1b[31m_steps",  # hostile escape riding the trigger
            progress_summary="12/40 campaigns",
            target="",
        )
        self.assertIn("17 / 500 limit this campaign (633 total this run)", banner)
        self.assertIn("graph_steps", banner)  # escape removed, text intact
        self.assertIn("--resume", banner)
        self.assertIsNone(
            _CONTROL_CHARS.search(banner),
            "pause banner leaked a control character; it is pasted into a terminal",
        )


class TestUnboundedBudget(unittest.TestCase):
    """--no-budget: a ceiling of 0 disables that dimension, and ONLY that dimension.

    The dangerous regression here is silent inversion: before the > 0 guards,
    a zeroed ceiling meant "pause instantly" (anything >= 0), the exact
    opposite of what the operator asked for.
    """

    def test_zero_spend_ceilings_never_trip(self):
        cfg = _quiet_config(max_wall_clock_seconds=0, max_tokens=0)
        ctrl = BudgetController(config=cfg, run_id="r_unbounded")
        ctrl.record_tokens(999_999_999)  # would exceed any real ceiling
        ctrl.check_budget()  # must not raise
        self.assertFalse(ctrl.is_paused)

    def test_loop_guards_still_fire_with_spend_ceilings_disabled(self):
        """No budget is not no guards: a wedged loop produces zero findings at
        ANY budget, so the campaign-scoped step guard must still pause."""
        cfg = _quiet_config(max_wall_clock_seconds=0, max_tokens=0, max_graph_steps=3)
        ctrl = BudgetController(config=cfg, run_id="r_unbounded_guard")
        ctrl.begin_campaign()
        with self.assertRaises(BudgetExceededError) as ctx:
            for _ in range(10):
                ctrl.record_step("looper")
        self.assertEqual(ctx.exception.trigger, "graph_steps")

    def test_zero_step_guard_is_disabled_individually(self):
        cfg = _quiet_config(max_graph_steps=0, max_node_visits=0)
        ctrl = BudgetController(config=cfg, run_id="r_no_step_guard")
        ctrl.begin_campaign()
        for _ in range(1000):
            ctrl.record_step("busy")  # must not raise

    def test_banner_renders_unlimited_not_zero_limits(self):
        cfg = _quiet_config(max_wall_clock_seconds=0, max_tokens=0, max_graph_steps=0)
        ctrl = BudgetController(config=cfg, run_id="r_unbounded_banner")
        banner = ctrl.format_pause_banner(trigger="node_visits_ceiling")
        self.assertIn("unlimited", banner)
        self.assertNotIn("/ 0.0h limit", banner)
        self.assertNotIn("/ 0 limit", banner)

    def test_parsers_accept_unlimited_spellings(self):
        from core.budget import parse_duration_seconds, parse_token_budget

        for spelling in ("unlimited", "none", "off", " UNLIMITED "):
            self.assertEqual(parse_duration_seconds(spelling), 0.0)
            self.assertEqual(parse_token_budget(spelling), 0)

    def test_estimate_scan_unbounded_covers_whole_plan(self):
        """A zero token budget means disabled, so the estimate must report the
        full plan as covered -- not 'covers ~0 campaigns'."""
        from core.cost import estimate_scan

        est = estimate_scan(["a.py", "b.py", "c.py"], 0)
        self.assertEqual(est.affordable_campaigns, 3)
        self.assertTrue(est.fits)
        self.assertIn("covers all 3", est.describe())


# =====================================================================================
# H-3: planner deterministic gates
# =====================================================================================


def _make_plan(proposals, rationale=""):
    """Builds a real CampaignPlan schema instance the way propose_campaigns would."""
    plan_cls, proposal_cls = planner._plan_schemas()
    return plan_cls(
        campaigns=[proposal_cls(**p) for p in proposals], rationale=rationale
    )


class _PlannerGateBase(unittest.TestCase):
    """Shared fixture: a real scan root with real files, and a real outside dir,
    because validate_scan_target demands existing, non-symlinked paths."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="mantis_test_planner_")
        self.addCleanup(tmp.cleanup)
        outside = tempfile.TemporaryDirectory(prefix="mantis_test_planner_outside_")
        self.addCleanup(outside.cleanup)

        # resolve() up front: on macOS tempdirs live under /var -> /private/var,
        # and the containment check compares RESOLVED paths.
        self.root = Path(tmp.name).resolve()
        self.outside = Path(outside.name).resolve()
        (self.root / "a.py").write_text("a = 1\n", encoding="utf-8")
        (self.root / "b.py").write_text("b = 2\n", encoding="utf-8")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "c.py").write_text("c = 3\n", encoding="utf-8")
        self.outside_file = self.outside / "secret.txt"
        self.outside_file.write_text("TOP_SECRET\n", encoding="utf-8")


class TestGateRejectsEscapingPaths(_PlannerGateBase):
    """Test 8: CP-3 containment -- the model chooses among authorized places,
    it can never add one."""

    def test_absolute_path_outside_root_is_rejected_and_counted(self):
        """An existing absolute path outside the scan root passes
        validate_scan_target but must fail relative_to(base) containment. If it
        survives, planner prose written by an earlier (attacker-influenced) run
        can point a campaign at ~/.ssh, and the rejected_paths counter -- the
        operator's only signal that a model tried to leave the root -- goes
        silent."""
        plan = _make_plan(
            [
                {
                    "kind": "coverage",
                    "mode": "cross-functional",
                    "target_paths": [str(self.root / "a.py"), str(self.outside_file)],
                    "hypothesis": "look at a.py",
                }
            ]
        )
        gated = planner._gate_proposals(plan, str(self.root), 0, "", "cross-functional")
        self.assertTrue(gated["available"])
        self.assertEqual(gated["counts"]["rejected_paths"], 1)
        self.assertEqual(gated["counts"]["proposed_paths"], 2)
        self.assertEqual(gated["targets"], [str(self.root / "a.py")])
        self.assertNotIn(str(self.outside_file), json.dumps(gated))

    def test_proposal_with_no_surviving_path_is_dropped_whole_including_hypothesis(self):
        """When every path of a proposal fails validation, the proposal -- and
        crucially its hypothesis TEXT -- must vanish. The only thing tying that
        prose to the run is a validated target; without one, injecting the text
        anywhere would hand un-anchored LLM prose to a campaign prompt."""
        plan = _make_plan(
            [
                {
                    "kind": "coverage",
                    "mode": "cross-functional",
                    "target_paths": [str(self.root / "b.py")],
                    "hypothesis": "SURVIVOR_HYPOTHESIS",
                },
                {
                    "kind": "hypothesis",
                    "mode": "whole",
                    "target_paths": [str(self.outside_file)],
                    "hypothesis": "MARKER_MUST_BE_DROPPED",
                },
            ]
        )
        gated = planner._gate_proposals(plan, str(self.root), 0, "", "cross-functional")
        self.assertTrue(gated["available"])
        self.assertEqual(len(gated["proposals"]), 1)
        self.assertNotIn("MARKER_MUST_BE_DROPPED", json.dumps(gated))
        self.assertIn("SURVIVOR_HYPOTHESIS", json.dumps(gated))

    def test_plan_with_zero_surviving_proposals_is_unavailable(self):
        """A plan where nothing survives the gates must degrade to
        `available: False` so the caller keeps the Surveyor's CP-3 validated
        list (INV-6), not an empty scan."""
        plan = _make_plan(
            [
                {
                    "kind": "coverage",
                    "mode": "file-by-file",
                    "target_paths": [str(self.outside_file)],
                    "hypothesis": "all paths escape",
                }
            ]
        )
        gated = planner._gate_proposals(plan, str(self.root), 0, "", "cross-functional")
        self.assertFalse(gated["available"])
        self.assertEqual(gated["targets"], [])
        self.assertEqual(gated["proposals"], [])


class TestGateRejectsTraversalAndNonexistent(_PlannerGateBase):
    """Test 9: relative traversal and phantom paths die at validate_scan_target."""

    def test_dotdot_traversal_and_nonexistent_paths_are_rejected_via_cp3(self):
        """'../x' is anchored at the validated root and must be refused by
        validate_scan_target's '..'-component ban (the normalization
        differential), and a path that does not exist must be refused by its
        existence check. If either survives, the planner has a path-invention
        primitive the Surveyor pipeline does not."""
        plan = _make_plan(
            [
                {
                    "kind": "coverage",
                    "mode": "cross-functional",
                    "target_paths": [
                        "../evil.py",  # traversal, anchored at root
                        "no_such_dir/nope.py",  # does not exist
                        "a.py",  # valid, keeps the proposal alive
                    ],
                    "hypothesis": "traversal probe",
                }
            ]
        )
        gated = planner._gate_proposals(plan, str(self.root), 0, "", "cross-functional")
        self.assertTrue(gated["available"])
        self.assertEqual(gated["counts"]["rejected_paths"], 2)
        self.assertEqual(gated["targets"], [str(self.root / "a.py")])


class TestAffordabilityCap(_PlannerGateBase):
    """Test 10: the cap trims the tail in model order; a broken estimate never
    costs the plan."""

    def _twelve_proposals(self):
        proposals = []
        for i in range(12):
            f = self.root / f"f{i:02d}.py"
            f.write_text(f"x = {i}\n", encoding="utf-8")
            proposals.append(
                {
                    "kind": "coverage" if i % 2 == 0 else "hypothesis",
                    "mode": "cross-functional",
                    "target_paths": [str(f)],
                    "hypothesis": f"idea {i}",
                }
            )
        return proposals

    def test_estimate_affording_three_keeps_first_three_in_model_order(self):
        """The model was asked to lead with its best ideas, so trimming must
        keep the HEAD of its ordering; re-sorting here would silently
        relitigate the model's ranking. The trimmed count must surface in
        counts and in the operator summary line."""
        plan = _make_plan(self._twelve_proposals())
        stub_est = _pytypes.SimpleNamespace(
            cost_per_campaign=50_000,
            affordable_campaigns=3,
            basis="observed",
            observations=7,
        )
        with patch("core.cost.estimate_scan", return_value=stub_est):
            gated = planner._gate_proposals(
                plan, str(self.root), 1_000_000, "", "cross-functional"
            )
        self.assertTrue(gated["available"])
        self.assertEqual(
            gated["targets"],
            [str(self.root / f"f{i:02d}.py") for i in range(3)],
            "trim must keep the model's own leading order",
        )
        self.assertEqual(gated["counts"]["trimmed_by_budget"], 9)
        self.assertEqual(gated["counts"]["accepted_campaigns"], 3)
        self.assertEqual(len(gated["proposals"]), 3)
        summary = planner.summarize_campaign_plan(gated)
        self.assertIn("9 trimmed by budget", summary)

    def test_estimate_failure_leaves_plan_untrimmed_because_controller_owns_the_ceiling(self):
        """cost.estimate_scan raising must leave the plan whole with no trim:
        the budget controller enforces the real ceiling, and 'the estimator
        broke' must never shrink a plan (INV-6 fail-safe direction is
        untrimmed, not empty)."""
        plan = _make_plan(self._twelve_proposals())
        with patch(
            "core.cost.estimate_scan", side_effect=RuntimeError("ledger corrupt")
        ):
            gated = planner._gate_proposals(
                plan, str(self.root), 1_000_000, "", "cross-functional"
            )
        self.assertTrue(gated["available"])
        self.assertEqual(len(gated["targets"]), 12)
        self.assertEqual(gated["counts"]["trimmed_by_budget"], 0)
        self.assertEqual(gated["estimate"], {})


class TestHasPlanningHistory(unittest.TestCase):
    """Test 11: the gate main.py consults before paying for a planning call."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="mantis_test_history_")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_missing_and_empty_db_report_no_history(self):
        """A first audit has nothing to plan FROM; reporting history would send
        main.py off to pay for an LLM call that can only dress a guess up as a
        plan."""
        missing = str(self.tmp / "never_created.db")
        self.assertFalse(planner.has_planning_history(missing, str(self.tmp)))

        from core.database import init_db

        empty = str(self.tmp / "empty.db")
        init_db(empty)
        self.assertFalse(planner.has_planning_history(empty, str(self.tmp)))

    def test_campaign_spend_row_counts_as_history(self):
        """One recorded campaign_spend row is real history (where budget went is
        planning evidence). If this stops counting, the planner never activates
        on deployments that have spend but no findings yet."""
        from core.cost import record_spend

        db = str(self.tmp / "spend.db")
        self.assertTrue(
            record_spend(
                db,
                run_id="r_hist",
                target=str(self.tmp / "svc"),
                scan_mode="cross-functional",
                tokens=50_000,
                llm_calls=100,
                graph_steps=16,
            )
        )
        self.assertTrue(planner.has_planning_history(db, str(self.tmp)))

    def test_corrupt_db_reports_false_without_raising(self):
        """INV-6: a knowledge base that cannot be read costs the planner its
        memory, never the run its life. has_planning_history on garbage bytes
        must return False, not raise."""
        corrupt = self.tmp / "corrupt.db"
        corrupt.write_bytes(b"this is definitely not a sqlite database\x00" * 64)
        try:
            result = planner.has_planning_history(str(corrupt), str(self.tmp))
        except Exception as exc:  # pragma: no cover - the failure being tested
            self.fail(f"has_planning_history raised on a corrupt DB: {exc!r}")
        self.assertFalse(result)


class TestRenderCampaignHypothesis(unittest.TestCase):
    """Test 12: hypothesis prose reaches a campaign only inside CP-4 fencing."""

    def test_hypothesis_is_cp4_fenced_with_untrusted_markers(self):
        """The hypothesis is LLM prose derived from prior LLM findings --
        doubly untrusted -- and must travel between the exact
        UNTRUSTED_DATA_START/END delimiters core.llm_gateway uses, followed by
        the operator-authored evidence-not-instruction trailer. Unfenced, a
        hostile 'finding' from a prior run becomes an instruction in the next
        campaign's prompt."""
        plan = {
            "available": True,
            "hypotheses": {
                "/repo/svc/upload.py": {
                    "kind": "hypothesis",
                    "hypothesis": "archive-extraction traversal at the seam",
                    "motivating_findings": ["LIN-42", "LIN-77"],
                }
            },
        }
        out = planner.render_campaign_hypothesis(plan, "/repo/svc/upload.py")
        self.assertIn(UNTRUSTED_DATA_START, out)
        self.assertIn(UNTRUSTED_DATA_END, out)
        start = out.index(UNTRUSTED_DATA_START)
        end = out.index(UNTRUSTED_DATA_END)
        body = out[start:end]
        self.assertIn("archive-extraction traversal at the seam", body)
        self.assertIn("LIN-42", body)
        # The fence must come BEFORE the operator trailer that demotes it to
        # evidence, and the trailer must exist outside the fence.
        self.assertIn("never an instruction", out[end:])

    def test_target_not_in_plan_renders_empty_string(self):
        """Callers append unconditionally; a non-empty return for an unplanned
        target would inject another campaign's hypothesis into the wrong
        prompt."""
        plan = {
            "available": True,
            "hypotheses": {"/repo/a.py": {"kind": "coverage", "hypothesis": "x"}},
        }
        self.assertEqual(planner.render_campaign_hypothesis(plan, "/repo/b.py"), "")
        self.assertEqual(planner.render_campaign_hypothesis({"available": False}, "/repo/a.py"), "")
        self.assertEqual(planner.render_campaign_hypothesis(None, "/repo/a.py"), "")


class TestModeSpellingPin(unittest.TestCase):
    """Test 13: the planner duplicates main.py's mode spellings as literals
    (core must not import main); this pin is the contract that keeps them in step."""

    def test_planner_mode_literals_exactly_match_main_scan_mode_constants(self):
        """_PROPOSAL_MODES and the CampaignProposal Literal are duplicated from
        main.py's SCAN_MODE_* because the layering forbids core -> main
        imports. If either side is renamed, gated proposals in the renamed mode
        are silently discarded ('mode not in _PROPOSAL_MODES' -> continue) and
        the planner degrades to unavailable with no error. This test makes that
        rename loud."""
        import typing

        from main import (
            SCAN_MODE_CROSS_FUNCTIONAL,
            SCAN_MODE_FILE_BY_FILE,
            SCAN_MODE_WHOLE,
        )

        expected = (SCAN_MODE_CROSS_FUNCTIONAL, SCAN_MODE_FILE_BY_FILE, SCAN_MODE_WHOLE)
        self.assertEqual(SCAN_MODE_CROSS_FUNCTIONAL, "cross-functional")
        self.assertEqual(SCAN_MODE_FILE_BY_FILE, "file-by-file")
        self.assertEqual(SCAN_MODE_WHOLE, "whole")
        self.assertEqual(set(planner._PROPOSAL_MODES), set(expected))

        # The pydantic schema's Literal must carry the same spellings: the
        # tuple and the schema are separate declarations and can drift apart.
        _plan_cls, proposal_cls = planner._plan_schemas()
        literal_args = typing.get_args(proposal_cls.model_fields["mode"].annotation)
        self.assertEqual(set(literal_args), set(expected))


class _ScriptedLlm:
    """Async stub matching the generate_content_async surface propose_campaigns
    consumes: yields one response whose single part carries `text`."""

    def __init__(self, text):
        self._text = text

    async def generate_content_async(self, req, stream=False):
        part = _pytypes.SimpleNamespace(function_call=None, text=self._text)
        yield _pytypes.SimpleNamespace(
            content=_pytypes.SimpleNamespace(parts=[part]), usage_metadata=None
        )


class _RaisingLlm:
    """Async stub that raises before yielding anything."""

    def __init__(self, exc):
        self._exc = exc

    async def generate_content_async(self, req, stream=False):
        raise self._exc
        yield  # pragma: no cover - makes this an async generator


class TestProposeCampaignsFailSafe(unittest.IsolatedAsyncioTestCase):
    """Test 14: every planning failure returns the safe empty plan; budget
    exhaustion propagates instead of becoming a silent surveyor fallback."""

    async def asyncSetUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="mantis_test_propose_")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        (self.root / "a.py").write_text("a = 1\n", encoding="utf-8")
        self.db = str(self.root / "knowledge.db")

        # Seed real history so propose_campaigns gets past the context gate and
        # actually consults the (scripted) model.
        from core.cost import record_spend

        self.assertTrue(
            record_spend(
                self.db,
                run_id="r_seed",
                target=str(self.root / "a.py"),
                scan_mode="cross-functional",
                tokens=60_000,
                llm_calls=120,
                graph_steps=16,
            )
        )
        self.assertTrue(planner.has_planning_history(self.db, str(self.root)))

    async def _propose(self, llm):
        return await planner.propose_campaigns(
            db_path=self.db,
            target=str(self.root),
            surveyor_targets=[str(self.root / "a.py")],
            llm_model=llm,
            token_budget=0,
            scan_mode="cross-functional",
        )

    async def test_refusal_text_degrades_to_safe_empty_plan(self):
        """A model that answers in prose instead of the schema produced no
        plan. The caller must get `available: False` and keep the Surveyor's
        list -- the behaviour that shipped before H-3 existed (INV-6)."""
        result = await self._propose(
            _ScriptedLlm("I cannot produce a campaign plan for this request.")
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["targets"], [])

    async def test_malformed_json_degrades_to_safe_empty_plan(self):
        """Truncated / unparseable JSON must yield None from the text parser --
        never a partially-trusted dict -- and therefore the safe empty plan."""
        result = await self._propose(
            _ScriptedLlm('```json\n{"campaigns": [{"kind": "coverage", "mode":\n```')
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["targets"], [])

    async def test_empty_campaigns_list_degrades_to_safe_empty_plan(self):
        """A schema-valid plan with zero campaigns is a valid refusal; it must
        gate to unavailable rather than to an empty-but-available plan that
        would replace the Surveyor's list with nothing."""
        result = await self._propose(
            _ScriptedLlm('{"campaigns": [], "rationale": "nothing worth planning"}')
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["targets"], [])

    async def test_llm_exception_degrades_but_budget_exceeded_propagates(self):
        """An ordinary model failure costs the plan, never the run. But
        BudgetExceededError means the RUN cannot continue: swallowing it here
        would turn 'out of budget' into a silent surveyor fallback that then
        spends MORE budget."""
        # Ordinary failure: degrade.
        result = await self._propose(_RaisingLlm(RuntimeError("503 model melted")))
        self.assertFalse(result["available"])

        # Budget exhaustion: propagate.
        boom = BudgetExceededError(
            trigger="token_budget",
            current_value=10_000_000,
            limit_value=10_000_000,
            run_id="r_seed",
        )
        with self.assertRaises(BudgetExceededError):
            await self._propose(_RaisingLlm(boom))

    async def test_auth_error_propagates_like_budget_exhaustion(self):
        """MantisAuthError shares BudgetExceededError's re-raise path: an
        unauthenticated run cannot continue and must not silently fall back."""
        from core.config import MantisAuthError

        with self.assertRaises(MantisAuthError):
            await self._propose(_RaisingLlm(MantisAuthError("401 Unauthorized")))


# =====================================================================================
# Dynamic-planner sprint: groups payload, chain hypotheses, replan, operator steering
# =====================================================================================


class _RecordingLlm:
    """Async stub that records every LlmRequest it receives and yields one canned
    text response, so tests can assert on the PROMPT the planner actually built."""

    def __init__(self, text):
        self._text = text
        self.requests = []

    async def generate_content_async(self, req, stream=False):
        self.requests.append(req)
        part = _pytypes.SimpleNamespace(function_call=None, text=self._text)
        yield _pytypes.SimpleNamespace(
            content=_pytypes.SimpleNamespace(parts=[part]), usage_metadata=None
        )

    def prompt(self):
        return self.requests[0].contents[0].parts[0].text


class TestGroupsPayload(_PlannerGateBase):
    """The gate's 'groups' key: one accepted proposal == one multi-target campaign
    group, capped at _MAX_GROUP_TARGETS, chain carried, prior keys untouched."""

    def test_single_target_group_has_exact_shape_and_prior_keys_survive(self):
        """Fail direction pinned: if the group shape drifts (missing/renamed keys)
        the scan loop KeyErrors mid-run, and if 'groups' stops being purely
        additive, callers that predate groups (reading targets/hypotheses) break
        -- the INV-6 backward-compat promise of the payload."""
        plan = _make_plan(
            [
                {
                    "kind": "coverage",
                    "mode": "cross-functional",
                    "target_paths": [str(self.root / "a.py")],
                    "hypothesis": "single-target group",
                }
            ]
        )
        gated = planner._gate_proposals(plan, str(self.root), 0, "", "cross-functional")
        self.assertTrue(gated["available"])
        self.assertEqual(len(gated["groups"]), 1)
        group = gated["groups"][0]
        self.assertEqual(set(group.keys()), {"targets", "mode", "hypothesis", "chain"})
        self.assertEqual(group["targets"], [str(self.root / "a.py")])
        self.assertEqual(group["mode"], "cross-functional")
        self.assertEqual(group["hypothesis"], "single-target group")
        self.assertIsNone(group["chain"])  # no chain proposed -> None, not absent
        # Prior payload keys keep their exact prior shapes.
        self.assertEqual(gated["targets"], [str(self.root / "a.py")])
        self.assertIn(str(self.root / "a.py"), gated["hypotheses"])
        self.assertEqual(len(gated["proposals"]), 1)

    def test_six_target_proposal_trims_to_five_and_counts_trimmed_by_group(self):
        """Fail direction pinned: without the cap a proposal becomes a queue
        wearing one hypothesis and the group no longer fits a single context;
        without the COUNT the operator never learns targets were dropped. The
        trim must keep the model's leading order (its best-ranked targets) and
        must never reject the proposal outright -- pydantic-style rejection
        would throw away five good targets to punish a sixth."""
        files = []
        for i in range(6):
            f = self.root / f"g{i}.py"
            f.write_text(f"g = {i}\n", encoding="utf-8")
            files.append(str(f))
        plan = _make_plan(
            [
                {
                    "kind": "coverage",
                    "mode": "cross-functional",
                    "target_paths": files,
                    "hypothesis": "six targets",
                }
            ]
        )
        gated = planner._gate_proposals(plan, str(self.root), 0, "", "cross-functional")
        self.assertTrue(gated["available"])
        self.assertEqual(gated["groups"][0]["targets"], files[:5])
        self.assertEqual(gated["counts"]["trimmed_by_group"], 1)
        self.assertEqual(gated["targets"], files[:5])

    def test_chain_hypothesis_is_carried_into_the_group_as_plain_dict(self):
        """Fail direction pinned: a chain that survives sanitization but is not
        carried into the group never reaches the campaign that was supposed to
        investigate it -- the entire point of the chain schema. It must arrive
        as a PLAIN dict (never a pydantic instance), because everything past
        the gate is data that outlives the pydantic import."""
        plan = _make_plan(
            [
                {
                    "kind": "hypothesis",
                    "mode": "cross-functional",
                    "target_paths": [str(self.root / "a.py")],
                    "hypothesis": "seam between upload and extract",
                    "chain_hypothesis": {
                        "description": "upload feeds extractor",
                        "links": [
                            {
                                "source_component": "upload_handler",
                                "sink_component": "archive_extractor",
                                "claim": "filename crosses unsanitized",
                            }
                        ],
                    },
                }
            ]
        )
        gated = planner._gate_proposals(plan, str(self.root), 0, "", "cross-functional")
        self.assertTrue(gated["available"])
        chain = gated["groups"][0]["chain"]
        self.assertIsInstance(chain, dict)
        self.assertEqual(chain["description"], "upload feeds extractor")
        self.assertEqual(len(chain["links"]), 1)
        self.assertEqual(
            chain["links"][0],
            {
                "source_component": "upload_handler",
                "sink_component": "archive_extractor",
                "claim": "filename crosses unsanitized",
            },
        )

    def test_hollow_chain_becomes_none_but_proposal_survives(self):
        """Fail direction pinned: losing a chain costs context, losing the
        proposal would cost coverage -- _sanitize_chain returning None must
        never take the proposal down with it. If this regresses toward
        drop-the-proposal, a model emitting one malformed chain silently
        cancels a whole campaign group."""
        plan = _make_plan(
            [
                {
                    "kind": "hypothesis",
                    "mode": "whole",
                    "target_paths": [str(self.root / "b.py")],
                    "hypothesis": "kept despite hollow chain",
                    "chain_hypothesis": {"description": "", "links": []},
                }
            ]
        )
        gated = planner._gate_proposals(plan, str(self.root), 0, "", "cross-functional")
        self.assertTrue(gated["available"])
        self.assertEqual(len(gated["groups"]), 1)
        self.assertIsNone(gated["groups"][0]["chain"])
        self.assertEqual(gated["groups"][0]["hypothesis"], "kept despite hollow chain")


class TestChainSanitization(unittest.TestCase):
    """_sanitize_chain: filter_claim at every level, hard length caps, and
    None-never-partial on failure."""

    def _sanitize(self, raw):
        from core.evidence import TIER_INTENT, filter_claim

        return planner._sanitize_chain(raw, filter_claim, TIER_INTENT)

    def test_verdict_fields_are_stripped_at_chain_and_link_level(self):
        """Fail direction pinned: a chain is TIER_INTENT prose about how findings
        might compose; if a 'status'/'reached_sink' smuggled into a chain or a
        link survives sanitization, LLM prose acquires verdict authority that
        INV-1/INV-2 reserve for reached-sink evidence. The output must carry
        exactly the fixed key sets and nothing that rode along."""
        raw = {
            "description": "end to end story",
            "status": "confirmed",  # verdict smuggled at chain level
            "reached_sink": True,
            "links": [
                {
                    "source_component": "src",
                    "sink_component": "snk",
                    "claim": "hop claim",
                    "status": "confirmed",  # verdict smuggled at link level
                    "repro_status": "reproduced",
                },
                "not-a-dict-link-is-skipped",
            ],
        }
        out = self._sanitize(raw)
        self.assertIsNotNone(out)
        self.assertEqual(set(out.keys()), {"description", "links"})
        self.assertEqual(len(out["links"]), 1)
        self.assertEqual(
            set(out["links"][0].keys()),
            {"source_component", "sink_component", "claim"},
        )
        self.assertNotIn("status", json.dumps(out))
        self.assertNotIn("reached_sink", json.dumps(out))

    def test_overlong_strings_are_capped_at_500_chars(self):
        """Fail direction pinned: every chain string is LLM prose that re-enters
        a later prompt; uncapped, a hostile 50KB 'description' becomes a prompt
        stuffing primitive. The cap must TRIM (chain kept), not reject -- an
        over-long description losing its tail is the chosen fail direction."""
        out = self._sanitize(
            {
                "description": "D" * 600,
                "links": [
                    {
                        "source_component": "S" * 600,
                        "sink_component": "K" * 600,
                        "claim": "C" * 600,
                    }
                ],
            }
        )
        self.assertEqual(len(out["description"]), planner._MAX_CHAIN_CHARS)
        link = out["links"][0]
        self.assertEqual(len(link["source_component"]), planner._MAX_CHAIN_CHARS)
        self.assertEqual(len(link["sink_component"]), planner._MAX_CHAIN_CHARS)
        self.assertEqual(len(link["claim"]), planner._MAX_CHAIN_CHARS)

    def test_link_list_is_capped_at_max_chain_links(self):
        """Fail direction pinned: without the link cap a model can emit an
        unbounded hop list that the render layer will dutifully replay into
        every later prompt. Excess hops are dropped from the tail, chain kept."""
        out = self._sanitize(
            {
                "description": "many hops",
                "links": [
                    {"source_component": f"s{i}", "sink_component": f"k{i}", "claim": "c"}
                    for i in range(planner._MAX_CHAIN_LINKS + 2)
                ],
            }
        )
        self.assertEqual(len(out["links"]), planner._MAX_CHAIN_LINKS)

    def test_unsanitizable_input_returns_none_never_partial(self):
        """Fail direction pinned: None-never-partial is the contract the gate is
        written against; a partial dict here would flow into a group as a
        half-trusted chain. Non-dict input and a hollow chain (no description,
        no links) must both come back as None."""
        self.assertIsNone(self._sanitize("just a string"))
        self.assertIsNone(self._sanitize(["a", "list"]))
        self.assertIsNone(self._sanitize(None))
        self.assertIsNone(self._sanitize({"description": "", "links": []}))
        self.assertIsNone(self._sanitize({"description": "   ", "links": ["junk"]}))


class TestReplanFailSafes(unittest.IsolatedAsyncioTestCase):
    """replan_campaigns: any failure keeps the caller's queue (available False),
    zero surviving campaigns is a floor not an instruction, auth/budget re-raise."""

    async def asyncSetUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="mantis_test_replan_")
        self.addCleanup(tmp.cleanup)
        outside = tempfile.TemporaryDirectory(prefix="mantis_test_replan_outside_")
        self.addCleanup(outside.cleanup)
        self.root = Path(tmp.name).resolve()
        self.outside = Path(outside.name).resolve()
        (self.root / "a.py").write_text("a = 1\n", encoding="utf-8")
        self.outside_file = self.outside / "escape.py"
        self.outside_file.write_text("x = 1\n", encoding="utf-8")

    async def _replan(self, llm, **overrides):
        kwargs = dict(
            scan_root=str(self.root),
            completed=[{"target": str(self.root / "a.py"), "route": "confirm",
                        "findings_by_status": {"confirmed": 1}}],
            remaining_groups=[{"targets": [str(self.root / "a.py")],
                               "mode": "cross-functional", "hypothesis": "h"}],
            token_budget=0,
            spent_tokens=0,
        )
        kwargs.update(overrides)
        return await planner.replan_campaigns(llm, **kwargs)

    async def test_runtime_error_returns_unavailable_so_caller_keeps_queue(self):
        """Fail direction pinned: an ordinary model failure mid-run must cost the
        REPLAN, never the run -- available False tells the caller 'keep the
        queue you already had'. If this starts raising, one flaky 503 aborts a
        half-finished sweep."""
        result = await self._replan(_RaisingLlm(RuntimeError("503 mid-run melt")))
        self.assertFalse(result["available"])
        self.assertEqual(result["groups"], [])
        self.assertEqual(result["targets"], [])

    async def test_all_targets_outside_root_hits_never_cancel_floor(self):
        """Fail direction pinned: a replan whose every proposed path escapes the
        scan root gates to zero campaigns, and ZERO surviving campaigns must
        mean 'replan not applied', never 'cancel remaining work'. If this
        returns available True with empty groups, a hostile dossier (or a
        hallucinated root) silently cancels the rest of the run -- the exact
        outcome the floor exists to make unreachable."""
        plan_json = json.dumps(
            {
                "campaigns": [
                    {
                        "kind": "coverage",
                        "mode": "cross-functional",
                        "target_paths": [str(self.outside_file)],
                        "hypothesis": "escape attempt",
                    }
                ]
            }
        )
        result = await self._replan(_ScriptedLlm(plan_json))
        self.assertFalse(result["available"])
        self.assertEqual(result["groups"], [])

    async def test_zero_campaign_plan_hits_never_cancel_floor(self):
        """Fail direction pinned: an empty campaigns list is a valid REFUSAL of
        the initial plan but must never be a valid CANCELLATION of a replan --
        'return the full revised plan, omission means removal' makes an empty
        plan indistinguishable from 'drop everything', and no model output is
        allowed to mean that."""
        result = await self._replan(
            _ScriptedLlm('{"campaigns": [], "rationale": "drop everything"}')
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["groups"], [])

    async def test_auth_and_budget_errors_reraise(self):
        """Fail direction pinned: auth failure and budget exhaustion are facts
        about the RUN; swallowing them into available False would turn 'out of
        budget' into a silent kept-queue that then spends MORE budget.
        MantisAuthError (isinstance) and BudgetExceededError must re-raise."""
        from core.config import MantisAuthError

        with self.assertRaises(MantisAuthError):
            await self._replan(_RaisingLlm(MantisAuthError("401 Unauthorized")))

        boom = BudgetExceededError(
            trigger="token_budget",
            current_value=10_000_000,
            limit_value=10_000_000,
            run_id="r_replan",
        )
        with self.assertRaises(BudgetExceededError):
            await self._replan(_RaisingLlm(boom))

    # REGRESSION PIN (bug found by this test, then fixed): planner.py's INNER
    # stream handlers re-raise name-matched auth errors via is_auth_error, but
    # that raise unwinds into the OUTER `except Exception` handlers -- which
    # originally only re-raised isinstance(exc, (MantisAuthError,
    # BudgetExceededError)) and never consulted is_auth_error. A google-auth
    # shaped exception (RefreshError, DefaultCredentialsError,
    # AuthenticationError, ...) was therefore swallowed into
    # {'available': False} -- the silent kept-queue-then-spend-more outcome the
    # inner handler's own comment forbids. Both outer handlers now call
    # is_auth_error(exc) too; this test keeps them honest.
    async def test_name_matched_auth_error_reraises_through_outer_handler(self):
        """Fail direction pinned: is_auth_error's NAME matching exists so that
        google-auth exceptions the harness does not import still count as auth
        failures; if only isinstance(MantisAuthError) survives the outer
        handler, a RefreshError mid-replan silently keeps the queue and the
        run continues unauthenticated."""

        class AuthenticationError(Exception):
            """Name-matched by core.config.is_auth_error's class-name list."""

        with self.assertRaises(AuthenticationError):
            await self._replan(_RaisingLlm(AuthenticationError("token refresh died")))

    async def test_affordability_is_priced_against_remaining_not_original_budget(self):
        """Fail direction pinned: a replan priced against the ORIGINAL budget
        plans money that is already spent -- the gate must receive
        max(0, token_budget - spent_tokens). Verified by capturing the
        max_tokens argument the gate hands to cost.estimate_scan; the happy
        path must also come back available with the surviving group."""
        plan_json = json.dumps(
            {
                "campaigns": [
                    {
                        "kind": "coverage",
                        "mode": "cross-functional",
                        "target_paths": [str(self.root / "a.py")],
                        "hypothesis": "still worth it",
                    }
                ]
            }
        )
        captured = {}

        def fake_estimate(targets, max_tokens, db_path="", scan_mode=""):
            captured["max_tokens"] = max_tokens
            return _pytypes.SimpleNamespace(
                cost_per_campaign=50_000,
                affordable_campaigns=100,
                basis="observed",
                observations=3,
            )

        with patch("core.cost.estimate_scan", side_effect=fake_estimate):
            result = await self._replan(
                _ScriptedLlm(plan_json), token_budget=1_000_000, spent_tokens=600_000
            )
        self.assertEqual(captured["max_tokens"], 400_000)
        self.assertTrue(result["available"])
        self.assertEqual(result["groups"][0]["targets"], [str(self.root / "a.py")])


class TestSteeringTrustSplit(unittest.IsolatedAsyncioTestCase):
    """Operator steering: focus is TRUSTED instruction (unfenced, control-stripped,
    capped 2000); seed is UNTRUSTED evidence (capped 8000, CP-4 fenced at render);
    empty steering leaves the prompts untouched."""

    async def asyncSetUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="mantis_test_steer_")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        (self.root / "a.py").write_text("a = 1\n", encoding="utf-8")

    def test_sanitize_focus_strips_controls_caps_2000_and_renders_unfenced(self):
        """Fail direction pinned in both directions: fencing the focus would tag
        the operator's own ask as evidence-that-may-not-instruct (a neutered
        focus the operator believes took effect), while skipping the control
        strip lets a copy-pasted escape sequence ride an operator variable into
        the prompt. Cap 2000, control chars gone, NO untrusted markers."""
        hostile = "hunt \x1b[31mIDOR\x07 in the api layer " + "F" * 3000
        cleaned = planner._sanitize_focus(hostile)
        self.assertLessEqual(len(cleaned), planner._MAX_FOCUS_CHARS)
        self.assertIsNone(_CONTROL_CHARS.search(cleaned))
        self.assertIn("hunt", cleaned)
        self.assertIn("IDOR", cleaned)

        rendered = planner._render_focus_section(cleaned)
        self.assertIn("OPERATOR FOCUS", rendered)
        self.assertIn("IDOR", rendered)
        self.assertNotIn(UNTRUSTED_DATA_START, rendered)
        self.assertNotIn(UNTRUSTED_DATA_END, rendered)

    def test_sanitize_seed_caps_8000_and_rendered_section_is_fenced(self):
        """Fail direction pinned: the seed is bug-report prose -- a classic
        prompt-injection carrier -- and the operator supplying the FILE does
        not make its BYTES operator-authored. The rendered section must carry
        the seed text INSIDE the CP-4 fence, with the ask (hunt variants)
        outside it; an unfenced seed is an instruction channel for whoever
        filed the report."""
        seed = "SEED_BODY_MARKER: overflow in tar header parse " + "S" * 9000
        cleaned = planner._sanitize_seed(seed)
        self.assertLessEqual(len(cleaned), planner._MAX_SEED_CHARS)
        self.assertIn("SEED_BODY_MARKER", cleaned)

        rendered = planner._render_seed_section(cleaned)
        self.assertIn("SEED REPORT", rendered)
        start = rendered.index(UNTRUSTED_DATA_START)
        end = rendered.index(UNTRUSTED_DATA_END)
        self.assertIn("SEED_BODY_MARKER", rendered[start:end])
        # The operator-authored ask lives OUTSIDE the fence.
        self.assertIn("SAME PATTERN", rendered[end:])

    def test_empty_steering_renders_no_sections(self):
        """Fail direction pinned: empty steering must be byte-identical to the
        pre-steering prompt -- a phantom OPERATOR FOCUS or SEED REPORT header
        over no content invites the model to hallucinate a directive that was
        never given."""
        for hollow in ("", None, "   ", "\x1b[2J\x07"):
            self.assertEqual(planner._render_focus_section(planner._sanitize_focus(hollow)), "")
            self.assertEqual(planner._render_seed_section(planner._sanitize_seed(hollow)), "")

    async def test_afford_line_never_states_zero_campaigns(self):
        """Fail direction pinned by the first steered live run: with a ledger
        average (~1.45M/campaign) above the remaining budget, the prompt said
        'affords roughly 0 campaign(s)... propose at most that many' -- and the
        model complied with an empty plan, tripping the zero-campaign floor and
        silently degrading a FUNDED, operator-steered run to the surveyor. The
        prompt must state the same max(1, ...) floor the gate enforces: one
        campaign is always affordable to ATTEMPT, the budget controller owns
        the real ceiling."""
        from core.cost import record_spend

        db = str(self.root / "afford.db")
        self.assertTrue(
            record_spend(
                db,
                run_id="r_expensive",
                target=str(self.root / "a.py"),
                scan_mode="cross-functional",
                tokens=500_000,
                llm_calls=100,
                graph_steps=10,
            )
        )
        llm = _RecordingLlm('{"campaigns": []}')
        await planner.propose_campaigns(
            db_path=db,
            target=str(self.root),
            surveyor_targets=[str(self.root / "a.py")],
            llm_model=llm,
            token_budget=100_000,  # far below the 500k observed average
            scan_mode="cross-functional",
            focus="look for IDOR",  # steering gets a first run past the history gate
        )
        prompt = llm.prompt()
        self.assertIn("affords roughly 1 campaign(s)", prompt)
        self.assertNotIn("affords roughly 0", prompt)

    async def test_replan_prompt_places_focus_unfenced_and_seed_fenced(self):
        """Fail direction pinned: the prompt assembly order IS the trust story --
        operator instruction travels unfenced beside the charter, evidence
        travels fenced. If the focus marker ends up inside a fence it is
        demoted to inert data; if the seed marker escapes its fence it is
        promoted to instruction. Both directions are pinned on the REAL prompt
        a recording stub captured."""
        llm = _RecordingLlm('{"campaigns": []}')
        await planner.replan_campaigns(
            llm,
            scan_root=str(self.root),
            completed=[],
            remaining_groups=[],
            token_budget=0,
            spent_tokens=0,
            focus="FOCUS_MARKER_XYZ",
            seed_report="SEED_MARKER_ABC",
        )
        prompt = llm.prompt()
        self.assertIn("OPERATOR FOCUS", prompt)
        self.assertIn("SEED REPORT", prompt)
        first_fence = prompt.index(UNTRUSTED_DATA_START)
        fence_end = prompt.index(UNTRUSTED_DATA_END)
        # Focus: present, and strictly BEFORE any fenced region opens.
        self.assertIn("FOCUS_MARKER_XYZ", prompt)
        self.assertLess(prompt.index("FOCUS_MARKER_XYZ"), first_fence)
        # Seed: present, and strictly INSIDE the fence.
        self.assertIn("SEED_MARKER_ABC", prompt[first_fence:fence_end])

    async def test_unsteered_replan_prompt_contains_no_steering_sections(self):
        """Fail direction pinned: with no focus, no seed, no completed rows, no
        remaining groups and no db, the replan brief has nothing untrusted to
        carry -- the prompt must contain neither steering header and no
        untrusted fence at all. A fence around nothing teaches the model that
        fences are furniture."""
        llm = _RecordingLlm('{"campaigns": []}')
        await planner.replan_campaigns(
            llm,
            scan_root=str(self.root),
            completed=[],
            remaining_groups=[],
            token_budget=0,
            spent_tokens=0,
        )
        prompt = llm.prompt()
        self.assertNotIn("OPERATOR FOCUS", prompt)
        self.assertNotIn("SEED REPORT", prompt)
        self.assertNotIn(UNTRUSTED_DATA_START, prompt)

    async def test_focus_alone_enables_planning_without_history(self):
        """Fail direction pinned: a run steered at 'find IDOR' on a fresh
        knowledge base is a legitimate ask -- the no-history early-returns
        must YIELD when steering is present, and must still hold when it is
        not (unsteered no-history planning would dress a guess up as a plan
        at LLM prices)."""
        plan_json = json.dumps(
            {
                "campaigns": [
                    {
                        "kind": "coverage",
                        "mode": "cross-functional",
                        "target_paths": [str(self.root / "a.py")],
                        "hypothesis": "steered from nothing",
                    }
                ]
            }
        )
        # A db path that exists nowhere: an empty string would fall back to the
        # INSTALLATION's default knowledge base, which on a developer machine
        # can hold real history and silently turn the control arm steered.
        no_history_db = str(self.root / "never_created" / "knowledge.db")

        # Control: unsteered + no history -> the model is never consulted.
        control = _RecordingLlm(plan_json)
        result = await planner.propose_campaigns(
            db_path=no_history_db,
            target=str(self.root),
            surveyor_targets=[str(self.root / "a.py")],
            llm_model=control,
        )
        self.assertFalse(result["available"])
        self.assertEqual(control.requests, [])

        # Steered: the focus is the thing to plan from; history is optional.
        steered = _RecordingLlm(plan_json)
        result = await planner.propose_campaigns(
            db_path=no_history_db,
            target=str(self.root),
            surveyor_targets=[str(self.root / "a.py")],
            llm_model=steered,
            focus="find IDOR in the api layer",
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["targets"], [str(self.root / "a.py")])
        self.assertEqual(len(steered.requests), 1)
        self.assertIn("OPERATOR FOCUS", steered.prompt())

    def test_fence_failure_omits_seed_section_entirely(self):
        """Fail direction pinned: when CP-4 fencing itself fails, the seed
        section is OMITTED -- seed bytes must never travel unfenced. The
        fallback of embedding the raw seed text would hand a bug report (a
        classic prompt-injection carrier) an unfenced channel into the
        planner prompt precisely when the defense mechanism is broken."""
        seed = "hostile seed body that must not appear unfenced"
        with patch(
            "core.llm_gateway.wrap_untrusted_content",
            side_effect=RuntimeError("fence machinery down"),
        ):
            rendered = planner._render_seed_section(seed)
        self.assertEqual(rendered, "")


class TestWideningUnexpressible(unittest.TestCase):
    """The plan schemas cannot express widening: no field for a tool, a sandbox
    tier, a trust level, or a verdict exists on any of the four classes."""

    def _all_schema_classes(self):
        import typing

        plan_cls, proposal_cls = planner._plan_schemas()
        chain_cls = typing.get_args(
            proposal_cls.model_fields["chain_hypothesis"].annotation
        )[0]
        link_cls = typing.get_args(chain_cls.model_fields["links"].annotation)[0]
        return plan_cls, proposal_cls, chain_cls, link_cls

    def test_hostile_widening_kwargs_are_dropped_not_stored(self):
        """Fail direction pinned: a plan may direct attention but never widen
        tools, sandbox, or trust. extra='ignore' must silently DROP unknown
        kwargs -- the instances must not carry the attributes, because a
        widening field that survives construction is a widening field some
        downstream consumer will eventually read."""
        _plan_cls, proposal_cls, chain_cls, link_cls = self._all_schema_classes()
        widening = {"tools": ["bash"], "sandbox": "host", "trust": "code"}

        link = link_cls(
            source_component="src", sink_component="snk", claim="c", **widening
        )
        chain = chain_cls(description="d", links=[link], **widening)
        proposal = proposal_cls(
            kind="coverage",
            mode="whole",
            target_paths=["x.py"],
            hypothesis="h",
            chain_hypothesis=chain,
            **widening,
        )
        for instance in (link, chain, proposal):
            for attr in ("tools", "sandbox", "trust"):
                self.assertFalse(
                    hasattr(instance, attr),
                    f"{type(instance).__name__} stored widening attribute {attr!r}",
                )
                self.assertNotIn(attr, instance.model_dump())

    def test_no_schema_declares_a_widening_or_verdict_field(self):
        """Fail direction pinned: the schema IS the first deterministic gate --
        'a plan cannot widen anything because the shape that would express
        widening does not exist to be parsed'. The day someone adds a
        tool/sandbox/trust/status field to any of the four classes, this test
        is the tripwire that fires before the tier system has to."""
        from core.evidence import VERDICT_FIELDS

        forbidden = {
            "tool", "tools", "sandbox", "sandbox_tier", "sandbox_type",
            "trust", "trust_tier", "trust_level",
        } | set(VERDICT_FIELDS)
        for cls in self._all_schema_classes():
            declared = set(cls.model_fields.keys())
            self.assertEqual(
                declared & forbidden,
                set(),
                f"{cls.__name__} declares widening/verdict field(s): "
                f"{declared & forbidden}",
            )


class TestSummarizeReplan(unittest.TestCase):
    """summarize_replan: every number stated, none judged, no LLM-authored bytes."""

    def test_unapplied_replan_still_earns_an_explicit_line(self):
        """Fail direction pinned: 'the queue did not change' is a fact the
        operator deserves stated, not inferred from silence -- an unavailable
        replan must render the not-applied line, never an empty string."""
        line = planner.summarize_replan({"available": False}, 0, 0, 0)
        self.assertEqual(
            line, "LLM replan: not applied; remaining queue kept unchanged."
        )
        self.assertEqual(
            planner.summarize_replan(None, 0, 0, 0),
            "LLM replan: not applied; remaining queue kept unchanged.",
        )

    def test_applied_replan_states_kept_added_dropped_and_gate_counts(self):
        """Fail direction pinned: the kept/added/dropped diff belongs to the
        CALLER (the only party that knows what it swapped); this line must
        relay those numbers verbatim plus the gate's rejected/trimmed counts,
        so a model that tried to leave the root is visible every time."""
        plan = {
            "available": True,
            "counts": {"rejected_paths": 2, "trimmed_by_budget": 1},
        }
        line = planner.summarize_replan(plan, kept=3, added=2, dropped=1)
        self.assertIn("3 group(s) kept", line)
        self.assertIn("2 added", line)
        self.assertIn("1 dropped", line)
        self.assertIn("2 proposed path(s) rejected at validation", line)
        self.assertIn("1 trimmed by remaining budget", line)


class TestCampaignBudgetScope(unittest.TestCase):
    """Parallel campaign execution: shared run ceilings, isolated campaign guards.

    With --parallel N the scan loop hands each campaign a CampaignBudgetScope
    over one shared BudgetController. Two things must both hold or parallel
    mode silently corrupts the budget model: the RUN ceilings (tokens, wall
    clock) must fire on what all campaigns spent TOGETHER, and the CAMPAIGN
    guards (runaway-loop steps/visits/tool calls) must count each campaign
    alone -- a sibling's begin_campaign() or spend must never reset or inflate
    another campaign's guard.
    """

    def _controller(self, **cfg) -> BudgetController:
        base = dict(max_tokens=0, max_wall_clock_seconds=0, max_graph_steps=0)
        base.update(cfg)
        return BudgetController(config=BudgetConfig(**base), run_id="run-par")

    def test_run_tokens_mirror_into_the_shared_parent(self):
        from core.budget import CampaignBudgetScope

        parent = self._controller()
        a = CampaignBudgetScope(parent)
        b = CampaignBudgetScope(parent)
        a.record_tokens(1_000)
        b.record_tokens(2_000)
        self.assertEqual(parent.accumulated_tokens, 3_000)
        self.assertEqual(parent.llm_calls, 2)
        # Attribution stays exact per campaign: siblings never move each
        # other's counters, so the ledger's before/after delta is the
        # campaign's own cost, not a share of the sweep's.
        self.assertEqual(a.accumulated_tokens, 1_000)
        self.assertEqual(b.accumulated_tokens, 2_000)

    def test_the_token_ceiling_fires_on_combined_spend(self):
        from core.budget import CampaignBudgetScope

        parent = self._controller(max_tokens=3_000)
        a = CampaignBudgetScope(parent)
        b = CampaignBudgetScope(parent)
        a.record_tokens(2_000)  # under the ceiling alone
        with self.assertRaises(BudgetExceededError):
            b.record_tokens(2_000)  # 4,000 combined: over
        self.assertTrue(parent.is_paused)

    def test_campaign_guards_are_isolated_between_scopes(self):
        from core.budget import CampaignBudgetScope

        parent = self._controller(max_graph_steps=3)
        a = CampaignBudgetScope(parent)
        b = CampaignBudgetScope(parent)
        a.record_step()
        a.record_step()
        # A sibling starting its campaign must not reset A's guard...
        b.begin_campaign()
        b.record_step()
        self.assertEqual(a.campaign_graph_steps, 2)
        self.assertEqual(b.campaign_graph_steps, 1)
        # ...and B's steps must not push A over its own ceiling: only A's
        # third step trips A.
        with self.assertRaises(BudgetExceededError):
            a.record_step()

    def test_run_steps_accumulate_in_the_parent_for_the_ledger(self):
        from core.budget import CampaignBudgetScope

        parent = self._controller()
        a = CampaignBudgetScope(parent)
        b = CampaignBudgetScope(parent)
        a.record_step()
        b.record_step()
        b.record_step()
        self.assertEqual(parent.graph_steps, 3)

    def test_the_pause_banner_reports_the_run_not_the_campaign(self):
        from core.budget import CampaignBudgetScope

        parent = self._controller()
        parent.accumulated_tokens = 123_456
        scope = CampaignBudgetScope(parent)
        banner = scope.format_pause_banner(trigger="token_budget")
        self.assertIn("123,456", banner)

    def test_pipeline_parallel_defaults_to_sequential(self):
        """INV-6: absence of --parallel is the pre-parallel behaviour."""
        import inspect

        import main

        sig = inspect.signature(main.pipeline)
        self.assertIn("parallel", sig.parameters)
        self.assertEqual(sig.parameters["parallel"].default, 1)


if __name__ == "__main__":
    unittest.main()
