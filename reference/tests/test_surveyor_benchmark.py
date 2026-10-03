"""Tests for the surveyor ranking benchmark (evals/surveyor_benchmark.py).

Two layers:

1. Scoring math (recall@k, nDCG@k, MRR, coverage ceiling) checked against
   hand-computed values.
2. A hermetic end-to-end run: the synthetic webapp is built in a tempdir and
   surveyed with the real core.surveyor.survey. Asserts that seeded groups
   outrank the decoys, every truth file lands in a slice, and the survey is
   deterministic.

No LLM and no network. Churn is disabled, so git is not required either;
the one git-dependent test skips when git is missing.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REF_ROOT = str(Path(__file__).resolve().parent.parent)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)

from evals.surveyor_benchmark import (
    extract_upstream_pin,
    format_report,
    resolve_target_commit,
    run_benchmark,
    score_ranking,
    validate_ground_truth,
)
from evals.synthetic_repo import build_synthetic_webapp


def _astm(roots):
    """Minimal ASTM shape: one slice per root, in the given order."""
    return {
        "slices": [
            {"priority": i + 1, "root_paths": [root]} for i, root in enumerate(roots)
        ]
    }


def _gt(files):
    return [{"id": f"gt-{i}", "file": f, "class": ""} for i, f in enumerate(files)]


class ScoreRankingTest(unittest.TestCase):
    def test_hand_computed_metrics(self):
        result = score_ranking(
            _astm(["a", "b", "c"]),
            _gt(["a/x.py", "c/y.py", "d/z.py"]),
            ks=(1, 3),
        )
        self.assertEqual(result["ground_truth_files"], 3)
        self.assertAlmostEqual(result["recall"]["recall@1"], 1 / 3, places=4)
        self.assertAlmostEqual(result["recall"]["recall@3"], 2 / 3, places=4)
        # DCG@3 = 1/log2(2) + 1/log2(4) = 1.5.
        # IDCG@3 includes the missed file: 1 + 1/log2(3) + 1/log2(4) = 2.1309.
        self.assertAlmostEqual(result["ndcg"]["ndcg@3"], 0.7039, places=4)
        self.assertEqual(result["mrr"], 1.0)
        self.assertAlmostEqual(result["coverage_ceiling"], 2 / 3, places=4)
        ranks = {e["file"]: e["rank"] for e in result["per_file"]}
        self.assertEqual(ranks, {"a/x.py": 1, "c/y.py": 3, "d/z.py": None})

    def test_dropping_files_lowers_ndcg(self):
        # One truth file in slice #1 and two never sliced must not score 1.0.
        result = score_ranking(
            _astm(["a"]), _gt(["a/x.py", "b/y.py", "c/z.py"]), ks=(3,)
        )
        # DCG@3 = 1; IDCG@3 = 1 + 1/log2(3) + 1/log2(4) = 2.1309.
        self.assertAlmostEqual(result["ndcg"]["ndcg@3"], 0.4693, places=4)

    def test_missed_files_in_one_directory_form_one_ideal_slice(self):
        # Both missed files live in b/, so the ideal ranking covers them with
        # a single slice of gain 2 at rank 1: harsher than scattering them.
        result = score_ranking(
            _astm(["a"]), _gt(["a/x.py", "b/y.py", "b/z.py"]), ks=(3,)
        )
        # DCG@3 = 1; IDCG@3 = 2 + 1/log2(3) = 2.6309.
        self.assertAlmostEqual(result["ndcg"]["ndcg@3"], 0.3801, places=4)

    def test_first_containing_slice_wins_attribution(self):
        # Root "." contains everything; a file must be attributed to the
        # earliest containing slice exactly once, never double counted.
        result = score_ranking(_astm(["lib", "."]), _gt(["lib/a.py", "app/b.py"]), ks=(1, 2))
        ranks = {e["file"]: e["rank"] for e in result["per_file"]}
        self.assertEqual(ranks, {"lib/a.py": 1, "app/b.py": 2})
        self.assertEqual(result["recall"]["recall@2"], 1.0)
        # Perfectly covered at the earliest possible positions => ideal order.
        self.assertEqual(result["ndcg"]["ndcg@2"], 1.0)

    def test_empty_inputs_do_not_divide_by_zero(self):
        result = score_ranking(_astm([]), _gt([]), ks=(1,))
        self.assertEqual(result["recall"]["recall@1"], 0.0)
        self.assertEqual(result["ndcg"]["ndcg@1"], 0.0)
        self.assertEqual(result["mrr"], 0.0)
        self.assertEqual(result["coverage_ceiling"], 0.0)

    def test_prefix_containment_is_component_aligned(self):
        # 'server/routes' must not contain 'server/routes2/x.js'.
        result = score_ranking(
            _astm(["server/routes"]), _gt(["server/routes2/x.js"]), ks=(1,)
        )
        self.assertEqual(result["coverage_ceiling"], 0.0)


class GroundTruthValidationTest(unittest.TestCase):
    def test_rejects_shapes_that_would_score_zero_silently(self):
        for bad in (
            [],
            {},
            {"vulnerable_files": []},
            {"vulnerable_files": [{"id": "x"}]},
            {"vulnerable_files": [{"file": "a.py"}, {"file": "a.py"}]},
        ):
            with self.assertRaises(ValueError, msg=repr(bad)):
                validate_ground_truth(bad)

    def test_normalizes_paths(self):
        entries = validate_ground_truth(
            {"vulnerable_files": [{"id": "x", "file": "/routes\\login.ts/"}]}
        )
        self.assertEqual(entries[0]["file"], "routes/login.ts")


class WrongTargetTest(unittest.TestCase):
    def test_all_ground_truth_missing_raises(self):
        # Zero matches means the wrong directory, not a 0.0 score.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "unrelated.py").write_text("x = 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                run_benchmark(
                    Path(tmp), _gt(["nope/missing.js"]), include_churn=False
                )


class SyntheticEndToEndTest(unittest.TestCase):
    """The hermetic ranking pin: real survey, generated target, known seeds."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="mantis_bench_test_")
        cls.target = Path(cls._tmp.name).resolve()
        cls.gt = validate_ground_truth(build_synthetic_webapp(cls.target))
        cls.result = run_benchmark(cls.target, cls.gt, include_churn=False)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_truth_file_is_sliced(self):
        self.assertEqual(self.result["coverage_ceiling"], 1.0, self.result["per_file"])
        self.assertEqual(self.result["missing_files"], [])

    def test_truth_groups_fill_the_top_ranks(self):
        # 5 seeded files across 3 groups; all of them must land in the top 3
        # slices, and the #1 slice must contain truth.
        self.assertEqual(
            self.result["recall"]["recall@3"], 1.0, self.result["slice_order"]
        )
        self.assertEqual(self.result["mrr"], 1.0)

    def test_demoted_decoys_rank_below_every_truth_group(self):
        # tests/ and vendor/ carry the SAME sink patterns as the seeded files;
        # only demotion separates them. This is the regression the benchmark
        # exists to catch.
        order = [entry["root"] for entry in self.result["slice_order"]]
        truth_groups = {"server/routes", "server/lib", "native/parser"}
        decoys = {"tests/unit", "vendor/bundled"}
        worst_truth = max(order.index(g) for g in truth_groups if g in order)
        for decoy in decoys & set(order):
            self.assertGreater(order.index(decoy), worst_truth, order)

    def test_survey_is_deterministic(self):
        again = run_benchmark(self.target, self.gt, include_churn=False)
        self.assertEqual(
            [e["root"] for e in again["slice_order"]],
            [e["root"] for e in self.result["slice_order"]],
        )
        self.assertEqual(again["recall"], self.result["recall"])
        self.assertEqual(again["ndcg"], self.result["ndcg"])


class UpstreamPinTest(unittest.TestCase):
    """The upstream pin must label drift without ever blocking a run."""

    def test_absent_or_malformed_pin_is_none(self):
        self.assertIsNone(extract_upstream_pin({"vulnerable_files": []}))
        self.assertIsNone(extract_upstream_pin({"upstream": "v20.2.0"}))
        self.assertIsNone(extract_upstream_pin({"upstream": {"tag": "v1"}}))
        self.assertIsNone(extract_upstream_pin(["not", "a", "dict"]))

    def test_pin_normalizes_commit_case_and_whitespace(self):
        pin = extract_upstream_pin(
            {"upstream": {"repo": "r", "tag": "v1", "commit": " ABCdef0123 "}}
        )
        self.assertEqual(pin, {"repo": "r", "tag": "v1", "commit": "abcdef0123"})

    def test_non_git_target_resolves_none_not_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(resolve_target_commit(Path(tmp)))

    def test_git_target_resolves_head(self):
        git = shutil.which("git")
        if git is None:
            self.skipTest("git unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([git, "-C", tmp, "init", "-q"], check=True)
            (Path(tmp) / "f.txt").write_text("x", encoding="utf-8")
            subprocess.run([git, "-C", tmp, "add", "."], check=True)
            subprocess.run(
                [
                    git, "-C", tmp,
                    "-c", "user.email=t@test", "-c", "user.name=t",
                    "commit", "-qm", "c",
                ],
                check=True,
            )
            head = subprocess.run(
                [git, "-C", tmp, "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip().lower()
            self.assertEqual(resolve_target_commit(Path(tmp)), head)

    def test_report_renders_all_three_match_states(self):
        base = {"recall": {}, "ndcg": {}, "per_file": []}
        match = dict(
            base,
            upstream_pin={
                "pinned_commit": "a" * 40, "pinned_tag": "v20.2.0",
                "target_commit": "a" * 40, "commit_match": True,
            },
        )
        self.assertIn("matches v20.2.0", format_report(match))
        mismatch = dict(
            base,
            upstream_pin={
                "pinned_commit": "a" * 40, "pinned_tag": "v20.2.0",
                "target_commit": "b" * 40, "commit_match": False,
            },
        )
        self.assertIn("not comparable", format_report(mismatch))
        unknown = dict(
            base,
            upstream_pin={
                "pinned_commit": "a" * 40, "pinned_tag": None,
                "target_commit": None, "commit_match": None,
            },
        )
        self.assertIn("cannot verify", format_report(unknown))


if __name__ == "__main__":
    unittest.main()
