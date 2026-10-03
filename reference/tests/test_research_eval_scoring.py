"""Tests for the researcher-stage eval (evals/research_eval.py).

Covers the deterministic layer only: the finding matcher, the scorer, the
ground-truth contract against the committed corpus, the jail-copy exclusion,
and token extraction. No LLM calls.
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

_REF_ROOT = str(Path(__file__).resolve().parent.parent)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)

from evals.research_eval import (
    DEFAULT_TARGET,
    GROUND_TRUTH_BASENAME,
    TOOLSETS,
    _copy_target,
    extract_event_tokens,
    load_research_ground_truth,
    match_finding,
    score_research_findings,
)


def _entry(id_, files, phrases, cross=False):
    return {
        "id": id_,
        "files": files,
        "match_any": [p.lower() for p in phrases],
        "cross_file": cross,
        "class": "",
        "cwe": "",
    }


def _finding(filepath="", title="", description="", code_paths=None):
    return {
        "filepath": filepath,
        "title": title,
        "description": description,
        "code_paths": code_paths or [],
    }


class MatchFindingTest(unittest.TestCase):
    GT = _entry("sqli", ["database.py", "app.py"], ["sql injection", "cwe-89"])

    def test_path_suffix_variants_match(self):
        for path in ("database.py", "app/database.py", "./app/database.py"):
            self.assertTrue(
                match_finding(_finding(path, "SQL injection in find_user"), self.GT),
                path,
            )

    def test_basename_collision_does_not_match(self):
        self.assertFalse(
            match_finding(_finding("app/xdatabase.py", "SQL injection"), self.GT)
        )

    def test_file_match_without_phrase_is_rejected(self):
        # Right file, wrong mechanism: naming the file is not naming the bug.
        self.assertFalse(
            match_finding(_finding("app/database.py", "Debug mode enabled"), self.GT)
        )

    def test_phrase_may_live_in_description(self):
        self.assertTrue(
            match_finding(
                _finding("app/database.py", "Injection flaw", "classic CWE-89 shape"),
                self.GT,
            )
        )

    def test_structured_cwe_field_can_carry_the_phrase_match(self):
        finding = _finding("app/database.py", "Injection in find_user")
        finding["cwe"] = "CWE-89"
        self.assertTrue(match_finding(finding, self.GT))

    def test_code_paths_can_carry_the_file_match(self):
        finding = _finding(
            "", "SQL injection", code_paths=["app/database.py:28"]
        )
        self.assertTrue(match_finding(finding, self.GT))


class ScoreResearchFindingsTest(unittest.TestCase):
    def setUp(self):
        self.gt = [
            _entry("sqli", ["database.py", "app.py"], ["sql injection"], cross=True),
            _entry("md5", ["auth.py"], ["md5", "weak hash"], cross=False),
            _entry("ssrf", ["fetcher.py", "app.py"], ["ssrf"], cross=True),
        ]

    def test_mixed_outcome_arithmetic(self):
        findings = [
            _finding("app/database.py", "SQL injection in find_user"),
            _finding("app/auth.py", "Weak password hashing", "unsalted MD5 digest"),
            _finding("database.py", "sql injection via name parameter"),  # duplicate
            _finding("app/helpers.py", "Hypothetical issue"),  # false positive
        ]
        result = score_research_findings(findings, self.gt)
        self.assertEqual(result["detected"], 2)
        self.assertAlmostEqual(result["recall"], 2 / 3, places=4)
        self.assertEqual(result["cross_file_total"], 2)
        self.assertEqual(result["cross_file_detected"], 1)
        self.assertAlmostEqual(result["cross_file_recall"], 0.5, places=4)
        self.assertEqual(result["findings_reported"], 4)
        self.assertEqual(result["findings_true"], 3)
        # Unique results: 2 detected + 1 unmatched; the duplicate counts for
        # neither, so it cannot dilute the false positive.
        self.assertAlmostEqual(result["precision"], 2 / 3, places=4)
        self.assertEqual(result["duplicates"], 1)
        self.assertEqual(len(result["unmatched_findings"]), 1)
        by_id = {e["id"]: e for e in result["per_entry"]}
        self.assertTrue(by_id["sqli"]["detected"])
        self.assertFalse(by_id["ssrf"]["detected"])

    def test_no_findings_scores_zero_not_crash(self):
        result = score_research_findings([], self.gt)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["findings_true"], 0)


class GroundTruthContractTest(unittest.TestCase):
    """The committed corpus and its answer key must stay in lockstep."""

    @classmethod
    def setUpClass(cls):
        cls.entries = load_research_ground_truth(
            DEFAULT_TARGET / GROUND_TRUTH_BASENAME
        )

    def test_every_cited_file_exists_in_the_corpus(self):
        for entry in self.entries:
            for file in entry["files"]:
                self.assertTrue(
                    (DEFAULT_TARGET / file).is_file(), f"{entry['id']}: {file}"
                )

    def test_cross_file_entries_dominate(self):
        # Most entries must stay cross-file: cross-file reasoning is the
        # capability this corpus exists to measure.
        cross = [e for e in self.entries if e["cross_file"]]
        self.assertGreaterEqual(len(cross), 4, [e["id"] for e in self.entries])
        for entry in cross:
            self.assertGreaterEqual(
                len(entry["files"]), 2,
                f"{entry['id']}: cross_file entries must allow source or sink file",
            )

    def test_match_phrases_are_lowercase_nonempty(self):
        for entry in self.entries:
            self.assertTrue(entry["match_any"], entry["id"])
            for phrase in entry["match_any"]:
                self.assertEqual(phrase, phrase.lower(), entry["id"])


class CorpusLeakTest(unittest.TestCase):
    """Corpus files must read like production code, not an answer key."""

    FORBIDDEN = (
        "seeded",
        "eval target",
        "benign counterpart",
        "vulnerab",
        "attacker",
        "exploit",
        "injection",
        "traversal",
        "ssrf",
    )

    def test_corpus_files_do_not_leak_ground_truth(self):
        for path in sorted(DEFAULT_TARGET.glob("*.py")):
            text = path.read_text(encoding="utf-8").lower()
            for marker in self.FORBIDDEN:
                self.assertNotIn(marker, text, f"{path.name}: {marker!r}")


class JailCopyTest(unittest.TestCase):
    def test_ground_truth_is_excluded_from_the_jail(self):
        with tempfile.TemporaryDirectory(prefix="mantis_jail_test_") as tmp:
            app_dir = _copy_target(DEFAULT_TARGET, Path(tmp))
            copied = sorted(p.name for p in app_dir.iterdir())
            self.assertIn("app.py", copied)
            self.assertIn("database.py", copied)
            self.assertNotIn(GROUND_TRUTH_BASENAME, copied)


class ExtractEventTokensTest(unittest.TestCase):
    def test_total_token_count_preferred(self):
        event = types.SimpleNamespace(
            usage_metadata=types.SimpleNamespace(
                total_token_count=321, prompt_token_count=300, candidates_token_count=21
            )
        )
        self.assertEqual(extract_event_tokens(event), 321)

    def test_falls_back_to_prompt_plus_candidates(self):
        event = types.SimpleNamespace(
            usage_metadata=types.SimpleNamespace(
                total_token_count=None, prompt_token_count=10, candidates_token_count=5
            )
        )
        self.assertEqual(extract_event_tokens(event), 15)

    def test_absent_or_malformed_usage_counts_zero(self):
        self.assertEqual(extract_event_tokens(types.SimpleNamespace()), 0)
        self.assertEqual(
            extract_event_tokens(
                types.SimpleNamespace(
                    usage_metadata=types.SimpleNamespace(
                        total_token_count="not-a-number",
                        prompt_token_count=None,
                        candidates_token_count=None,
                    )
                )
            ),
            0,
        )


class ToolsetContractTest(unittest.TestCase):
    def test_structural_is_a_strict_superset_of_baseline(self):
        self.assertTrue(set(TOOLSETS["baseline"]) < set(TOOLSETS["structural"]))

    def test_resolve_toolset_against_live_registry(self):
        # Requires the production tool registry (and its ADK dependency); the
        # contract still holds without it, so skip rather than fail.
        try:
            from evals.research_eval import resolve_toolset
            resolved = resolve_toolset("baseline")
        except ImportError as exc:  # pragma: no cover - minimal environments
            self.skipTest(f"tool registry unavailable: {exc}")
        self.assertEqual(resolved, TOOLSETS["baseline"])
        with self.assertRaises(ValueError):
            resolve_toolset("nonexistent")


if __name__ == "__main__":
    unittest.main()
