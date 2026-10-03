"""Benchmark for the surveyor's file ranking. Deterministic, no LLM calls.

Surveys a repository and measures how early the ranked slices cover files
known to be vulnerable:

* recall@k: fraction of ground-truth files inside the top-k slices.
* ndcg@k: rank-discounted gain, normalized to the ideal ordering.
* mrr: 1/rank of the first slice containing a truth file.
* coverage_ceiling: fraction of truth files in any slice at all; separates
  "ranked late" from "never sliced".

Ground truth format:

    {
      "format_version": "1.0",
      "target": "name",
      "upstream": {"repo": "https://...", "tag": "v20.2.0", "commit": "<sha>"},
      "vulnerable_files": [
        {"id": "sqli-login", "file": "routes/login.ts", "class": "sql_injection"}
      ]
    }

Paths are repo-root-relative. Entries missing from the target are reported as
stale and excluded from scoring. The optional "upstream" block records the
commit the ground truth was curated against; a mismatched target still runs,
but the report and JSON flag it (upstream_pin.commit_match).

Usage:

    # Hermetic self-check against the generated synthetic webapp:
    python3 evals/surveyor_benchmark.py --synthetic

    # External target with a curated ground truth:
    python3 evals/surveyor_benchmark.py --target ~/src/juice-shop \\
        --ground-truth evals/ground_truth/juice_shop.json

    # Machine-readable output for A/B comparison:
    python3 evals/surveyor_benchmark.py --synthetic --json-out /tmp/baseline.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):  # pragma: no cover - direct CLI invocation
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_KS = (1, 3, 5, 10, 20)


# --- Ground truth -------------------------------------------------------------------


def load_ground_truth(path: Path) -> List[Dict[str, str]]:
    """Loads and validates a ground-truth file. Raises ValueError on bad shape.

    Validation is strict on purpose: a malformed ground truth would otherwise
    score 0.0 and look like a ranking regression.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_ground_truth(raw)


def validate_ground_truth(raw: Any) -> List[Dict[str, str]]:
    if not isinstance(raw, dict):
        raise ValueError("ground truth must be a JSON object")
    entries = raw.get("vulnerable_files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("ground truth must carry a non-empty 'vulnerable_files' list")
    seen: set = set()
    out: List[Dict[str, str]] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"vulnerable_files[{i}] is not an object")
        file = str(entry.get("file") or "").replace("\\", "/").strip("/")
        if not file:
            raise ValueError(f"vulnerable_files[{i}] is missing 'file'")
        if file in seen:
            raise ValueError(f"vulnerable_files[{i}] duplicates '{file}'")
        seen.add(file)
        out.append(
            {
                "id": str(entry.get("id") or f"gt-{i}"),
                "file": file,
                "class": str(entry.get("class") or ""),
            }
        )
    return out


def extract_upstream_pin(raw: Any) -> Optional[Dict[str, str]]:
    """Returns the optional upstream pin ({repo, tag, commit}) or None."""
    if not isinstance(raw, dict):
        return None
    pin = raw.get("upstream")
    if not isinstance(pin, dict):
        return None
    commit = str(pin.get("commit") or "").strip().lower()
    if not commit:
        return None
    return {
        "repo": str(pin.get("repo") or ""),
        "tag": str(pin.get("tag") or ""),
        "commit": commit,
    }


def resolve_target_commit(target: Path) -> Optional[str]:
    """Returns HEAD of the target checkout, or None if it has no usable git.

    Never raises: the benchmark must still run on plain directory snapshots.
    The .git check stops git from walking up to a parent repository when the
    target itself is not a checkout.
    """
    target = Path(target)
    if not (target / ".git").exists():
        return None
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(target),
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env-specific
        return None
    if proc.returncode != 0:
        return None
    head = proc.stdout.strip().lower()
    return head or None


# --- Scoring --------------------------------------------------------------------------


def _slice_contains(root: str, file: str) -> bool:
    """True if a repo-relative file falls under a slice root (directory prefix)."""
    root = str(root).replace("\\", "/").strip("/")
    if root in ("", "."):
        return True
    return file == root or file.startswith(root + "/")


def score_ranking(
    astm: Dict[str, Any],
    gt_entries: Sequence[Dict[str, str]],
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, Any]:
    """Scores one ranked map against one ground truth. Pure function, no I/O.

    Each file counts toward the first (best-ranked) slice that contains it,
    so nested slice roots are not double counted.
    """
    slices = [s for s in (astm.get("slices") or []) if isinstance(s, dict)]
    roots: List[str] = []
    for entry in slices:
        root = (entry.get("root_paths") or ["."])[0]
        roots.append(str(root))

    per_file: List[Dict[str, Any]] = []
    gains = [0] * len(roots)
    for gt in gt_entries:
        rank: Optional[int] = None
        for idx, root in enumerate(roots):
            if _slice_contains(root, gt["file"]):
                rank = idx + 1
                gains[idx] += 1
                break
        per_file.append({**gt, "rank": rank})

    found_ranks = sorted(e["rank"] for e in per_file if e["rank"] is not None)
    total = len(per_file)

    def _recall_at(k: int) -> float:
        if not total:
            return 0.0
        return sum(1 for r in found_ranks if r <= k) / total

    def _dcg(values: Sequence[int], k: int) -> float:
        return sum(v / math.log2(i + 2) for i, v in enumerate(values[:k]))

    ndcg = {}
    # The ideal ranking also has to place the files the survey never sliced;
    # otherwise dropping files would not lower ndcg at all. Missed files are
    # grouped the way the surveyor groups candidate slices, so the ideal is a
    # ranking the surveyor could actually have emitted.
    from core.surveyor import _group_key

    missed_groups: Dict[str, int] = {}
    for e in per_file:
        if e["rank"] is None:
            key = _group_key(e["file"])
            missed_groups[key] = missed_groups.get(key, 0) + 1
    ideal = sorted(gains + list(missed_groups.values()), reverse=True)
    for k in ks:
        idcg = _dcg(ideal, k)
        ndcg[f"ndcg@{k}"] = round(_dcg(gains, k) / idcg, 4) if idcg > 0 else 0.0

    return {
        "ground_truth_files": total,
        "slices_emitted": len(roots),
        "recall": {f"recall@{k}": round(_recall_at(k), 4) for k in ks},
        "ndcg": ndcg,
        "mrr": round(1.0 / found_ranks[0], 4) if found_ranks else 0.0,
        "coverage_ceiling": round(len(found_ranks) / total, 4) if total else 0.0,
        "per_file": per_file,
    }


# --- Running --------------------------------------------------------------------------


def run_benchmark(
    target: Path,
    gt_entries: Sequence[Dict[str, str]],
    max_slices: Optional[int] = None,
    include_churn: bool = True,
    ks: Sequence[int] = DEFAULT_KS,
    pin: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Runs the real core.surveyor.survey on target and scores the result."""
    from core import surveyor

    target = Path(target)
    present: List[Dict[str, str]] = []
    missing: List[str] = []
    for entry in gt_entries:
        if (target / entry["file"]).is_file():
            present.append(entry)
        else:
            missing.append(entry["file"])
    if gt_entries and not present:
        # Partial staleness is reported; a target with zero matches is almost
        # certainly the wrong directory, and 0.0 scores would hide that.
        raise ValueError(
            f"none of the {len(gt_entries)} ground-truth files exist under "
            f"{target}; wrong --target?"
        )

    astm = surveyor.survey(
        target,
        max_slices=max_slices or surveyor.MAX_SLICES,
        include_churn=include_churn,
    )
    result = score_ranking(astm, present, ks=ks)
    provenance = astm.get("provenance") or {}
    result.update(
        {
            "target": str(target),
            "snapshot_id": astm.get("snapshot_id"),
            "surveyor_version": astm.get("schema_version"),
            "missing_files": missing,
            "survey_provenance": {
                "effective_weights": provenance.get("effective_weights"),
                "inactive_signals": provenance.get("inactive_signals"),
                "attack_surface_weak": (provenance.get("coverage") or {}).get(
                    "attack_surface_weak"
                ),
                "elapsed_seconds": provenance.get("elapsed_seconds"),
            },
            "slice_order": [
                {
                    "priority": s.get("priority"),
                    "root": (s.get("root_paths") or ["?"])[0],
                    "risk_score": s.get("risk_score"),
                    "archetype": s.get("domain_archetype"),
                }
                for s in (astm.get("slices") or [])
            ],
        }
    )
    if pin:
        target_commit = resolve_target_commit(target)
        result["upstream_pin"] = {
            "pinned_commit": pin["commit"],
            "pinned_tag": pin.get("tag") or None,
            "target_commit": target_commit,
            # None when HEAD is unknown (target is not a git checkout).
            "commit_match": (
                target_commit == pin["commit"] if target_commit else None
            ),
        }
    return result


def format_report(result: Dict[str, Any]) -> str:
    lines = [
        f"Surveyor ranking benchmark -- {result.get('target')}",
        f"  snapshot: {result.get('snapshot_id')}  "
        f"slices: {result.get('slices_emitted')}  "
        f"truth files: {result.get('ground_truth_files')}",
        "  " + "  ".join(f"{k}={v}" for k, v in result.get("recall", {}).items()),
        "  " + "  ".join(f"{k}={v}" for k, v in result.get("ndcg", {}).items()),
        f"  mrr={result.get('mrr')}  coverage_ceiling={result.get('coverage_ceiling')}",
    ]
    pin = result.get("upstream_pin")
    if pin:
        tag = pin.get("pinned_tag") or "pinned commit"
        if pin.get("commit_match") is True:
            lines.append(f"  target matches {tag} @ {pin['pinned_commit'][:12]}")
        elif pin.get("commit_match") is False:
            lines.append(
                f"  WARNING: target HEAD {str(pin.get('target_commit'))[:12]} != "
                f"{tag} @ {pin['pinned_commit'][:12]} -- scores are not comparable "
                "to pinned-baseline runs"
            )
        else:
            lines.append(
                f"  NOTE: target is not a git checkout; cannot verify {tag} @ "
                f"{pin['pinned_commit'][:12]} -- treat comparisons with care"
            )
    if result.get("missing_files"):
        lines.append(
            "  STALE GROUND TRUTH (excluded from scoring): "
            + ", ".join(result["missing_files"])
        )
    lines.append("  placement:")
    for entry in result.get("per_file", []):
        rank = entry.get("rank")
        where = f"slice #{rank}" if rank else "NOT IN ANY SLICE"
        lines.append(f"    {entry['id']:<24} {entry['file']:<48} {where}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=Path, help="Repository root to survey.")
    parser.add_argument("--ground-truth", type=Path, help="Ground-truth JSON path.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Build the hermetic synthetic webapp in a tempdir and benchmark it.",
    )
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument(
        "--no-churn",
        action="store_true",
        help="Skip the git churn signal (faster; identical on non-git targets).",
    )
    parser.add_argument("--json-out", type=Path, help="Also write the result as JSON.")
    args = parser.parse_args(argv)

    if args.synthetic == bool(args.target or args.ground_truth):
        parser.error("use either --synthetic or both --target and --ground-truth")

    if args.synthetic:
        from evals.synthetic_repo import build_synthetic_webapp

        with tempfile.TemporaryDirectory(prefix="mantis_surveyor_bench_") as tmp:
            gt = validate_ground_truth(build_synthetic_webapp(Path(tmp)))
            result = run_benchmark(
                Path(tmp), gt, args.max_slices, include_churn=not args.no_churn
            )
    else:
        raw = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
        gt = validate_ground_truth(raw)
        result = run_benchmark(
            args.target,
            gt,
            args.max_slices,
            include_churn=not args.no_churn,
            pin=extract_upstream_pin(raw),
        )

    print(format_report(result))
    if args.json_out:
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nJSON written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
