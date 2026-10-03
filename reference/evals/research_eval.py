"""Researcher-stage eval: runs one researcher campaign over a seeded target.

Scores detection recall, cross-file recall, precision, and tokens per true
finding. The other stage benchmarks measure the stages that filter findings;
this one measures the stage that produces them. The toolset is the A/B
variable: same model, same instruction, same target, different tools.

How it works:

* The target (evals/research_target/) is copied into a disposable jail.
  ground_truth.json is excluded from the copy so the model cannot read it.
* The researcher runs with the production tool implementations and reports
  via report_findings; scoring reads the same database the pipeline uses.
* A finding matches a ground-truth entry only if it names one of the entry's
  files and mentions one of its match phrases. Duplicates of a matched entry
  do not hurt precision; deduplication is a different stage's job.
* cross_file_recall is reported separately because source-to-sink flows that
  cross files are what structural navigation tools should improve.

Usage:

    # Via the benchmark runner (model sweep, table output):
    python3 evals/run_eval.py --stage research --runs 3

    # Direct, single configuration, explicit toolset:
    python3 evals/research_eval.py --model vertex_ai/gemini-3.7-flash \\
        --effort low --toolset baseline --json-out /tmp/baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):  # pragma: no cover - direct CLI invocation
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

DEFAULT_TARGET = Path(__file__).resolve().parent / "research_target"
GROUND_TRUTH_BASENAME = "ground_truth.json"

RESEARCH_PROMPT = (
    "You are the researcher stage of a security review. Audit the application "
    "source under the directory 'app/' for real, exploitable vulnerabilities.\n"
    "1. Call list_files('app') to enumerate the source files.\n"
    "2. Read the files and trace data flow from untrusted inputs (HTTP request "
    "parameters) to dangerous operations, including flows that cross file "
    "boundaries.\n"
    "3. Report every distinct vulnerability you confirm via report_findings, "
    "with a precise title, the file where the flaw lives (filepath), affected "
    "line numbers, and a description naming the mechanism end to end.\n"
    "Report only real flaws; do not pad with style issues or hypotheticals."
)

# The A/B variable. 'structural' lists future tool names in advance: names
# not registered in tools.TOOLS are skipped with a warning, so the A/B runs
# today and picks the new tools up automatically once they land.
TOOLSETS: Dict[str, List[str]] = {
    "baseline": [
        "read_file",
        "write_file",
        "list_files",
        "get_summary",
        "get_threat_model",
        "get_plan",
        "report_findings",
        "get_findings",
    ],
    "structural": [
        "read_file",
        "write_file",
        "list_files",
        "get_summary",
        "get_threat_model",
        "get_plan",
        "report_findings",
        "get_findings",
        # Future structural tools; skipped until registered:
        "get_function_boundary",
        "find_symbol",
        "find_callers",
        "find_callees",
    ],
}


# --- Ground truth & scoring (pure, no LLM, unit-tested) -------------------------------


def load_research_ground_truth(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("findings") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("research ground truth must carry a non-empty 'findings' list")
    out: List[Dict[str, Any]] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"findings[{i}] is not an object")
        files = [str(f).replace("\\", "/").strip("/") for f in entry.get("files") or []]
        phrases = [str(p).lower() for p in entry.get("match_any") or [] if str(p).strip()]
        if not files or not phrases:
            raise ValueError(f"findings[{i}] needs non-empty 'files' and 'match_any'")
        out.append(
            {
                "id": str(entry.get("id") or f"gt-{i}"),
                "files": files,
                "match_any": phrases,
                "cross_file": bool(entry.get("cross_file")),
                "class": str(entry.get("class") or ""),
                "cwe": str(entry.get("cwe") or ""),
            }
        )
    return out


def _norm_path(path: Any) -> str:
    return str(path or "").replace("\\", "/").removeprefix("./").strip("/")


def _file_matches(finding_path: str, gt_file: str) -> bool:
    """Suffix match on path-component boundaries: 'app/database.py' matches
    'database.py'; 'db/database.py' matches 'database.py'; 'xdatabase.py'
    does not."""
    fp, gt = _norm_path(finding_path), _norm_path(gt_file)
    if not fp or not gt:
        return False
    return fp == gt or fp.endswith("/" + gt)


def match_finding(finding: Dict[str, Any], gt_entry: Dict[str, Any]) -> bool:
    """True when the finding names one of the entry's files and mentions one
    of its phrases. Phrases are checked over title, description, and cwe."""
    paths = [finding.get("filepath") or ""]
    paths.extend(str(cp).rsplit(":", 1)[0] for cp in finding.get("code_paths") or [])
    if not any(_file_matches(p, f) for p in paths for f in gt_entry["files"]):
        return False
    prose = " ".join(
        str(finding.get(key) or "") for key in ("title", "description", "cwe")
    ).lower()
    return any(phrase in prose for phrase in gt_entry["match_any"])


def score_research_findings(
    findings: Sequence[Dict[str, Any]],
    gt_entries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Scores reported findings against ground truth. Pure function.

    Returns recall, cross_file_recall, precision, and per-entry placement.
    Precision is computed over unique results -- detected entries vs unmatched
    reports -- so duplicate reports of a matched entry neither help nor hurt.
    """
    matched: Dict[str, List[int]] = {gt["id"]: [] for gt in gt_entries}
    unmatched_findings: List[Dict[str, Any]] = []
    for idx, finding in enumerate(findings):
        hits = [gt["id"] for gt in gt_entries if match_finding(finding, gt)]
        if hits:
            for gt_id in hits:
                matched[gt_id].append(idx)
        else:
            unmatched_findings.append(
                {
                    "index": idx,
                    "title": str(finding.get("title") or "")[:120],
                    "filepath": _norm_path(finding.get("filepath")),
                }
            )

    detected = [gt for gt in gt_entries if matched[gt["id"]]]
    cross = [gt for gt in gt_entries if gt["cross_file"]]
    cross_detected = [gt for gt in cross if matched[gt["id"]]]
    true_findings = len(findings) - len(unmatched_findings)
    unique_results = len(detected) + len(unmatched_findings)

    return {
        "ground_truth_total": len(gt_entries),
        "detected": len(detected),
        "recall": round(len(detected) / len(gt_entries), 4) if gt_entries else 0.0,
        "cross_file_total": len(cross),
        "cross_file_detected": len(cross_detected),
        "cross_file_recall": round(len(cross_detected) / len(cross), 4) if cross else 0.0,
        "findings_reported": len(findings),
        "findings_true": true_findings,
        "precision": (
            round(len(detected) / unique_results, 4) if unique_results else 0.0
        ),
        "duplicates": sum(max(0, len(v) - 1) for v in matched.values()),
        "per_entry": [
            {
                "id": gt["id"],
                "class": gt["class"],
                "cross_file": gt["cross_file"],
                "detected": bool(matched[gt["id"]]),
                "finding_indexes": matched[gt["id"]],
            }
            for gt in gt_entries
        ],
        "unmatched_findings": unmatched_findings,
    }


def extract_event_tokens(event: Any) -> int:
    """Best-effort token count for one ADK/LiteLLM event. Never raises.

    Each LLM call reports its own usage, so summing per-event totals gives
    the campaign total. Missing or malformed metadata counts as zero.
    """
    try:
        usage = getattr(event, "usage_metadata", None)
        if usage is None:
            return 0
        total = getattr(usage, "total_token_count", None)
        if isinstance(total, int) and total > 0:
            return total
        prompt = getattr(usage, "prompt_token_count", None) or 0
        candidates = getattr(usage, "candidates_token_count", None) or 0
        return int(prompt) + int(candidates)
    except Exception:
        return 0


# --- Jail setup -----------------------------------------------------------------------


def _copy_target(src: Path, jail_dir: Path) -> Path:
    """Copies the target into ``jail_dir``/app, excluding ground_truth.json.

    The exclusion is required: the model under evaluation must not be able
    to read the answers.
    """
    app_dir = Path(jail_dir) / "app"
    shutil.copytree(
        src,
        app_dir,
        ignore=shutil.ignore_patterns(GROUND_TRUTH_BASENAME, "__pycache__", "*.pyc"),
    )
    copied_gt = app_dir / GROUND_TRUTH_BASENAME
    if copied_gt.exists():  # pragma: no cover - belt and braces
        raise RuntimeError("ground truth leaked into the eval jail")
    return app_dir


def resolve_toolset(name: str) -> List[str]:
    """Resolves a toolset name to registered tool names, warning on gaps."""
    from tools import TOOLS

    requested = TOOLSETS.get(name)
    if requested is None:
        raise ValueError(f"unknown toolset '{name}'; known: {sorted(TOOLSETS)}")
    missing = [t for t in requested if t not in TOOLS]
    if missing:
        logger.warning(
            "Toolset '%s': %s not registered yet; running without them.",
            name,
            ", ".join(missing),
        )
    return [t for t in requested if t in TOOLS]


# --- The eval -------------------------------------------------------------------------


async def eval_researcher(
    model_id: str,
    effort: str,
    target_dir: Optional[Path] = None,
    toolset: str = "baseline",
) -> Dict[str, Any]:
    """Runs one researcher campaign over the seeded target and scores it."""
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from core.database import init_db, read_findings
    from evals.stage_agents import build_stage_agent, install_eval_run_context

    target_dir = Path(target_dir or DEFAULT_TARGET)
    gt_entries = load_research_ground_truth(target_dir / GROUND_TRUTH_BASENAME)
    tool_names = resolve_toolset(toolset)

    temp_dir = tempfile.mkdtemp(prefix="mantis_eval_research_")
    tokens = 0
    run_errors: List[str] = []
    try:
        # The jail root is a subdirectory so eval.db stays outside it: the
        # jailed read_file/list_files tools must not be able to reach the
        # findings database.
        jail_root = Path(temp_dir) / "jail"
        jail_root.mkdir()
        app_dir = _copy_target(target_dir, jail_root)
        db_path = os.path.join(temp_dir, "eval.db")
        init_db(db_path)
        ctx, reset_token = install_eval_run_context(
            str(jail_root), db_path, target_file=str(app_dir)
        )
        try:
            agent = build_stage_agent(
                "researcher",
                model_id=model_id,
                reasoning_effort=effort,
                tools_override=tool_names,
            )
            session_service = InMemorySessionService()
            runner = Runner(
                agent=agent, session_service=session_service, app_name="eval_app"
            )
            session = await session_service.create_session(
                app_name="eval_app", user_id="eval_user"
            )
            content = types.Content(
                role="user", parts=[types.Part.from_text(text=RESEARCH_PROMPT)]
            )
            started = time.time()
            try:
                async for event in runner.run_async(
                    session_id=session.id, user_id="eval_user", new_message=content
                ):
                    tokens += extract_event_tokens(event)
            except Exception as exc:
                from core.config import is_auth_error

                if is_auth_error(exc):
                    # A credential problem is an operator error, not a score.
                    raise
                logger.warning("Researcher run aborted: %r", exc)
                run_errors.append(repr(exc)[:200])
            latency = time.time() - started
        finally:
            from core.context import current_run_context

            current_run_context.reset(reset_token)

        findings = read_findings(db_path)
        result = score_research_findings(findings, gt_entries)
        result.update(
            {
                "toolset": toolset,
                "tools_resolved": tool_names,
                "latency": latency,
                "run_errors": run_errors,
                "total_tokens": tokens,
                "tokens_per_detected": (
                    round(tokens / result["detected"])
                    if tokens and result["detected"]
                    else None
                ),
            }
        )
        return result
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="Model ID (vertex_ai/...).")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--toolset", default="baseline", choices=sorted(TOOLSETS))
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    os.environ.setdefault("VERTEXAI_LOCATION", "global")
    result = await eval_researcher(
        args.model, args.effort, target_dir=args.target, toolset=args.toolset
    )
    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
