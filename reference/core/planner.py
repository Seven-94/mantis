"""Coverage planner: decides what to examine FIRST, given what earlier runs examined.

WHAT THIS IS FOR
----------------
The Surveyor ranks areas by how much attack surface they appear to hold. That ranking
is memoryless: it produces the same order on the tenth audit as on the first, so a
repository audited weekly spends its budget re-walking the same top-ranked ground while
areas that were never opened stay never opened. Rank is a statement about a repository;
it is not a statement about what has already been done to it.

This module supplies the missing half. It joins three records that already exist but
have never been read together:

  * the coverage ledger (`core.memory.load_coverage`) -- what was actually EXAMINED,
  * the survey diff  (`core.surveyor.diff_surveys`)   -- what CHANGED since last time,
  * recall           (`core.memory.recall`)           -- what was FOUND.

and reorders the campaign list so that unknown ground and changed ground are examined
before ground that was walked recently and was clean.

WHAT IT IS NOT ALLOWED TO DO
----------------------------
It reorders. It never adds, never removes, never substitutes.

That boundary is the whole security argument for this module, because its three inputs
are untrustworthy in three different ways: the coverage ledger and the survey diff are
derived from repository content (a directory can be named anything), and recall is prose
written by earlier LLM runs. Under the settled trust model such data may direct
attention -- which is exactly what reordering is -- but may never widen tools, sandbox,
or trust, and may never introduce a path that did not come out of the CP-3 validated set
`resolve_scan_targets` produced.

So the output is enforced to be a permutation of the input, checked structurally rather
than by inspection: if the result is not the same multiset, the original order is
returned unchanged. A hostile repository that games the ordering achieves, at absolute
worst, the order it would have gotten from the Surveyor alone.

For the same reason the ordering decision reads only validated and structural fields --
path strings, status tokens, snapshot identifiers, rank integers. No `title`,
`description` or `learning` prose reaches the sort. Prose is for the human and the agent
to read; it is not an input to control flow.

DEGRADATION (INV-6)
-------------------
Every failure path returns the Surveyor's original order. A planner that cannot plan
costs a run its prioritization, never its results.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Ordering bands, best-first. The value is the sort rank; the key is what the operator
# and the agent are told, so these names are part of the published contract.
#
# The ordering encodes four judgements, in decreasing confidence:
#
#   1. Ground nobody has walked is worth more than ground that was walked. This is the
#      only band justified by certainty rather than estimate: we know we have no
#      information, whereas every other band rests on an earlier run's conclusion.
#   2. Changed code invalidates earlier conclusions. An area that was clean at the last
#      commit is not thereby clean now, and an area whose rank moved has seen activity.
#   3. An area with a confirmed prior defect is worth revisiting even unchanged --
#      defects cluster, and an incomplete fix looks exactly like a fixed one from here.
#   4. Examined, unchanged, and clean is the weakest claim on a budget. Note this is
#      LAST, not EXCLUDED: it is still scanned. A prior clean result is one earlier
#      run's opinion, and arranging for an area to look clean once is precisely what an
#      attacker with commit access would do.
BAND_NEVER_EXAMINED = "never_examined"
BAND_CHANGED_AND_ACTIVE = "changed_and_active"
BAND_CHANGED = "changed"
BAND_PRIOR_DEFECTS = "prior_defects"
BAND_CLEAN_UNCHANGED = "clean_unchanged"

_BAND_ORDER: Tuple[str, ...] = (
    BAND_NEVER_EXAMINED,
    BAND_CHANGED_AND_ACTIVE,
    BAND_CHANGED,
    BAND_PRIOR_DEFECTS,
    BAND_CLEAN_UNCHANGED,
)

_BAND_RANK = {name: index for index, name in enumerate(_BAND_ORDER)}

# Operator-authored explanation of each band, shown to the agent examining that area.
# Every byte here is fixed at import: the repository selects which one applies, it never
# supplies the text. Selection is influence; authorship would be injection.
_BAND_NOTE = {
    BAND_NEVER_EXAMINED: (
        "No previous run of this knowledge base examined this area. Nothing here has "
        "been ruled out by anyone; treat the whole area as unreviewed."
    ),
    BAND_CHANGED_AND_ACTIVE: (
        "A previous run examined this area, but the code has changed since and this "
        "area has seen enough activity to move in the risk ranking. Earlier "
        "conclusions about it may no longer hold."
    ),
    BAND_CHANGED: (
        "A previous run examined this area, but the code has changed since. Earlier "
        "conclusions about it describe a different commit."
    ),
    BAND_PRIOR_DEFECTS: (
        "A previous run examined this area and confirmed at least one defect here. "
        "Defects cluster, and an incomplete fix is indistinguishable from a complete "
        "one without checking; the surrounding code deserves the same scrutiny."
    ),
    BAND_CLEAN_UNCHANGED: (
        "A previous run examined this area at this same commit and confirmed nothing. "
        "That is one earlier run's opinion, not a guarantee: it bounded the search, it "
        "did not prove the area safe. Prefer depth over re-tracing the obvious."
    ),
}


def _norm(path: Any) -> str:
    """Normalizes a path for comparison. Never raises."""
    return str(path or "").replace("\\", "/").rstrip("/")


def _suffixes(path: str) -> List[str]:
    """Every path-component-aligned suffix of `path`, longest first.

    `/a/b/c` -> `['/a/b/c', 'b/c', 'c']`. Used to answer containment in O(depth)
    rather than O(number of recorded areas); see `_AreaIndex`.
    """
    out = [path]
    idx = path.find("/")
    while idx != -1:
        tail = path[idx + 1 :]
        if tail:
            out.append(tail)
        idx = path.find("/", idx + 1)
    return out


def _ancestors(path: str) -> List[str]:
    """`path` and every directory containing it. `/a/b/c.py` -> `['/a/b/c.py','/a/b','/a']`.

    Needed because the planner's inputs are recorded at different granularities: the
    coverage ledger and the survey name AREAS (directories), while findings name FILES.
    A confirmed defect at `/repo/svc/handler.py` is a fact about the area `/repo/svc`,
    and matching the two by suffix alone finds nothing -- which silently turned the
    prior-defect band into dead code until a probe went looking for it.
    """
    out = [path]
    idx = path.rfind("/")
    while idx > 0:
        out.append(path[:idx])
        idx = path.rfind("/", 0, idx)
    return out


def _same_area(left: Any, right: Any) -> bool:
    """Whether two path strings name the same area.

    Suffix matching on a path-component boundary, matching `surveyor._slice_for_target`.
    Necessary because the three inputs disagree about form by construction: campaign
    targets are absolute and CP-3 resolved, survey slice roots are relative to the
    repository, and ledger entries hold whatever the recording run used as a root.
    Equality alone silently matches nothing, which would look exactly like a clean
    first run.
    """
    a, b = _norm(left), _norm(right)
    if not a or not b:
        return False
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


class _AreaIndex:
    """Membership test for a set of recorded paths, in O(path depth) per query.

    The obvious implementation -- compare each target against each recorded path -- is
    quadratic, which is invisible on ten subsystems and fatal in file-by-file mode where both
    sides can hold hundreds of thousands of paths. Precomputing every component-aligned
    suffix of every recorded path turns both directions of the suffix test into set
    lookups.

    Set `ancestors=True` for records held at FILE granularity (findings). Each record
    then also registers the directories containing it, so a defect in
    `svc/auth/token.py` is found when asking about the area `svc/auth`. Left off for
    records already held at area granularity, where it would make a recorded area match
    every one of its parents -- and so make the whole repository look examined.
    """

    __slots__ = ("_exact", "_suffixes")

    def __init__(self, paths: Any, ancestors: bool = False) -> None:
        self._exact: set = set()
        self._suffixes: set = set()
        if not isinstance(paths, (list, tuple, set)):
            return
        for raw in paths:
            path = _norm(raw)
            if not path:
                continue
            for entry in _ancestors(path) if ancestors else (path,):
                self._exact.add(entry)
                self._suffixes.update(_suffixes(entry))

    def __bool__(self) -> bool:
        return bool(self._exact)

    def matches(self, target: Any) -> bool:
        path = _norm(target)
        if not path:
            return False
        # `recorded.endswith("/" + target)` and equality, via the precomputed suffixes.
        if path in self._suffixes:
            return True
        # `target.endswith("/" + recorded)`: test the target's own suffixes.
        return any(suffix in self._exact for suffix in _suffixes(path))



def _order_by_band(banded: List[Tuple[str, str]]) -> List[str]:
    """Sorts `(target, band)` pairs into the scan order.

    Stable sort on (band, original position). Ties keep the Surveyor's risk order, so
    this layer only ever expresses the coverage judgement and never quietly relitigates
    the ranking. The position is carried rather than looked up: an `original.index(t)`
    key is quadratic, which file-by-file mode would feel.

    Separate from `plan_coverage` so the permutation guard there has something it can
    actually catch.
    """
    return [
        target
        for target, _band, _pos in sorted(
            ((t, b, i) for i, (t, b) in enumerate(banded)),
            key=lambda row: (_BAND_RANK.get(row[1], 0), row[2]),
        )
    ]


def plan_coverage(
    targets: List[str],
    coverage: Optional[Dict[str, Any]] = None,
    survey_diff: Optional[Dict[str, Any]] = None,
    memory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Orders `targets` by what earlier runs already covered.

    Pure function: the caller performs the three loads and passes the results, so this
    can be reasoned about and tested without a database, and so the layering stays
    one-directional.

    Returns:
        {
          "available":  whether any prior record informed the order,
          "order":      a PERMUTATION of `targets`, never anything else,
          "bands":      {target: band key},
          "counts":     {band key: how many targets landed in it},
          "reordered":  whether the order actually differs from the input,
        }

    Never raises (INV-6). On any failure the Surveyor's original order is returned.
    """
    original = [str(t) for t in (targets or [])]
    fallback: Dict[str, Any] = {
        "available": False,
        "order": list(original),
        "bands": {},
        "counts": {},
        "reordered": False,
    }
    if not original:
        return fallback

    try:
        coverage = coverage if isinstance(coverage, dict) else {}
        survey_diff = survey_diff if isinstance(survey_diff, dict) else {}
        memory = memory if isinstance(memory, dict) else {}

        have_coverage = bool(coverage.get("available"))
        have_diff = bool(survey_diff.get("available"))
        if not have_coverage and not have_diff:
            # First audit of this target, or no usable history. Every area is unknown
            # ground and the Surveyor's risk ranking is the best available order --
            # reporting a "plan" here would dress up the absence of information as a
            # decision.
            return fallback

        # A repository-wide fact: the commit differs from the one last surveyed. Absent
        # a prior survey this is unknowable, and assuming "changed" would put every area
        # in a band it has not earned.
        code_changed = have_diff and not survey_diff.get("unchanged_snapshot")

        examined_index = _AreaIndex(coverage.get("examined_ever") or [])
        new_index = _AreaIndex(survey_diff.get("new_areas") or [])
        moved_index = _AreaIndex(
            [
                entry.get("root")
                for entry in (survey_diff.get("moved") or [])
                if isinstance(entry, dict)
            ]
        )
        # Only CONFIRMED findings count toward the defect band. A dismissal is context,
        # not a verdict, and `reported` never enters recall at all -- an unreviewed claim
        # must not be able to reorder a campaign.
        #
        # `ancestors=True` because findings name files while targets name areas: a
        # confirmed defect in `svc/auth/token.py` is what makes the area `svc/auth`
        # worth revisiting. Without it this band matched nothing and was dead code.
        defect_index = _AreaIndex(
            [
                item.get("filepath")
                for item in (memory.get("confirmed") or [])
                if isinstance(item, dict)
            ],
            ancestors=True,
        )

        def _band_for(target: str) -> str:
            # An area the last survey did not have is unknown ground whatever the ledger
            # says, because a ledger entry that suffix-matches a brand new path is a
            # coincidence of naming, not evidence anyone looked.
            if new_index.matches(target):
                return BAND_NEVER_EXAMINED
            if not have_coverage or not examined_index.matches(target):
                return BAND_NEVER_EXAMINED
            if code_changed:
                return (
                    BAND_CHANGED_AND_ACTIVE
                    if moved_index.matches(target)
                    else BAND_CHANGED
                )
            if defect_index.matches(target):
                return BAND_PRIOR_DEFECTS
            return BAND_CLEAN_UNCHANGED

        banded = [(target, _band_for(target)) for target in original]
        bands: Dict[str, str] = dict(banded)

        order = _order_by_band(banded)

        # The permutation invariant, enforced rather than assumed. If the ordering step
        # ever loses, duplicates or invents a target it must not be able to change what
        # gets scanned: the Surveyor's list -- already CP-3 validated -- is used instead.
        #
        # `_order_by_band` is a separate function so this guard is REACHABLE. Inlined, a
        # `sorted()` of a list is a permutation by construction and the check could never
        # trip, which made it untestable -- and an untestable safety check is one nobody
        # will notice has stopped working.
        if sorted(order) != sorted(original):
            logger.warning("Coverage plan was not a permutation of its input; discarding.")
            return fallback

        # Counted over positions rather than over `bands`, so a duplicated target is
        # reported once per campaign it will actually cost.
        counts: Dict[str, int] = {}
        for _target, band in banded:
            counts[band] = counts.get(band, 0) + 1

        return {
            "available": True,
            "order": order,
            "bands": bands,
            "counts": counts,
            "reordered": order != original,
        }
    except Exception as exc:
        logger.warning("Coverage planning failed: %s", exc)
        return fallback


def render_coverage_note(plan: Dict[str, Any], scan_item: str) -> str:
    """Tells the agent examining `scan_item` what prior runs did to this area.

    Returns "" when there is nothing to say, so callers can append unconditionally.

    Emits no repository-derived bytes -- only operator-authored text selected by a band
    key -- so, unlike the slice briefing and the prior-audit history, it needs no CP-4
    fence. The path it describes is already in the prompt as the scan target.
    """
    if not isinstance(plan, dict) or not plan.get("available"):
        return ""
    band = (plan.get("bands") or {}).get(str(scan_item))
    note = _BAND_NOTE.get(band)
    if not note:
        return ""
    return "\n\nCOVERAGE HISTORY FOR THIS AREA:\n  " + note


def summarize_plan(plan: Dict[str, Any]) -> str:
    """One line of counts for the operator. Contains no repository-derived bytes."""
    if not isinstance(plan, dict) or not plan.get("available"):
        return ""
    counts = plan.get("counts") or {}
    parts = [
        f"{counts[band]} {band.replace('_', ' ')}"
        for band in _BAND_ORDER
        if counts.get(band)
    ]
    if not parts:
        return ""
    lead = "Coverage plan: " if plan.get("reordered") else "Coverage plan (order unchanged): "
    return lead + ", ".join(parts) + "."
