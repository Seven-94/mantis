"""OKF concept-id scoping regression tests.

record_okf_concept must store explicit concept ids verbatim: a regression
qualified every resourced concept with "@<resource>", mutating direct CRUD
ids and shattering bundle export layout on the "/" in the resource. The
qualification exists to stop same-run multi-file sweeps from evicting each
other's frontmatter-less repo documents under UNIQUE(run_id, concept_id),
and lives at the artifact layer where that collision is minted.
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

from core.database import (
    export_okf_bundle,
    import_okf_bundle,
    init_db,
    read_okf_concepts,
    record_artifact,
    record_okf_concept,
)


class ExplicitIdStabilityTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="mantis_okf_scope_")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(os.path.realpath(self._tmp.name))
        self.db = str(self.tmp / "test.db")
        init_db(self.db)

    def test_direct_crud_keeps_explicit_id(self):
        record_okf_concept(
            self.db,
            "run-1",
            {
                "concept_id": "entities/auth",
                "type": "Component Entity",
                "title": "Auth Entity",
                "resource": "src/auth.py",
            },
        )
        rows = read_okf_concepts(self.db, run_id="run-1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["concept_id"], "entities/auth")

    def test_bundle_roundtrip_preserves_id_and_layout(self):
        record_okf_concept(
            self.db,
            "run-1",
            {
                "concept_id": "entities/crypto_vault",
                "type": "Component Entity",
                "title": "Crypto Vault",
                "resource": "src/vault.py",
                "body_markdown": "# Crypto Vault\n",
            },
        )
        bundle = self.tmp / "bundle"
        export_okf_bundle(self.db, str(bundle), run_id="run-1")
        entities = sorted(os.listdir(bundle / "entities"))
        self.assertTrue(
            any(f.startswith("crypto_vault-") and f.endswith(".md") for f in entities),
            entities,
        )
        dst = str(self.tmp / "dst.db")
        init_db(dst)
        self.assertGreater(import_okf_bundle(dst, str(bundle), run_id="run-2"), 0)
        rows = read_okf_concepts(dst, resource="src/vault.py", run_id="run-2")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "Crypto Vault")
        # Import derives its own id from the bundle filename; the regression
        # pin is that no "@<resource>" qualification leaks into it.
        self.assertNotIn("@", rows[0]["concept_id"])


class SweepEvictionProtectionTest(unittest.TestCase):
    """The relocated qualification still prevents same-run row eviction."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="mantis_okf_sweep_")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(os.path.realpath(self._tmp.name))
        self.db = str(self.tmp / "test.db")
        init_db(self.db)

    def test_single_file_scoped_threat_models_coexist(self):
        for name in ("a.py", "b.py"):
            (self.tmp / name).write_text("x = 1\n")
        # The pipeline records resources jail-relative with the jail as cwd;
        # absolute paths canonicalize to empty and bypass file scoping.
        old_cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            for scanned in ("a.py", "b.py"):
                record_artifact(
                    self.db,
                    "run-1",
                    "threat_model",
                    "workspace/kb/THREAT_MODEL.md",
                    "# Threat Model\n\nSame default id for every file.\n",
                    metadata={"resource": scanned},
                )
        finally:
            os.chdir(old_cwd)
        rows = read_okf_concepts(self.db, run_id="run-1", include_repo_wide=True)
        ids = sorted(r["concept_id"] for r in rows)
        self.assertEqual(len(rows), 2, ids)
        self.assertTrue(all("@" in i for i in ids), ids)


if __name__ == "__main__":
    unittest.main()
