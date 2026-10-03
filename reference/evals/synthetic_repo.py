"""Generates the synthetic target for the surveyor ranking benchmark.

Builds a small polyglot web-app tree with known vulnerable files so the
benchmark and its tests run hermetically: no network, no external clone,
identical bytes on every build. Generated instead of committed so the seeded
content stays reviewable in one place and nobody "fixes" a seeded bug.

Directory layout:

* server/routes: request handlers, 3 seeded vulnerable files.
* server/lib: helpers, 1 seeded weak-crypto file.
* native/parser: C code, 1 seeded memory-safety file.
* webclient/components: benign UI code; must rank below everything seeded.
* tests/unit and vendor/bundled: decoys with real sink patterns inside
  demoted path segments; must rank below every seeded group.
* docs: markdown only.

ground_truth() returns the matching ground truth in the schema
surveyor_benchmark.load_ground_truth accepts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

# --- Seeded vulnerable files ----------------------------------------------------------

_SEARCH_JS = """\
const models = require('../models');

module.exports = function searchProducts() {
  return (req, res, next) => {
    let criteria = req.query.q || '';
    // Classic string-concatenated query: criteria flows into SQL unescaped.
    models.sequelize.query(
      "SELECT * FROM Products WHERE name LIKE '%" + criteria + "%'")
      .then(([products]) => res.json({ status: 'success', data: products }))
      .catch(error => next(error));
  };
};
"""

_EXEC_JS = """\
const { execSync } = require('child_process');

module.exports = function renderReport() {
  return (req, res) => {
    const format = req.body.format || 'pdf';
    // Attacker-chosen format string interpolated into a shell command.
    const output = execSync('report-gen --format ' + format + ' /srv/data');
    res.send(output);
  };
};
"""

_DOWNLOAD_JS = """\
const path = require('path');

module.exports = function serveDocument() {
  return (req, res) => {
    const requested = req.params.file;
    // No normalization or containment check before the send.
    res.sendFile(path.join('/srv/documents', requested));
  };
};
"""

_INSECURITY_JS = """\
const crypto = require('crypto');
const jwt = require('jsonwebtoken');

// Hardcoded signing secret shared by every deployment.
const privateKey = 'ebd7af94-8d75-4a1a-b2f1-synthetic';

exports.hash = data => crypto.createHash('md5').update(data).digest('hex');

exports.authorize = (user = {}) =>
  jwt.sign({ user }, privateKey, { expiresIn: 3600 * 5 });

exports.verify = token => (token ? jwt.verify(token, privateKey) : null);
"""

_DECODE_C = """\
#include <string.h>
#include <stdio.h>

#include "tokens.h"

/* Frame header gives the payload length; the buffer does not. */
int decode_frame(const unsigned char *wire, unsigned int wire_len, frame_t *out) {
  char scratch[64];
  unsigned int claimed = wire[0] | (wire[1] << 8);
  memcpy(out->payload, wire + 2, claimed);
  strcpy(out->label, (const char *)wire + 2 + claimed);
  sprintf(scratch, "frame:%s", out->label);
  return log_frame(scratch, wire_len);
}
"""

# --- Benign filler --------------------------------------------------------------------

_BENIGN_ROUTE = """\
module.exports = function {name}() {{
  return (req, res) => {{
    const page = Number(req.query.page) || 0;
    res.json({{ status: 'success', page: page, items: [] }});
  }};
}};
"""

_BENIGN_LIB = """\
// {name}: pure data-shaping helpers; no I/O, no handlers, no sinks.
exports.{name} = value => {{
  if (value === null || value === undefined) return '';
  return String(value).trim();
}};
"""

_BENIGN_C = """\
#include "tokens.h"

/* {name}: bounded, index-checked token bookkeeping. */
int {name}_count(const token_list_t *list) {{
  int total = 0;
  for (int i = 0; i < list->length && i < MAX_TOKENS; i++) {{
    total += list->items[i].width;
  }}
  return total;
}}
"""

_BENIGN_TSX = """\
export function {name}(props: {{ label: string }}) {{
  return <section className="{name}">{{props.label}}</section>;
}}
"""

_DECOY_TEST_EXEC = """\
const { execSync } = require('child_process');

// Test fixture: drives the CLI end to end inside CI.
it('renders the sample report', () => {
  const out = execSync('report-gen --format pdf tests/fixtures/sample');
  expect(out.length).toBeGreaterThan(0);
});
"""

_DECOY_TEST_SQL = """\
// Test fixture reproducing the legacy concatenated-query bug shape.
it('escapes quotes in product names', () => {
  const name = "O'Brien";
  const q = "SELECT * FROM Products WHERE name = '" + name + "'";
  expect(q).toContain("O'Brien");
});
"""

_DECOY_VENDOR_EVAL = """\
/* bundled-{name} v1.0.2 | (c) upstream | minified */
function u(e){{return eval('(' + e + ')');}}
module.exports={{parse:u,stringify:function(e){{return JSON.stringify(e)}}}};
"""

_BENIGN_VENDOR = """\
/* bundled-{name} v2.1.0 | (c) upstream | minified */
module.exports={{clamp:function(n,a,b){{return n<a?a:n>b?b:n}},noop:function(){{}}}};
"""

_DOC_MD = """\
# {title}

Operational notes for the synthetic webapp. This page is documentation only
and holds no executable code.
"""


def _tree() -> Dict[str, str]:
    """Full path -> content map. Pure data; identical on every call."""
    files: Dict[str, str] = {
        # server/routes: 8 files, 3 seeded.
        "server/routes/search.js": _SEARCH_JS,
        "server/routes/exec.js": _EXEC_JS,
        "server/routes/download.js": _DOWNLOAD_JS,
        # server/lib: 7 files, 1 seeded.
        "server/lib/insecurity.js": _INSECURITY_JS,
        # native/parser: 6 files, 1 seeded.
        "native/parser/decode.c": _DECODE_C,
    }
    for name in ("products", "address", "feedback", "languages", "health"):
        files[f"server/routes/{name}.js"] = _BENIGN_ROUTE.format(name=name)
    for name in ("format", "dates", "strings", "arrays", "logger", "clock"):
        files[f"server/lib/{name}.js"] = _BENIGN_LIB.format(name=name)
    for name in ("lexer", "tokens", "ast", "walker", "printer"):
        files[f"native/parser/{name}.c"] = _BENIGN_C.format(name=name)
    for name in ("Banner", "Card", "Footer", "Header", "Menu", "Table", "Badge"):
        files[f"webclient/components/{name}.tsx"] = _BENIGN_TSX.format(name=name)
    files["tests/unit/report.spec.js"] = _DECOY_TEST_EXEC
    files["tests/unit/query.spec.js"] = _DECOY_TEST_SQL
    for name in ("routes", "models", "helpers", "config"):
        files[f"tests/unit/{name}.spec.js"] = _BENIGN_LIB.format(name=name)
    for name in ("json2", "shim"):
        files[f"vendor/bundled/{name}.js"] = _DECOY_VENDOR_EVAL.format(name=name)
    for name in ("curry", "defer", "merge", "range"):
        files[f"vendor/bundled/{name}.js"] = _BENIGN_VENDOR.format(name=name)
    for name in ("install", "deploy", "faq", "style", "glossary"):
        files[f"docs/{name}.md"] = _DOC_MD.format(title=name.capitalize())
    return files


def build_synthetic_webapp(dest: Path) -> Dict[str, Any]:
    """Writes the tree under ``dest`` and returns the matching ground truth.

    ``dest`` must already exist and should be empty (a tempdir in tests).
    """
    dest = Path(dest)
    for rel, content in _tree().items():
        target = dest / rel
        os.makedirs(target.parent, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return ground_truth()


def ground_truth() -> Dict[str, Any]:
    """Ground truth in the schema ``surveyor_benchmark.load_ground_truth`` accepts."""
    return {
        "format_version": "1.0",
        "target": "synthetic_webapp",
        "notes": (
            "Hermetic benchmark target generated by evals/synthetic_repo.py. "
            "Every entry is seeded by construction; see that module for the bytes."
        ),
        "vulnerable_files": [
            {
                "id": "sqli-search",
                "file": "server/routes/search.js",
                "class": "sql_injection",
                "notes": "String-concatenated criteria into sequelize.query.",
            },
            {
                "id": "cmdi-exec",
                "file": "server/routes/exec.js",
                "class": "command_injection",
                "notes": "req.body.format interpolated into execSync command line.",
            },
            {
                "id": "traversal-download",
                "file": "server/routes/download.js",
                "class": "path_traversal",
                "notes": "req.params.file joined and sent without containment.",
            },
            {
                "id": "crypto-insecurity",
                "file": "server/lib/insecurity.js",
                "class": "crypto_misuse",
                "notes": "MD5 hashing and a hardcoded JWT signing secret.",
            },
            {
                "id": "memsafety-decode",
                "file": "native/parser/decode.c",
                "class": "memory_safety",
                "notes": "memcpy/strcpy/sprintf driven by wire-controlled length.",
            },
        ],
    }
