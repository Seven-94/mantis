# ADK Reference Implementation with Mantis Skills

This directory contains a reference implementation of Mantis built directly on
top of the **Agent Development Kit (ADK)** using the full suite of canonical
**Mantis Skills** and **isolated sandboxed execution environments**.

## Getting Started

Mantis requires **Python 3.12 or newer**. First, install python3-venv such as
with `sudo apt install python3-venv`, then run the install script. Mantis comes
with automated configuration and launcher tools (`mantis-configure` and
`mantis-launch`):

```bash
cd reference && ./install.sh

# 0. Authenticate Google Cloud Application Default Credentials (ADC) if using Vertex AI
gcloud auth application-default login

# 1. Fast Configuration & Capability Auto-Detection (or --interactive wizard)
python3 scripts/configure.py --auto

# 2. Fast Preflight Validation (~1s) & Live Reachability Probe
python3 scripts/configure.py --test --probe

# 3. Launch Vulnerability Review Campaign (file or repository)
./run.sh path/to/code            # a file or a directory
```

### Local Configuration Overlay (`workflow.local.json`)

Mantis uses a layered configuration pattern:

- **`workflow.json` (Tracked)**: Contains base pipeline definitions, nodes,
  edges, and default placeholder configurations (`YOUR_PROJECT_ID`).
- **`workflow.local.json` (Gitignored)**: Contains machine-specific settings
  (such as auto-resolved GCP projects, custom sandbox paths, and model
  configurations). When present, it automatically merges on top of
  `workflow.json`.

When you run `./run.sh` or `scripts/configure.py --auto`, Mantis auto-heals
unconfigured placeholders and writes the resolved settings into
`workflow.local.json`. This ensures your `git status` remains clean after
running campaigns. To opt out of auto-healing, pass `--no-auto-configure`. To
explicitly save changes to the base tracked `workflow.json`, use
`--save-tracked` (or `--global`).

You can customize the sandbox execution mechanism (`static-only`, `gvisor`,
`microsandbox`, `gce`) or AI model at any time:

```bash
# Switch to Static-only (zero host virtualization requirements)
python3 scripts/configure.py --sandbox static-only

# Or pass runtime overrides directly to launch:
./run.sh path/to/code --sandbox static-only --model gemini-3.7-flash
```

Once you have run it you can add the mantis-advise skill to your favorite coding
agent and use that while developing your code to have your coding agent attempt
to create fewer vulnerabilities. To try it manually you can run the script:

```
python3 scripts/advise.py --file path/to/file.py   # query accumulated knowledge
```

## Benchmarks

Benchmarks for the pipeline live in [`evals/`](evals/README.md): a
deterministic, zero-LLM surveyor ranking benchmark and LLM stage evals
(researcher, dedupe, review, critic, calibrate).

## Configuration & Launch Skills

- **`mantis-configure`**
  ([`skills/mantis-configure/SKILL.md`](skills/mantis-configure/SKILL.md) /
  [`scripts/configure.py`](scripts/configure.py)): Manages pipeline settings via
  `workflow.local.json` overlay or base `workflow.json`, auto-detects host
  virtualization and cloud capabilities, configures sandboxes and LLM providers,
  and executes preflight sanity checks and live reachability probes (`--probe`).
- **`mantis-launch`**
  ([`skills/mantis-launch/SKILL.md`](skills/mantis-launch/SKILL.md) /
  [`scripts/launch.py`](scripts/launch.py)): Autonomous campaign launcher.
  Auto-heals unconfigured placeholders (e.g. `YOUR_PROJECT_ID`) into
  `workflow.local.json`, validates preflight readiness, accepts CLI overrides,
  and executes the 16-agent review graph over target files or repositories.
- **`mantis-advise`** ([`scripts/advise.py`](scripts/advise.py)): Developer
  security advisor. Queries threat models, historical lineages, verified patch
  diffs, and triaged false positives from `knowledge.db`.

## Research Graph Synthesis & Evolution Flywheel

Mantis features **research graph synthesis**, enabling autonomous construction
of tailored multi-agent review topologies for specific vulnerability classes or
audit objectives:

```bash
# Synthesize a specialized research graph tailored to an audit objective:
./run.sh path/to/code --objective "Audit for memory safety, bounds checks, and use-after-free in packet parsers"

# Inspect the synthesized graph structure without executing:
./run.sh path/to/code --objective "Audit for SSRF in webhook handlers" --inspect --dry-run
```

### Synthesis Archetypes & Deterministic Gates

Research graph synthesis generates specialized graph specifications validated
through deterministic architectural gates:

1. **Tool Registry Gate**: Only strictly whitelisted ADK tools (`read_file`,
   `list_files`, `get_findings`, `report_findings`, etc.) are permitted.
2. **Topological & Cycle Validation Gate**: Synthesizer ensures connected, valid
   DAG structures with validated cyclical feedback loops (e.g. patch
   verification loops).
3. **Structured Verdict Mapping**: Review, Critic, and Reproducer nodes are
   automatically bound to structured Pydantic schemas (`ReviewVerdict`,
   `CriticVerdict`, `ReproVerdict`).
4. **Sandbox Policy Gate**: Sandbox backends are strictly clamped to operator
   policy or archetype defaults (`static-only`), preventing untrusted LLM
   outputs from escalating execution privileges.

### Hardened Budgets & Checkpointing

Mantis enforces multi-dimensional ceilings across every campaign:

- **Wall-Clock Time**: `--max-time 2h` (ISO duration / time format).
- **Token Budget**: `--token-budget 10M` (raw integer or human-readable format).
- **Graph Steps & Node Visits**: `--max-steps 500 --max-node-visits 50`.
- **State Resumption**: `--resume <run_id>` seamlessly resumes paused campaigns
  from SQLite checkpoints with monotonic status preservation.

#### Budgets and resumption

**A resumed run starts from a fresh budget.** `--resume` does not carry forward
the tokens, steps or elapsed time already spent: the run keeps its findings,
coverage ledger and status history, but its ceilings start again at zero.

This is deliberate, so that resuming works out of the box with no arithmetic
from the operator. The consequence to be aware of is that `max_tokens` bounds
**one process**, not the total ever spent under a given run ID. A campaign
resumed five times may spend five times its configured ceiling. The pause banner
reflects this by suggesting a larger budget on the resume line.

If you need a hard total across resumptions, enforce it outside Mantis — the
per-campaign spend ledger below records what each run actually cost.

#### Sizing a scan to its budget

Before any campaign runs, Mantis states what the token budget covers:

```
📋 Work plan — file-by-file: 462,079 campaign(s), up to 2000 LLM call(s) each.
   Budget covers ~340 of 462,079 campaign(s) (0.1%) at ~25,000 tokens each
   (estimated, no observed runs yet). The run pauses there and resumes with --resume.
```

The estimate **never withholds or caps a scan**. An explicitly requested mode
runs exactly what was asked for; the line reports coverage so the number is
visible in a CI log afterwards. The only place the estimate *chooses* anything
is `scan_mode=auto`, which is already defined as the mode that decides on the
operator's behalf — there it sizes the number of surveyed subsystems, and only
ever downward from the configured `max_slices`.

Each completed campaign's actual cost is recorded to the `campaign_spend` table,
and later estimates use that deployment's own observed mean in place of the
built-in seed. Accuracy improves after the first run; the first run on a new
deployment is the only one working from an assumption. `basis` in the printed
line always names which of the two it is.

The seed is `core.cost.DEFAULT_CAMPAIGN_TOKENS`. It is an assumption about how
many turns a campaign takes, not a measurement, and is the number to tune if
estimates are consistently wrong for your workflow.

## Scan Modes

The two modes answer different questions and are **complementary, not a quality
ladder**. Their names describe *coverage* (how much of the tree is looked at)
and *reach* (how much context one question spans), which vary independently.

| Mode               | Coverage          | Reach        | What it is for                                                                                                                                                                                                                       |
| ------------------ | ----------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `whole`            | one campaign      | whole target | Small targets that fit in a single listing.                                                                                                                                                                                          |
| `file-by-file`     | every source file | one file     | Localized defects: the bad `memcpy`, the unchecked index, the missing authz call. Nothing is skipped for being unglamorous. Cost scales with file count.                                                                             |
| `cross-functional` | ranked subsystems | many files   | Defects that span files, repositories or systems — invisible to a per-file pass by construction, because no single file contains the bug.                                                                                            |
| `auto`             | —                 | —            | Whole target if it fits. Past that: `file-by-file` over every file no recorded campaign has covered — all files on a first scan, only the gap after a partial one — else `cross-functional`. An unreadable ledger counts as covered. |

```bash
# Ask one researcher about every file, then dedupe downstream.
python3 main.py /path/to/repo --scan-mode file-by-file

# Go straight at composite defects, with no per-file pass first.
python3 main.py /path/to/repo --scan-mode cross-functional

# Skip the work-plan confirmation (CI, cron, nohup).
python3 main.py /path/to/repo --scan-mode file-by-file --yes
```

Running `file-by-file` first gives a later `cross-functional` pass real evidence
to plan from, and `auto` encodes exactly that workflow, file by file: it sweeps
whatever the spend ledger has never seen covered (a first scan sweeps
everything; scanning `repo` after a run over `repo/lib` sweeps everything except
`lib`), and selects `cross-functional` once coverage is complete. Either may
still be run alone with an explicit `--scan-mode`. Any plan of 50 campaigns or
more asks for confirmation on an interactive terminal; the plan is printed
either way, so a CI log records what the run committed to. The superseded
spellings `file-sweep` and `slices` are still accepted.

## Evidence Sources

Every fact carries the tier of the source it came from, and a tier states what a
claim is *allowed to do* — not whether it is true:

- **`code`** — bytes on disk under the scan root. The only tier that may support
  a verdict (`status`, `repro_status`, `reattack_status`, `patch_status`).
- **`history`** — version-control metadata. Says what changed, never whether the
  change was correct.
- **`intent`** — human prose: wikis, tickets, design docs. Directs attention and
  establishes the real threat model, so it can show that something is
  *irrelevant* as deployed. It can never establish that something is *safe*.

A deployment can add its own sources by naming them in `workflow.json`:

```jsonc
{
  "evidence_sources": [
    {
      "type": "mycompany.mantis_sources:InternalWiki",  // package.module:ClassName
      "source_id": "eng-wiki",
      "trust_tier": "intent",
      "options": {"space": "SEC"}
    }
  ]
}
```

A source runs only if the configuration names it. Nothing self-registers: no
entry points, no plugin-directory scanning, so the set of things that can inject
evidence is readable from the config file by a reviewer rather than inferred
from what happens to be installed.

> [!IMPORTANT] **The seam is in place; no consumer reads it yet.** Nothing in
> the pipeline calls `documents()`, `history()` or `enumerate_files()`, so a
> configured source is validated, attributed and disclosed — but its content
> does not reach the analysis. The run banner says `NOT YET READ` for exactly
> this reason. Treat this as the contract to write against, not as working wiki
> ingestion.

Three rules are enforced structurally, not by convention:

1. **`trust_tier` is assigned by Mantis, and overwrites whatever the class
   declares.** A hook cannot promote itself.
2. **`code` is not configurable.** Code authority means "bytes CP-1 vetted and
   CP-3 contained"; a hook is not that. Asking for it is a configuration error,
   refused loudly rather than downgraded silently.
3. **A source that fails to build stops the run.** A silently dropped source
   leaves a report that looks identical to one that had all its evidence.

> There is no networked source, and a configuration declaring `url`, `base_url`
> or `network` is refused before its module is imported. The egress contract a
> future one would have to satisfy is written down in
> [`core/evidence.py`](core/evidence.py). The absence is a security property,
> not an unfinished feature.

## Custom Tools

The 28 built-in tools are a closed set, and a workflow naming anything else
fails validation. Without a way in, a site with an internal taint engine or a
house SAST tool has exactly one option: fork. A fork never receives the next
security fix, so leaving the set closed does not prevent custom tools — it makes
the unsafe way the only way.

### Why a declared tool inherits no trust

The built-in 28 are not safe because they are confined. Most run on the host
with the host's full privilege: nothing in the process stops `read_file` from
opening `~/.ssh/id_rsa`. They are safe because they **are** the enforcement —
`read_file` is the containment for reads, `run_sandbox` for execution,
`_run_safe_git_command` is the git jail. A chokepoint is a chokepoint only while
nothing goes around it, and an unreviewed function added to that dict is a path
around it.

A tool is also a sharper problem than an evidence source. A source is called
**once, by Mantis, before analysis, with arguments Mantis chose**. A tool is
called **repeatedly, by the model, with arguments it composed after reading the
target's source code**. `acme_taint(path=<something the repository suggested>)`
is the shape of every injection-to-execution chain, so argument handling — not
registration — is where the safety has to live.

### Lane A — declarative (no code runs)

You ship a *specification*; Mantis executes it with an audited primitive. Today
that is a parameterised, read-only query against the knowledge base.

```jsonc
{
  "config": {
    "tools": {
      "acme_prior_findings": {
        "lane": "declarative",
        "kind": "sql_query",
        "query": "SELECT title, severity FROM findings WHERE status = ?",
        "params": ["confirmed"],
        "max_rows": 50
      }
    }
  }
}
```

No third-party code enters the process, so no new path around CP-1/2/3/4 is
created and no sandbox is required. The query must be a single `SELECT`, every
`FROM`/`JOIN` target must be on the allow-list, and the tool **takes no
arguments from the model** — it is a fixed question with fixed parameters. A
declarative tool the model could parameterise would be an injection surface
wearing a safe label.

### Lane B — executable (your binary, under a sandbox floor)

Your binary never becomes a Python entry point. It is a command dispatched
through the same `ctx.sandbox.execute()` path `run_sandbox` already uses.

```jsonc
{
  "config": {
    "sandbox": {"type": "gvisor"},
    "tools": {
      "acme_taint": {
        "lane": "executable",
        "command": "/opt/acme/taint --json {filepath}",
        "minimum_sandbox": "gvisor"
      }
    }
  }
}
```

> [!WARNING] **`static-only` is the no-op environment, not a weak sandbox — and
> it is the default.** Running a third-party binary under it means running it on
> the host, in the process holding the LLM credentials and the knowledge base.
> Lane B therefore requires `gvisor` or stronger and **fails the whole run**
> when the floor is unmet, rather than warning or skipping the tool.

The floor **never raises the ceiling**. If the operator ceiling is `static-only`
and a tool asks for `gvisor`, the tool is refused; the sandbox is never upgraded
to satisfy it. A configuration that could escalate its own containment is the
exact bug the ceiling exists to prevent. An unknown backend name satisfies no
floor.

`gvisor` requires Linux (`runsc`). On a macOS host, use `gce` — a remote
hardened VM, which ranks above the floor.

### What a declared tool can never do

- **Set a verdict field.** `may_set_verdict` remains the single audit point
  (INV-1/INV-2). A "query" tool that could `UPDATE findings.status` would route
  around it entirely.
- **Shadow a built-in.** A name collision is refused, not resolved: letting the
  custom tool win replaces the containment, and letting the built-in win hands
  the operator a tool that is not the one they declared.
- **Widen the sandbox, the ceiling, or any trust tier.** A floor is a condition
  for running, never a request for more capability.
- **Reach the network.** Lane A never executes; Lane B inherits the guest's
  networkless configuration.

### Threat model: what Mantis guarantees, and what you must uphold

You can fork this harness and change any of it. The point of writing this down
is that the easy path is the safe one, and leaving it should be a decision
rather than an accident.

**Mantis guarantees, for tools declared through this registry:**

1. Your Lane B command runs in the guest, never in the host process, and never
   below its declared sandbox floor — rechecked against the live sandbox at call
   time, not just at load time.
2. Any `{filepath}` Mantis substitutes is shell-quoted as a single argument, so
   a model-chosen path cannot break out into a second command.
3. Your tool's output is secret-scrubbed and wrapped in untrusted-data
   delimiters before the model sees it, the same treatment `run_sandbox` gives
   guest output.
4. Your tool cannot set a verdict, rebind a built-in, or widen containment.
5. A tool that cannot be built or cannot be honoured **fails the run loudly**.
   It is never silently dropped, because a report from a run that skipped your
   analyser is indistinguishable from one where the analyser found nothing.

**You must uphold, and Mantis cannot check:**

1. **Your tool will be called with arguments derived from the target's source
   code.** Treat every input as attacker-controlled. Quoting protects the guest
   shell from injection; it does not protect your parser from a malicious input
   file.
2. **Your binary's own behaviour is yours.** Mantis confines it to the guest; it
   does not audit what it does there. A Lane B tool that phones home defeats the
   networkless guarantee from inside.
3. **Do not rely on the fence.** Mantis marks your tool's output as data, which
   makes a model less likely to act on instructions inside it. That is a
   mitigation, not a boundary: if your tool relays attacker-controlled text, it
   is relaying it to a model that may act on it anyway.
4. **If you widen the allow-lists or bypass the registry, these guarantees
   lapse** — including the ones you did not change, since they depend on the
   chokepoints staying chokepoints.

## Output: SARIF

By default a run's findings live only in the SQLite knowledge base, which is
fine for Mantis talking to itself and useless for anything else. Naming a path
also writes
[SARIF 2.1.0](https://docs.oasis-open.org/sarif/sarif/v2.1.0/os/sarif-v2.1.0-os.html),
the standard interchange format for static-analysis results:

```jsonc
{
  "config": {
    "sarif_output": "workspace/mantis.sarif"
  }
}
```

Opt-in, because writing a file nobody asked for is an unexpected egress and the
path is the operator's to choose. Suppressed findings — false positives,
duplicates, findings dropped at review — are excluded, so the export carries
what Mantis stands behind rather than everything it considered.

> [!NOTE] This is the opposite trade-off from
> [`mantis-pipeline-adapter`](../mantis-pipeline-adapter/SKILL.md), which argues
> against SARIF and for a minimal IR. That is about **ingestion**: every scanner
> emits a different subset, so a reader absorbs all of the variance. Emission
> inverts it — we control what we produce, and the consumer needs no custom
> parser.

### Paths are re-derived, never copied

Every URI is recomputed relative to the scan root rather than taken from the
stored finding. `canonical_filepath` falls back to an **absolute** path when it
cannot relativize one, which is harmless in a local database row and two
separate problems in a file designed to be uploaded:

- it discloses the host filesystem layout, the username, and often the project
  name; and
- an absolute or `file://` URI is the single most common cause of SARIF that
  *appears* to work — the upload succeeds and the consumer then displays
  nothing, because it cannot match the path to a repository file.

No `uriBaseId` or `originalUriBaseIds` is emitted either; the conventional
`%SRCROOT%` entry is defined as an absolute host path, which would put back
exactly what the relative URIs remove.

A finding whose path cannot be made relative is **dropped and named** in the run
output. It is never emitted absolute, and never dropped quietly — an export that
silently disagrees with the knowledge base is worse than one that is visibly
incomplete.

### Unverified findings say so

> [!IMPORTANT] SARIF has no field for "nobody reproduced this." Every entry is a
> `result` with a `level`, so a finding with a working proof-of-concept and one
> that was never checked have the same shape.

Mantis spends its whole design keeping that distinction — the evidence tiers,
the reproduction gate, INV-1. Flattening it at the export would undo that at the
last step, so the verification state is written into `message.text` where a
reader sees it without knowing any Mantis-specific property keys:

```
Stack overflow in parse_token

unbounded memcpy

[Mantis unverified: no reproduction or patch verification is recorded.
 Triage status: VALID.]
```

Machine-readable copies also go in the result property bag as `mantis-status`
and `mantis-patch-status`.

### What else is emitted

- **`level`** from Mantis severity, restricted to SARIF's four values
  (`none`/`note`/`warning`/`error`).
- **`properties["security-severity"]`** from `mantis_risk_score`, which is
  already on the 0–10 CVSS scale GitHub expects. It is emitted as a **string**;
  a JSON number is accepted on upload and then silently never assigned a
  severity. When no score exists the band floor is used rather than the middle,
  since the band is all that is actually known.
- **`partialFingerprints`** from the `lineage_id` Mantis already maintains for
  INV-3 regression tracking, so the consumer's notion of "the same finding as
  last run" agrees with ours instead of being guessed from surrounding text.
- **Rules** grouped by CWE, so findings of one weakness class share a rule.

The document is validated against the invariants that cause rejection or silent
data loss before anything is written. A document that fails is **not written at
all**: a file whose consumer rejects it or shows nothing is not a partial
success, and writing it would let the run report success anyway.

## Core Pipeline Stages

The pipeline in `workflow.json` orchestrates 15 canonical stages across the
complete vulnerability campaign lifecycle:

01. **`history`**: Extracts commit history, churn hotspots, and developer
    activity logs.
02. **`structural_index`**: Deterministically builds the tree-sitter symbol,
    call-edge, and function-boundary catalog that backs the structural
    navigation tools (no LLM calls).
03. **`architect`**: Constructs the structured Markdown Knowledge Base
    (`workspace/kb/`).
04. **`threat_modeler`**: Maps threat actors, entry points, and trust boundaries
    (`workspace/kb/THREAT_MODEL.md`).
05. **`planner`**: Formulates prioritized review targets and questions
    (`workspace/plan.json`).
06. **`researcher`**: Executes deep static analysis sweeps and flags potential
    flaws.
07. **`deduplicator`**: Clusters and deduplicates candidate findings across
    passes.
08. **`reviewer`**: Filters out false positives and evaluates reachability
    (`ReviewVerdict`).
09. **`critic`**: Conducts adversarial viability review (`CriticVerdict`).
10. **`reproducer`**: Synthesizes and runs dynamic exploit PoCs inside the
    isolated sandbox (`ReproVerdict`).
11. **`chainer`**: Chains related findings into multi-stage exploit
    trajectories.
12. **`patcher`**: Autonomous remediation conductor with strict separation of
    duties; delegates code authoring to isolated subagents, commissions
    independent third-party re-attackers with parallel adversarial trajectory
    search, and enforces dual-gate sandbox verification.
13. **`calibrator`**: Calibrates final risk scores (0–100) and justification.
14. **`reflector`**: Rotates learnings and feedback into the knowledge base
    (`workspace/learnings.jsonl`).
15. **`reporter`**: Compiles the final review packet and executive summary
    (`workspace/report/review_packet-latest.md`).

## Sandboxing & Isolation

The reference harness implements ADK's `BaseEnvironment` interface:

- **`GceEnvironment`**: Hardened Google Compute Engine (GCE) ephemeral VM
  isolation (single-VM only). Golden machine image, private non-internet VPC,
  link-local DNS blackholing, IAM token suppression, and IAP SSH tunneling. See
  [GCE Sandbox Setup Guide](docs/gce_sandbox_setup.md).
- **`MicrosandboxEnvironment`**: Hardware microVM isolation (libkrun / KVM).
  Networkless (`Network.none()`), guest-isolated filesystem at `/workspace`.
- **`GvisorEnvironment`**: OCI container isolation via gVisor (`runsc`).
  Networkless (`--network=none`), container-isolated filesystem at `/workspace`.
- **`StaticOnlyEnvironment`**: Safe no-op environment for static-only scans.

### Configuring the Sandbox Backend in `workflow.json`

To change the sandbox backend, update the `"config.sandbox"` block in
[`workflow.json`](workflow.json):

#### 1. Static-Only (`"static-only"`)

Zero dependencies. Dynamic exploit execution and patch testing are skipped.

```json
"sandbox": {
  "type": "static-only"
}
```

#### 2. gVisor (`"gvisor"`)

Local OCI container isolation via Docker/Podman with gVisor `runsc` and
`--network=none`.

```json
"sandbox": {
  "type": "gvisor",
  "options": {
    "image": "mantis-sandbox:latest",
    "runtime": "runsc",
    "timeout_seconds": 600
  }
}
```

#### 3. MicroSandbox (`"microsandbox"`)

In-process hardware microVM isolation via `libkrun` and `Network.none()`.

```json
"sandbox": {
  "type": "microsandbox",
  "options": {
    "image": "mantis-sandbox:latest",
    "timeout_seconds": 600
  }
}
```

#### 4. Hardened GCE VM (`"gce"`)

Ephemeral cloud VM in an isolated VPC with link-local DNS blackholing.

```json
"sandbox": {
  "type": "gce",
  "options": {
    "project": "YOUR_PROJECT_ID",
    "zone": "us-central1-b",
    "image": "mantis-sandbox-image",
    "subnet": "mantis-isolated-subnet",
    "workdir": "/workspace",
    "tunnel_through_iap": true,
    "no_service_account": true,
    "no_external_ip": true,
    "verify_isolation": true,
    "timeout_seconds": 600
  }
}
```

| Sandbox Type         | Dynamic Execution | Prerequisites                                       |
| :------------------- | :---------------: | :-------------------------------------------------- |
| **`"static-only"`**  |        ❌         | None                                                |
| **`"gvisor"`**       |        ✅         | Docker/Podman + `runsc` runtime                     |
| **`"microsandbox"`** |        ✅         | Hardware virtualization (`/dev/kvm`)                |
| **`"gce"`**          |        ✅         | GCP Project, Isolated VPC/Subnet, Custom Disk Image |

### Target Boundary & Isolation Model

- **Boundary vs. Filter**: The target repository jail is a strict boundary
  around the target checkout directory, not an in-tree content filter. Pointing
  Mantis at a directory exposes the files within that directory to analysis by
  design. Outside the target directory, access is strictly blocked. Inside the
  target directory, only protected VCS directories (`.git`, `.hg`, `.svn`,
  `.jj`) and 11 sensitive metadata/credential filenames (`.gitconfig`,
  `.gitmodules`, `.gitattributes`, `.git-credentials`, `.netrc`, `.env`,
  `.env.local`, `.npmrc`, `.pypirc`, `.pre-commit-config.yaml`,
  `.pre-commit-config.yml`) are denied. Other files inside the target (e.g.
  `.env.production`, `.aws/credentials`, `secrets.yaml`) are within the analyzed
  target scope and will be read if requested.
- **Untrusted Git Hardening**: All host git inspection tools (`get_git_diff`,
  `get_git_log`, `ls-files`) are hardened with `--no-ext-diff`, `--no-textconv`,
  `-c diff.external=`, `-c diff.tool=`, `-c core.attributesFile=/dev/null`, and
  `GIT_CONFIG_NOSYSTEM=1`. This prevents repositories carrying attacker-authored
  `.git/config` or `.gitattributes` files from executing arbitrary binaries or
  external diff drivers on the host during commit history or diff analysis.
- **Workflow Discovery**: By default, Mantis discovers `workflow.json` strictly
  from the reference package installation paths and will not probe an arbitrary
  `workflow.json` located in the current working directory, preventing untrusted
  repository graph hijacking. Custom workflows must be explicitly specified via
  `--workflow <path>`.

### Quickstart: Isolated GCE Sandbox Setup

An automated setup script is provided at
[`reference/scripts/setup_gce_sandbox.sh`](scripts/setup_gce_sandbox.sh):

```bash
# Automated setup (provisions VPC, subnet, firewall, DNS policy):
PROJECT_ID=your-gcp-project SOURCE_INSTANCE=your-dev-vm ./reference/scripts/setup_gce_sandbox.sh
```

Or run the parameterized commands manually:

```bash
# Configuration variables
REGION="us-central1"
ZONE="us-central1-a"
VPC_NAME="mantis-isolated-vpc"
SUBNET_NAME="mantis-isolated-subnet"
IMAGE_NAME="mantis-golden-image-v1"
DEV_BUILD_VM="my-dev-build-vm"

# 1. Custom Isolated VPC & Subnet (no internet, no Cloud NAT, no Google API access)
gcloud compute networks create "${VPC_NAME}" --subnet-mode=custom
gcloud compute networks subnets create "${SUBNET_NAME}" \
    --network="${VPC_NAME}" \
    --region="${REGION}" \
    --range=10.0.0.0/24 \
    --no-enable-private-ip-google-access

# 2. Allow SSH strictly from Google Cloud Identity-Aware Proxy (IAP)
gcloud compute firewall-rules create "allow-iap-ssh-${VPC_NAME}" \
    --network="${VPC_NAME}" \
    --allow=tcp:22 \
    --source-ranges=35.235.240.0/20

# 3. Block recursive public DNS exfiltration via Cloud DNS Response Policy
gcloud dns response-policies create mantis-block-public-dns \
    --project="${PROJECT_ID}" \
    --networks="${VPC_NAME}" \
    --description="Block all public DNS lookups"
gcloud dns response-policies rules create block-all-domains \
    --project="${PROJECT_ID}" \
    --response-policy=mantis-block-public-dns \
    --dns-name="*." \
    --local-data=name="*.",type="A",ttl=300,rrdatas="0.0.0.0"

# 4. Create Assessment Disk Image from your pre-warmed build/dev VM
gcloud compute images create "${IMAGE_NAME}" \
    --project="${PROJECT_ID}" \
    --source-disk="${DEV_BUILD_VM}" \
    --source-disk-zone="${ZONE}" \
    --force \
    --description="Assessment disk image with pre-warmed build dependencies for Mantis"
```

## Typed Domain Tools Suite

For maximum reliability and structured database grounding, the harness provides
strictly-typed domain tools backed by Pydantic models and SQLite persistence:

- **`report_findings(report)`**: Validates `VulnerabilityReport` and writes
  findings.
- **`get_findings()`**: Retrieves recorded findings for the current target file.
- **`dedupe_findings(primary_title, duplicate_titles, reason)`**: Merges
  duplicates.
- **`record_plan(plan)`**: Validates `ReviewPlan` and records
  `workspace/plan.json`.
- **`record_threat_model(threat_model)`**: Validates `ThreatModel` and records
  `THREAT_MODEL.md`.
- **`record_summary(summary)`**: Validates `CodebaseSummary` and records
  `mantis-summary.md`.
- **`record_exploit_chain(chain)`**: Validates and records `ExploitChain`.
- **`score_risk(score, reasoning)`**: Validates $0 \\le \\text{score} \\le 100$
  and records risk calibration.
- **`record_learning(learning)`**: Validates `LearningEntry` and rotates
  learnings into SQLite.
- **`generate_report(report)`**: Validates `ExecutiveReport` and writes
  `review_packet-latest.md`.

## Schema Single-Source-of-Truth

All state contracts and Pydantic models in `core/schemas.py` implement the root
canonical `schema.json`, providing runtime type safety and invariant enforcement
(INV-1 through INV-6) across all Mantis skills, external orchestrators, and the
ADK reference harness.

## Integration Pattern

Each agent node in `workflow.json` declares its node identifier and attached
tools:

```json
{
  "id": "researcher",
  "type": "agent",
  "tools": ["read_file", "write_file", "list_files", "report_findings", "get_findings"]
}
```

When compiled by `core/graph_loader.py`, the agent node's `id` automatically
maps to its high-density system prompt in `core/prompts.py` (or literal
`system_prompt`), with domain tools attached directly to the agent and stage
isolation enforced (`include_contents="none"`).

### Custom System Prompt Alternative

`core/graph_loader.py` also supports configuring an agent node with literal
instruction text via `system_prompt`:

```json
{
  "id": "custom_auditor",
  "type": "agent",
  "system_prompt": "You are a specialized auditor inspecting cryptographic primitives. Focus on key reuse and weak RNG.",
  "tools": ["read_file", "list_files", "report_findings", "get_findings"]
}
```

When `system_prompt` is specified, `core/graph_loader.py` uses the literal
instruction text directly rather than resolving the default prompt by `id`.
