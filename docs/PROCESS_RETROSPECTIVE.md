# Building Climbing ELO: A Process Retrospective

A working session log of decisions, debugging, and design tradeoffs from one day of work on the climbing-elo project — a FastAPI + SQLAlchemy rating system for IFSC competition climbing.

This document is structured by *what was learned*, not by *what was shipped*. Specific PR numbers and commit hashes are included so the work is auditable, but the goal is to document **how the project actually evolved under pressure**.

---

## Where the day started

- ~~10 open GitHub issues~~, three of which had code on `main` but were never closed
- A Vercel deployment that was failing on every request (500s on `/`, `/predictions`, `/favicon.ico`)
- A Supabase Postgres database that was supposed to be production, but had only Lead-discipline ratings (Boulder/Speed empty)
- A `v2` monochrome frontend co-existing with the legacy `v1` frontend under a `/v2/` prefix
- An R&D research doc (`docs/RATING_SYSTEM_RESEARCH.md`) outlining five rating-system improvements, none of them implemented

By end of day:

- 19 issues closed, 10 new follow-up issues filed
- Production live at `https://climb-elo.vercel.app` with all three disciplines + combined ratings
- v2 promoted to root, v1 deleted entirely
- Glicko-2 RD integration (#51), 538-style MOV gap-conditioning (#53), and learned Boulder+Lead composite weights (#54) all shipped
- 442 → 524 tests passing
- Two planning documents in `docs/`: one for Glicko-2 integration, one for field-strength + activity-weighted leaderboard

The trajectory wasn't a straight line. Each section below documents a *turn* — a point where a decision changed what the project was.

---

## Turn 1: The deployment crisis (and what production actually demands)

The deployment was broken in a cascade of four failures, each masking the next.

**Failure 1: `DATABASE_URL` not set.** The Vercel deployment was running, the build succeeded, but every request crashed at `init_db()`. Easy fix: add the env var via the Vercel dashboard.

**Failure 2: Vercel's Python runtime couldn't find `app`.** I had wrapped `create_app()` in a try/except (to surface startup errors with a traceback instead of an opaque `FUNCTION_INVOCATION_FAILED`). But Vercel's static analyzer parses `api/index.py` looking for a top-level `app = ...` assignment. Putting it inside a `try` block hides it from the analyzer.

The fix:

```python
app = FastAPI()  # top-level declaration, always visible
try:
    from climbing_elo.api.app import create_app
    app = create_app()  # override
except Exception:
    _startup_error = traceback.format_exc()
    @app.get("/{path:path}")
    async def _error(...): ...
```

Two assignments to `app` — one outside the try, one inside. The outer assignment satisfies the static analyzer; the inner assignment is what actually runs.

**Lesson:** Vercel's "deploy any FastAPI app" promise hides assumptions about how your entry point is written. Tools that introspect your code statically constrain how you can express runtime logic.

**Failure 3: IPv6.** With the entry point fixed, the function started — and immediately crashed: `psycopg2.OperationalError: connection to server at "db.PROJECT.supabase.co" (2600:1f14:...): Network is unreachable`.

Supabase's direct connection URL is **IPv6-only**. Vercel's serverless functions don't have IPv6 connectivity to external hosts.

The fix is to use Supabase's connection pooler URL instead — same database, different network path. Supabase exposes **three** connection URLs:

| URL | Port | Network | Use for |
|---|---|---|---|
| `db.PROJECT.supabase.co` (direct) | 5432 | IPv6-only | Local dev only |
| `aws-0-REGION.pooler.supabase.com` (transaction) | 6543 | IPv4 | Serverless (Vercel) |
| `aws-0-REGION.pooler.supabase.com` (session) | 5432 | IPv4 | Bulk ops (GH Actions) |

Each pooler has tradeoffs. Transaction pooler doesn't support session-level features (advisory locks, `SET` statements, prepared statements). Session pooler does, but is intended for longer-lived connections.

**Failure 4: GitHub Actions hit the same IPv6 wall.** I had set up a daily scrape workflow expecting to use the direct URL (since GH Actions runs in AWS and I assumed IPv4 was fine). Same network-unreachable error. The fix is the **session pooler URL** — same hostname as Vercel's transaction pooler, just port 5432 instead of 6543.

**Lesson:** "Postgres is Postgres" is the *application* abstraction. The *operational* abstraction is "three different connection strings depending on which compute environment you're calling from." This wasn't visible until things broke; no document I read up front would have predicted this. Production knowledge is paid for in outages.

The full set of fixes ended up in `CLAUDE.md` under a new "Connection strings (Supabase)" subsection — a future visitor will not have to learn this the way I did.

---

## Turn 2: When the fix is to delete the thing

Several hours later, while implementing learned weights for combined ratings (#54), an agent reported back: *"the fitter couldn't produce real numbers — the local SQLite has uppercase enum values like `WORLD_CHAMPIONSHIP` instead of `world_championship`, and is missing the `livestream_url` column added in #23."*

My first instinct was to file an issue: "Local SQLite drift — add a refresh script." That became **#79**.

Reviewing the issue body, I realized I was solving the wrong problem. The local SQLite file existed only because `database.py` falls back to it when `DATABASE_URL` is unset. That fallback was useful pre-Supabase-migration but is now a footgun: it means scripts silently run against a stale schema instead of failing loudly.

Worse, the `snapshot.yml` workflow had been running daily, snapshotting an *empty* local SQLite file in CI, uploading the empty file to a GitHub Release named `db-snapshots`. The "Daily Snapshots" feature documented in CLAUDE.md was a non-feature.

I closed #79 as outdated. Filed **#82** as a replacement: *"Require `DATABASE_URL` for scripts + delete broken snapshot workflow."*

The right fix wasn't to refresh the local DB. It was to **delete the local DB code path entirely**. One source of truth (Supabase). Tests use in-memory SQLite directly (no schema drift possible — every test run creates fresh schema from `models.py`). Scripts now fail fast with a useful error message if `DATABASE_URL` isn't set.

The snapshot workflow died with it. Supabase has its own backups; we don't need ours.

**Lesson:** Sometimes the right fix isn't to maintain the thing that's broken. It's to ask whether the thing should exist at all. Removing 743 lines (`scripts/snapshot_db.py` + `scripts/restore_snapshot.py` + `tests/test_snapshot.py` + `.github/workflows/snapshot.yml`) made the project simpler, not poorer.

---

## Turn 3: Plan before you build

The R&D track was different from the deployment work. The research synthesis already existed (`docs/RATING_SYSTEM_RESEARCH.md`) — five candidate improvements, each estimated 2 days to 2 weeks. The temptation was to start coding.

Instead, I dispatched a **planning agent** for the highest-payoff issue (#51 — Glicko-2 RD integration). The agent's deliverable was a written plan, not code: `docs/PLAN_GLICKO2_RD_INTEGRATION.md` (488 lines). It included:

- File:line references for every integration point in `engine/elo.py`
- The exact Glicko-2 formulas (from Glickman 2013) that would replace the existing math
- Migration strategy (full re-backfill vs forward-only)
- Three open decisions flagged for me:
  1. Inactivity inflation semantics (calendar-time vs event-count)
  2. Whether margin-of-victory folds into the outcome score `s_j` or into the K factor
  3. Whether to reuse Glicko-2's `φ` for the Monte Carlo projector

A research agent costs ~$1 in API calls. An implementation agent that re-litigates open architectural decisions mid-stream costs more in token churn — and produces inconsistent code if those decisions get answered differently across multiple attempts.

When I dispatched the implementation agent for R1, I **baked the three decisions into the prompt** as locked-in answers. The agent didn't ask. It executed.

The result: R1 shipped in a single commit (`47dae01`) with the full test suite passing, including a cold-start trajectory test using real production athlete IDs (Sorato Anraku at #1/611 Men's Boulder, matching AscentStats). The backtest beat baseline by +37.5pp on Lead and +77.8pp on Boulder.

**Lesson:** For changes that touch foundational math, the deliverable of Phase 1 should be a document, not code. The cost of re-doing implementation work is much higher than the cost of producing a written plan that locks in the questions whose answers shape the work.

I used the same pattern for the field-strength + activity-weighting plan (#88) and got a 606-line plan back, including a Supabase data audit showing that 14 of the top-30 men's lead athletes are inactive >24 months and a heuristic classifier (3-year gap rule) that correctly distinguishes Coxsey-style retirees from Garnbret-style break-takers.

---

## Turn 4: Orchestrating parallel agents (and the conflicts they create)

By mid-day there were four background agents running in isolated git worktrees:

| Agent | Task | Files |
|---|---|---|
| A | #80 K-factor regrid | `scripts/regrid_k_factors.py`, reads `engine/elo.py` |
| B | #88 field-strength research | `docs/` only |
| C | #86 rich climber profile | `models.py`, `scraper/`, `routes.py`, `templates/` |
| D | #76+#77+#78 R4 fitter polish | `scripts/fit_combined_weights.py`, `tests/test_combined.py` |

Disjoint file sets. No expected merge conflicts. All four ran in parallel.

But there were close calls:

- The earlier-dispatched R3 agent (gap-conditioned MOV, #53) touched `engine/elo.py`. The K-regrid agent also reads `engine/elo.py`. If I'd started K-regrid before R3 finished, they would have collided.
- The #83 Target 3 agent (`EloConfig` dataclass) also touched `engine/elo.py`. I queued it to start **only after R3 landed** — and queued K-regrid to start **only after #83 Target 3 landed**, because K-regrid's whole value proposition was using the new `EloConfig` cleanly instead of monkey-patching module globals.

This forced me to think in dependency graphs:

```
R1 (Glicko-2)     ──►  R3 (MOV gap-cond) ──►  #83 Target 3 (EloConfig) ──►  #80 (K regrid)
                              │
                              └──►  CLAUDE.md update
R4 (B+L weights)  ──►  parallel, isolated  ──►  #76+#77+#78 (R4 polish)
#82 (DATABASE_URL cleanup)  ──►  parallel
#86 (climber profile)       ──►  parallel
```

Each arrow is a "must come before." Each lack-of-arrow is "safe to run in parallel."

**Lesson:** Parallelism isn't free. The constraint isn't "is the agent intelligent enough to handle the task" — it's "do the agents' edits collide at git merge time?" Worktree isolation makes the *agent* safe; it doesn't make the *merge* safe.

The cost of a wrong call is a non-fast-forward push and a few minutes of manual rebase. The cost of a right call is 4× wall-clock throughput.

---

## Turn 5: Mistakes & corrections

A selection of things that went wrong and what I learned from each.

### The `git add -A` near-miss

While finishing #83 Target 3 inline (after the agent hit a session token limit mid-work), I ran `git add -A` to stage the agent's WIP changes. That swept in **66 files including 84,131 line additions** — agent-generated backtest output dirs, design handoff bundles, and (worst) `.mcp.json` containing my Vercel personal access token.

GitHub's push protection blocked the push. The token never made it to a public commit.

The fix: `git reset HEAD~1`, tighten `.gitignore` to explicitly block `.mcp.json`, `data/backtests/`, and `agent-resources/`, then `git add` only the intended files by name.

**Lesson:** `git add -A` is a foot-gun in any working tree where agents are leaving artifacts. The `.gitignore` should be exhaustive about agent-generated paths *before* you start dispatching agents. GitHub's secret scanning is the last line of defense, not the first.

### Out-of-order agent commits caused doc staleness

The #71 agent (CLAUDE.md refresh) ran in parallel with #72 (delete `requirements.txt`) and #74 (strip startup wrapper). The #71 agent's brief was written before #72 and #74 had specific outcomes. When #71 committed, its CLAUDE.md described `requirements.txt` as "hand-written" — but #72 had already deleted that file.

This happened *twice*: commits `fc67b6a` and `bccdbcd` are both "fix CLAUDE.md staleness from out-of-order agent commits."

**Lesson:** Documentation changes that span agents' scopes get stale fast. Either centralize all CLAUDE.md edits in a single agent (no parallelism on docs) or audit the doc after every batch of commits lands. I now lean toward the latter — the audit is cheap; the parallelism is valuable.

### The stale-DB rabbit hole

The fitter for #54 couldn't produce real numbers because the local SQLite was on a pre-migration schema. My first issue (#79) tried to fix the local SQLite. My second issue (#82) deleted it.

Three weeks ago (before this session, in a session log I don't have), someone — probably me — added the SQLite fallback to `database.py` as a convenience for offline dev. It made sense at the time. By today it was actively harmful.

**Lesson:** Convenience code paths accumulate. Periodically audit "does this fallback / shim / compat layer still serve someone?" Removing one is a meaningful improvement, even when "nothing was broken."

---

## Process patterns that emerged

Looking back at the day's commits, I can name the patterns I leaned on. Most of these aren't novel, but applying them deliberately mattered.

### 1. Decisions get baked into prompts, not left open

When an agent reaches an architectural choice point, it either makes a defensible default (and documents it) or asks. Either outcome is worse than my having decided in advance. For Glicko-2's three open questions, I baked answers into the implementation agent's prompt. The agent didn't deliberate; it executed.

### 2. Research agents are cheap

A planning agent that produces a 500-line markdown doc with file:line references costs a fraction of an implementation agent that has to investigate the same code from scratch. The plan acts as a shared mental model between me and the implementation agent.

### 3. Backtest-gated changes

Every PR touching rating math (R1, R3, R4) had to keep `scripts/run_backtest.py` above the 15pp baseline-beat threshold. This isn't a CI check yet; it's a discipline applied per-PR. Filed as a follow-up to make it a hard gate.

### 4. File the follow-ups; don't bundle scope

R1 (Glicko-2) explicitly deferred two things: a full K-factor regrid sweep (#80) and the full Glicko-2 volatility update (#81). R4 (combined weights) deferred three: scipy.optimize (#76), k-fold CV (#77), σ-weights (#78). Each deferral became its own issue. The PRs stayed scoped.

### 5. Surface findings as new issues

The #88 research agent discovered that σ-ceiling is binding for 80-91% of athletes — meaning R1's σ-inflation mechanism is currently invisible. That observation didn't fit in the field-strength plan, so I filed **#89** as its own investigation. The research agent's deliverable doesn't have to solve every problem it notices; it just has to *notice* them and hand them off.

### 6. Worktree isolation enables parallelism; merge planning preserves it

Four agents in parallel works when their file sets are disjoint. The thinking happens *before* dispatch, not during. Once an agent is running, course-correction is expensive.

### 7. Doc-the-doc

CLAUDE.md got updated three times today. Each update was a reaction to a change in production reality. The doc is a feedback loop, not a delivery artifact.

### 8. Mid-task corrections via inter-agent messages

The #88 research agent was already running when I realized the plan should also address a smart "active vs retired" classifier. Rather than waiting and dispatching a follow-up, I sent the agent a message via `SendMessage` adding the consideration to its scope. The agent folded it into the plan's existing Gap 2 section. Mid-flight refinement saved a full second pass.

---

## Where we ended

| Category | Count |
|---|---|
| Issues closed | 19 |
| New issues filed | 10 |
| Commits to `main` | 30+ |
| Agent dispatches (background) | 12 |
| Tests | 442 → 524 passing |
| Planning documents written | 2 (Glicko-2, field-strength) |
| Production deployment failures | 4 (each fixed) |
| Production deployment uptime at end of day | 100% |

The remaining open work splits into three categories:

**Immediate (gates Glicko-2 going live in prod):**
- #80 K-factor regrid sweep (in flight) — without sensible K values, the prod leaderboard would show μ in 3000-3400 range after re-backfill

**Field-strength + activity (plan ready, decisions made):**
- Gap 2 first: dual-view leaderboard with retirement classifier
- Gap 1 second: tournament participation bonus (gated on #80)

**Polish & deferred (R&D backlog):**
- #56 R6 Speed bracket model (Phase 2 per phased plan)
- #52 R2 WHR/ILSR batch refit (Phase 3, biggest scope)
- #84, #85 G-Elo benchmark + MOV grid search (research)
- #83 Targets 1-2 (split `evaluation.py` + `routes.py`)
- #89 σ-ceiling investigation

---

## The thing I'd say to someone starting on a project like this

Production knowledge is paid for in outages. Plan documents are cheaper than implementation re-runs. Parallel agents work only if their file sets are disjoint and you've thought about the merge graph. The right fix is sometimes to delete the thing causing the problem.

Most of all: **the work is the iteration**. The codebase that exists at the end of this document is not the codebase that existed at the start. Neither is final. The next iteration is already queued.
