# Security Lockdown — revert guide

This document records the security/restrictive settings applied to the `milwil-2/climb-elo` repository on **2026-05-26** as part of a "public but restricted" posture. Each section lists the change, why it was made, and the exact command/UI step to revert.

> When you're confident in the codebase and want to invite contributions, walk down this file and revert items in the order listed (most restrictive → least).

---

## 1. Branch protection on `main`

**Applied:** `pytest (3.11)` + `pytest (3.12)` required as status checks; `strict: true` (must be up to date before merging); no force pushes; no branch deletion.

**Why:** prevent broken code from landing on `main` even via direct push.

**Revert:**
```bash
gh api -X DELETE repos/milwil-2/climb-elo/branches/main/protection
```
Or **Settings → Branches → Branch protection rules → Delete rule for `main`**.

---

## 2. Restricted PR review permissions

**Applied (via UI):** "Limit to users explicitly granted read or higher access" enabled.

**Why:** prevents random GitHub users from posting Approve / Request-changes reviews on PRs.

**Revert (UI):** Settings → General → scroll to "Code review limits" → uncheck.

---

## 3. Projects disabled

**Applied:** `has_projects=false`.

**Why:** unused feature; reduces spam attack surface.

**Revert:**
```bash
gh api -X PATCH repos/milwil-2/climb-elo -F has_projects=true
```
Or **Settings → General → Features → check "Projects"**.

---

## 4. Actions allowlist (verified marketplace + GitHub-owned + named patterns)

**Applied:** `allowed_actions=selected`, with these patterns explicitly allowed:
- `actions/*` (GitHub-owned)
- All verified-marketplace actions
- `astral-sh/setup-uv@*`
- `peter-evans/create-issue-from-file@*`

**Why:** prevents an unreviewed third-party Action from running in our workflows and exfiltrating data.

**Revert (re-allow all):**
```bash
gh api -X PUT repos/milwil-2/climb-elo/actions/permissions \
  -F enabled=true -F allowed_actions=all
```
Or **Settings → Actions → General → "Allow all actions and reusable workflows"**.

---

## 5. Dependabot alerts + security updates

**Applied:** alerts enabled (auto via API); security updates enabled (also auto).

**Why:** auto-PR when a dependency has a CVE.

**Revert:**
```bash
gh api -X DELETE repos/milwil-2/climb-elo/vulnerability-alerts
gh api -X PATCH repos/milwil-2/climb-elo -F security_and_analysis.dependabot_security_updates.status=disabled
```
Or **Settings → Code security → toggle off**.

---

## 6. Dependabot weekly version updates

**Applied:** `.github/dependabot.yml` configures weekly grouped updates for Python deps + GitHub Actions.

**Why:** keep deps fresh without overwhelming the issue tracker.

**Revert (full):** delete `.github/dependabot.yml`.
**Revert (less frequent):** change `interval: "weekly"` → `"monthly"` in that file.

---

## 7. Private vulnerability reporting

**Applied:** enabled via API.

**Why:** gives security researchers a private channel to disclose findings instead of opening a public issue.

**Revert:**
```bash
gh api -X DELETE repos/milwil-2/climb-elo/private-vulnerability-reporting
```
Or **Settings → Code security → toggle "Private vulnerability reporting" off**.

---

## 8. Workflow `GITHUB_TOKEN` default permissions

**Applied (already by default):** `default_workflow_permissions: read`. No change made — confirmed existing.

**Why:** workflows that need write (e.g., `snapshot.yml`) must opt in explicitly via a `permissions:` block.

**Revert:**
```bash
gh api -X PUT repos/milwil-2/climb-elo/actions/permissions/workflow \
  -F default_workflow_permissions=write
```
Or **Settings → Actions → General → "Workflow permissions" → "Read and write"**.

---

## 9. SECURITY.md disclosure policy

**Applied:** added `SECURITY.md` with disclosure instructions.

**Why:** GitHub displays this file in the Security tab and recommends it to anyone filing a security issue.

**Revert:** delete `SECURITY.md`.

---

## Settings that did NOT change (still pending)

These need to be toggled in the UI — the API either rejected them on personal-public repos or the toggle isn't exposed:

| Setting | Why it failed | Where to enable |
|---------|---------------|-----------------|
| **Disable forking** | API only allows this on org-owned **private** repos | Settings → General → uncheck "Allow forking" |
| **Secret scanning** | API returned "disabled" despite write success — UI toggle is the reliable path | Settings → Code security → "Secret scanning" → enable |
| **Push protection for secrets** | Same as above | Settings → Code security → "Push protection" → enable |
| **CodeQL / code scanning** | Complex API; UI is faster | Settings → Code security → "Code scanning" → "Set up" → Default |

---

## Quick revert (most → least restrictive)

If you want to fully revert all repo-side restrictions (keeping the file-based ones), run:

```bash
# Reverse order: re-allow first, lock-down configs last
gh api -X PUT repos/milwil-2/climb-elo/actions/permissions -F enabled=true -F allowed_actions=all
gh api -X PATCH repos/milwil-2/climb-elo -F has_projects=true
gh api -X DELETE repos/milwil-2/climb-elo/private-vulnerability-reporting
gh api -X DELETE repos/milwil-2/climb-elo/vulnerability-alerts
gh api -X PATCH repos/milwil-2/climb-elo -F security_and_analysis.dependabot_security_updates.status=disabled
gh api -X DELETE repos/milwil-2/climb-elo/branches/main/protection
```
Then in the UI: re-enable Code review limits (Settings → General).

You'll still want to keep `SECURITY.md` and `.github/dependabot.yml` even after opening up the repo — they're best practices regardless of contribution posture.
