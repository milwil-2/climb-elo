# Security Policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities. Use one of these channels instead:

1. **Private vulnerability report** (preferred): <https://github.com/milwil-2/climb-elo/security/advisories/new>
2. **Email**: milanwillett@gmail.com with subject line `[climbing-elo security]`

I'll acknowledge within 7 days and aim to ship a fix or mitigation within 30 days for confirmed issues.

## Scope

In scope:
- The Python application (`src/climbing_elo/`)
- The public REST API (`/api/v1/*`)
- The HTML dashboard (`/`, `/predictions`, `/head-to-head`, etc.)
- The SSE live endpoint (`/live/{event_id}/stream`)
- GitHub Actions workflows in `.github/workflows/`
- Scripts in `scripts/`

Out of scope:
- The underlying IFSC results API (`ifsc.results.info`) — report to IFSC directly
- The hosting platform (when deployed) — report to the provider
- Third-party dependencies — please report upstream first; we'll bump our pin

## What counts as a vulnerability

| Severity | Examples |
|----------|----------|
| Critical | RCE, auth bypass, SQL injection, sensitive data leak |
| High     | Stored XSS, SSRF, CSRF on a state-changing endpoint, broken access control |
| Medium   | Reflected XSS, info disclosure, predictable identifiers |
| Low      | Verbose error messages, missing security headers |
| Info     | Best-practice suggestions, defense-in-depth |

This project is a personal research tool with no auth and no user data. Most "high"-tier issues in commercial apps are reduced to "medium" here. That said, please still report them — defense in depth matters.

## What I will NOT pay for

This is a personal project with no bug-bounty program. Acknowledgement in the release notes (with your permission) is the only compensation offered.
