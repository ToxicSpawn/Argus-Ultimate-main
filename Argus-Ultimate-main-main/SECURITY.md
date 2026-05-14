# Security Policy

## Supported Versions

| Version | Supported |
|---------|----------|
| 3.x (current) | ✅ |
| 2.x | ⚠️ Security fixes only |
| 1.x | ❌ End of life |

---

## Reporting a Vulnerability

**Do not file a public GitHub issue for security vulnerabilities.**

To report a vulnerability:

1. Use [GitHub Private Security Advisories](https://github.com/ToxicSpawn/Argus-Ultimate-main/security/advisories/new) to submit a confidential report.
2. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if known)

You can expect an acknowledgement within **72 hours** and a resolution timeline within **14 days** for critical issues.

---

## Security Design

- **No hardcoded credentials.** All API keys, secrets, and credentials are loaded exclusively from environment variables or `.env` files.
- **`.env` files are gitignored** and must never be committed.
- **Secret scanning** via `detect-secrets` pre-commit hook blocks credential commits automatically.
- **CodeQL analysis** runs on every push and weekly to detect security anti-patterns.
- **Bandit** runs in CI and as a pre-commit hook to catch common Python security issues.
- **Audit logging** records all trading decisions with trace IDs for forensic replay.

---

## Dependency Security

Dependabot is configured to open automated PRs for:
- Python pip dependency CVEs and version bumps (weekly)
- GitHub Actions version bumps (weekly)

Review and merge Dependabot PRs promptly, especially for `ccxt`, `aiohttp`, and `websockets` which have direct financial risk exposure.
