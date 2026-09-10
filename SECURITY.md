# Security Policy

## Supported versions

UniMRI is in pre-alpha (0.0.x). Only the latest commit on `main` and the most
recent release receive fixes. There is no long-term-support branch yet.

| Version | Supported |
| --- | --- |
| latest `main` / latest release | ✅ |
| anything older | ❌ |

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately via GitHub's
[private vulnerability reporting](https://github.com/istiyakamin/UniMRI/security/advisories/new)
("Report a vulnerability" on the repository's *Security* tab). If that is
unavailable, email the maintainer at the address in `CITATION.cff`.

Please include:

- a description of the issue and its impact,
- steps or a proof of concept to reproduce it,
- affected version(s) / commit,
- any suggested fix.

## What to expect

- Acknowledgement of your report within **7 days**.
- An initial assessment and severity estimate within **14 days**.
- Coordinated disclosure: we will agree on a timeline with you, aim to release a
  fix before public disclosure, and credit you in the advisory and changelog
  unless you prefer to remain anonymous.

## Scope

In scope: the `unimri` package and its build/release tooling in this repository.

Out of scope: vulnerabilities in third-party dependencies (report those
upstream), and issues that require a malicious raw-data file the user already
trusts — though we still want to know about parser crashes and will harden
readers against malformed input.
