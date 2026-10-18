# Git Commit Workflow and GitHub Actions Policy

## Current user authorization (2026-10-18)
The user explicitly requested that every major completed step be pushed to https://github.com/ArnavGoel28/Titanic-Dataset and that commit history be maintained. `origin` is this repository; the approved branch is `main`. After each coherent step, run its checks, stage explicit paths, create a meaningful local commit and push it with `git push origin main`. This authorization persists; do not ask again for ordinary milestone pushes. Preserve all existing history. It does not authorize destructive history rewriting, other repositories, or unrelated remote settings changes.

## Objectives
A clean, intelligible history proving incremental engineering and validation. Do not confuse frequent commits with well-designed commits: one complete, tested logical unit per change.

## Local workflow for Codex
1. Before work: `git status --short`, `git branch --show-current`, `git log -5 --oneline`. If repository is not initialized, initialize only the intended project root.
2. Create/keep a feature branch when appropriate; never overwrite another person's changes. Inspect `git diff` and `.gitignore`.
3. Implement a coherent milestone. Run `python -m compileall -q src tests` (when directories exist), `python -m pytest -q` (when tests/deps exist), and notebook execution smoke check where applicable.
4. `git diff --check`, inspect changes and file sizes. `git add <explicit paths>`; `git diff --cached --stat` and `git diff --cached`.
5. `git commit -m "type(scope): imperative summary"` after checks pass; record SHA and evidence in next progress entry. If only documentation is touched, use `docs:`. If nothing changed, do not commit.
6. Do not push, create a remote repository, set GitHub secrets, or change branch protection until explicit user authorization. Once authorized, `git push -u origin <branch>` only to the approved repo/branch, preferably through a PR for review.

## Commit conventions
Conventional Commit prefixes: `chore`, `docs`, `feat`, `fix`, `test`, `refactor`, `ci`. Scope examples `data`, `models`, `eval`, `report`. Short subject, present tense, factual content. No claims like '100% accurate', 'production ready' or 'all tests pass' without evidence.

## GitHub Actions
The included `.github/workflows/ci.yml` triggers on push/PR. It checks Python syntax, unit tests and optional notebook execution (only once a real notebook exists and dependencies are available). It does NOT commit or push code. Extend CI during implementation to reproduce the fixed dataset/model experiment if data access/license permits. Don't rely on remote access in CI for datasets requiring authorization.

## Hygiene
- Add `.gitignore` before datasets/caches are generated.
- Don't commit `.env`, credentials, checkpoints, model pickle files, raw restricted datasets, giant plot artifacts, `__pycache__`, `.pytest_cache` or OS files.
- Keep the source and instructions to reproduce plots/tables; include only selected final figures needed for report.
- No force pushes, destructive resets, amend on shared history, or secret commits. If a secret is found, stop and inform the user; deleting it from a new commit is not sufficient remediation.

## Recommended branch protection once GitHub repo is approved
Require CI pass and at least one reviewer if collaborating; use a default main branch and short-lived feature branches/PRs. Branch protection setup requires user authorization and repository permission.
