# Working rules

For anyone, human or agent, who changes this repository.

- **Every push to `main` is a release.** Give it a version in `pyproject.toml`, `open_dneo/__init__.py` and
  `CHANGELOG.md`. There is no Unreleased section. A tag `vX.Y.Z` on that commit publishes its `CHANGELOG.md` section
  as the GitHub release.
- **One fact, one place.** `docs/plan.md` holds the goal, the stages and the gates. `docs/survey.md` and the later
  stage reports hold what was read and measured. The README points to them and repeats only the headline.
- **A number needs a source.** Cite the table, the file or the command it came from, and read a table's column
  heads before quoting a cell. A placeholder says it is one, and what will replace it.
- **Gates are written before the numbers** they judge, and a failed gate is published as it is.
- **Receipts.** A measurement's raw output goes under `receipts/<date>-<subject>/` with the commands and a
  `SHA256SUMS`. Tools write UTF-8 with LF.
- **Nothing private.** No machine names, addresses, local paths, e-mail addresses or tokens, in files or in commit
  messages. Commits use a GitHub noreply address.
- **Third-party material.** Before adding code, weights or data from elsewhere, record its source, revision and
  licence in `NOTICE`. A published checkpoint is trained only on public datasets whose licences were checked.
- **Language.** Documents are in English. The README is also kept in Japanese (`README.ja.md`).
- **Before a commit:** `ruff check .`, `ruff format --check .` and `python -m pytest`.
