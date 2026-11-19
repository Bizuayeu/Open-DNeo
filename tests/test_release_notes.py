from pathlib import Path

import pytest

import open_dneo
from tools import release_notes

REPO = Path(__file__).resolve().parents[1]

CHANGELOG = """# Changelog

## 0.3.0 (not released)

Written, not released.

## 0.2.0 (2026-11-21)

A paragraph that wraps
onto a second line.

A second paragraph.

- An item that wraps
  onto a second line, with `code`.
- A short item.
  1. A numbered item under it
     that wraps.

| A | table |
|---|---|
| stays | in rows |

```sh
kept as
  it is,
every line
```
After the block.

## 0.1.1 (2026-11-20)

## 0.1.0 (2026-11-19)

The first.
"""

# What GitHub shows for 0.1.0, written by hand before this tool existed.
PUBLISHED = """The plan and the survey. No weights, and no result yet on whether the approach works.

- `docs/plan.md`: the goal, DNeo's parts by what each costs to train, the four moves away from the paper's recipe, \
the stages and the gates, written before any number.
- `docs/survey.md`: the paper, the public code and weights, the tensor layouts of GLM's DFlash2 drafter and the \
published DNeo side by side, TensorFold's GLM drafter path, precedents upstream, and the cost of the paper's recipe \
on two GB10 machines.
- `open-dneo shapes`: a Hub checkpoint's files, config and tensor shapes, from the safetensors header alone.
- `open-dneo entrance`: the diagonal share of a dense feature encoder, per layer block. GLM's DFlash2 entrance is \
at chance level (0.024%) in every block.
- `receipts/2026-11-19-survey`: the outputs of both tools on the pinned revisions.
"""


def test_a_section_is_the_notes_under_its_dated_heading():
    assert release_notes.section(CHANGELOG, "0.1.0") == "The first.\n"
    middle = release_notes.section(CHANGELOG, "0.2.0")
    assert middle.startswith("A paragraph that wraps\nonto a second line.\n\nA second")
    assert middle.endswith("```\nAfter the block.\n")


@pytest.mark.parametrize(
    ("version", "why"),
    [
        ("0.2", "not a release version"),
        ("v0.2.0", "not a release version"),
        ("0.4.0", "no dated section"),
        ("0.3.0", "no dated section"),  # a heading without a date is not released
        ("0.1.1", "no dated section"),  # a heading without notes
    ],
)
def test_a_section_that_is_not_a_release_is_refused(version, why):
    with pytest.raises(ValueError, match=why):
        release_notes.section(CHANGELOG, version)


UNWRAPPED = (
    "A paragraph that wraps onto a second line.\n"
    "\n"
    "A second paragraph.\n"
    "\n"
    "- An item that wraps onto a second line, with `code`.\n"
    "- A short item.\n"
    "  1. A numbered item under it that wraps.\n"
    "\n"
    "| A | table |\n"
    "|---|---|\n"
    "| stays | in rows |\n"
    "\n"
    "```sh\n"
    "kept as\n"
    "  it is,\n"
    "every line\n"
    "```\n"
    "After the block.\n"
)


def test_wrapped_lines_go_out_on_one_line_and_blocks_stay_apart():
    section = release_notes.section(CHANGELOG, "0.2.0")
    assert release_notes.unwrapped(section) == UNWRAPPED


def test_a_heading_or_a_quote_begins_its_own_line():
    wrapped = "text\n### Sub\n> a\n> b\nmore\n"
    assert release_notes.unwrapped(wrapped) == "text\n### Sub\n> a\n> b more\n"


def test_the_first_release_comes_out_as_it_was_published():
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    notes = release_notes.unwrapped(release_notes.section(changelog, "0.1.0"))
    assert notes == PUBLISHED


def test_this_checkout_can_be_released_at_its_own_version():
    # the three places a version is written agree, and the changelog has its notes
    assert release_notes.notes(REPO, open_dneo.__version__)


def project(root: Path, pyproject: str, package: str) -> Path:
    (root / "open_dneo").mkdir()
    (root / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nversion = "{pyproject}"\n', encoding="utf-8"
    )
    (root / "open_dneo" / "__init__.py").write_text(
        f'"""A package."""\n\n__version__ = "{package}"\n', encoding="utf-8"
    )
    return root


def test_notes_are_the_unwrapped_section_of_a_project_at_that_version(tmp_path):
    root = project(tmp_path, "0.2.0", "0.2.0")
    assert release_notes.notes(root, "0.2.0") == UNWRAPPED


@pytest.mark.parametrize(
    ("pyproject", "package", "where"),
    [
        ("0.2.0", "0.1.0", r"pyproject\.toml is at 0\.2\.0, not 0\.1\.0"),
        ("0.1.0", "0.2.0", r"open_dneo/__init__\.py is at 0\.2\.0, not 0\.1\.0"),
    ],
)
def test_a_tag_on_a_commit_at_another_version_publishes_nothing(
    tmp_path, pyproject, package, where
):
    with pytest.raises(ValueError, match=where):
        release_notes.notes(project(tmp_path, pyproject, package), "0.1.0")


def test_the_command_prints_the_notes_and_names_what_is_wrong(
    tmp_path, capfdbinary, monkeypatch
):
    monkeypatch.setattr(release_notes, "REPO", project(tmp_path, "0.1.0", "0.1.0"))
    release_notes.main(["0.1.0"])
    assert capfdbinary.readouterr().out == b"The first.\n"
    with pytest.raises(SystemExit, match="no dated section"):
        release_notes.main(["0.4.0"])
