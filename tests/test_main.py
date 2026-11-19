from pathlib import Path

import pytest

from open_dneo import __main__ as cli
from open_dneo import entrance, recording, shapes


def test_a_report_is_written_as_utf8_with_lf_only(monkeypatch, capfdbinary):
    monkeypatch.setattr(shapes, "report", lambda repo, revision: ["a", "é"])
    assert cli.main(["shapes", "o/n"]) == 0
    assert capfdbinary.readouterr().out == "a\né\n".encode()


def test_entrance_takes_the_tensor_name(monkeypatch, capfdbinary):
    seen = []
    monkeypatch.setattr(
        entrance, "report", lambda repo, revision, tensor: seen.append(tensor) or []
    )
    assert cli.main(["entrance", "o/n", "--tensor", "enc.weight"]) == 0
    assert seen == ["enc.weight"]


def test_recording_fails_when_its_report_found_something(monkeypatch, capfdbinary):
    seen = []

    def report(folder, each):
        seen.append((folder, each))
        return ["findings 1"], False

    monkeypatch.setattr(recording, "report", report)
    assert cli.main(["recording", "some/folder", "--streams"]) == 1
    assert seen == [(Path("some/folder"), True)]
    assert capfdbinary.readouterr().out == b"findings 1\n"


def test_recording_reads_real_files_and_passes_a_sound_one(tmp_path, capfdbinary):
    (tmp_path / "0.json").write_text(
        '{"rows":0,"first_row":0,"prompt_len":4,"taps":[45],"row_bytes":0,'
        '"dtype":"","top_k":0,"gap":null}'
    )
    for ext in (".tok", ".pos", ".kind", ".st"):
        (tmp_path / f"0{ext}").write_bytes(b"")
    assert cli.main(["recording", str(tmp_path)]) == 0
    assert capfdbinary.readouterr().out.splitlines()[-1] == b"findings 0"


def test_a_command_is_required(capfdbinary):
    with pytest.raises(SystemExit):
        cli.main([])


def test_the_torch_commands_pass_their_folders_and_device(monkeypatch, capfdbinary):
    from open_dneo import greedy, score

    seen = []
    monkeypatch.setattr(greedy, "report", lambda *a: seen.append(a) or (["g"], True))
    monkeypatch.setattr(score, "report", lambda *a: seen.append(a) or (["s"], False))
    assert cli.main(["greedy", "rec", "--target", "ck", "--streams"]) == 0
    asked = ["score", "rec", "--drafter", "dr", "--target", "ck", "--device", "cuda"]
    assert cli.main(asked) == 1
    assert cli.main([*asked[:6], "--served", "--streams", "--max-tokens", "48"]) == 1
    assert cli.main([*asked[:6], "--served", "--run", "run.jsonl"]) == 1
    logged = ["--run", "run.jsonl", "--drafts-log", "serve.log"]
    assert cli.main([*asked[:6], *logged]) == 1
    assert seen == [
        (Path("rec"), Path("ck"), "cpu", True),
        (Path("rec"), Path("dr"), Path("ck"), "cuda", False, False, None, None, None),
        (Path("rec"), Path("dr"), Path("ck"), "cpu", True, True, 48, None, None),
        (
            Path("rec"),
            Path("dr"),
            Path("ck"),
            "cpu",
            False,
            True,
            None,
            Path("run.jsonl"),
            None,
        ),
        (
            Path("rec"),
            Path("dr"),
            Path("ck"),
            "cpu",
            False,
            False,
            None,
            Path("run.jsonl"),
            Path("serve.log"),
        ),
    ]
    assert capfdbinary.readouterr().out == b"g\ns\ns\ns\ns\n"


def test_a_torch_command_says_what_it_needs_where_torch_is_not(monkeypatch):
    def absent(name):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(cli.importlib, "import_module", absent)
    with pytest.raises(SystemExit, match="open-dneo\\[torch\\]"):
        cli.main(["greedy", "rec", "--target", "ck"])


def test_another_missing_module_is_not_blamed_on_torch(monkeypatch):
    def absent(name):
        raise ModuleNotFoundError("No module named 'other'", name="other")

    monkeypatch.setattr(cli.importlib, "import_module", absent)
    with pytest.raises(ModuleNotFoundError, match="other"):
        cli.main(["score", "rec", "--drafter", "dr", "--target", "ck"])
