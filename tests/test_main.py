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
    assert (
        cli.main(["greedy", "rec", "--target", "ck", "--save", "nx", "--top", "16"])
        == 0
    )
    asked = ["score", "rec", "--drafter", "dr", "--target", "ck", "--device", "cuda"]
    assert cli.main(asked) == 1
    assert cli.main([*asked[:6], "--served", "--streams", "--max-tokens", "48"]) == 1
    assert cli.main([*asked[:6], "--served", "--run", "run.jsonl"]) == 1
    logged = ["--run", "run.jsonl", "--drafts-log", "serve.log"]
    assert cli.main([*asked[:6], *logged]) == 1
    assert (
        cli.main([*asked[:6], "--served", "--bridge", "b.st", "--bridge-served"]) == 1
    )
    assert cli.main([*asked[:6], "--dneo-mask", "--dneo-head", "16"]) == 1
    assert seen == [
        (Path("rec"), Path("ck"), "cpu", True, None, 0),
        (Path("rec"), Path("ck"), "cpu", False, Path("nx"), 16),
        (
            Path("rec"),
            Path("dr"),
            Path("ck"),
            "cuda",
            False,
            False,
            None,
            None,
            None,
            None,
            False,
            False,
            None,
        ),
        (
            Path("rec"),
            Path("dr"),
            Path("ck"),
            "cpu",
            True,
            True,
            48,
            None,
            None,
            None,
            False,
            False,
            None,
        ),
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
            None,
            False,
            False,
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
            None,
            False,
            False,
            None,
        ),
        (
            Path("rec"),
            Path("dr"),
            Path("ck"),
            "cpu",
            False,
            True,
            None,
            None,
            None,
            Path("b.st"),
            True,
            False,
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
            None,
            None,
            None,
            False,
            True,
            16,
        ),
    ]
    assert capfdbinary.readouterr().out == b"g\ng\ns\ns\ns\ns\ns\ns\n"


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


def test_rows_passes_its_folders_run_and_weights(monkeypatch, capfdbinary):
    from open_dneo import rows

    seen = []
    monkeypatch.setattr(rows, "report", lambda *a: seen.append(a) or (["r"], True))
    asked = ["rows", "rec", "--drafter", "dr", "--target", "ck", "--run", "run.jsonl"]
    assert cli.main([*asked, "--served"]) == 0
    assert cli.main([*asked, "--device", "cuda"]) == 0
    assert seen == [
        (Path("rec"), Path("dr"), Path("ck"), Path("run.jsonl"), "cpu", True),
        (Path("rec"), Path("dr"), Path("ck"), Path("run.jsonl"), "cuda", False),
    ]
    assert capfdbinary.readouterr().out == b"r\nr\n"


def test_probe_passes_its_folders_and_choices(monkeypatch, capfdbinary):
    from open_dneo import probe

    seen = []
    monkeypatch.setattr(probe, "report", lambda *a: seen.append(a) or (["p"], True))
    asked = ["probe", "rec", "--target", "ck", "--next", "nx"]
    assert cli.main(asked) == 0
    more = ["--k", "1", "--set", "6,15", "--parts", "p.tsv", "--ridge", "0.1"]
    assert cli.main([*asked, *more, "--held-every", "5"]) == 0
    assert seen == [
        (
            Path("rec"),
            Path("ck"),
            Path("nx"),
            (1, 2, 3),
            (),
            None,
            1e-3,
            10,
            4,
            8,
            "cpu",
        ),
        (
            Path("rec"),
            Path("ck"),
            Path("nx"),
            (1,),
            ((6, 15),),
            Path("p.tsv"),
            0.1,
            5,
            4,
            8,
            "cpu",
        ),
    ]
    assert capfdbinary.readouterr().out == b"p\np\n"


def test_bridge_passes_its_folders_weights_and_split(monkeypatch, capfdbinary):
    from open_dneo import bridge

    seen = []
    monkeypatch.setattr(bridge, "report", lambda *a: seen.append(a) or (["b"], True))
    asked = ["bridge", "r1", "r2", "--drafter", "dr", "--out", "o"]
    assert cli.main(asked) == 0
    more = ["--weights", "a=1,2", "--weights", "3,4", "--parts", "p.tsv"]
    more += ["--ridge", "0.1", "--held-every", "5", "--train-every", "2"]
    more += ["--inputs", "40,41"]
    assert cli.main([*asked, *more]) == 0
    assert seen == [
        ([Path("r1"), Path("r2")], Path("dr"), Path("o"), [], [], 1e-3, 10, 4, ()),
        (
            [Path("r1"), Path("r2")],
            Path("dr"),
            Path("o"),
            ["a=1,2", "3,4"],
            [Path("p.tsv")],
            0.1,
            5,
            2,
            (40, 41),
        ),
    ]
    assert capfdbinary.readouterr().out == b"b\nb\n"


def test_train_passes_its_folders_numbers_and_form(monkeypatch, capfdbinary):
    from open_dneo import trainer

    seen = []

    def command(*given, **named):
        seen.append((given, named))
        return ["== train finished"], True

    monkeypatch.setattr(trainer, "command", command)
    asked = [
        "train", "rec", "rec2", "--next", "nx", "--drafter", "dr", "--target", "ck",
        "--out", "o", "--steps", "50", "--bridge", "b.st", "--dneo-mask",
        "--dneo-head", "0", "--parts", "p.tsv", "--shares", "code=1,chat=3",
        "--peak", "1e-4", "--off-path", "0.25", "--held-every", "5",
        "--held-streams", "4", "--rates", "layers=0.1",
    ]  # fmt: skip
    assert cli.main(asked) == 0
    ((given, named),) = seen
    folders, nexts, drafted, checkpoint, out, settings = given
    assert folders == [Path("rec"), Path("rec2")] and nexts == Path("nx")
    assert (drafted, checkpoint, out) == (Path("dr"), Path("ck"), Path("o"))
    assert settings == trainer.Settings(
        steps=50,
        peak=1e-4,
        off_path=0.25,
        shares={"code": 1.0, "chat": 3.0},
        held_streams=4,
        rates={"layers": 0.1},
    )
    assert named == {
        "parts": [Path("p.tsv")], "bridge": Path("b.st"), "dneo_mask": True,
        "dneo_head": 0, "held_every": 5, "device": "cpu", "resume": False,
    }  # fmt: skip
    assert capfdbinary.readouterr().out == b"== train finished\n"
    seen.clear()
    assert cli.main([*asked[:13], "--resume"]) == 0
    assert seen[0][1]["resume"] is True


def test_shares_are_read_as_part_equals_number():
    from open_dneo import trainer

    assert trainer.shares(None) is None and trainer.shares("") is None
    assert trainer.shares("code=0.3, chat=0.7") == {"code": 0.3, "chat": 0.7}
    with pytest.raises(ValueError, match="a share is part=number, not 'code'"):
        trainer.shares("code")
    assert trainer.rates("layers=0.1,head=1") == {"layers": 0.1, "head": 1.0}
    assert trainer.rates(None) is None
    with pytest.raises(ValueError, match="a rate's factor is part=number, not 'x'"):
        trainer.rates("x")
