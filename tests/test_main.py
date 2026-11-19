import pytest

from open_dneo import __main__ as cli
from open_dneo import entrance, shapes


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


def test_a_command_is_required(capfdbinary):
    with pytest.raises(SystemExit):
        cli.main([])
