import json

from open_dneo import hub, shapes
from tests.test_hub import safetensors_blob, serving


def test_tensor_lines_are_sorted_and_totalled():
    header = {
        "__metadata__": {"format": "pt"},
        "b.weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
        "a.weight": {"dtype": "BF16", "shape": [4], "data_offsets": [12, 20]},
    }
    lines = shapes.tensor_lines(header)
    assert [line.split()[0] for line in lines[:2]] == ["a.weight", "b.weight"]
    assert lines[-1] == "  total parameters 10"
    assert "__metadata__" not in "".join(lines)


def test_report_lists_files_config_and_tensors():
    info = {
        "sha": "abc",
        "cardData": {"license": "apache-2.0", "base_model": "x/y"},
        "siblings": [
            {"rfilename": "config.json", "size": 13},
            {"rfilename": "model.safetensors", "size": 99},
        ],
    }
    files = {
        f"{hub.HUB}/api/models/o/n/revision/main?blobs=true": json.dumps(info).encode(),
        hub.file_url("o/n", "abc", "config.json"): b'{"block": 8}\n',
        hub.file_url("o/n", "abc", "model.safetensors"): safetensors_blob(
            {"fc.weight": ("BF16", [2, 4], b"\x00" * 16)}
        ),
    }
    lines = shapes.report("o/n", "main", serving(files))
    assert lines[0] == "repo o/n commit abc"
    assert lines[1] == "license apache-2.0 base_model x/y"
    assert "file model.safetensors 99" in lines
    assert lines[lines.index("== config.json") + 1] == '{"block": 8}'
    assert "== model.safetensors: 1 tensors" in lines
    assert lines[-1] == "  total parameters 8"
