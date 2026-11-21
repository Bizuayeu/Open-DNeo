"""The bridge: DNeo's weighted sum mapped onto what DFlash2's entrance gave.

The recordings here are random rows, and the drafter's `fc` is made from a known
bridge: fitted with the weights it was made with, the bridge comes back.
"""

import json

import numpy as np
import pytest
import torch

from open_dneo import bridge, drafter

from .test_hub import safetensors_blob
from .test_score import STORED, TAKEN, folders, on_disk

HIDDEN = 8
TRUE = (0.25, 0.75)  # the weights fc is made with, a tap each of TAKEN


def plain(array) -> tuple[str, list[int], bytes]:
    array = np.ascontiguousarray(array, "<f4")
    return ("F32", list(array.shape), array.tobytes())


def made_fc(matrix: np.ndarray, weights=TRUE) -> np.ndarray:
    """fc [hidden, taps * hidden] that gives matrix @ (the weighted sum of the taps)."""
    return np.concatenate([w * matrix for w in weights], axis=1).astype(np.float32)


def dflash2_folder(folder, fc):
    """A drafter folder with what the bridge reads: config.json and fc.weight."""
    folder.mkdir(parents=True)
    config = {
        "layer_types": ["full_attention"],
        "is_causal": False,
        "dflash_config": {
            "mask_token_id": 22,
            "target_layer_ids": [tap - 1 for tap in TAKEN],
        },
    }
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "model.safetensors").write_bytes(
        safetensors_blob({"fc.weight": plain(fc)})
    )
    return folder


def recorded(root, streams=10, rows=40, seed=3):
    """Streams of random rows of the STORED taps, every one a prompt's."""
    rng = np.random.default_rng(seed)
    for n in range(streams):
        values = rng.standard_normal((rows, len(STORED), HIDDEN)).astype(np.float32)
        values *= rng.uniform(0.1, 30.0, (rows, 1, 1)).astype(np.float32)  # sizes vary
        tokens = rng.integers(0, 20, rows)
        on_disk(root / "7-0", values, tokens, first=0, prompt=rows, n=n)
    return root


@pytest.fixture
def made(tmp_path):
    matrix = np.random.default_rng(5).standard_normal((HIDDEN, HIDDEN))
    drafted = dflash2_folder(tmp_path / "drafter", made_fc(matrix))
    return recorded(tmp_path / "rec"), drafted, matrix.astype(np.float32)


def line(lines, label):
    return next(each for each in lines if each.startswith(f"{label} weights "))


def test_the_bridge_fc_was_made_from_comes_back_with_its_weights(made, tmp_path):
    root, drafted, matrix = made
    out = tmp_path / "out"
    lines, sound = bridge.report(
        [root], drafted, out, ["true=1,3"], ridge=1e-9, held_every=5, train_every=1
    )
    assert sound
    assert line(lines, "true").startswith(
        "true weights 0.250,0.750: explained 1.0000 fitted, 1.0000 held; cos 1.0000 held"
    )
    fitted, weights, taps = bridge.read(out / "bridge-true.safetensors")
    assert taps == TAKEN
    assert weights == pytest.approx(TRUE)
    np.testing.assert_allclose(fitted.numpy(), matrix, rtol=1e-3, atol=1e-3)
    # the uniform sum loses what the taps hold apart: it cannot give fc's output
    assert "explained 1.0000" not in line(lines, "uniform")
    assert (out / "bridge-uniform.safetensors").exists()
    assert (out / "bridge-even.safetensors").exists()


def test_each_row_is_divided_by_the_size_of_its_target():
    rng = np.random.default_rng(1)
    rows = rng.standard_normal((5, 2, HIDDEN)).astype(np.float32)
    rows[3] = 0.0  # no target: no direction to fit
    fc = made_fc(rng.standard_normal((HIDDEN, HIDDEN)))
    states, targets = bridge.scaled(rows, fc)
    kept = rows[[0, 1, 2, 4]]
    given = kept.reshape(4, -1) @ fc.T
    size = np.sqrt((given**2).mean(-1, keepdims=True))
    np.testing.assert_allclose(targets, given / size, rtol=1e-5)
    np.testing.assert_allclose(states, kept.reshape(4, -1) / size, rtol=1e-5)


def test_sums_hold_each_pair_of_taps_so_any_weights_can_be_fitted_after():
    rng = np.random.default_rng(2)
    rows = rng.standard_normal((30, 2, HIDDEN)).astype(np.float32)
    fc = made_fc(rng.standard_normal((HIDDEN, HIDDEN)))
    whole, halves = bridge.Sums(2, HIDDEN), bridge.Sums(2, HIDDEN)
    whole.add(*bridge.scaled(rows, fc))
    halves.add(*bridge.scaled(rows[:13], fc))
    halves.add(*bridge.scaled(rows[13:], fc))
    states, targets = bridge.scaled(rows, fc)
    summed = states.reshape(30, 2, HIDDEN).astype(np.float64)
    weights = np.array([0.4, 0.6])
    mixed = np.einsum("t,ntd->nd", weights, summed)
    for sums in (whole, halves):
        gram, cross = sums.matrices(weights)
        np.testing.assert_allclose(gram, mixed.T @ mixed, rtol=1e-5)
        np.testing.assert_allclose(cross, mixed.T @ targets, rtol=1e-5)
        assert sums.rows == 30
        assert sums.energy == pytest.approx(
            float((targets.astype(np.float64) ** 2).sum())
        )


def test_even_weights_bring_each_tap_to_the_same_size():
    rng = np.random.default_rng(4)
    rows = rng.standard_normal((50, 2, HIDDEN)).astype(np.float32)
    rows[:, 1] *= 10.0
    sums = bridge.Sums(2, HIDDEN)
    sums.add(*bridge.scaled(rows, made_fc(rng.standard_normal((HIDDEN, HIDDEN)))))
    sizes = sums.tap_sizes()
    assert sizes[1] / sizes[0] == pytest.approx(10.0, rel=0.1)
    assert bridge.even(sizes) == pytest.approx(
        [sizes[1] / sizes.sum(), sizes[0] / sizes.sum()]
    )


@pytest.mark.parametrize(
    ("text", "want"),
    [("1,3", (None, [0.25, 0.75])), ("deep=0,2", ("deep", [0.0, 1.0]))],
)
def test_weights_are_given_as_a_list_with_a_name_or_without(text, want):
    assert bridge.parse(text) == want


@pytest.mark.parametrize("text", ["-1,2", "0,0", "a,b", "x=", "uniform=1,1"])
def test_weights_that_are_not_a_sum_are_refused(text):
    with pytest.raises(ValueError):
        bridge.parse(text)


def test_weights_for_another_number_of_taps_are_refused(made, tmp_path):
    root, drafted, _ = made
    with pytest.raises(ValueError, match="2 taps"):
        bridge.report([root], drafted, tmp_path / "out", ["1,1,1"])


def test_a_bridge_already_written_is_not_written_over(made, tmp_path):
    root, drafted, _ = made
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "bridge-even.safetensors").write_bytes(b"kept")
    with pytest.raises(FileExistsError):
        bridge.report([root], drafted, tmp_path / "out")
    assert (tmp_path / "out" / "bridge-even.safetensors").read_bytes() == b"kept"


def test_held_rows_are_scored_by_part_across_folders(made, tmp_path):
    root, drafted, _ = made
    other = tmp_path / "rec2"
    rng = np.random.default_rng(8)
    values = rng.standard_normal((40, 2, HIDDEN)).astype(np.float32)
    on_disk(
        other / "8-0", values, rng.integers(0, 20, 40), first=0, prompt=40, taps=TAKEN
    )
    parts = tmp_path / "parts.tsv"
    parts.write_text(
        "".join(f"7-0/{n}\t{'a' if n < 5 else 'b'}\n" for n in range(10)) + "8-0/0\tc\n"
    )
    lines, sound = bridge.report(
        [root, other], drafted, tmp_path / "out", parts=[parts], held_every=2
    )
    assert sound
    assert lines[0].startswith(
        "11 streams in 2 folders, 5 held out (one in every 2, by name); the sum of"
        " taps (5, 9), the target fc on taps (5, 9) as DFlash2 reads them"
    )
    said = line(lines, "uniform")
    assert "; by part: a " in said and ", b " in said and ", c " not in said
    # 8-0/0 sorts last, the 11th: it is fitted, not held


def test_streams_of_the_same_name_in_two_folders_are_refused(made, tmp_path):
    root, drafted, _ = made
    with pytest.raises(ValueError, match="7-0/0"):
        bridge.report([root, root], drafted, tmp_path / "out")


def test_a_bridged_entrance_norms_the_bridged_weighted_sum(tmp_path):
    drafted, _, (model, _, _) = folders(tmp_path)
    rng = np.random.default_rng(6)
    matrix = rng.standard_normal((model.shape.hidden, model.shape.hidden))
    path = tmp_path / "b.safetensors"
    bridge.write(path, matrix.astype(np.float32), [0.0, 1.0], TAKEN)
    fitted, weights, taps = bridge.read(path)
    assert (weights, taps) == ((0.0, 1.0), TAKEN)
    new = bridge.entrance(model, fitted, weights)
    assert isinstance(new, drafter.Bridged)
    features = torch.from_numpy(
        rng.standard_normal((1, 3, 2, model.shape.hidden)).astype(np.float32)
    )
    want = model.entrance.norm(features[:, :, 1] @ fitted.T)
    torch.testing.assert_close(new(features), want)


def test_only_dflash2s_entrance_is_bridged(tmp_path):
    from .test_score import made as small

    model, _, _ = small("dneo")
    with pytest.raises(ValueError, match="DFlash2"):
        bridge.entrance(model, torch.zeros(HIDDEN, HIDDEN), (0.5, 0.5))


def test_the_sum_can_read_other_taps_than_fc_does(tmp_path):
    """fc reads taps 5 and 9; where tap 2 holds tap 5's rows twice over, a sum of 2 and 9
    weighed 1 to 6 gives fc's output, B coming back at 7/8 of fc's matrix."""
    matrix = np.random.default_rng(5).standard_normal((HIDDEN, HIDDEN))
    drafted = dflash2_folder(tmp_path / "drafter", made_fc(matrix))
    rng = np.random.default_rng(3)
    root = tmp_path / "rec"
    for n in range(10):
        values = rng.standard_normal((40, len(STORED), HIDDEN)).astype(np.float32)
        values[:, STORED.index(2)] = 2 * values[:, STORED.index(5)]
        on_disk(root / "7-0", values, rng.integers(0, 20, 40), first=0, prompt=40, n=n)
    out = tmp_path / "out"
    lines, sound = bridge.report(
        [root], drafted, out, ["true=1,6"], ridge=1e-9, held_every=5, train_every=1,
        inputs=(2, 9),
    )  # fmt: skip
    assert sound
    assert (
        "; the sum of taps (2, 9), the target fc on taps (5, 9) as DFlash2" in lines[0]
    )
    assert line(lines, "true").startswith(
        "true weights 0.143,0.857: explained 1.0000 fitted, 1.0000 held; cos 1.0000 held"
    )
    fitted, weights, taps = bridge.read(out / "bridge-true.safetensors")
    assert taps == (2, 9)
    np.testing.assert_allclose(fitted.numpy(), matrix * 7 / 8, rtol=1e-3, atol=1e-3)


def test_a_sum_of_another_number_of_taps_than_fc_reads_is_refused(made, tmp_path):
    root, drafted, _ = made
    with pytest.raises(ValueError, match="2 taps"):
        bridge.report([root], drafted, tmp_path / "out", inputs=(2, 5, 9))


def test_a_stream_without_the_sums_taps_is_refused(made, tmp_path):
    root, drafted, _ = made
    with pytest.raises(ValueError, match="not \\[4\\]"):
        bridge.report([root], drafted, tmp_path / "out", inputs=(4, 9))
