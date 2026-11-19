# Reference outputs

`tests/fixtures/dflash2_small.npz` and `tests/fixtures/dneo_small.npz` hold what the public definitions of the two
drafters compute at a small size with every parameter random: the configuration, the parameters, the inputs, each
layer's output, the final states and the head's drafts. `tests/test_drafter.py` runs `open_dneo.drafter` on the same
parameters and inputs and compares.

The public code is not copied into this repository, and the package does not import it. The two scripts here run it
from files fetched at pinned revisions. They are kept so the fixtures can be made again; the tests do not run them,
because each definition needs its own versions of torch and transformers.

| Fixture | Definition | Revision | sha256 (first 16) | Made with |
|---|---|---|---|---|
| `dflash2_small.npz` | [z-lab/dflash](https://github.com/z-lab/dflash) `dflash/model.py`, MIT | `07ebd93db9f472af339b644bb70221ad8428328a` | `f55b7fe0a4c0b307` | torch 2.13.0 (CPU), transformers 5.15.0 |
| `dneo_small.npz` | [J-sandler/Qwen3-4B-DNeo](https://huggingface.co/J-sandler/Qwen3-4B-DNeo) `configuration_dneo.py`, `modeling_dneo.py`, `reference.py` | `8451a2e8446c0bfa09be47623ff044d896540ec4` | `f50bbb2021b7e1e2`, `c9371489171ede8c`, `f79c36aece1380e5` | torch 2.15.0 (CPU), transformers 4.57.1 |
| | [J-sandler/DNeo](https://github.com/J-sandler/DNeo) `src/dneo/runtime/dneo.py` (the head, `FoldedDF2Head`), MIT | `8449a3ce891f6b3321efafc1e5c0c7e5c40e11c7` | `d1e98f35efad2463` | |

## What the small size covers

Both fixtures use 6 attention heads over 2 key-value heads, 6 convolution groups, 3 tapped target layers for 2 draft
layers, and a head rank that differs from every other size, so a port that confuses two of them does not match.

- DFlash2: a context of 50 rows, longer than the sliding window of 40; two rows of the batch at different positions.
- DNeo: three blocks a row at different anchors; rotary frequencies rounded to bf16, as the published checkpoint's
  are; no rejection memory.

## Making them again

Each definition in its own environment, with the versions in the table.

```sh
# DFlash2
curl -L -o model.py https://raw.githubusercontent.com/z-lab/dflash/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py
python -I tools/reference/run_dflash2.py model.py tests/fixtures/dflash2_small.npz

# DNeo: the model repository's three files as a package, and the runtime's file
mkdir -p pkg/dneo_ref && touch pkg/dneo_ref/__init__.py
for f in configuration_dneo.py modeling_dneo.py reference.py; do
  curl -L -o pkg/dneo_ref/$f https://huggingface.co/J-sandler/Qwen3-4B-DNeo/resolve/8451a2e8446c0bfa09be47623ff044d896540ec4/$f
done
curl -L -o dneo.py https://raw.githubusercontent.com/J-sandler/DNeo/8449a3ce891f6b3321efafc1e5c0c7e5c40e11c7/src/dneo/runtime/dneo.py
python -I tools/reference/run_dneo.py pkg dneo.py tests/fixtures/dneo_small.npz
```

Random numbers come from a fixed seed, but a different torch version may draw different ones. That changes the
fixtures' bytes and not what the tests show.
