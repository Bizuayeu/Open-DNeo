# Open-DNeo

Tools and a procedure for making [DNeo](https://arxiv.org/abs/2611.21925) drafter weights for a target model on modest
hardware, from rows a serving engine records. An independent, unofficial project.

**Status (0.6.0, 2026-11-21): a plan, a survey, an offline instrument that counts as serving does, and Stage 1's
pilot recording, layer probe and step-0 bridge. There are no weights yet, and nothing here shows that the approach works.** Gate G0 found the instrument's tokens a round within
0.013 of serving's in every class of the evaluation set's open half
([receipts](receipts/2026-11-20-g0/README.md)). From the rows of a prompt pass, the cheap ones to record, the
published DFlash2 drafter proposes nearly what it proposes from reply rows of the same tokens
([receipts](receipts/2026-11-20-rows/README.md)). On a pilot of 609,309 such rows, the five taps before the final
row hold more of the target's next tokens than the five DFlash2 reads
([receipts](receipts/2026-11-20-pilot/README.md)). With DNeo's entrance bridged by least squares onto DFlash2's
layers, the drafter lands 0.853 to 0.962 of DFlash2's drafts a round before any training. A bridge from those five
deeper taps led one from DFlash2's own in three classes of five, short of the rule set before the numbers, so
Stage 2 reads DFlash2's taps ([receipts](receipts/2026-11-21-g1/README.md)). Each stage will add its tools, its
numbers and what did not pay.

[日本語](README.ja.md)

## What this is

DNeo is a parallel drafter for speculative decoding: a small model proposes a block of tokens in one pass and the
target model verifies them, so a reply stays the target's own. The paper reports 19-27% more tokens a round than
DFlash-2 at temperature 0 on Qwen3-4B. Its authors published weights for Qwen3-4B and inference code. The training
code is not public.

This project works out how to make DNeo weights for another target, and publishes the tools, the steps and the
measurements. The first target is GLM-5.3-Flash served by [TensorFold](https://github.com/ashhart/TensorFold) on two
GB10 machines (DGX Spark class).

## Why not the paper's recipe

The paper trains on rollouts the target writes: 1.6 million of them, up to 1,024 tokens each, and 100,000 for its
ablations. On our pair the target decodes at 52-73 tok/s over four streams and reads prompts at 1,420-1,540 tok/s:

| To produce | Tokens, at most | Path | Time on one pair |
|---|---|---|---|
| The full 1.6 million rollouts | 1.64 billion | decode | 260-365 days |
| The 100,000 of the ablations | 102 million | decode | 16-23 days |
| The same 102 million tokens | | prompt pass | 18-20 hours |

A prompt pass is about 20-30 times cheaper per token than decoding. The plan ranks every idea by which of the two it
needs.

## The approach

1. **Record, don't generate.** Run existing broad text through the target as prompts and record the rows a drafter
   reads. TensorFold's recipe for training a drafter reports that this generalised on another target.
2. **Start from the publisher's DFlash2 drafter**, not from zero. Its layers and head carry over by layout. Its
   dense entrance does not (see the survey), so the entrance is bridged by least squares, and re-initialised if the
   bridge is poor.
3. **Leave rejection memory for last.** It is the one part of DNeo that the paper says needs rollouts drafted by the
   drafter itself, and it has the smallest of the four measured effects.
4. **Measure offline before writing engine code.** A PyTorch instrument scores a checkpoint on recorded rows, under
   the limits it would be served with. The engine path is written only after a checkpoint beats both drafters the
   target has today.

The stages, the gates and the stopping rules are in [docs/plan.md](docs/plan.md).

## What the survey found

- The published DNeo checkpoint and GLM's DFlash2 drafter share the layout of their layers (15 tensors a layer),
  the final norm and two head tensors. DNeo replaces the dense entrance with one scalar a target layer.
- GLM's DFlash2 entrance mixes every coordinate: the diagonal share of each layer block is 0.024%, the chance level.
  Its layers were trained on those mixed coordinates, so the entrance cannot be carried over as it is.
- In TensorFold's GLM path, the rows of rejected drafts are already in the verify buffer the drafter reads. That is
  the input rejection memory needs.

Sources, revisions and the full numbers are in [docs/survey.md](docs/survey.md). Raw outputs are in
[receipts/](receipts/).

## Tools in this release

```sh
python -m pip install -e .                        # shapes, entrance and recording need numpy alone
python -m pip install -e ".[torch]"               # greedy and score compute with torch
open-dneo shapes incoai/GLM-5.3-Flash-DFlash2     # files, config and every tensor's shape
open-dneo entrance incoai/GLM-5.3-Flash-DFlash2   # the diagonal share of its dense entrance
open-dneo recording DIR                           # check what a server recorded with --record DIR
open-dneo greedy DIR --target CHECKPOINT          # how much of a recording its target would have written
open-dneo greedy DIR --target CHECKPOINT --save NEXT   # and its choice at every row, kept
open-dneo score DIR --drafter DRAFTER --target CHECKPOINT   # a DFlash2 drafter's tokens a round on it
open-dneo rows DIR --drafter DRAFTER --target CHECKPOINT --run RUN   # its drafts from prompt rows and reply rows
open-dneo probe DIR --target CHECKPOINT --next NEXT   # which taps hold the target's next tokens
open-dneo bridge DIR --drafter DRAFTER --out OUT   # DNeo's entrance fitted onto the drafter's layers
open-dneo score DIR --drafter DRAFTER --target CHECKPOINT --bridge BRIDGE   # and scored with it
```

`shapes` and `entrance` read a public Hugging Face repository by HTTP range requests, without credentials. `shapes`
reads only the safetensors header. `entrance` fetches one tensor (160 MiB in this example) and needs a few gigabytes
of memory.

`recording` reads the files a TensorFold server writes with `--record`: each stream's tokens, positions, kinds and
tapped rows, and the target's best tokens where a request asked for logprobs. It checks every finished stream
against what the recorder writes, totals the rows by layout, and exits with 1 if a stream is not sound. `--streams`
lists each stream. From Python, `open_dneo.recording.open_stream` gives a stream's rows as arrays.

`greedy` and `score` run on real weights: a recording, the target checkpoint's folder and, for `score`, the
drafter's folder. They read GLM-5.3-Flash's tensor names, and a head stored plain or as MXFP8. A checkpoint folder
needs its `config.json`, its index and the files that hold the embedding and the head.

`greedy` multiplies the rows after the final norm by the target's head and counts, for a recording's reply tokens
and its prompt tokens, how many are the token the target would write next. A reply decoded without sampling should
be that all the way, which checks the head, its orientation and the tap before a drafter is scored against them.
`--save NEXT` keeps the target's choice at every row of each stream counted, a little-endian int32 a row in
`NEXT/<stream>.next`, for counts by part or role and for `probe`.

`score` takes every reply row as an anchor and runs the DFlash2 drafter on the block serving would build there, with
the stream's earlier rows as its context. It cuts each chain of drafts where TensorFold's GLM lanes cut it and walks
each reply twice. The first walk takes every round as the drafter's. The second takes the rounds as serving takes
them for a stream decoded alone: where the stream's last tokens repeat eight or more of its earlier ones, a round
verifies a copy of what followed in place of the drafter's chain. `--served` scores with the weights as serving
holds them by default: the drafter's linears and the target's head at 4 bits, and the head's scores in bfloat16.
`--max-tokens` gives the requests' limit of new tokens, which bounds the rounds near the end of a reply it cut.

`--run RUN` reads a run file (`open_dneo.runs`): a JSON line a request, with the prompt's and the reply's token ids,
its `max_tokens` and serving's rounds. Each request is paired with the stream that recorded it and walked over its
whole prompt and reply, to its own limit, so a stream that begins after the prompt's start, where the server
resumed a stored prefix, still finds copies in the prompt's head. The report sets serving's tokens a round beside
the instrument's, class by class. With it, `--drafts-log LOG` reads the DFlash2 rounds a TensorFold server logs
with `TF_GLM_DFLASH2_LOG` and counts, class by class, the rounds whose first draft is the instrument's at the same
anchor, the drafts that agree from the first on, and the rounds whose every draft does.

The second walk is the one gate G0 in the [plan](docs/plan.md) compares with serving. With `--run` and the weights
as served it agreed within 0.013 in every class ([receipts/2026-11-20-g0](receipts/2026-11-20-g0/README.md)). It
still differs from serving in two things. Serving chooses the drafts a reply's first round holds from measured
round times; the report also gives the rounds if that round held one draft and if it held every draft. And without
`--run` the walk is over the tokens a recording holds, which can be one fewer than the reply's.

`rows` compares the two kinds of row a drafter can be trained on. Where a recording holds each reply of a run file
and then the same tokens sent again as one prompt, each reply position has a row from the reply's rounds and a row
from the prompt pass. `rows` pairs each request with both streams, reads both from the later of their first rows,
and has the drafter propose at the reply's positions from each kind of row. Per class it prints the drafter-alone
walk over each kind and how often the two agree on the first draft and on every draft. With the weights as served,
the walks differed by 0.004 to 0.029 tokens a round ([receipts/2026-11-20-rows](receipts/2026-11-20-rows/README.md)).

`probe` asks which taps of a recording hold what the target writes next. For each tap, and for sets of taps side
by side (`--set 6,15,25,34,43`), it fits a linear map with an intercept by ridge least squares, from the rows at a
position p to the target's final row at p + k (`--k`, by default 1, 2 and 3), on most of the streams. On the
streams held out, one in `--held-every` by name, it sends the mapped row through the target's head and counts how
often the best token is the target's own choice at p + k, as `greedy --save` kept it. It also gives the floor,
how often the commonest choice is right, and with `--parts` (a stream's name and its part a line) the scores by
part. The final tap at k 0 maps a row to itself and scores 1, which checks that the choices sit beside their rows.

`bridge` fits DNeo's entrance onto a DFlash2 drafter's layers. DNeo's entrance is a weighted sum of tapped rows;
DFlash2's sets its five taps' rows side by side and mixes them through `fc`. `bridge` fits a matrix B by ridge least
squares from the weighted sum to `fc`'s output, both divided by the size of `fc`'s output, on the rows of the folders
given, and holds out one stream in `--held-every` by name. It reports the share of `fc`'s output explained and the
cosine on the held rows, by part with `--parts`, and writes `bridge-LABEL.safetensors` (B, the weights and the taps)
for uniform weights, for weights that make the taps' sizes even, and for any given with `--weights`. `--inputs` lets
the sum read other taps than the drafter's, still fitted to `fc` on the drafter's own. `score --bridge FILE` then
scores the drafter with that entrance: DNeo's weighted sum, B, and DFlash2's norm, layers, mask and head as they are,
with B in float32 (`--bridge-served` rounds it to 4 bits with the rest). With `--run`, `score` also gives, class by
class, the walk in which the drafter makes every round, without copies. At step 0 the bridged drafter landed 0.853 to
0.962 of DFlash2's drafts a round ([receipts/2026-11-21-g1](receipts/2026-11-21-g1/README.md)).

The recording these commands need: tap k is the target's state after k layers, for GLM the mean of its four
residual streams, as its DFlash2 drafter reads them, and the tap numbered as the layer count is the row after the
final norm. `greedy` and `probe` read that last tap, 45 for GLM-5.3-Flash. `score` and `rows` read the tap one above each of
the drafter's `target_layer_ids`: 6, 15, 25, 34 and 43 for the drafter above, or with `--bridge` the taps the
bridge names. TensorFold's `--record` gives no rows for
GLM yet, and the recorder for GLM that this project uses is not published yet. What its first check showed is in
[receipts/2026-11-19-recorder-check](receipts/2026-11-19-recorder-check/README.md).

## Credits and licences

- DNeo: Sandler et al., [arXiv:2611.21925](https://arxiv.org/abs/2611.21925). Reference inference code:
  [J-sandler/DNeo](https://github.com/J-sandler/DNeo) (MIT). Weights:
  [J-sandler/Qwen3-4B-DNeo](https://huggingface.co/J-sandler/Qwen3-4B-DNeo) (Apache-2.0).
- DFlash: [z-lab/dflash](https://github.com/z-lab/dflash) (MIT). GLM's DFlash2 drafter:
  [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (Apache-2.0).
- [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0): the engine, its recorder and its recipe for
  training a drafter.

This release contains none of their weights and copies none of their files. Some modules were written to compute
what their code computes; [NOTICE](NOTICE) names each module and its source. Open-DNeo is licensed under
[Apache-2.0](LICENSE). It is not affiliated with the DNeo authors, Cerebras, incoai, Z.ai or the TensorFold
maintainers.
