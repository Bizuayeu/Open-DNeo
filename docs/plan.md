# Plan

What this project is trying to reach, in what order, and what decides each step. The facts behind it, with their
sources, are in [survey.md](survey.md). Durations are estimates until a stage reports its own.

## Goal

- `--drafter <a DNeo folder>` serves a GLM-5.3-Flash checkpoint at TP=2 on two GB10 machines with DNeo drafts.
  Replies are token for token what the target writes without drafts. Decoding is faster than with the better of the
  two drafters the target has today, on the same prompts (the gates below).
- The gain is not known. The paper reports 19-27% more tokens a round than DFlash-2 at temperature 0 on Qwen3-4B, a
  dense 4B model on one H100. Nobody has tried DNeo on a MoE target, a 4-bit target or one split across machines.
- The bar is not DFlash2 alone. GLM carries its own MTP head, which is the default drafter here: its acceptance
  length on our serving checks is 3.2, 2.0 and 3.3 tokens on count, prose and code.
- Not in scope yet: other model families and backends, tuning for temperature 1, and block 16. The first version
  keeps GLM's block of 8, since on a MoE target every extra verify row brings its own experts.

## DNeo's parts and what each costs

| Part | Effect in the paper when removed (tokens a round, 100,000 rollouts) | Training data it needs | Engine work |
|---|---|---|---|
| A non-mixing entrance (MoFA) and the choice of target layers | -0.651 | prompt-pass rows are enough | which layers are tapped, how context rows are taken in |
| Two-tap convolutions | -0.331 | none new | already in DFlash2 |
| The fused causal head | -0.217 | prompt-pass rows are enough | the head's chain, over the whole vocabulary or the best k |
| No attention between draft positions | -0.004 to -0.027, that is, with the attention put back; its main gain is a cheaper round | none new | the attention mask |
| Rejection memory | -0.182 | **rollouts drafted by the drafter itself, at decode cost** | the rows are already in the verify buffer; a path to the next block |

## Four moves away from the paper's recipe

| # | The recipe assumes | We do instead | Why | What we don't know |
|---|---|---|---|---|
| 1 | The target writes the training data | Run existing broad text through the target as prompts and record the rows. Add a little of the target's own output only if that falls short | A prompt pass is about 20-30 times cheaper per token. TensorFold's recipe reports it generalised on another target | Whether it carries over to GLM |
| 2 | Train from zero | Start from the publisher's DFlash2 drafter. Keep its layers and head, bridge the entrance by least squares | The layouts match, see the survey | How good the bridge is. If poor, re-initialise the entrance and the key and value projections, and keep the rest |
| 3 | All parts at once | No rejection memory in the first version | It is the only part that needs on-policy rollouts, and the smallest effect | How far the rest gets without it |
| 4 | Port to the engine, then measure | Score checkpoints offline first, under serving's limits (block 8, 4-bit linears, a best-k head), and write engine code only for a checkpoint that wins | Precedent in TensorFold issues #1485 and #2592 | Whether the instrument agrees with serving (gate G0) |

One rule of ours for the labels: within a block, the text's tokens are the target's own greedy continuation up to
the first position where the text and the target's argmax differ. Positions after that get a lower weight. Whether
this helps is compared in Stage 2.

## Not doing

| Idea | Why not |
|---|---|
| The paper's recipe, at 1.6 million or 100,000 rollouts | 260 days or more on one pair, and 16 days or more for the small set |
| Renting GPUs to run the target | Move 1 removes the need, and the recordings are hundreds of gigabytes to move |
| Fine-tuning DFlash2 as the goal | An adapted drafter did not transfer to unseen prompts in TensorFold issue #2667. We run it only as the **control arm**: it needs no engine change, so if it wins as much, no engine code is written |
| Copying the paper's layer formula and block 16 | GLM's taps are defined differently (the mean of four residual streams). A probe on a pilot recording picks the layers |
| Proposing anything upstream first | There are no weights and no numbers yet |

## Stages

| Stage | What | Why | Exit | Estimate |
|---|---|---|---|---|
| 0 Tools | In a fork only: recording rows from GLM (prompt and reply rows, selectable tap layers, the target's best k per row). The offline instrument (DFlash2 and DNeo forwards in PyTorch, under serving's limits). The training script | To be able to measure | Replies have equal `token_sha` with and without recording. **G0** | 3-4 days of development |
| 1 Pilot | An evaluation set (code, agent turns, chat, Japanese, prose; split from training by task; MTP's and DFlash2's tokens a round and round time taken first). A pilot prompt-pass recording of 0.5-1 million rows over a wide choice of layers. The layer probe. The bridge's step-0 score. Disk and training speed | To turn the bet into numbers | **G1** | 1-1.5 days |
| 2 Training | The main prompt-pass recording (45 million rows as a first size) and training. Arm A: DNeo's form, from DFlash2. Arm C: DFlash2 as it is, the control. Arm B if G1 is poor: DNeo's form with the entrance re-initialised. If the first run misses G2, a second run mixes in 1-2 million tokens of the target's own output | To decide whether a winning checkpoint exists | **G2** | Recording 8.1-8.8 hours of prompt passes, a night to a day with cooling pauses. Training: from Stage 1's measurement |
| 3 Engine | A DNeo path in TensorFold's GLM family, without memory: loading, taps, context intake (the entrance matrix goes away), the mask, the head's chain. The ring, the convolutions, the split MLP and the 4-bit linears are reused | To serve it | The engine's drafts equal the PyTorch reference's at bf16. Replies have the `token_sha` of undrafted, alone and serial runs. Upstream's own gates | 4-6 days of development |
| 4 Served | One binary, launches in A-B-B-A order | The verdict that matters | **G3** | 0.5-1 day |
| 5 Memory (optional) | Rejection memory: the engine path, rejected rows in the recording, 3 million tokens drafted by DNeo itself, a finishing run | To take the remaining effect | **G4** | 3-4 days |
| 6 Release | Weights under Apache-2.0 with notices, and a proposal upstream, starting with an issue | To share it | | |

A number that says whether a winning checkpoint exists (the end of Stage 2) is about 6-8 days out, and the served
verdict about 11-15.

## Gates

Written before any number is seen. The evaluation prompts train nothing and are split from training by task. Every
gate is read per class, Japanese included.

- **G0, the instrument.** Its tokens a round for DFlash2 are within 0.1 of serving's on the same prompts, in every
  class. Nothing proceeds until it holds.
- **G1, the starting point.** Not pass or fail. If the bridged checkpoint scores 70% or more of DFlash2 at step 0,
  arm A leads. Below that, arms A and B run side by side for the first 5 million tokens and the better one stays.
- **G2, winning offline.** Under serving's limits, on code or agent turns, arm A (or B) does both of these, and no
  class falls by more than 1%. (i) 5% or more tokens a round than DFlash2. (ii) 5% or more estimated speed than MTP,
  where estimated speed is tokens a round over the round's time from serving's logs, draft time included. MTP and a
  block drafter pay different prices a round, so token counts alone don't compare them. The gap to arm C is reported
  too: if the control gains as much, the gain is not DNeo's.
- **G3, winning served.** Decode tok/s is 5% or more above today's better drafter on code or agent turns, with no
  class more than 1% slower and replies unchanged.
- **G4, memory.** 3% or more tokens a round over the first version. Otherwise it stays out.

Where the numbers come from: the 5% and 1% in G2 and G3 follow the gate TensorFold issue #2667 used, and 45 million
rows is the size TensorFold's recipe reports. G0's 0.1, G1's 70% and 5 million tokens, G4's 3%, Stage 5's 3 million
tokens and the 1-2 million tokens of the target's own output are placeholders. Stage 1 replaces them with values
from what it measures: the spread of the evaluation set and the training speed.

Stopping: if G0 fails after three fixes, or G2 is missed again by the second run, the work stops and the result is
published as it is. If only arm C wins, the outcome is a better DFlash2 checkpoint and no engine change.

## Decisions so far (2026-11-19)

- Rows are recorded from `Bizuayeu/GLM-5.3-Flash-Jammed-XSFP4`, the checkpoint served at TP=2. Other GLM-family
  checkpoints are scored on the evaluation set only.
- The engine's DNeo path (Stage 3) waits for G2.
- Training text is public datasets whose licences have been checked. Private logs never train a published
  checkpoint.
- Every push to this repository is a release with a version. Documents are in English, with the README also in
  Japanese.

## Constraints

- **Heat.** These machines must not run long prompt passes without a break. Recording uses short documents with
  pauses, which is why its wall time is longer than the arithmetic.
- **Disk.** A row is 8,192 bytes a tap. Three taps over 45 million rows is 1.1 TB, and one mixed tap is 369 GB. The
  taps are chosen in Stage 1 and the row count is computed first.
- **Memory.** The target and the trainer cannot share one 128 GB machine. Recording uses the pair and training a
  third machine.
- **Upstream.** TensorFold's GLM lane file is at its 600-line limit, so DNeo wiring there has to move something out
  first. Nothing is proposed upstream before weights and numbers exist.
- **Licences.** A released checkpoint carries notices for the DFlash2 weights it starts from and the DNeo code it
  follows, and a manifest of its training datasets.
- **Training code.** DNeo's is private, so the training script here is written from the paper and the released
  model definition.
