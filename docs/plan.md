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

Move 1 leaves a gap of its own. A drafter is trained on rows of a prompt pass, and it is served rows that verify
passes computed over the cache. In the recorder's check the same positions computed both ways lay 9 to 11% apart
at the four deeper taps, and the target's head picked the same tokens from both
([receipts](../receipts/2026-11-19-recorder-check/README.md)). TensorFold's recipe reports about 1 to 5% on other
targets and trains on both kinds of row. Stage 1 measures what the gap does to a drafter's proposals, before the
pilot recording. Measured on 2026-11-20 with the published DFlash2 drafter, on the evaluation set's open half: from
prompt rows it walks 0.004 to 0.029 fewer tokens a round than from reply rows of the same tokens, by class, and its
first draft is the same at 95.7 to 99.2% of positions ([receipts](../receipts/2026-11-20-rows/README.md)). That
drafter was trained on its publisher's data, not on prompt rows, and the numbers come from one launch with the
weights as serving holds them.

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
| 0 Tools | In a fork only: recording rows from GLM (prompt and reply rows, selectable tap layers, and the row after the final norm, from which the target's best k are computed offline). The offline instrument (DFlash2 and DNeo forwards in PyTorch, under serving's limits). The training script | To be able to measure | Replies have equal `token_sha` with and without recording. The instrument's forwards give the outputs of the public definitions | 3-4 days of development |
| 1 Pilot | An evaluation set (code, agent turns, chat, Japanese, prose; split from training by task; MTP's and DFlash2's tokens a round and round time taken first). G0 on it, before anything else is scored. DFlash2's proposals from prompt-pass rows against those from reply rows of the same tokens. A pilot prompt-pass recording of 0.5-1 million rows over a wide choice of layers. The layer probe. The bridge's step-0 score. Disk and training speed | To turn the bet into numbers | **G0**, then **G1** | 1-1.5 days |
| 2 Training | The main prompt-pass recording (45 million rows as a first size) and training. Arm A: DNeo's form, from DFlash2. Arm C: DFlash2 as it is, the control. Arm B if G1 is poor: DNeo's form with the entrance re-initialised. If the first run misses G2, a second run mixes in 1-2 million tokens of the target's own output | To decide whether a winning checkpoint exists | **G2** | Recording 8.1-8.8 hours of prompt passes, a night to a day with cooling pauses. Training: from Stage 1's measurement |
| 3 Engine | A DNeo path in TensorFold's GLM family, without memory: loading, taps, context intake (the entrance matrix goes away), the mask, the head's chain. The ring, the convolutions, the split MLP and the 4-bit linears are reused | To serve it | The engine's drafts equal the PyTorch reference's at bf16. Replies have the `token_sha` of undrafted, alone and serial runs. Upstream's own gates | 4-6 days of development |
| 4 Served | One binary, launches in A-B-B-A order | The verdict that matters | **G3** | 0.5-1 day |
| 5 Memory (optional) | Rejection memory: the engine path, rejected rows in the recording, 3 million tokens drafted by DNeo itself, a finishing run | To take the remaining effect | **G4** | 3-4 days |
| 6 Release | Weights under Apache-2.0 with notices, and a proposal upstream, starting with an issue | To share it | | |

A number that says whether a winning checkpoint exists (the end of Stage 2) is about 6-8 days out, and the served
verdict about 11-15. The recording times assume the prompt-pass speed of the default path. Recording takes its rows
by another path, which computed 0.925 of the default path's prompt tokens a second in a first check, on prompts of
about 2,000 tokens ([receipts](../receipts/2026-11-19-recorder-check/README.md)). At that rate the recording takes
about 8% longer.

Stage 0's two exit conditions were met on 2026-11-19. The recorder's check is in the same receipts, and the tests
hold the instrument's forwards to the outputs of the public definitions (`tests/test_drafter.py`). The stage is not
finished: its training script is still to be written.

Stage 1 so far, on 2026-11-20: G0 passed, prompt rows were compared with reply rows, and a pilot of 609,309
prompt-pass rows was recorded and probed ([receipts](../receipts/2026-11-20-pilot/README.md)). The bridge's step-0
score, the disk and the training speed are still to be reported.

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

### How G0 is measured

Fixed on 2026-11-19, before the evaluation set existed and before either of the numbers it compares was taken. Made
exact later that day, after the set was built and before serving's first number on it was read: the set's sealed
half, the route the prompts take, which launch is judged and what its second launch differs in, and the walk over
whole replies.

- **The set.** 64 prompts in each of five classes (code, agent turns, chat, Japanese chat, prose), drawn by fixed
  rules from public datasets, and as many again drawn alike and sealed. G0, G1 and comparisons during training read
  the open half. The sealed half is not served, scored or read before G2, which is read on it.
- **Serving.** The evaluation prompts go one at a time, decoded greedily, to a server started with `--drafter`
  naming the DFlash2 drafter at its default 4 bits. Each prompt is its messages under the chat template with thinking
  off, sent as token ids to the completions route where those ids are the ones the chat route runs for the same
  messages, and to the chat route where they are not. `TF_GLM_DFLASH2_LOG` is set high enough to log every round's
  drafts. `TF_GLM_COPIES` stays unset, which keeps the opt-in root and context copies off, and
  `TF_GLM_DRAFT_CONFIDENCE` stays unset, which leaves the chain's cut at 0.7. A launch whose log reports a restarted
  DFlash2 context is not counted.
- **A round is not always the drafter's.** Where a stream's last tokens repeat a run of at least 8 tokens from
  earlier in the stream, and at least 2 tokens can be copied from what followed that run, the round verifies the
  copy in place of the drafter's chain (`copyProposal` and `plan` in TensorFold's `zig/src/core/lanes/windows.zig`).
  A reply's round count takes in every round, whatever drafted it.
- **Serving's tokens a round.** A reply's first token comes from the prompt pass, and no round gives it. The
  reply's `speculative.tokens_per_round` divides all of its tokens by its rounds, so it is not used as it is: a
  reply counts `completion_tokens - 1` tokens over `speculative.rounds` rounds. A class's number is its replies'
  tokens summed, over their rounds summed.
- **Which launch is judged.** The first counted launch with the drafter that serves the whole open half and records
  its rows. Where that launch is not counted, or does not serve every prompt, it is launched again, and no launch
  without rows is judged in its place. A launch without the drafter runs before it, to find which route gives the
  chat route's ids, and is not counted. A reply's rounds depend on times the engine measures, so they need not be the
  same in every launch. The same binary is launched once more with the drafter on the same prompts, and how far its
  number is from the first is reported. That second launch records no rows, so the round times it measures can
  differ for that reason too.
- **The instrument's tokens a round.** The rows recorded for the same replies are scored with the weights as
  serving holds them: the drafter's linears and the head at 4 bits, and the head's scores in bfloat16. Each reply
  is walked over all of its tokens, a round being a copy where serving's rule gives one and the drafter's chain
  elsewhere. The walk is given the request's `max_tokens`, which bounds a copy and the drafts a round asks for near
  that limit. A class's number is again tokens summed over rounds summed.
- **Whole replies.** `open-dneo score --served --run RUN` pairs each request of a run file (the prompt's and the
  reply's token ids, `max_tokens`, serving's rounds) with the stream that recorded it, and walks the request's whole
  prompt and reply to its own `max_tokens`. A stream can begin after the prompt's start, where the server resumed a
  stored prefix, so the prompt the walk looks for copies in is the run file's. It prints serving's number beside the
  instrument's, class by class. The walk's rounds are the lane core's own on 16 cases (`tests/test_lane_core.py`).
- **Pass.** In every class the two numbers differ by 0.1 or less.
- **Result (2026-11-20).** It passes: the largest gap is 0.013, in the agent class, with no fix
  ([receipts/2026-11-20-g0](../receipts/2026-11-20-g0/README.md)).
- **Reported, not judged.** The instrument's number with the weights as stored, how often the drafts serving
  logged at a position are the instrument's at that position, and serving's number in its second launch.
- **A fix** is a change to how the instrument copies serving, made after its number missed. Three are allowed.
  Correcting a misreading of the data, such as the wrong tensor or a reply paired with another's rows, is not a
  fix; it is reported with the result.

Where the instrument is known to differ from serving, each a candidate for a fix:

- The drafts a reply's first round holds. Serving chooses their number from measured round times. The instrument
  holds 4, which an engine that has measured nothing chooses. It is one round a reply, and the instrument's rounds
  with a first round of 1 draft and of every draft, the least and the most serving can choose, are reported beside
  it.
- The instrument's products are float32. Serving keeps its rows in bfloat16.
- Candidates with equal scores at the sixteenth place may be chosen differently, and bfloat16 scores make such ties
  likelier.

Serving also sizes a drafter's window by what its rows cost, and the instrument stops at the chain's cut. At the
default cut that is no difference. A lone stream's round is priced at 76 ms and 4.1 ms a row more (`shared_costs` in
TensorFold's `zig/src/families/glm/cuda_copy_policy.zig`), so a draft is kept where its chance is over 4.1/76 of the
tokens expected without it, which are 1 for the first draft and 7 at most. A first draft's chance is 1/16 at the
least, being the best share among 16 candidates, and a later draft that the cut kept has 0.7 or more. The first
draft's margin is thin, 4.1/76 being 0.054 and 1/16 being 0.0625, so both prices were read again in the source of
the server G0 ran on, TensorFold `e3fb56cb09`, whose prices the recorder's commit leaves as they are: they are the
same.

Checked on 2026-11-20: one launch records rows and serves DFlash2 drafts together, and its replies are the same
tokens as those of launches that record nothing, so the rows G0 read came from the judged launch itself.

## Decisions so far (2026-11-20)

- Rows are recorded from `Bizuayeu/GLM-5.3-Flash-Jammed-XSFP4`, the checkpoint served at TP=2. Other GLM-family
  checkpoints are scored on the evaluation set only.
- The engine's DNeo path (Stage 3) waits for G2.
- Training text is public datasets whose licences have been checked. Private logs never train a published
  checkpoint.
- A reply in the training text is written by people under a permissive licence, by an open-weight model published
  under MIT or Apache-2.0, or by the target itself. Each item keeps its dataset, revision, row and licence (fixed on
  2026-11-20, before the pilot recording).
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
