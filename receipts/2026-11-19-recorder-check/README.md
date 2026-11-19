# Recorder check receipts (2026-11-19)

Ten launches of a TensorFold server with this project's row recorder for GLM, and what they show. This is the check
behind the first of Stage 0's exit conditions in [docs/plan.md](../../docs/plan.md): replies are the same tokens with
and without recording.

The recorder is one commit over TensorFold `e3fb56cb09061a979cb02080601fc6f7870d3362`, in a branch that is not
published yet. It writes what `open-dneo recording` reads. For GLM, tap k is the mean of the four residual streams
after k layers, and tap 45 is the row after the final norm.

## What ran

Two GB10 machines at TP=2 served `Bizuayeu/GLM-5.3-Flash-Jammed-XSFP4` at revision
`0a22bb46ecef3d0f5f96e41dc6011be12974fbf1`. Drafts came from the checkpoint's own MTP head and the lanes' copies: the
server's startup line reads `drafts: MTP + copies`. No DFlash2 drafter was loaded, so no number here is one that
gate G0 compares.

| Launch | Records | Prompt pass | Taps | Lists |
|---|---|---|---|---|
| recb | no | default | | R, S |
| recp | no | `TF_GLM_PROMPT_PAIR=0` | | S |
| recg | no | `TF_GLM_PREFILL_REDUCE=gather`, `TF_GLM_PROMPT_PAIR=0` | | S |
| recf | yes | default | 45 | R, S |
| rect | yes | `TF_GLM_PREFILL_REDUCE=gather` | 6, 15, 25, 34, 43, 45 | R, F, S |
| recu | yes | as rect | as rect | R |
| recv | yes | as rect | 6, 25, 45 | R |
| recg2, recp2, recb2 | no | as recg, recp and recb, in that order | | S |

Recording a layer's tap at TP=2 needs the gather path, and a recording launch takes no paired prompt steps.

- **R**: four prompts of 257 to 341 tokens, each sent as a plain completion with drafts and with `"draft": false`,
  for 48 tokens at temperature 0.
- **F** (rect only): the tokens each drafted R stream recorded, its prompt's and its reply's, sent again as one prompt
  of token ids. A reply row and the row at the same position of this prompt are the same tokens computed two ways.
- **S**: eight prompts of 1,986 to 2,584 tokens, for 1 token. Their prompt passes are timed.

## Files

`SHA256SUMS` covers the six outputs. The recordings (1.4 GB) and the replies are not in this repository. `REC` is the
folder of each launch's replies and `OUT` the folder of the recordings, one folder a launch.

```sh
python judge.py REC OUT > judge.txt
open-dneo recording OUT --streams > reader.txt
python copies.py OUT/rect > copies.txt
python twins.py OUT/rect > twins.txt
open-dneo greedy R --target CHECKPOINT --streams > greedy.txt        # R: rect's streams 0 to 7, in a folder r
open-dneo greedy F --target CHECKPOINT --streams > greedy-again.txt  # F: rect's streams 8 to 11, in a folder f
```

`judge.py`, `copies.py` and `twins.py` are here as they ran, on a machine with open-dneo installed. The two
`greedy` runs were on an aarch64 machine, with the checkpoint above as `CHECKPOINT` and this release's code as it
stood before the release was numbered.

## What they show

- **The gates pass** (`judge.txt`, last line `judge: 0 failures`). A reply's tokens are the same with drafts and
  without, and the same in every launch as in recb, which neither records nor leaves the default prompt path. Each
  recording passes the reader's checks and holds one closed stream a request. The same launch twice writes the same
  bytes. Launches with different taps write the same bytes for the taps they share. A reply's rows are near the same
  positions of its F prompt (relative L2 0.0887 at the median) and far from the next position (0.9835).
- **The recorded rows are the target's.** The row after the final norm, multiplied by the target's head, gives the
  token the reply went on with for 375 of 377 reply tokens (`greedy.txt`). The two that differ are the same token
  of one reply, which was asked for twice.
- **Recording costs prompt-pass speed.** Six taps on the gather path computed 0.9246 of the default path's prompt
  tokens a second: the median over the eight S prompts, each prompt's speed taken over its mean in recb and recb2
  (`judge.txt`). The gather path alone is at 0.9458 and 0.9476. Recording the final tap alone, on the default path,
  is at 0.9984. One launch each, on prompts of about 2,000 tokens.
- **A reply's rows.** A reply of n tokens leaves n - 1 rows, since its last token is never read, and n rows when the
  token limit ended it inside a round (`judge.txt`: 47 and 48 rows for 48 tokens).
- **A reply's rounds differ between launches** of one binary on the same prompts: the same 48 tokens of the code
  prompt took 18 rounds in two launches and 16 in two others (`judge.txt`). These are MTP's rounds.
- **Where a copy could take a round** (`copies.txt`): at 68 of 377 reply rows the stream repeats eight or more of its
  earlier tokens with at least two to copy, 27 of 47 in the prose reply and none in the code and list replies.
- **Reply rows are not prompt-pass rows** (`twins.txt`). The same position computed as a reply's row and as a
  prompt's differs by 0.3% at tap 6 and 1.7% at tap 15, and by 9 to 11% at taps 25, 34, 43 and 45 (relative L2, the
  median over 185 rows). Each path gives the same bytes every time, so the difference is between the paths. The
  head picks the same tokens from both: from the F prompts' rows it gives 47 of 47, 48 of 48, 46 of 47 and 47 of 47
  of the four replies' tokens (`greedy-again.txt`: each F stream's count less its R stream's prompt count in
  `greedy.txt`), the same as from the replies' own rows. TensorFold's recipe for training a drafter reports about
  1 to 5% on other targets and that both kinds of row are fine to train on
  (`docs/recipes/training-a-drafter.md` at the commit above). What the larger difference here does to a drafter,
  which is trained on prompt-pass rows and served reply rows, is for Stage 1 to measure.
