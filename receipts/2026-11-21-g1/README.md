# G1: the bridge's step-0 score, and the taps of Stage 2's entrance (2026-11-20 and 2026-11-21)

Stage 1 in [docs/plan.md](../../docs/plan.md) scores the bridged drafter at step 0, before any training: DNeo's
weighted sum of taps, a matrix B fitted to DFlash2's entrance, then DFlash2's norm, layers, mask and head as they are.
G1 is read as drafts landed a round, tokens a round less the one the target adds to every round. This receipt holds
two sets of numbers on the evaluation set's open half: the bridge on DFlash2's own taps (2026-11-20, `evd1/`), and
the bridge on the five deepest taps beside it (2026-11-21, `evd3/`), which decided the taps Stage 2's entrance reads.

## What ran

- **The fits.** `open-dneo bridge` fits B by ridge least squares on recorded prompt-pass rows of the pilot
  ([receipts](../2026-11-20-pilot/README.md)), from the weighted sum of the taps it is given to the output of
  DFlash2's `fc` on taps 6, 15, 25, 34 and 43, both divided by the target's size. One stream in ten is held out;
  every report gives the share of the target explained and the cosine on the held rows, by part.
  - `fit-pr1-pa1.txt` (2026-11-20): DFlash2's taps, on the nine parts at 18 taps and the agent part, with three
    choices of weights (uniform, `fc`'s block sizes, and sizes made even).
  - `fit-pr1.txt` and `fit-pr1-deep.txt` (2026-11-21): on the nine parts alone (the agent part has no tap 42), DFlash2's
    taps and taps 40, 41, 42, 43 and 44 (`--inputs`), the target still DFlash2's `fc`.
- **The scores.** `open-dneo score --served --run` walks each of the 320 open prompts' replies under serving's
  limits, with the drafter's linears and head at 4 bits and B in float32 (or at 4 bits, `--bridge-served`).
  - `evd1/` (2026-11-20) on G0's judged recording, taps 6, 15, 25, 34, 43 and 45: DFlash2 as it is, the uniform and
    `fc` bridges, the uniform bridge at 4 bits, and a bridge file whose matrix is zero (`null`, the entrance reading
    nothing).
  - `evd3/` (2026-11-21) on a new recording of the same 320 prompts at ten taps (6, 15, 25, 34 and 40 to 45),
    `recording-evd3.txt` its check: DFlash2 as it is, the deep bridge (`deep.txt`), DFlash2's taps fitted on the nine
    parts (`pr1.txt`), and the bridge of 2026-11-20 (`pr1-pa1.txt`).
- **evd3 was served by MTP, not by DFlash2.** It was recorded without `--drafter`, so in its reports serving's tokens
  a round, and how far the instrument is "apart" from them (0.021 to 1.068), are MTP's: they compare nothing here.
  G0's comparison is evd1's. Only the instrument's lines are read in `evd3/`.
- **The bridge of 2026-11-20 on evd3.** It was fitted before bridge files held their taps, so it was written again
  with the taps its sum read (6, 15, 25, 34 and 43), its matrix and weights byte for byte the same, before
  `pr1-pa1.txt` was scored.

`SHA256SUMS` covers the reports and the script. The recordings, the prompts, the run files and the bridge files are
not in this repository: the recordings hold the text.

```sh
open-dneo bridge PR1 PA1 --drafter DRAFTER --out FIT --weights fc=W --parts PR1_PARTS --parts PA1_PARTS > fit-pr1-pa1.txt
open-dneo bridge PR1 --drafter DRAFTER --out DFL --parts PR1_PARTS > fit-pr1.txt
open-dneo bridge PR1 --drafter DRAFTER --out DEEP --parts PR1_PARTS --inputs 40,41,42,43,44 > fit-pr1-deep.txt
open-dneo score EVD1 --drafter DRAFTER --target CHECKPOINT --served --run EVD1_RUN > evd1/dflash2.txt
open-dneo score EVD1 --drafter DRAFTER --target CHECKPOINT --served --run EVD1_RUN --bridge FIT/bridge-uniform.safetensors > evd1/uniform.txt
open-dneo score EVD1 ... --bridge FIT/bridge-uniform.safetensors --bridge-served > evd1/uniform4.txt   # fc.txt, null.txt alike
open-dneo recording EVD3 > recording-evd3.txt
open-dneo score EVD3 --drafter DRAFTER --target CHECKPOINT --served --run EVD3_RUN --bridge DEEP/bridge-uniform.safetensors > evd3/deep.txt
python table.py evd1 uniform fc null uniform4
python table.py evd3 deep pr1 pr1-pa1
```

## What they show

- **The bridge carries most of DFlash2's entrance** (`evd1/`). Drafts landed a round, as a share of DFlash2's:

  | Class | uniform | `fc` weights | uniform, B at 4 bits | null (B = 0) |
  |---|---|---|---|---|
  | code | 0.928 | 0.936 | 0.925 | 0.002 |
  | agent | 0.853 | 0.838 | 0.841 | 0.000 |
  | chat | 0.962 | 0.960 | 0.959 | 0.006 |
  | ja | 0.929 | 0.924 | 0.931 | 0.001 |
  | prose | 0.947 | 0.933 | 0.944 | 0.014 |

  Above G1's 70% in every class, so arm A leads. The weights of the sum and B's rounding to 4 bits move the share by
  0.015 at most. Tokens a round give higher shares, because a round of drafts that all miss still makes one token:
  the null bridge's share of tokens a round with copies is 0.24 to 0.51.
- **DFlash2's numbers did not move.** Walked by the bridge's path, DFlash2 as it is gives G0's tokens a round with
  copies and the rows receipt's drafter-alone tokens a round, digit for digit.
- **evd3 reproduces evd1.** On evd3, DFlash2's and the 2026-11-20 bridge's instrument lines are those of evd1, every
  one, the serving clauses aside: the shared taps hold the same rows.
- **The deep five against DFlash2's five** (`evd3/`), drafts landed a round as a share of DFlash2's, both fitted on
  the nine parts:

  | Class | DFlash2 | deep (40 to 44) | share | DFlash2's taps | share |
  |---|---|---|---|---|---|
  | code | 4.018 | 3.633 | 0.904 | 3.720 | 0.926 |
  | agent | 2.431 | 2.285 | 0.940 | 2.114 | 0.869 |
  | chat | 1.402 | 1.336 | 0.953 | 1.350 | 0.963 |
  | ja | 1.089 | 1.031 | 0.946 | 1.011 | 0.928 |
  | prose | 1.034 | 0.994 | 0.961 | 0.980 | 0.948 |

  The deep five are ahead in agent (by 0.071), Japanese and prose, and behind in code (by 0.022) and chat. Their fit
  explains less of DFlash2's entrance (0.920 of the held rows against 0.939), as a sum without taps 6 to 34 would.
- **The rule, and what it chose.** Written on 2026-11-20, before any evd3 number: on the same recording, with both
  uniform bridges fitted on the nine parts alone, if the deep five's share is above DFlash2's five's in four classes of five or more (an equal share not counting), and below it
  by more than 0.02 in none, Stage 2's entrance reads the deep five; otherwise DFlash2's five. The 0.02 was a
  placeholder. The deep five lead in three classes, and fall by 0.022 in code: **Stage 2's entrance reads DFlash2's
  five taps**.
- **What this does not say.** Step 0 reads B through DFlash2's layers as trained, on rows B was fitted to make look
  like DFlash2's entrance, which favours the taps that entrance was trained on. Training moves the layers and B.
  After these numbers were seen, the first training run was set to compare four sets of taps
  ([docs/plan.md](../../docs/plan.md)). The shares are exact for these 64 prompts a class: the instrument gives the
  same lines on two recordings. How far they would move on other prompts was not measured.
