# Stage 2's end: G2's price of the trained drafters (2026-11-23 to 2026-11-26)

Stage 2 in [docs/plan.md](../../docs/plan.md) trains drafters and asks whether one wins. These receipts hold the
runs that decided where it ended: drafters in DNeo's form (arm A), DFlash2 trained further (arm C, the control) and
a halfway form, each read on the evaluation set's open half at G2's time of a round. No drafter in DNeo's form
reached DFlash2's predicted speed. By a line written before the last numbers, development stopped there.

## What was trained

Every drafter starts from the published DFlash2 drafter (`incoai/GLM-5.3-Flash-DFlash2`, revision
`6cc6c481eee930a5139619ed919f78726371adbc`) and is trained with `open-dneo train` on rows recorded from prompt
passes. Its labels are the target's own greedy choices at those rows (`greedy --save`).

- **The rows.** Recorded by the server of the [G0 receipts](../2026-11-20-g0/README.md), on two GB10 machines at
  tensor parallel 2, from `Bizuayeu/GLM-5.3-Flash-Jammed-XSFP4` at revision
  `0a22bb46ecef3d0f5f96e41dc6011be12974fbf1`, at taps 6, 15, 25, 34, 43 and 45. Each drafter was trained on one
  GB10 machine. `runs.txt` gives each run's streams, those held out to measure it, and its parts' shares.
- **The text** comes in two kinds:
  - *The target's own replies* (parts `chat-glm`, `code-glm`, `ja-glm`): replies GLM-5.3-Flash wrote greedily, with
    thinking off and up to 1,024 tokens, from a checkpoint built from the publisher's MIT-licensed weights. The
    prompts are user turns from `OpenAssistant/oasst2`, `nvidia/OpenCodeInstruct` and `llm-jp/magpie-sft-v1.0`
    (Japanese), at the revisions in the [pilot receipts](../2026-11-20-pilot/README.md). There are two sizes: 267
    replies (241 streams trained, 26 held) and all 1,199 (1,080 and 119).
  - *A public set written by the same model* (parts `chat-hv`, `code-hv`, `chat-hv2`, `code-hv2`):
    `Zek-Takai/glm53-flash-harvest` (MIT), revision `f912e86060f4b928ef2421d1398f3ea33c79c701`, round 1, written
    from the publisher's FP8 weights with thinking off, at temperatures 0.6 and 0.8. `chat-hv2` holds its chat,
    prose, structured, music and operator replies, and `code-hv2` its code: 21,838 replies, cut at 1,024 tokens
    (`public-set.txt`). The `hv2` arms take nested sets of them (`public-sets.txt`).
    Before recording, its prompts were checked against the messages of both halves of the evaluation set (the
    same text, or 50 characters in common). The check printed counts only: 0 rows overlap with either half
    (`overlap.txt`).
- **The arms.** A drafter's form is one of three:
  - *DNeo's form* (arm A): DNeo's entrance (a weighted sum of taps 6, 15, 25, 34 and 43, through the uniform
    bridge on DFlash2's taps fitted on the pilot's nine parts, `fit-pr1.txt` in the
    [G1 receipts](../2026-11-21-g1/README.md)), DNeo's mask and DNeo's head over the whole vocabulary. The rate
    rises to 2e-4 and falls to 2e-5.
  - *The bridge alone* (`form0-*`): the same entrance, with DFlash2's mask and head. The same rates.
  - *Arm C, the control* (`own-c`, `own-cb`): DFlash2 as it is, at a tenth of the rates, 2e-5 to 2e-6.

  | Label | Form | Text | Steps |
  |---|---|---|---|
  | `own-3`, `own-3s1` | DNeo's | the 267 own replies; seeds 0 and 1 | 200 |
  | `own-3b`, `own-3bs1` | DNeo's | the 1,199 own replies; seeds 0 and 1 | 200 |
  | `own-3m` | DNeo's | the 1,199, weighted by part as for the 267 | 200 |
  | `hv-3` | DNeo's | as `own-3m`, with the chat and code replies swapped for as many from the public set at 0.6 | 200 |
  | `own-3b8` | DNeo's | as `own-3b` | 800 |
  | `hv2-1x`, `hv2-4x`, `hv2-16x` | DNeo's | the 300 own Japanese replies, and 899, 3,596 or 14,384 replies from the public set | 800 |
  | `hv2-4x-long`, `hv2-16x-long` | DNeo's | as `hv2-4x` and `hv2-16x` | 2,400 |
  | `form0-own`, `form0-b` | the bridge alone | the 267 and the 1,199 own replies | 200 |
  | `own-c`, `own-cb` | arm C | the 267 and the 1,199 own replies | 200 |

  The three `hv2` sets are nested: 1× is the head of 4×, and 4× the head of 16×. Their 1× has as many chat and
  code replies as the 1,199 own replies have. The `hv2` arms fix the parts' shares at those of `own-3b`.
- **The temperature.** `hv2-16x-t1.25`, `-t1.5` and `-t2.0` are `hv2-16x`'s step 800 scored with the head's chances
  taken at that temperature. The drafts do not change, and at temperature 1.0 the report is the arm's own.
- **The code** is this repository at 0.7.0. The arms' source differs from the 0.7.0 release in its version string
  only, and the one DFlash2's report was scored with also in how a checkpoint is saved. So the reports have the
  sweep by the cut alone. The sweep with serving's price of a row came in 0.8.0 (see below). The temperature is
  `score.proposals`'s keyword `temperature`, which the command line does not take, so those reports were made
  through a wrapper that sets it (below).

## How they are read

Every drafter is scored with `open-dneo score --served --run` on the recording of the open half's 320 prompts at
ten taps that the [G1 receipts](../2026-11-21-g1/README.md) call evd3. `table.py` reads two numbers from those
reports. Both are shares of DFlash2's number from its own report on the same recording.

- **The cut-free sum.** Per class, the shares of anchors whose drafts land whole to position 1, 2, … 7, summed: the
  drafts an anchor lands with every draft held, without copies. It reads how often drafts land, not how a drafter
  states its chances.
- **G2's price.** A drafter's predicted serving speed. Per class and cut from 0.3 to 0.9, the report's line "by the
  cut alone" gives tokens and drafts verified a round. A class's time is its reply tokens over tokens a round,
  times 41.87 ms + 5.87 ms a draft verified (G2's time of a round,
  [receipts/2026-11-23-round-time](../2026-11-23-round-time/README.md)). Each drafter takes its fastest cut over
  the 320 prompts (the lower cut on a tie), and its speed there is the reply tokens over that time. In
  `g2-price.txt` this is the block "g2t 5 launches", the fit over all of that receipt. The blocks "g2t without
  c20" (the same launches without the one at cut 0.2) and "evd2 requests" (a fit across the requests of G0's
  second launch) are two other fits, not in that receipt, given for comparison.
- **The price of a row.** [docs/plan.md](../../docs/plan.md) picks a drafter's cut by the sweep with serving's
  price of a row. These reports have the sweep by the cut alone only. From cut 0.3 up the two differ only where a
  round's first draft has a chance under 4.1/76, which a head over the whole vocabulary can give. Each "price
  bound" line in `g2-price.txt` bounds that difference from the drafts under 0.1. For own-3 and own-c, scored
  again with the price, the speeds are the same to four digits (`g2-priced.txt`).
- **The stop line**, written on 2026-11-25 before the temperature was tried and before the 2,400-step arms ran,
  and after the 800-step arms' numbers had been seen. The temperature with the highest price on `hv2-16x`'s step
  800 is taken. The higher of the two 2,400-step arms' prices at their last step, at that temperature, is compared
  with 1.009, own-c's price. At 1.009 or more, the work in DNeo's form goes on. Below, it stops. Within 0.002 of
  the line, the decision is taken with the project's owner. The 0.002 is the larger spread between two seeds in
  `g2-price.txt` (own-3s1 − own-3 −0.0015, own-3bs1 − own-3b −0.0022).

None of this is gate G2. G2 is read on the sealed half, which was not served, scored or read (the check of the
public set's prompts above printed counts only), and its clause against MTP needs MTP's round time, which is not
measured.

## Files

`SHA256SUMS` covers every file but this README. Each line end is LF.

- `reports/LABEL.txt`: the `score` report of each drafter in the table, and of every kept step of the two
  2,400-step arms (`-sSTEP`). `reports/DFlash2.txt` is DFlash2's.
- `table.py` and `table.txt`: the cut-free sums and G2's price from the reports (numpy-free Python). Its prices are
  the lines of the block "g2t 5 launches" in `g2-price.txt`, digit for digit.
- `g2-price.txt`: the price tool's whole output on the same reports: a control of DFlash2's lines against serving's
  tokens and drafts a round (`instrument plain`, the round-time receipt's), the three fits, the differences
  between pairs, each drafter's seconds of rounds by cut, the price bounds and MTP's tokens a round.
- `g2-priced.txt`: own-3 and own-c scored by the cut alone and with serving's price of a row, against DFlash2
  scored alike, with `rounds.priced` as 0.8.0 released it.
- `chance-drift.txt`: how each drafter's chances stand against how its drafts land. Section 2 gives, per class, the
  share of drafts landed less the middle of their chance's bin, over the bins 0.1 to 0.9 (negative: the chances
  are too high). Section 3 gives, at each drafter's cut, the chains just above it and their share landed whole.
- `runs.txt`: each run's entrance, form and `run:` line, and its last step's loss and rate, from its training
  output.
- `train/`: the whole training output of `hv2-4x`, `hv2-16x` and the two 2,400-step arms. Its last line named
  the run's folder, written `OUT` here.
- `overlap.txt`: the check of the public set's prompts against the evaluation set. A third line, which named a
  local file, is left out.
- `public-set.txt` and `public-sets.txt`: the public set's replies by part and temperature as recorded, and the
  three nested sets the `hv2` arms take. A last line of `public-sets.txt`, which named a local file, is left out.

The recordings, the labels, the run file, the checkpoints and the tools that made `g2-price.txt` and
`chance-drift.txt` are not in this repository: the recordings hold the text, and the tools read a layout of
folders this repository does not have. `table.py` computes the price and the sums from the reports here.

```sh
open-dneo train ROWS --next NEXT --parts PARTS --drafter DFLASH2 --target CHECKPOINT --out OUT --steps N \
    --bridge BRIDGE --dneo-mask --dneo-head 0 --peak 2e-4 --end 2e-5 --anchors 384 \
    --measure-every M --keep-every K --device cuda                  # DNeo's form; --seed 1, --shares S as above
open-dneo train ROWS ... --bridge BRIDGE --peak 2e-4 --end 2e-5 ...  # the bridge alone
open-dneo train ROWS ... --peak 2e-5 --end 2e-6 ...                  # arm C
open-dneo score EVD3 --drafter OUT/step-N --target CHECKPOINT --served --run EVD3_RUN --device cuda > reports/LABEL.txt
python -c 'import functools, sys; from open_dneo import __main__ as cli, score
score.proposals = functools.partial(score.proposals, temperature=float(sys.argv[1]))
sys.exit(cli.main(sys.argv[2:]))' T score EVD3 ...                   # at a temperature T
python table.py reports > table.txt
```

## What they show

- **G2's price** (`g2-price.txt`, block "g2t 5 launches"; DFlash2 = 1, at 45.6 tokens a second of rounds):

  | Drafter | Steps | Cut | All 320 | code | agent | chat | ja | prose |
  |---|---|---|---|---|---|---|---|---|
  | DFlash2 | | 0.7 | 1 | 1 | 1 | 1 | 1 | 1 |
  | own-c (arm C) | 200 | 0.3 | **1.0091** | 1.005 | 0.990 | 1.015 | 1.008 | 1.011 |
  | own-cb (arm C) | 200 | 0.3 | **1.0085** | 1.015 | 1.013 | 1.012 | 1.005 | 1.004 |
  | form0-own (the bridge alone) | 200 | 0.5 | 0.9859 | 0.980 | 1.022 | 0.984 | 0.985 | 0.986 |
  | form0-b (the bridge alone) | 200 | 0.3 | 0.9946 | 0.998 | 1.062 | 0.993 | 0.997 | 0.981 |
  | own-3 | 200 | 0.5 | 0.9670 | 0.959 | 0.996 | 0.965 | 0.969 | 0.967 |
  | own-3s1 | 200 | 0.5 | 0.9655 | 0.956 | 1.013 | 0.957 | 0.970 | 0.966 |
  | own-3b | 200 | 0.3 | 0.9849 | 0.966 | 1.031 | 0.983 | 0.996 | 0.979 |
  | own-3bs1 | 200 | 0.3 | 0.9827 | 0.969 | 1.035 | 0.978 | 0.988 | 0.980 |
  | own-3m | 200 | 0.3 | 0.9855 | 0.964 | 1.021 | 0.981 | 0.998 | 0.984 |
  | hv-3 | 200 | 0.3 | 0.9875 | 0.962 | 1.027 | 0.983 | 0.995 | 0.993 |
  | own-3b8 | 800 | 0.5 | 0.9792 | 0.967 | 1.039 | 0.971 | 0.997 | 0.966 |
  | hv2-1x | 800 | 0.4 | 0.9813 | 0.970 | 1.039 | 0.970 | 0.994 | 0.976 |
  | hv2-4x | 800 | 0.4 | 0.9896 | 0.976 | 1.041 | 0.981 | 0.998 | 0.987 |
  | hv2-16x | 800 | 0.4 | **0.9922** | 0.967 | 1.081 | 0.983 | 1.001 | 0.992 |
  | hv2-16x at temperature 1.25 | 800 | 0.3 | 0.9858 | 0.962 | 1.062 | 0.974 | 1.001 | 0.983 |
  | hv2-16x at temperature 1.5 | 800 | 0.3 | 0.9504 | 0.924 | 1.019 | 0.931 | 0.979 | 0.945 |
  | hv2-16x at temperature 2.0 | 800 | 0.3 | 0.8485 | 0.700 | 0.876 | 0.841 | 0.912 | 0.898 |
  | hv2-4x-long | 2,400 | 0.5 | 0.9769 | 0.967 | 1.058 | 0.967 | 0.977 | 0.977 |
  | hv2-16x-long | 2,400 | 0.5 | **0.9832** | 0.969 | 1.080 | 0.976 | 0.983 | 0.982 |

  No drafter in DNeo's form reaches 1 over the 320 prompts. The best is hv2-16x at 0.9922. The best with DNeo's
  entrance is the bridge alone on the 1,199 own replies, 0.9946. Most drafters with DNeo's entrance are faster
  than DFlash2 on agent turns (up to 1.081), and every one has some class more than 1% slower.
- **The verdict by the stop line.** The temperature with the highest price on hv2-16x's step 800 is 1.0. At
  temperature 1.0 the two 2,400-step arms' last steps price at 0.9769 and 0.9832. The higher, 0.9832, is 0.0258
  below 1.009, outside the 0.002 band. So the work stopped. In the other two fits the same drafter prices at
  0.9836 and 0.9843.
- **Arm C gained by its chances, not by landing more.** Its cut-free sums are DFlash2's within 0.2%: own-c 1.0016
  and own-cb 0.9993 (code, chat and ja, `table.txt`). Its chances are close to how its drafts land, −0.039 to
  +0.007 by class for own-c and −0.009 to +0.047 for own-cb, where DFlash2's are −0.121 to −0.253
  (`chance-drift.txt`, section 2). With such chances it takes cut 0.3 and keeps the drafts that pay, which reads
  as the source of its 0.9% over DFlash2.
- **A uniform temperature lowered the price.** The cut-free sums are the same at every temperature (`table.txt`):
  the temperature moved the chances, not the drafts. At 1.25 and 1.5 the chances come out too low (code +0.091
  and +0.220), and the price falls: 0.9922 at 1.0, 0.9858 at 1.25, 0.9504 at 1.5, 0.8485 at 2.0.
- **Longer training moved the held-out streams, not the evaluation.** On the streams held out of the same text,
  the drafts landed on the target's path an anchor were 2.001 at the end of the 800-step 16× run and 2.076 at the
  end of the 2,400-step one, and 1.922 and 1.966 for 4× (`train/`, the `held:` lines). On the evaluation set the
  cut-free sums fell, 0.9216 to 0.9115 for 16× and 0.9188 to 0.8952 for 4× (code, chat and ja, `table.txt`). The chances also
  grew too high: in code, −0.027 to −0.063 for 16× and −0.045 to −0.104 for 4× (`chance-drift.txt`). The fastest
  cut rose from 0.4 to 0.5, and the price fell from 0.9922 to 0.9832 and from 0.9896 to 0.9769.

## What they do not show

- Gate G2. These are numbers on the open half, at DFlash2's round time. The sealed half was not served, scored or
  read, and MTP's round is not timed. `g2-price.txt`'s last line gives MTP's tokens a round, unpriced.
- A round of DNeo's form. Its entrance and its head over the whole vocabulary are priced at DFlash2's round. If
  they cost more, DNeo's form is further behind.
- The price of a row for most drafters. It was scored only for own-3 and own-c, and the bounds cover the others. For
  the 2,400-step arms the bound is small (at most 0.0026 and 0.0038 of their time). At temperature 1.25 and above
  it is not (at most 0.0292 of the time and 0.0129 of the tokens at 1.25), so the temperature's prices are by the
  cut alone.
- A temperature for the 2,400-step arms. It was chosen on hv2-16x's step 800 and not taken again on them, though
  their chances are further off. A temperature by class, or one between 1.0 and 1.25, was not tried.
- More seeds. The arms in DNeo's form with two seeds differ by 0.0015 and 0.0022, and the rest ran once.
- Training to the served distribution. The public set's prompts are synthetic, and the evaluation set's are drawn
  from public datasets. Making the training text more like what is served was not tried.
- Stage 2's first runs: on the pilot's text, comparing four sets of taps for the entrance. They are not in this
  receipt.
- The plan's checkpoint. [docs/plan.md](../../docs/plan.md) said Stage 2 would record from, and be served on, a
  checkpoint built from the MIT-licensed weights alone. These rows were recorded from the Jammed XSFP4 checkpoint
  of Stage 1. No weights are published, so no released drafter depends on it.
- The device. DFlash2's report was scored on a CPU and the arms' on a GPU. A GPU score of DFlash2 was compared
  with it once and agreed. That check is not in this receipt.
