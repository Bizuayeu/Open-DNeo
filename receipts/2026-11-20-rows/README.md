# Prompt rows against reply rows (2026-11-20)

Stage 1 in [docs/plan.md](../../docs/plan.md) measures what the gap between two kinds of row does to a drafter's
proposals, before the pilot recording. A drafter would be trained on rows a prompt pass computed, the cheap ones to
record, and is served rows that its rounds' verify passes computed. On the evaluation set's open half, the published
DFlash2 drafter proposes nearly the same drafts from either. Walking alone, it takes 0.004 to 0.029 fewer tokens a
round from prompt rows, by class.

## What ran

- **The recording** is that of evd1, the launch G0 judged ([G0 receipts](../2026-11-20-g0/README.md)): the same
  server, target weights, drafter and taps. After the open half, that launch sent each reply again, the request's
  prompt and reply as one prompt of token ids, for 1 token. So for every position of a reply the recording holds two
  rows of the same tokens: one computed in the reply's rounds, and one computed by the prompt pass.
- **`open-dneo rows`** pairs each of the 320 requests with its reply's stream and with the stream whose prompt is the
  request's prompt and reply. It reads both from the later of their first rows, so the drafter's context starts at
  the same row on both sides, and at each reply position it has the drafter propose from each kind of row, with the
  weights as serving holds them (`--served`). Each side is walked with every round the drafter's, since a copy would
  be the same on both sides.
- All 320 requests found both streams, and no position lacked a row on either side.
- It ran on an aarch64 machine's CPU, with this release's code, under a soft limit of 1,024 open files.

## Files

`SHA256SUMS` covers the outputs and the script. The recording and the run file are not in this repository: the run
file holds the prompts' token ids. `EVD1` and `RUN` are named as in the G0 receipts.

```sh
open-dneo rows EVD1 --drafter DRAFTER --target CHECKPOINT --run evd1/RUN --served > rows.txt
python table.py rows.txt > table.txt
```

`open-files.txt` holds the open files the process held while it ran, and how the same command ended with the reading
of a recording as released in 0.3.0.

## What they show

- **The table** (`table.txt`). Positions are the reply positions where the drafter proposed. A walk's number is its
  tokens over its rounds, and apart is the reply rows' number less the prompt rows', from the counts in `rows.txt`.

  | Class | Replies | Positions | Reply rows | Prompt rows | Apart | First draft the same | All seven the same |
  |---|---|---|---|---|---|---|---|
  | code | 64 | 16,107 | 16,094 over 3,207: 5.018 | 16,094 over 3,214: 5.007 | 0.011 | 99.2% | 88.4% |
  | agent | 64 | 4,010 | 4,004 over 1,167: 3.431 | 4,004 over 1,177: 3.402 | 0.029 | 98.2% | 79.4% |
  | chat | 64 | 15,715 | 15,693 over 6,532: 2.402 | 15,693 over 6,543: 2.398 | 0.004 | 96.8% | 65.3% |
  | ja | 64 | 15,755 | 15,736 over 7,531: 2.089 | 15,736 over 7,545: 2.086 | 0.004 | 95.8% | 56.5% |
  | prose | 64 | 16,123 | 16,109 over 7,919: 2.034 | 16,109 over 7,956: 2.025 | 0.009 | 95.7% | 59.6% |

- **From prompt rows the drafter walks a little shorter in every class**, by 0.004 to 0.029 tokens a round. G0
  allows the instrument 0.1 from serving.
- **The first draft is the same from both kinds of row at 95.7 to 99.2% of positions**, and all seven drafts at
  56.5 to 88.4%.
- **What this does not say.** The drafter was trained by its publisher, not on prompt rows. Whether a drafter trained
  on prompt rows does as well on the rows serving gives it is for Stage 2. The numbers come from one launch, on the
  open half, with the weights as serving holds them.
- **One open file a stream.** On the 640 streams, this release's `rows` held 645 open files a minute after it started,
  640 of them the maps of the tapped rows, and finished under the soft limit of 1,024. With the reading of a recording as released in 0.3.0,
  the same command stopped after 7 seconds with `OSError: [Errno 24] Too many open files` while it mapped the
  streams (`open-files.txt`).
- **The change to reading changed no number.** `rows.txt` is, byte for byte, the report of the same command run on
  the same recording before that change, without the limit.
