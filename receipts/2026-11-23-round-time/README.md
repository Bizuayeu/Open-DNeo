# G2's time of a round (2026-11-23)

Gate G2 in [docs/plan.md](../../docs/plan.md) estimates a block drafter's speed as tokens a round over the round's
time in serving. These receipts give that time: 41.87 ms a round and 5.87 ms for each draft verified
(`rounds.ROUND_MS` and `rounds.DRAFT_MS`), fitted to the seconds serving took for DFlash2's replies at five cuts.

## What ran

The two GB10 machines, the server, the checkpoint and the drafter of the [G0 receipts](../2026-11-20-g0/README.md):
TensorFold `e3fb56cb09061a979cb02080601fc6f7870d3362` with this project's recorder for GLM, at TP=2, serving
`Bizuayeu/GLM-5.3-Flash-Jammed-XSFP4` at revision `0a22bb46ecef3d0f5f96e41dc6011be12974fbf1`, with
`incoai/GLM-5.3-Flash-DFlash2` at revision `6cc6c481eee930a5139619ed919f78726371adbc` as the drafter.

Five launches ran in this order, each fresh and none recording rows:

| Launch | `TF_GLM_DRAFT_CONFIDENCE` (the chain's cut) |
|---|---|
| c70a | 0.7 |
| c90 | 0.9 |
| c50 | 0.5 |
| c20 | 0.2 |
| c70b | 0.7 |

- **Every launch**: `TF_GLM_DFLASH2_LOG=1000000`, as in G0's evd2, and `TF_GLM_EXPERT_FORM=factored`, the expert form
  G0's evd1 chose by timing, so that all five share it. `TF_GLM_COPIES` was unset.
- **Every request**: the open half's 320 prompts as token ids to `/v1/completions`, one at a time, temperature 0,
  `max_tokens` 256, as in G0.
- **Checked before the fit**: each launch's server logs on both ranks show the drafts line of G0's evd1, the expert
  form above, no fallback or failed compile, and 0 restarted DFlash2 contexts. The heat watch switched the clock cap 0
  times.
- A sixth launch, with the drafter's stage timer on, was only a check: its seconds are not serving's, and it is not
  in `runs.tsv`.

## Files

`SHA256SUMS` covers all three. The run files and the server logs are not in this repository: the run files hold
the prompts' token ids.

- `runs.tsv`: a row for each request in each launch, from its run file. It has the launch, the request's id and
  class, the cut, its status, its cached tokens, the sha256 of its reply's token ids, its seconds, and the rounds and
  the drafts verified (`drafted`) that serving counted for it. The launch named evd2 is G0's second launch, at the
  default cut 0.7: it is the reference for the replies' tokens and is not fitted.
- `fit.py`: the fit (numpy).
- `fit.txt`: its output.

```sh
python fit.py runs.tsv > fit.txt
```

## What they show

- **All five launches count** (`fit.txt`): 320 requests each, status 200, no cached tokens, and every reply the same
  tokens as evd2's. The cut changes how many rounds and drafts a reply takes, not its tokens.
- **The fit.** For request i and launch c, `seconds = s_i + a * rounds + b * drafted`, within each request: `s_i`
  holds what the cut does not move, such as the prompt pass. The rounds and drafts are serving's own counts, not
  the instrument's.

  | Class | a (a round) | b (a draft verified) | Within R² |
  |---|---|---|---|
  | code | 42.77 ms | 6.02 ms | 0.891 |
  | agent | 41.70 | 5.86 | 0.799 |
  | chat | 41.47 | 5.96 | 0.987 |
  | ja | 43.08 | 5.62 | 0.974 |
  | prose | 41.80 | 5.97 | 0.991 |
  | all | **41.87** | **5.87** | 0.979 |

  G2 takes the fit over all.
- **A round by the fit over all**, at each launch's drafts a round: 52.9 ms at cut 0.9 (1.883 drafts), 55.9 at 0.7
  (2.391 and 2.390), 59.7 at 0.5 (3.033) and 69.9 at 0.2 (4.766).
- **The two launches at cut 0.7** differ by a median of −27.0 ms a request. 310 of the 320 requests took the same
  rounds and drafts in both.

## What they do not show

- MTP's round. MTP pays a different price a round and is not timed this way yet.
- A drafter whose round costs more than DFlash2's. G2 has such a drafter borrow this time, so a heavier round is not
  measured here.
- Other loads or machines: one request at a time, on these two machines at TP=2.
