# G0 receipts (2026-11-20)

Gate G0 in [docs/plan.md](../../docs/plan.md) asks whether the offline instrument counts a DFlash2 drafter's tokens
a round as serving does: within 0.1 in every class of the evaluation set's open half, on the same prompts. It
passes. The largest gap is 0.013, in the agent class, and no fix was needed.

## What ran

Two GB10 machines at TP=2 served `Bizuayeu/GLM-5.3-Flash-Jammed-XSFP4` at revision
`0a22bb46ecef3d0f5f96e41dc6011be12974fbf1`. The server is the one of the
[recorder check](../2026-11-19-recorder-check/README.md): TensorFold `e3fb56cb09061a979cb02080601fc6f7870d3362` with
this project's recorder for GLM, which is not published yet. The DFlash2 drafter is
`incoai/GLM-5.3-Flash-DFlash2` at revision `6cc6c481eee930a5139619ed919f78726371adbc`, loaded at its default 4 bits.
`TF_GLM_COPIES` and `TF_GLM_DRAFT_CONFIDENCE` were unset in every launch.

Four launches ran in this order, each fresh:

| Launch | Drafts (the startup line) | Rows recorded | Requests |
|---|---|---|---|
| evm1 | `MTP + copies` | yes, taps 6, 15, 25, 34, 43, 45 | the route check on 6 prompts, then the open half |
| evd1 | `DFlash2 (…) + copies`, `TF_GLM_DFLASH2_LOG` set | yes, the same taps | the open half, then each reply again |
| evd2 | `DFlash2 (…) + copies`, `TF_GLM_DFLASH2_LOG` set | no | the open half |
| evm2 | `MTP + copies` | no | the open half |

- **The open half**: 320 prompts, 64 in each of five classes (code, agent turns, chat, Japanese chat, prose). The
  prompts are not in this repository.
- **The route.** Each prompt is its messages under the chat template with thinking off. The route check sent six
  prompts, one of each class and a second agent prompt, as token ids to `/v1/completions` and then as messages to
  `/v1/chat/completions`. All six ran the same prompt ids on both routes (`probe.txt`). The recorded chat streams
  held those ids from row 0 for four of them; the two agent prompts' chat streams resumed from the prompt the
  completions route had just run, and held them from rows 2,048 and 4,096. So every launch sent the open half as
  token ids to `/v1/completions`.
- **Every request**: one at a time, temperature 0, `max_tokens` 256, the reply's ids returned
  (`return_token_ids`).
- **Again** (evd1): each reply's prompt and reply sent as one prompt of token ids, for 1 token. These rows are for
  Stage 1's comparison of reply rows with prompt rows, and G0 does not read them.
- evd1 is the launch G0 judges: the first counted launch with the drafter that serves the whole open half and
  records its rows. It served all 320 with status 200, and its log reports 0 restarted DFlash2 contexts.

## Files

`SHA256SUMS` covers the outputs and the two scripts. The recordings (42 GB), the run files and the server logs are not
in this repository: the run files hold the prompts' token ids. `EVD1` is evd1's recording, `RUN` the run file of a
launch (a JSON line a request, `open_dneo.runs`), `LOG` evd1's server log on rank 0.

```sh
open-dneo score EVD1 --drafter DRAFTER --target CHECKPOINT --served --run evd1/RUN --drafts-log LOG > served.txt
open-dneo score EVD1 --drafter DRAFTER --target CHECKPOINT --run evd1/RUN > stored.txt
python serving.py evd1/RUN evd2/RUN evm1/RUN evm2/RUN > serving.txt
python astray.py LOG evd1/RUN evd1/AGAIN > astray.txt                  # AGAIN: the again prompts' run
```

`probe.txt` is the route check's report, as the evaluation client printed it in evm1.

Both `score` runs were on an aarch64 machine's CPU, with the drafter and checkpoint above and this release's code as
it stood before the release was numbered.

## What they show

- **G0 passes** (`served.txt`, the weights as serving holds them). Tokens a round, a reply's tokens less one over
  its rounds, summed by class:

  | Class | Replies | Serving | The instrument | Apart |
  |---|---|---|---|---|
  | code | 64 | 16,094 over 3,375: 4.769 | 16,094 over 3,374: 4.770 | 0.001 |
  | agent | 64 | 4,004 over 1,117: 3.585 | 4,004 over 1,113: 3.597 | 0.013 |
  | chat | 64 | 15,693 over 6,540: 2.400 | 15,693 over 6,541: 2.399 | 0.000 |
  | ja | 64 | 15,736 over 7,548: 2.085 | 15,736 over 7,544: 2.086 | 0.001 |
  | prose | 64 | 16,109 over 7,929: 2.032 | 16,109 over 7,923: 2.033 | 0.002 |

  If each reply's first round had held 1 draft or every draft, the instrument would count 3,398 or 3,359 rounds for
  code, 1,135 or 1,106 for agent, 6,558 or 6,537 for chat, 7,553 or 7,538 for ja and 7,928 or 7,923 for prose.
- **The drafts serving logged are the instrument's.** The log holds 26,829 rounds. The 26,509 held by the 320
  requests are as many as serving's rounds in every class. At the same anchor, the first draft is the instrument's
  in 99.7% of code rounds, 99.5% of agent, 99.2% of chat, 99.1% of ja and 99.4% of prose rounds, and all seven drafts
  are in 96.9%, 94.5%, 91.0%, 87.9% and 89.8% (`served.txt`). The other 320 rounds are the again prompts': each
  sits at its prompt's end with the one token it gave, after every round of the 320 requests (`astray.txt`).
- **The second launch** (`serving.txt`, evd2) took one round more for code (4.767) and the same rounds in the other
  four classes.
- **One launch can record rows and serve DFlash2 drafts.** The replies are the same tokens in all four launches,
  with and without recording and with either drafter (`serving.txt`: 320 of 320 against evd1's).
- **The weights as stored** (`stored.txt`, reported, not judged): the instrument counts 4.788 for code, 3.673 for
  agent, 2.415 for chat, 2.101 for ja and 2.040 for prose, from 0.008 to 0.089 above serving. With the drafter's
  linears and the head at their stored precision the drafts land a little more often than at 4 bits. G0 reads the
  weights as serving holds them, as the plan fixed before either number was taken.
- **MTP's rounds** (`serving.txt`, evm1 and evm2): code 3.641 and 3.633, agent 4.242 and 4.237, chat 2.258 and
  2.229, ja 2.257 and 2.246, prose 2.032 and 2.036. MTP and a block drafter pay different prices a round, so these
  are not set against DFlash2's here; G2 compares estimated speed.
