# The pilot recording, the target's own agreement with its text, and the layer probe (2026-11-20)

Stage 1 in [docs/plan.md](../../docs/plan.md) records a pilot of prompt-pass rows over a wide choice of layers and
probes which layers hold the target's next tokens. Before recording, the text a drafter would learn from was chosen
again by one rule: a reply is written by people under a permissive licence, by an open-weight model published
under MIT or Apache-2.0, or by the target itself. The pilot holds 609,309 rows of ten parts, about 60,000 a part.

## What ran

- **The text.** Ten parts, each from one public dataset at one revision (`composition.tsv`). Four are people's
  text, five are open models' replies, and one is the target's own replies to the prompts of another part (chat-glm,
  to chat-human's prompts). Items whose task, prompt or repository the evaluation set draws from were left out.
  code-oci leaves out the tasks taken from competitive-programming sites, and code-r1 holds DeepSeek-R1's answers
  alone, without the problems they answer.
- **The recording.** One TensorFold server (GLM-5.3-Flash, tensor parallel over two machines, no drafter) recorded
  the prompt pass of each item, at most 1,536 tokens, and of the agent part 8,192. Nine parts were recorded at 18
  taps (6, 12, 15, 18, 24, 25, 30, 33, 34, 36, 38, 39, 40, 41, 42, 43, 44 and 45), each to about 60,000 rows; the agent
  part at 9 (6, 15, 25, 34, 40, 41, 43, 44, 45), 8 prompts of 8,192 rows. Tap k is the row after k layers, and tap 45
  is the final row the target's head reads.
- **The target's agreement.** `open-dneo greedy --save` kept the target's greedy choice at every row. `role_greedy.py`
  counts, by part and by the role GLM's template names, how often the choice at a position is the text's next token.

## Files

`SHA256SUMS` covers the outputs, the table of the composition and the script. The recordings, the prompts and the
run files are not in this repository: they hold the text. `pr1` is the recording of the nine parts at 18 taps, `pa1`
that of the agent part at 9.

```sh
open-dneo greedy PR1 --target CHECKPOINT --save PR1_NEXT
open-dneo greedy PA1 --target CHECKPOINT --save PA1_NEXT
python role_greedy.py PR1 PR1_NEXT PR1_STREAMS PR1_RUN PR1_PARTS   # and the same for PA1; both into greedy.txt
open-dneo probe PR1 --target CHECKPOINT --next PR1_NEXT --parts PR1_PARTS --k 0 --k 1 --k 2 --k 3 \
  --set 6,15,25,34,43 --set 40,41,42,43,44 --set 6,15,25,34,41,43,44 --held-every 10 > probe-pr1.txt
open-dneo probe PA1 --target CHECKPOINT --next PA1_NEXT --parts PA1_PARTS --k 0 --k 1 --k 2 --k 3 \
  --held-every 4 > probe-pa1.txt
```

## What they show

- **The composition** (`composition.tsv`). Revisions are the first 12 characters of the Hub's commit.

  | Part | Written by | Dataset | Licence | Streams | Rows |
  |---|---|---|---|---|---|
  | ja-book | people (Aozora Bunko) | globis-university/aozorabunko-clean | CC-BY-4.0 (the books are in the public domain) | 41 | 60,636 |
  | ja-web | people (the web) | HuggingFaceFW/fineweb-2, jpn_Jpan test | ODC-By 1.0 | 92 | 60,172 |
  | prose | people (the web) | HuggingFaceFW/fineweb, sample-10BT | ODC-By 1.0 | 108 | 60,053 |
  | ja-magpie | Qwen2.5-32B-Instruct | llm-jp/magpie-sft-v1.0 | Apache-2.0 | 143 | 60,009 |
  | ja-rewrite | Qwen3-14B | llm-jp/scaling-data-constrained-llms | CC-BY-4.0 | 112 | 60,918 |
  | code-oci | Qwen2.5-Coder-32B-Instruct | nvidia/OpenCodeInstruct | CC-BY-4.0 | 138 | 60,453 |
  | code-r1 | DeepSeek-R1 | nvidia/Nemotron-Post-Training-Dataset-v1, code | CC-BY-4.0 | 64 | 61,000 |
  | chat-human | people (oasst2) | OpenAssistant/oasst2, train | Apache-2.0 | 225 | 60,192 |
  | chat-glm | the target | OpenAssistant/oasst2's prompts, train | Apache-2.0 (the prompts) | 140 | 60,340 |
  | agent-swe | Qwen3-Coder-480B-A35B-Instruct | nvidia/Nemotron-SWE-v1, r2e_gym | CC-BY-4.0; the repositories' own | 8 | 65,536 |

- **The target writes its own text, and little of anyone else's** (`greedy.txt`). At the positions of a reply or of
  plain text, the target's greedy choice is the next token:

  | Part | Positions | Agreement |
  |---|---|---|
  | chat-glm, the replies | 54,607 | 0.973 |
  | code-oci, the replies | 29,943 | 0.875 |
  | code-r1, the text | 60,936 | 0.836 |
  | agent-swe, the replies | 4,997 | 0.836 |
  | ja-rewrite, the text | 60,806 | 0.597 |
  | ja-magpie, the replies | 50,911 | 0.568 |
  | ja-web, the text | 60,080 | 0.561 |
  | chat-human, the replies | 50,741 | 0.551 |
  | prose, the text | 59,945 | 0.543 |
  | ja-book, the text | 60,595 | 0.519 |

  Over the same 140 prompts, the target's choice is the next token at 0.543 of the people's reply positions and at
  0.973 of its own. A drafter is trained to propose what the target would write, so text the target writes itself
  teaches it most directly; at 0.55, about every second token of people's text is one the target would not have
  written. The agent part has few reply positions (8 prompts), and its number is weak.
- **Why the target's own replies are not 1.0.** Each of the 140 first differs at a reply's first position: with
  thinking off, the template closes the thinking block right after opening it, and the target does not predict the
  close. The other misses, about 2.7% of the reply positions, spread evenly from a reply's start to its end. We read
  them as near ties that flip: the replies were decoded, and these rows come from a prompt pass, which computes the
  same tokens a little differently ([rows receipts](../2026-11-20-rows/README.md)).

## The layer probe

- **What ran** (`probe-pr1.txt`). `open-dneo probe` on the nine parts at 18 taps: each tap alone and three sets of
  taps side by side, DFlash2's five (6, 15, 25, 34, 43), the deep five (40 to 44) and DFlash2's five with 41 and 44.
  k runs from 0 to 3. 106 of the 1,063 streams were held out, one in ten by name; rows were fitted one in four and
  scored one in eight. k 0 asks for the target's next token, which its own head gives; k 1 is the first token a
  drafter proposes beyond it, k 2 and 3 the ones after. It ran 3 hours 34 minutes on an aarch64 machine's CPU.
- **Checks.** Tap 45, the row the head reads, scores 1.000 at k 0 (0.999 in three parts): the saved choices sit
  beside their rows. The floor, the commonest choice taken every time, is 0.009 to 0.012. No ja-book stream fell
  among those held out, so the parts' scores cover eight parts.

  | Taps | k 0 | k 1 | k 2 | k 3 |
  |---|---|---|---|---|
  | the floor | 0.011 | 0.012 | 0.009 | 0.011 |
  | 6 | 0.236 | 0.127 | 0.075 | 0.070 |
  | 25 | 0.230 | 0.140 | 0.088 | 0.079 |
  | 34 | 0.316 | 0.172 | 0.097 | 0.078 |
  | 40 | 0.562 | 0.237 | 0.110 | 0.084 |
  | 43 | 0.756 | 0.294 | 0.123 | 0.089 |
  | 44 | 0.818 | 0.309 | 0.130 | 0.089 |
  | 45 | 1.000 | 0.332 | 0.137 | 0.096 |
  | DFlash2's five | 0.766 | 0.318 | 0.146 | 0.105 |
  | the deep five | 0.822 | 0.351 | 0.158 | 0.107 |
  | DFlash2's five, 41 and 44 | 0.829 | 0.353 | 0.162 | 0.110 |

- **The deeper a tap, the more of the target's next tokens it holds.** Alone, at k 1, from 0.127 at tap 6 to 0.332
  at tap 45, nearly in order of depth; the taps from 6 to 30 score close to one another. Every tap is in
  `probe-pr1.txt`.
- **The deep five hold more than DFlash2's five at every k**, by 0.056, 0.033, 0.012 and 0.002 from k 0 to 3, and in
  every part at k 0 and 1. At k 2 and k 3, three parts each go the other way, by 0.002 to 0.015. DFlash2's four
  shallower taps add 0.002 to 0.007 to the deep five. At k 1, tap 45 alone scores above DFlash2's five, which do not
  include it.
- **The agent part does not rank the taps** (`probe-pa1.txt`, 9 taps, 2 of its 8 streams held out). There, shallow
  taps score about 0.5 even at k 3, above the deep ones, over a floor of 0.03. The eight prompts share one system
  prompt and tool list, about 3,900 tokens and 47% of their rows, so the held streams' copies of it were in the fit.
  We read these scores as the map recalling repeated text. Scoring only the turns after the shared part would be
  needed, and was not done.
- **What this does not say.** A linear probe ranks taps by what a linear map recovers from them. A drafter's layers
  are not linear and read the context, so these numbers are not tokens a round. Which taps Stage 2's drafter reads is
  not decided here.
