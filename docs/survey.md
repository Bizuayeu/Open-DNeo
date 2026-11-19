# Survey (2026-11-19)

What was read and measured before the [plan](plan.md) was written. Nothing here ran a model: the only computation
is on one public tensor, on a CPU.

## Sources

| What | Where | Revision |
|---|---|---|
| The paper | [arXiv:2611.21925](https://arxiv.org/abs/2611.21925), read in full | v1, 2026-11-16 |
| DNeo inference code | [J-sandler/DNeo](https://github.com/J-sandler/DNeo), MIT | `8449a3ce891f6b3321efafc1e5c0c7e5c40e11c7` |
| DNeo weights | [J-sandler/Qwen3-4B-DNeo](https://huggingface.co/J-sandler/Qwen3-4B-DNeo), Apache-2.0 | `8451a2e8446c0bfa09be47623ff044d896540ec4` |
| GLM's DFlash2 drafter | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), Apache-2.0 | `6cc6c481eee930a5139619ed919f78726371adbc` |
| TensorFold | [ashhart/TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0 | `2a674b552498739e783a31c2573fc9ccf4d8cc0d` (1.2.10) |

## The paper

- DNeo is a parallel drafter in the DFlash line. It was tried on one target, Qwen3-4B (36 layers, width 2,560), on
  one H100 with SGLang, at block 16.
- Removing one part at a time from DNeo, at 100,000 training rollouts, costs this many tokens a round out of 3.112
  (Table 4): the non-mixing entrance (MoFA) 0.651, the two-tap convolutions 0.331, the fused head 0.217, rejection
  memory 0.182.
- **Entrance.** DFlash-2 joins the chosen layers' hidden states and shrinks them with one dense matrix. DNeo takes
  a weighted sum with one scalar a layer and no matrix, which needs the drafter to be as wide as the target. Giving
  DFlash-2 a non-mixing entrance alone moves it from 1.938 to 2.361 at 50,000 rollouts (Table 5).
- **Layers.** The first layer and layers from the later half of the target. The count matters little: 2.481 with
  five layers and 2.516 with fifteen.
- **Attention.** Draft positions do not attend to one another. The effect is +0.004 to +0.027.
- **Head.** DFlash-2's head re-scores the 16 best candidates. DNeo applies it to the whole vocabulary and adds
  DSpark's low-rank head in the same latent space, with two scalars a position.
- **Rejection memory.** The target's hidden states at the positions of rejected drafts, which verification has
  already computed, become extra attention context for the next round. It adds no parameters. The paper says this
  part must be trained on speculative rollouts, by on-policy distillation.
- **Training.** Prompts from the Llama-Nemotron post-training dataset, four rollouts a prompt, written by the
  target in thinking mode, capped at 1,024 tokens. 1.6 million rollouts for the main results and 100,000 for the
  ablations. Learning rate 0 to 1e-5 over 20,000 rollouts, then to 0. The DSpark loss. Compute and wall time are not
  stated. Every drafter compared was trained from zero under the same conditions; warm starts are not discussed.
- **Results at temperature 0** (Table 2): tokens a round rise 19-27% over DFlash-2 (GSM8K 3.49 to 4.42, MT-Bench
  2.59 to 3.09). Speed rises 7-13% over DFlash-2. Against the first DFlash it is 1-2% faster on math and code and
  slower on GPQA and MT-Bench. At temperature 1 it is 17-21% faster than DFlash-2.
- Reaching a given tokens a round takes up to a tenth of DFlash's training (Appendix A.1).

## What is public

- The code is inference only. The model card says no private training code is needed to run it; none is published.
- The head, from the runtime: `latent = a * folded_token_embed[previous] + b * df2_latent_projection(
  predecessor_codebook[previous] * hidden_projection(h))`, then `logits = (a + b) * base + folded_out_proj(latent)`,
  with `a` and `b` learned per position. The runtime also has a mode that scores only the 16 best candidates.
- The rows it reads from the target are each layer's attention input, after that layer's input norm.
- The released checkpoint's manifest records 2,476,000 prompts seen and 100,000 adaptation prompts. This revision
  stores one router where earlier ones stored five.
- For DFlash2, [z-lab/dflash](https://github.com/z-lab/dflash) (MIT) has the model and no training code.

## Tensor layouts side by side

GLM's DFlash2 drafter has 81 tensors and 1,171,080,448 parameters: 5 layers, width 4,096 (the target's), FFN
12,288, block 8, target layers 5, 14, 24, 33 and 42 of 45, head rank 256 over the 16 best. The listings are in
[receipts/2026-11-19-survey](../receipts/2026-11-19-survey/).

| Part | DFlash2 for GLM | DNeo as published | Carries over? |
|---|---|---|---|
| 15 tensors a layer, 5 layers (q, k, v, o, four norms, three MLP matrices, two convolutions) | `layers.N.*` | `draft.layers.N.*` | Same layout |
| Final norm | `norm.weight` | `draft.norm.weight` | Same layout |
| Entrance | `fc.weight` [4096, 20480], `hidden_norm.weight` | `conditioner.router_logits`, one scalar a target layer | **No**, see below |
| Head: state projection, previous-token table | `candidate_selector.hidden_projection`, `predecessor_codebook` | `ar_head.hidden_projection`, `predecessor_codebook` | Same layout |
| Head: candidate table | `successor_codebook` [vocabulary, 256] | `folded_out_proj` [vocabulary, 256] | The same score, with the latent projection as the identity, b = 1 and a = 0 |
| Head: DSpark's side | none | `folded_token_embed`, two scalars a position | New, to be trained |

"Same layout" means the same tensors in the same arrangement. The published DNeo is 2,560 wide and GLM's DFlash2
4,096, so a DNeo for GLM has DFlash2's shapes. DNeo keeps key and value projections in every layer; what differs
inside a block is the mask (a draft row sees context rows and its block's first row only). Even where shapes match,
a new entrance changes what the context rows hold, so whether the layers still work is a measurement, not a given.

## The entrance, measured

`open-dneo entrance` splits `fc.weight` into its five 4,096 x 4,096 layer blocks and reports the share of each
block's squared weight on its diagonal, that is, from a coordinate to the same coordinate. Chance is 1/4096.

```text
chance level of a diagonal share: 0.02441%
block 0: energy share 21.89%  diagonal share 0.024%  rms diagonal 0.16632  rms off-diagonal 0.16893  diagonal > 0 49.5%
block 1: energy share 26.81%  diagonal share 0.024%  rms diagonal 0.18560  rms off-diagonal 0.18699  diagonal > 0 50.0%
block 2: energy share 33.08%  diagonal share 0.024%  rms diagonal 0.20782  rms off-diagonal 0.20770  diagonal > 0 49.9%
block 3: energy share 9.46%  diagonal share 0.024%  rms diagonal 0.11063  rms off-diagonal 0.11107  diagonal > 0 49.5%
block 4: energy share 8.76%  diagonal share 0.024%  rms diagonal 0.10524  rms off-diagonal 0.10685  diagonal > 0 49.3%
all blocks: diagonal share 0.024%
singular directions holding the squared weight: 50%: 201, 90%: 1383, 99%: 2851 of 4096
```

- Every block is at chance, with signs split evenly. The entrance mixes all coordinates, and the five layers behind
  it were trained on the mixed ones. It cannot be moved into DNeo's non-mixing entrance as it is.
- To carry the layers over, a linear bridge from the new entrance's output to the old one's is fitted by least
  squares. Fitting it needs recorded rows of the target, so its quality is not known yet.
- The bridge cannot simply be folded into the layers. A context row reaches a layer only through its key and value
  projections, so a bridge composes with those two exactly. But the published DNeo layer, like DFlash2's, has one
  key and one value projection, and its block rows are read through the same two: folded there, the bridge would
  change the block rows too. A bridge needs either key and value projections for the context alone or a matrix kept
  in the entrance, and the published DNeo layout has neither. TensorFold's GLM path does pack the context's
  projections as a linear of their own, from the same two tensors. Which form to use is not decided.
- Limit: weight energy says nothing about how large each layer's input is, so the blocks' energy shares do not rank
  the layers.

## TensorFold's GLM drafter path today

- DFlash2 for GLM is its own implementation: about 990 lines of Zig in `zig/src/families/glm/dflash*.zig` and
  `cuda_taps.zig`, and one kernel file. Block 8, width, layer count, rank and vocabulary are constants in the source.
- A tap is the mean of the target's four residual streams after a fixed layer, not normed: 5 x 4,096 bf16 values a
  row. DNeo's published code taps one stream's normed attention input, so the definitions differ.
- Context rows go through `fc`, a norm and each layer's key and value projections into a 2,048-row ring per stream.
  A block is the anchor and seven masks. Each rank takes its 16 best candidates from a 4-bit copy of the target's
  head, and rank 0 re-scores them on the host.
- The verify pass's tap rows include the rejected rows: the engine notes every verified row, and the next draft
  takes only the accepted ones into its context. Rejection memory's input is already there.
- `--record`, the recorder for drafter training, is in the core, but GLM gives it no rows yet.
- The drafter release format (version 1) describes one architecture, the block drafter.

## Precedents upstream

- [The recipe for training a drafter](https://github.com/ashhart/TensorFold/blob/2a674b552498739e783a31c2573fc9ccf4d8cc0d/docs/recipes/training-a-drafter.md),
  from Kolibri-1 on one DGX Spark. `--record` writes prompt rows as well as reply rows. 45 million tokens of broad
  text, regenerated by a strong model and run through the target as prompts, generalised; about a million rows of
  the target's own output, repeated, were memorised. The loss is KL to the target's distribution. The released
  drafter took 213.6 million tokens. On a MoE target, deeper verification was slower.
- [#1485](https://github.com/ashhart/TensorFold/issues/1485): a feature dump and an offline acceptance tool, whose
  tokens a round should agree with a served drafter's within noise.
- [#2592](https://github.com/ashhart/TensorFold/issues/2592): a public drafter's acceptance measured in PyTorch on
  a GB10 before porting it.
- [#2667](https://github.com/ashhart/TensorFold/issues/2667): a released drafter adapted to a 4-bit target. In the table of
  the GB10 replication, the geometric mean of the per-prompt served-speed ratio, adapted over original and greedy,
  was 1.114 [1.038, 1.198] on the 16 agent prompts used in development and 0.989 [0.970, 1.008] on 110 independent
  ones (48 task groups), which failed the gate. The proposed gate included a DFlash2 fine-tune on the same data as a
  control.
- There is no issue or pull request about DNeo.

## The cost of the paper's recipe here

Two GB10 machines at TP=2, greedy, serving `Bizuayeu/GLM-5.3-Flash-Jammed-XSFP4`:

- Decode over four streams: 52-53 tok/s on prose and 71-73 on code, with the default MTP drafts
  ([TensorFold #3333](https://github.com/ashhart/TensorFold/pull/3333), the default-path table).
- A 38,960-token prompt: 1,420-1,540 tok/s (1,422.2 and 1,437.8 in the 3.10.0 table of the
  [measurements](https://github.com/Bizuayeu/GLM-5.3-Flash-DGX-Sparks-BIZ/blob/main/v3/docs/measurements.md);
  1,535.0 and 1,507.3 in the 3.12.0 table of the
  [v3 README](https://github.com/Bizuayeu/GLM-5.3-Flash-DGX-Sparks-BIZ/blob/main/v3/README.md)).
- MTP's acceptance length on the serving checks: 3.200, 1.964 and 3.262 on count, prose and code (the same 3.12.0
  table).

| To produce | Tokens, at most | Path | Time on one pair |
|---|---|---|---|
| 1.6 million rollouts of up to 1,024 tokens | 1.64 billion | decode | 260-365 days |
| 100,000 rollouts | 102 million | decode | 16-23 days |
| 102 million tokens | | prompt pass | 18-20 hours |

The token counts are upper bounds, since a rollout may end before its cap.

## Not read, not checked

- The paper's figures: it was read as text, so the curves of Figures 5 to 8 were not.
- The rest of the DNeo runtime, including how rejection rows are passed beyond the mask in the model definition.
- Other public training code for drafters, and its licences.
- Whether the mean of GLM's four streams meets the premise of a non-mixing entrance, that coordinates line up
  across layers.
- Training speed on a GB10, and disk on the machines.
