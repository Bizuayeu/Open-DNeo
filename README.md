# Open-DNeo

Tools and a procedure for making [DNeo](https://arxiv.org/abs/2611.21925) drafter weights for a target model on modest
hardware, from rows a serving engine records. An independent, unofficial project.

**Status (0.1.0, 2026-11-19): a plan and a survey. There are no weights yet, and nothing here shows that the
approach works.** Each stage will add its tools, its numbers and what did not pay.

[日本語](README.ja.md)

## What this is

DNeo is a parallel drafter for speculative decoding: a small model proposes a block of tokens in one pass and the
target model verifies them, so a reply stays the target's own. The paper reports 19-27% more tokens a round than
DFlash-2 at temperature 0 on Qwen3-4B. Its authors published weights for Qwen3-4B and inference code. The training
code is not public.

This project works out how to make DNeo weights for another target, and publishes the tools, the steps and the
measurements. The first target is GLM-5.3-Flash served by [TensorFold](https://github.com/ashhart/TensorFold) on two
GB10 machines (DGX Spark class).

## Why not the paper's recipe

The paper trains on rollouts the target writes: 1.6 million of them, up to 1,024 tokens each, and 100,000 for its
ablations. On our pair the target decodes at 52-73 tok/s over four streams and reads prompts at 1,420-1,540 tok/s:

| To produce | Tokens, at most | Path | Time on one pair |
|---|---|---|---|
| The full 1.6 million rollouts | 1.64 billion | decode | 260-365 days |
| The 100,000 of the ablations | 102 million | decode | 16-23 days |
| The same 102 million tokens | | prompt pass | 18-20 hours |

A prompt pass is about 20-30 times cheaper per token than decoding. The plan ranks every idea by which of the two it
needs.

## The approach

1. **Record, don't generate.** Run existing broad text through the target as prompts and record the rows a drafter
   reads. TensorFold's recipe for training a drafter reports that this generalised on another target.
2. **Start from the publisher's DFlash2 drafter**, not from zero. Its layers and head carry over by layout. Its
   dense entrance does not (see the survey), so the entrance is bridged by least squares, and re-initialised if the
   bridge is poor.
3. **Leave rejection memory for last.** It is the one part of DNeo that the paper says needs rollouts drafted by the
   drafter itself, and it has the smallest of the four measured effects.
4. **Measure offline before writing engine code.** A PyTorch instrument scores a checkpoint on recorded rows, under
   the limits it would be served with. The engine path is written only after a checkpoint beats both drafters the
   target has today.

The stages, the gates and the stopping rules are in [docs/plan.md](docs/plan.md).

## What the survey found

- The published DNeo checkpoint and GLM's DFlash2 drafter share the layout of their layers (15 tensors a layer),
  the final norm and two head tensors. DNeo replaces the dense entrance with one scalar a target layer.
- GLM's DFlash2 entrance mixes every coordinate: the diagonal share of each layer block is 0.024%, the chance level.
  Its layers were trained on those mixed coordinates, so the entrance cannot be carried over as it is.
- In TensorFold's GLM path, the rows of rejected drafts are already in the verify buffer the drafter reads. That is
  the input rejection memory needs.

Sources, revisions and the full numbers are in [docs/survey.md](docs/survey.md). Raw outputs are in
[receipts/](receipts/).

## Tools in this release

```sh
python -m pip install -e .
open-dneo shapes incoai/GLM-5.3-Flash-DFlash2     # files, config and every tensor's shape
open-dneo entrance incoai/GLM-5.3-Flash-DFlash2   # the diagonal share of its dense entrance
```

Both read a public Hugging Face repository by HTTP range requests, without credentials. `shapes` reads only the
safetensors header. `entrance` fetches one tensor (160 MiB in this example) and needs a few gigabytes of memory.

## Credits and licences

- DNeo: Sandler et al., [arXiv:2611.21925](https://arxiv.org/abs/2611.21925). Reference inference code:
  [J-sandler/DNeo](https://github.com/J-sandler/DNeo) (MIT). Weights:
  [J-sandler/Qwen3-4B-DNeo](https://huggingface.co/J-sandler/Qwen3-4B-DNeo) (Apache-2.0).
- DFlash: [z-lab/dflash](https://github.com/z-lab/dflash) (MIT). GLM's DFlash2 drafter:
  [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (Apache-2.0).
- [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0): the engine, its recorder and its recipe for
  training a drafter.

This release contains none of their code or weights. Open-DNeo is licensed under [Apache-2.0](LICENSE). It is not
affiliated with the DNeo authors, Cerebras, incoai, Z.ai or the TensorFold maintainers.
