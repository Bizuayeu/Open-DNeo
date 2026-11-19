# Changelog

## 0.1.0 (2026-11-19)

The plan and the survey. No weights, and no result yet on whether the approach works.

- `docs/plan.md`: the goal, DNeo's parts by what each costs to train, the four moves away from the paper's recipe,
  the stages and the gates, written before any number.
- `docs/survey.md`: the paper, the public code and weights, the tensor layouts of GLM's DFlash2 drafter and the
  published DNeo side by side, TensorFold's GLM drafter path, precedents upstream, and the cost of the paper's
  recipe on two GB10 machines.
- `open-dneo shapes`: a Hub checkpoint's files, config and tensor shapes, from the safetensors header alone.
- `open-dneo entrance`: the diagonal share of a dense feature encoder, per layer block. GLM's DFlash2 entrance is at
  chance level (0.024%) in every block.
- `receipts/2026-11-19-survey`: the outputs of both tools on the pinned revisions.
