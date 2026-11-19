# Changelog

## 0.2.0 (2026-11-19)

The offline instrument, and how gate G0 is measured. No weights, and no result yet on whether the approach works.
The instrument has run on real weights. Its tokens a round are not published before G0 compares it with serving.

- `open-dneo recording`: reads and checks the rows a TensorFold server writes with `--record`.
- `open_dneo.drafter`: DFlash2's and DNeo's forward passes in PyTorch. One layer class serves both, and they differ
  in the entrance, the mask and the head. Each gives the outputs of its public definition on a small random
  checkpoint (`tests/fixtures`, made by `tools/reference` from pinned revisions).
- `open-dneo greedy`: how much of a recording its target would have written itself, from the rows after the final
  norm and the target's head, stored plain or as MXFP8. It checks the head, its orientation and the tap before a
  drafter is scored against them.
- `open-dneo score`: a DFlash2 drafter's tokens a round on a recording, with the weights as stored or as serving
  holds them (4-bit linears, a 4-bit copy of the head, scores in bfloat16). It walks each reply twice: every round
  the drafter's, and the rounds as serving takes them for a stream decoded alone.
- `open_dneo.copies` and `open_dneo.rounds`: serving's rounds are not all the drafter's. Where a stream repeats
  eight or more of its earlier tokens, TensorFold's lanes verify a copy of what followed in place of the drafter's
  chain. The lookup and the rounds of a lone stream are ported from the lane core, and the lookup's tests include
  the lane core's own cases.
- `docs/plan.md`: "How G0 is measured", fixed before either of the numbers it compares was taken.
- `receipts/2026-11-19-recorder-check`: the first check of the recorder for GLM that this project uses, which is not
  published yet. Replies are the same tokens with and without recording. With the tests of the forwards that meets
  Stage 0's two exit conditions, while the stage's training script is still to be written. Recording six taps
  computed 0.925 of the default path's prompt tokens a second. A reply's rows lie 9 to 11% from the same positions
  computed as a prompt's at the deeper taps, and Stage 1 now measures what that does to a drafter's proposals.
- A tag `vX.Y.Z` publishes its section of this file as the GitHub release (`.github/workflows/release.yml`,
  `tools/release_notes.py`).
- CI installs torch, and the tests of the modules that compute with it fail without it instead of skipping.

Changes to what 0.1.0 said:

- `docs/plan.md` had gate G0 as Stage 0's exit. G0 is measured on the evaluation set, which Stage 1 builds, so it
  is now Stage 1's first exit. Stage 0's exit is that replies are the same tokens with and without recording, and
  that the instrument's forwards give the outputs of the public definitions. The gate's sentence and its width of
  0.1 are unchanged.
- `docs/plan.md` listed the target's best k per row among what the recorder writes. For GLM the recorder writes
  the row after the final norm, and the best k are computed from it offline.
- `docs/survey.md` said the bridge from a new entrance to the old one would be folded into each layer's key and
  value projections. A layer's block rows are read through the same two projections, so that fold would change
  them too. The bridge needs projections for the context alone or a matrix kept in the entrance, and which is not
  decided.

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
