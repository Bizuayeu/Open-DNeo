# The lane core's own rounds

`tests/fixtures/lane_core_rounds.jsonl` holds the rounds TensorFold's lane core took over one stream at a time, and
`tests/test_lane_core.py` checks that `open_dneo.rounds.served` takes the same ones. The walk was written from the
lane core's source, so its own tests share its reading; these cases come from the lane core running.

`open_dneo_oracle_test.zig` made them. It is a test for TensorFold's tree, not a part of this package:

| | |
|---|---|
| TensorFold commit | `e3fb56cb09061a979cb02080601fc6f7870d3362` |
| Zig | 0.17.0 |
| `open_dneo_oracle_test.zig` sha256 | `ceb5b154a305f4a0e12b21e38e091523c0c12f7047d08a89e93da28a2ca32a48` |
| `lane_core_rounds.jsonl` sha256 | `2cc9135f7234c1c3298dc557f1bb3f1008701f02bd191e446f3d744abbdd0a13` |

To make them again, in a TensorFold tree at that commit:

1. Copy `open_dneo_oracle_test.zig` to `zig/src/core/lanes/`.
2. In `zig/src/core/lanes/lanes.zig`, add `_ = @import("open_dneo_oracle_test.zig");` to the `test` block that
   imports `rollback_test.zig`.
3. Run `zig build -Dfixed-cpu test-shared-lane-regressions-host`. The test writes `open-dneo-oracle.jsonl` in the
   directory the build runs it from (the tree's root). An unchanged test is not run again: Zig answers the step from
   its cache and writes nothing.

## What the test sets up

- **The facts.** GLM-5.3-Flash's with a DFlash2 drafter and one stream, as `facts` in
  `zig/src/families/glm/cuda_lanes.zig` gives them: windows of 32 rows, a first copy of 4 rows, 7 drafts a round, the
  drafter's chances sizing each window, and the two cost tables of `zig/src/families/glm/cuda_copy_policy.zig`
  (52 + 6.8 ms a row, 76 + 4.1 ms a row). The copy lookup is the lane host's, with a least match of 4.
- **The target.** The lane core's fake backend, made to give a fixed reply (`cycle_after` 0 and a `pattern`).
  Its drafts are the reply's next tokens, one in five wrong.
- **The drafter's chances.** Each draft's chance is a fixed function of its anchor and its place in the block,
  k/1024 between 0.3 and 1, so that JSON carries it exactly. The test cuts and chains them as GLM does:
  `draftBlock`'s cut at 0.7 past the first draft, then `chancesFn`'s products. The fixture gives the chances before
  the cut, which the walk applies itself.
- **The clock.** A round takes 52 + 6.8 ms for each verified row past the first, and 1.6 ms for the drafter's pass:
  a model of serving's times, which only the window prices read.

## The cases

Each is one stream with its own reply: no repeats; periods of 1, 2, 5, 7 and 40 tokens, one of them with a token
changed once; a motif of 8 tokens followed by a new token each time, so that copies miss and the lookup falls silent;
two tokens at random; a reply that repeats the prompt; replies cut by the limit of new tokens; a reply that ends on
its end token; a first round whose chances are all 1, with room for many tokens and for 3. The last case, `low-first
(seen only)`, gives the first draft a chance of 1/16 at every third anchor. The window prices kept it there, as
`rounds.py` says they do, but that depends on the modelled clock.
