# Changelog

## 0.9.0 (2026-11-26)

The last release: development has ended, on a negative result. Stage 2's drafters in DNeo's form did not reach
DFlash2's predicted serving speed, and the work stopped by a line written before the last numbers. No weights are
published, and the package does not change.

- `receipts/2026-11-26-stage2`: the runs that decided it, on the evaluation set's open half. Drafters in three forms
  (DNeo's, arm A; the bridge alone; DFlash2's, arm C), trained on prompt-pass rows of the target's own replies
  and of a public set the same model wrote, for 200 to 2,400 steps, each read by G2's price: its predicted speed at
  G2's time of a round and its fastest cut, as a share of DFlash2's. The best in DNeo's form reached 0.9922, the
  bridge alone 0.9946, and arm C 1.009, from chances that match how its drafts land rather than from more drafts
  landing. A uniform temperature on the chances lowered the price. Training to 2,400 steps raised the drafts landed
  on held-out streams of the training text, and lowered the drafts landed and the price on the evaluation set. The
  stop line, arm C's 1.009, was set before the 2,400-step runs. Their best reached 0.9832. The receipt holds every
  drafter's score report, the price and calibration outputs, and `table.py`, which computes the price from the
  reports.
- `docs/plan.md`: where Stage 2 ended. G2 itself was not read: the sealed half was not served, scored or read and
  stays sealed, and MTP's round is not timed. Stages 3 to 6 were not started. The rows were recorded from Stage 1's
  checkpoint, not from the one of the MIT-licensed weights alone, and each cut was read by the cut alone, with the
  price of a row bounded.
- `README.md` and `README.ja.md`: the result at the top.
- `NOTICE`: the public dataset some of Stage 2's drafters were trained on, of which this project includes no text.

## 0.8.0 (2026-11-24)

The instrument counts the rounds as serving prices them, and reads them at G2's time of a round, measured in
serving. Still no weights, and this release holds no numbers from training.

- `open_dneo.rounds.priced`: of the drafts the chain's cut keeps, the prefix TensorFold's GLM lanes verify for its
  price, the one with the most expected tokens a ms at a lone window's price (76 ms and 4.1 ms a row for DFlash2,
  `allocateDrafts`). `rounds.served` takes it, so a drafter read at its own cut is counted as serving would take its
  rounds. At the default cut it drops no draft; a cut below about 0.29 lets the price end a chain, and a first draft
  under 4.1/76 leaves the round plain. `open-dneo score --run` sweeps the cuts both ways, by the cut alone and with
  the price.
- `open_dneo.rounds.ROUND_MS`, `DRAFT_MS` and `timed`: G2's time of a round, 41.87 ms and 5.87 ms a draft verified,
  fitted to serving's seconds for DFlash2's replies at five launches from cut 0.9 to 0.2, on two GB10s at tensor
  parallel 2. Over the whole run, `open-dneo score --run` gives the seconds of rounds at each cut at that time, and
  the cut with the least, as serving holds one cut a launch. A drafter whose round costs more borrows it, and MTP's
  round is not timed this way yet.
- `receipts/2026-11-23-round-time`: the fit of G2's time of a round, from five launches of G0's server on the open
  half's 320 prompts at cuts 0.7, 0.9, 0.5, 0.2 and 0.7 again: each request's seconds, rounds and drafts verified
  (no prompts or replies), the fit and its output, 41.87 ms and 5.87 ms over all with a within R² of 0.979.
- Tests: the lane core's allocator (`extendCosts` and `allocateDrafts`) written out as it stands, held to the
  engine's own test numbers, and `priced` held to it for a lone stream over 2,000 drawn chains, cuts, prices and
  overheads.
- `docs/plan.md`: where the instrument and serving part after the cut, and G2's time of a round. A cut set at launch
  (`TF_GLM_DRAFT_CONFIDENCE`) leaves a block drafter's row prices on; the plan said it took them out, which holds
  for MTP's `--mtp-confidence` only. Each drafter's cut is the one with the least time over the open half at that
  time and serving's price, decided on 2026-11-24, after trained drafters' numbers on the open half had been seen
  and before the sealed half was served or read; the plan had said the rule would come before those numbers.

## 0.7.0 (2026-11-23)

Stage 0's last tool: the command that trains a drafter on recorded rows, and the instrument's view of what training
changes. No weights yet, and this release holds no numbers from training.

- `open-dneo train FOLDER... --next NEXT --drafter D --target T --out O --steps N` (`open_dneo.train`,
  `open_dneo.trainer`, `open_dneo.feed`): a block is taught the target's own choices after its anchor, kept by
  `greedy --save`, as a cross-entropy, on the rows the target writes (an assistant's turn under GLM's chat template,
  or a plain text). A label past the first token the target would not have chosen has its own weight
  (`--off-path`, 0 by default), and a draft's weight falls with its position (`--decay`). Each step takes a window of
  anchors in a stream of every part (`--parts`, `--shares`) and an AdamW step at a rate warmed up and then down a
  cosine (`--peak`, `--end`, `--warmup`), times a factor for each part of the drafter (`--rates`). Streams held out
  by name are measured as the run goes, and `step-N` folders are kept that `score` reads and `--resume` goes on from.
  The defaults are placeholders, each with its source in `trainer.Settings`.
- `open-dneo score --dneo-mask --dneo-head K` (`drafter`): DNeo's mask, where a block row sees the context and its
  block's first row only, and DNeo's head carried over from DFlash2's, over the whole vocabulary (0) or the target
  head's best K. Carried over with K at DFlash2's own, it drafts as DFlash2 does. A trained drafter's folder carries
  its own entrance, mask and head.
- `open-dneo score --run`'s report, by class: how each position's drafts land once those before them have, every
  draft held; how chains land whole against the product of chances that serving's cut reads; the tokens and drafts
  verified a round at cuts from 0.2 to 0.9; how often drafts land against their chances; and how often the target head's best k on
  the drafter's states hold the next token, at the first draft and by position.
- `open-dneo greedy --save NEXT --top K`: also each row's best K tokens and their log probabilities
  (`<stream>.best`) and the next token's (`<stream>.nextlp`), so a recording need not keep the final row to train
  on.
- `open_dneo.weights.write`: a trained drafter saved as a safetensors file that `score` reads, its header from the
  arrays' shapes and the arrays written to the file one at a time, so a save copies one array at a time, not every
  tensor at once.
- `docs/plan.md`: Stage 0 is finished.

## 0.6.1 (2026-11-22)

The README explains parallel drafting for newcomers. No new tools or numbers, and still no weights.

- `README.md` and `README.ja.md`: how a target checks a drafter's guesses, also at a temperature; why the drafter
  guesses at temperature 0; how a parallel drafter differs from a sequential one; and what DNeo does instead of
  guessing one token at a time.

## 0.6.0 (2026-11-21)

Stage 1's step-0 bridge, gate G1: DNeo's entrance, a weighted sum of taps, fitted by least squares onto the published
DFlash2 drafter's layers, mask and head. Before any training, the bridged drafter lands 0.853 to 0.962 of DFlash2's
drafts a round, by class, so arm A leads. A bridge from the five deepest taps led one from DFlash2's own taps in three
classes of five; by the rule written before those numbers, Stage 2's entrance reads DFlash2's five taps, and the first
training run compares four sets of taps. No weights yet, and no result yet on whether the approach works.

- `receipts/2026-11-21-g1`: the fits and the step-0 scores on the evaluation set's open half. On G0's recording,
  DFlash2's taps with uniform weights, with `fc`'s block sizes, with B at 4 bits, and with B at zero (0.000 to 0.014
  of DFlash2's drafts). On a recording of the same prompts at ten taps, which gives G0's recording's instrument lines
  on the taps they share, the five deepest taps against DFlash2's five, both fitted on the pilot's nine parts.
- `open-dneo bridge FOLDER... --drafter D --out O` (`open_dneo.bridge`): B fitted by ridge least squares from DNeo's
  weighted sum of taps to the output of DFlash2's `fc`, with streams held out and scores by part; `--weights` for
  more weights than the uniform and even ones, `--inputs` for other taps in the sum. It writes
  `bridge-LABEL.safetensors`: B, the weights and the taps.
- `open-dneo score --bridge FILE` (`drafter.Bridged`): the drafter scored with that entrance, B in float32 or, with
  `--bridge-served`, at 4 bits. With `--run`, each class also gets the walk in which the drafter makes every round.
- `docs/plan.md`: G1 read as drafts landed and its result; the taps of Stage 2's entrance, the rule that chose them,
  and the first run's comparison of four sets; G2 read at each drafter's own cut, chosen before G2; the fifth move, a
  cross-entropy on the target's own greedy choices, and the first run's labels; the entrance that keeps its matrix,
  and what arm A's entrance does not test; Stage 2's body built from MIT-licensed weights alone; the target's own
  replies in the first recording, and the streams held out by prompt; where the 1% of G2 and G3 comes from, now a
  placeholder.
- `docs/survey.md`: the entrance, measured at step 0.

## 0.5.0 (2026-11-20)

Stage 1's pilot recording and layer probe: 609,309 prompt-pass rows of ten parts of public text, chosen by one rule
for who wrote the replies. The target's own greedy choice is the text's next token at 0.973 of the positions of its
own replies, at 0.84 to 0.88 of open models' code and agent replies, and at 0.52 to 0.60 of people's text and of
open models' Japanese. A linear probe finds that the deeper a tap, the more of the target's next tokens it holds:
the five taps 40 to 44 hold more than the five DFlash2 reads, at every k overall. No weights yet, and no result yet
on whether the approach works.

- `receipts/2026-11-20-pilot`: the composition (each part's writer, dataset, revision, licence, streams and rows,
  without the text), the target's agreement with the text by part and by role, and the probe's scores for 18 taps
  and three sets of taps from k 0 to 3. On the agent part the probe does not rank the taps: its eight prompts share
  a system prompt and tool list that the fit saw.
- `open-dneo greedy --save NEXT`: keeps the target's choice at every row of each stream counted, a little-endian
  int32 a row in `NEXT/<stream>.next`.
- `open-dneo probe FOLDER --target T --next NEXT` (`open_dneo.probe`): for each tap, and for sets of taps side by
  side, a linear map with an intercept fitted by ridge least squares from the rows at a position to the target's
  final row k positions on; on the streams held out, how often the head's best token on the mapped row is the
  target's own choice there, beside the floor, and by part with `--parts`. A configuration's sums are kept only
  while it is fitted, since a set's Gram can be gigabytes.
- `docs/plan.md`: who may write the training text's replies, among the decisions so far, and where Stage 1 stands.
- `NOTICE`: the datasets the receipt names.

## 0.4.0 (2026-11-20)

Stage 1's comparison of prompt rows with reply rows: from the rows a prompt pass computed, the published DFlash2
drafter proposes nearly what it proposes from the rows of the reply that wrote the same tokens. Walking alone, it
takes 0.004 to 0.029 fewer tokens a round from prompt rows, by class. No weights yet, and no result yet on whether
the approach works.

- `receipts/2026-11-20-rows`: the comparison on G0's judged launch, which recorded each reply and then the same
  tokens sent again as one prompt. Over 64 replies a class, the drafter alone takes 5.018 tokens a round from reply
  rows and 5.007 from prompt rows for code, 3.431 and 3.402 for agent, 2.402 and 2.398 for chat, 2.089 and 2.086
  for ja, and 2.034 and 2.025 for prose. Its first draft is the same from both at 95.7 to 99.2% of positions, and
  all seven drafts at 56.5 to 88.4%.
- `open-dneo rows FOLDER --drafter D --target T --run RUN` (`open_dneo.rows`): pairs each request with its reply's
  stream and with the stream whose prompt is the request's prompt and reply, and has the drafter propose at the
  reply's positions from each kind of row. Per class it prints the drafter-alone walk over each and how often their
  drafts agree. A reply whose prompt stream lacks a row or a context for one of its positions is counted and left
  out.
- Reading a recording holds one open file a stream. `score --run` and `rows` open every stream at once, and each
  stream held four memory maps, each an open file: G0's recording of 640 streams needed 2,560, past the 1,024 many
  systems allow a process by default. Tokens, positions and kinds are a few bytes a row, so they are now read whole,
  and only the tapped rows stay mapped. Under a soft limit of 1,024, the receipt's `rows` held 645 open files and
  finished, with the same report as before; with 0.3.0's reading it stopped at the start with too many open files.
- `score.proposals` takes the positions to propose at and the row the context starts from, and `score.loaded` is
  the loading that `score` and `rows` share.
- `docs/plan.md`: what the comparison found, where the plan names the gap.

## 0.3.0 (2026-11-20)

Gate G0 passes: the offline instrument counts a DFlash2 drafter's tokens a round within 0.013 of serving's in every
class of the evaluation set's open half, where the gate allows 0.1. No weights yet, and no result yet on whether
the approach works.

- `receipts/2026-11-20-g0`: the reading of G0, from four launches of one server on the open half's 320 prompts.
  The drafts serving logged are the instrument's at the same anchor: the first draft in 99.1 to 99.7% of rounds
  by class, all seven in 87.9 to 96.9%. A second launch with the drafter differs by one round, in one class. One
  launch can record rows and serve DFlash2 drafts, and recording changes no reply.
- `open-dneo score --run RUN` (`open_dneo.runs`): pairs each request of a run file with the stream that recorded it,
  walks the request's whole prompt and reply to its own limit of new tokens, and sets serving's tokens a round
  beside the instrument's, class by class.
- `open-dneo score --run RUN --drafts-log LOG` (`open_dneo.logged`): the drafts a TensorFold server logs with
  `TF_GLM_DFLASH2_LOG`, beside the instrument's at the same anchors, class by class.
- `score.lanes` takes a request's prompt and reply, so that copies are looked for in the whole prompt where a
  stream begins after it, and the reply's last token is walked.
- `tests/test_lane_core.py` and `tools/oracle`: the walk's rounds are those of TensorFold's lane core, which
  `tools/oracle` records on 16 cases at TensorFold `e3fb56cb09`.
- `docs/plan.md`: "How G0 is measured" was made exact after the evaluation set was built and before serving's
  number was read, and holds G0's result.

Changes to what 0.2.0 said:

- `docs/plan.md` had the prompts go as plain completions. Each is its messages under the chat template, sent as
  token ids to the completions route where those ids are the ones the chat route runs, which they were on the six
  prompts checked.
- The evaluation set has a sealed half that nothing reads before G2, and G2 is read on it.
- The launch G0 judges is the first counted one with the drafter that serves the whole open half and records its
  rows, and no launch without rows is judged in its place.
- The walk over a reply's own tokens, which 0.2.0 listed as not written, is `score --run`. Without `--run` the
  walk is still over the tokens a recording holds.
- Whether one launch can record rows and serve DFlash2 drafts was not checked. It can.

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
