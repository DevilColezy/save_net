# v25 depth-to-control policy

This package trains two causal students for the hierarchical expert: a
**30 Hz local-avoidance student** and a **5 Hz upper-planner student**.
on the non-privileged contract written by `il_dataset` schema v25:

- input: one depth image plus gravity direction (3), current FLU velocity
  (3), yaw rate (1), FLU goal direction (3), and clipped goal distance (1);
- output: FLU velocity `(vx, vy, vz)` and yaw rate;
- network: two-stage overlapping Mix-Transformer visual encoder, state MLP,
  and a three-layer LSTM inspired by ViTFly.

`train.py` trains only the 30 Hz local policy.  `train_macro.py` trains the
5 Hz macro policy from the same committed episodes.  The two trainers have
separate checkpoints and recurrent states; a 5 Hz decision is made only on
the real `macro_update_mask==1` rows.

`planner_status` and `hierarchical_mode` are teacher/diagnostic metadata —
they are never policy inputs (the loader strictly validates their values and
maps `planner_status` to a stable index; see `dataloader.py`).
`planner_status` is a validated categorical diagnostic, not a student input.

Audit a dataset before training:

```bash
python src/save_net/train.py --dataset-root il_data --audit-only
```

Train:

```bash
python src/save_net/train.py --dataset-root /path/to/il_data \
  --output-dir checkpoints/vitfly_v25 --epochs 40
```

Training defaults to stateful truncated BPTT: a trajectory is emitted as
chronological 32-frame chunks, and its LSTM `(h, c)` is passed from each chunk
to the next before being detached. The state is reset only for the first chunk
of a new trajectory. This matches streaming inference without retaining an
unbounded backward graph. The legacy independent-window mode is still
available for ablation with `--stateless-windows`; only that mode uses the
`--burn-in` context and weighted window sampling. Stateful augmentation uses
one consistent horizontal-flip choice for all chunks of a trajectory in an
epoch, so a hidden state is never transferred between incompatible frames.

The loader accepts only atomically committed, successful schema-v25 episodes,
checks contiguous frame indices and every referenced depth PNG, never joins
sequences across episodes, and uses a scene-disjoint validation split whenever
at least two scenes exist. A one-scene dataset falls back to an episode split
with a warning and must not be treated as a generalization result.

The collector now also includes `continuous_tracking` episodes.  Their local
guide point is refreshed at 5 Hz with small, smooth heading/lookahead changes
and a 2.5--4.5 m lookahead, so the LSTM sees the same piecewise-continuous goal input
expected from an upstream planner.  Fixed goals and abrupt one/two-switch
episodes remain in the mixture for endpoint and recovery behaviour.

> NOTE: that paragraph describes an OLD goal-switching collector.  The
> current `il_dataset` pipeline writes the NEW hierarchical expert: the
> 30 Hz goal input is the live effective target (PASS = original goal,
> NORMAL = re-expressed world-latched correction, TURN = re-expressed
> world-latched turn direction, norm == 1) and there are no
> `goal_switch_event`/`active_goal_*`/`abrupt_goal_switch` fields.  The
> `rollout.py` deployment tool still references the old fields and is NOT
> compatible with v25 `hierarchical_mode` data (see `src/il_dataset` README).

Horizontal mirror augmentation is disabled by default.  It is geometrically
valid (the loader mirrors every lateral state and command component) but a
synthetic, centred obstacle can have a deterministic expert tie-break side.
Mirroring such an otherwise indistinguishable observation supplies both
opposite commands and weakens the intended avoidance action.  Use
`--mirror-augmentation` only as an explicit ablation after the unmirrored
policy has passed the centred-pillar rollout tasks.

## 5 Hz upper-planner network

The dataset carries real 5 Hz directives every 6 committed frames
(`macro_update_mask==1` on `episode_frame_index % 6 == 0`).  The macro loader
filters those rows before constructing windows; it never trains on the six
zero-order-held copies.  Train it with:

```bash
python src/save_net/train_macro.py --dataset-root /path/to/il_data_joint_v2 \
  --output-dir checkpoints/macro_v25 --epochs 40
```

The macro network input is one depth frame plus 11 values:
`gravity_flu(3)`, current FLU velocity (3), yaw rate (1), and the **original**
navigation-goal direction plus clipped distance (4).  It outputs:

- directive type: `PASS_THROUGH`, `NORMAL_CORRECTION`, `TURN_LEFT`,
  `TURN_RIGHT`;
- direction token 0..12 for non-PASS directives;
- a unit FLU direction and normalized distance for the correction target.

The runtime adapter should treat the type head as authoritative, use the
token/direction/distance heads only for NORMAL/TURN, re-express the world-
latched target in the live body frame, and keep the GRU/LSTM state at 5 Hz.
`MacroPlannerPolicy.decode_directive(...)` applies these legality clamps
(PASS token `-1`, TURN distance `1`, ordinary NORMAL token range `1..11`).

The loader still performs the complete legality audit:

- **validates** every committed 5 Hz row at discovery time (item 三):
  mask ∈ {0,1}; mask==1 only at frame index %6==0; mask==1 ⇒
  `macro_label_valid==1`; `macro_correction_type` ∈ the four directive
  classes; PASS ⇒ token -1 + `macro_param_valid==0`; NORMAL ⇒ token in
  [1,11] + param 1 + `distance_norm < 1`; TURN_LEFT ⇒ token 0 + param 1 +
  `distance_norm == 1`; TURN_RIGHT ⇒ token 12 + param 1 + `distance_norm == 1`;
  all 5 Hz inputs/labels finite; NORMAL/TURN direction is a unit vector;
- returns `state_5hz` (original navigation goal direction FLU + distance
  norm) and `label_5hz` (encoded masks/types/token/direction/norm/param) per
  row for a future independent trainer;
- the future 5 Hz student may sample ONLY `macro_update_mask==1` rows —
  never the 6 zero-order-hold copies between decisions as if they were 6
  independent 5 Hz samples;
- mirror augmentation mirrors exactly the left axis (`state_5hz` index 1,
  `label_5hz` index 5) and swaps TURN_LEFT↔TURN_RIGHT (type index 2 and
  `hierarchical_mode`) and tokens over the full range 0↔12, 1↔11, … (PASS
  -1 unchanged), consistent with the expert `direction_bin_count=11` token
  contract (shared constants in `dataloader.py`).

The 5 Hz recurrent input is the depth / local observation at the macro
instant, not a re-sampled six-frame mean.  Validation uses the same
scene-disjoint split as the 30 Hz trainer.

The current collection task is deliberately 2.5D: procedural cylinders span
the flight volume and the expert avoids them horizontally. `vz` supervision
therefore covers altitude stabilization/transients, not learned over/under
obstacle avoidance. Adding true 3D avoidance requires a 3D expert and partial-
height scene distribution together; changing only obstacle heights would make
the labels inconsistent.

## Closed-loop rollout

`rollout.py` loads the checkpoint format written by `train.py` directly. It
strictly checks schema v25, the `ViTFlyLSTMPolicy` architecture, student field
order, depth range and normalization before connecting to Unity. The LSTM
state is initialized once per episode and is then carried across every frame,
including abrupt and continuous goal changes.

Start AvoidBench on the same ports used by collection, then run the basic
acceptance suite (the model file is inferred automatically):

```bash
python3 src/save_net/rollout.py \
  --checkpoint ./il_data_pilot/checkpoints/best.pt \
  --tasks basic --repeats 3 \
  --log-prefix ./il_data_pilot/rollout/basic
```

Useful alternatives are `--list-tasks`, a comma-separated task selection, and
`--tasks stress`. The default `basic` suite covers clear flight, isolated and
forced-side avoidance, a gate, a rear goal, an abrupt switch and smooth 5 Hz
goal updates. Stress tasks contain separated compound detours. All cylinders
are full height and every obstacle pair has at least 1.20 m surface separation,
matching the collection geometry contract. The old partial-height
`climb_over` task was removed because the v25 2.5D expert does not supervise
over/under avoidance.

The continuously flushed `*_steps.csv` records active goals, commands,
clearance, depth statistics and inference time. `*_summary.json` records
success/collision/timeout outcomes per task. `--lstm-reset-interval` is only an
ablation option; keep its default value of zero for normal evaluation.
