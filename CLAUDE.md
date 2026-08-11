# Masterthesis-rl-deploy — Claude context

Real-robot deployment of the end-to-end RL cooling-screw insertion policy.
iiwa7 + custom Y-gripper, wrist D405 RGB-D + wrist F/T + proprioceptive state → 5-DoF
policy delta action → damped-least-squares differential IK → FRI joint-position streaming @ 15 Hz.

For the practical run/build/safety flow, `README.md` and `docs/HOW_TO_DEPLOY.md` are the
source of truth. This file is orientation + working rules; the deeper "why" lives in memory
(see the bottom of this file).

## Source of truth (lives in the training repo)
- Policy is trained in `~/Masterthesis-rl-train` for task
  `Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0`. The deploy node loads the
  **rl_games actor only** and must rebuild the *exact* sim observation.
- Checkpoints: `~/Masterthesis-rl-train/logs/rl_games/Forge/<run>/nn/`.
  - `e2e_weld_curric` = the 85.2% baseline (no estimator, no gravity comp).
  - `w2_estimator_192` = the **DEPLOYED WINNER** (`nn/last_Forge_ep_2000_rew_162.13815.pth`):
    explicit-estimator policy, 83.2% @ full ±2.5 cm socket noise (vs 67.8% without). Same E2E policy
    as `e2e_weld_curric` + an internal vision estimator head. Two deploy consequences, both handled in
    code: (1) its `agent.yaml` adds a privileged `aux_label` obs group + `aux_head` — training-only,
    zero effect on the action; the loader auto-declares `aux_label` in the obs space and feeds dummy
    zeros `(1,4)`. (2) trained with GRAVITY COMPENSATION → `ft_force` is pure contact force, so keep
    `ft_bias_base_xyz: [0,0,0]`.

## Hard invariants (get these wrong and it silently fails)
- **Observation parity is non-negotiable.** The `policy` vector order and the `image`
  normalization (RGB `[0,1]` w/ per-image mean subtraction; depth inf→0, clamp `[0,far]`, /far;
  then temporal frame-stack oldest→newest) must match sim byte-for-byte. Use the `obs_parity`
  tool + `obs_dump_path` gate before enabling motion.
- **Socket = position anchor only.** Policy ignores socket orientation by design; only socket
  *position* feeds the action anchor. `socket_part_id: -1` holds motion closed.
- **Camera extrinsics = empirical at deploy.** The CAD `T_flange_cam` is locked in sim, but the
  CAD datum is the mount, not the COLOR optical frame (D405 is stereo → ~cm residual). Use
  `aligned_depth_to_color` and fix the residual empirically against a pose-matched real frame.
  See memory `d405-handeye-calib`.
- **The gripper has NO tactile sensor.** Only wrist F/T is real. Force safeguards latch hold
  above `ft_force_cap_n`; no tactile/force-history fusion exists.
- **Control is 15 Hz.** Sim was trained at 15 Hz with no latency buffer — watch real latency.
- Start guarded: motion disabled by default, `max_joint_step_rad: 0.010`,
  `e2e_pos_action_scale: 0.01`, `freeze_on_seat: true`. Walk the README safety gates in order.

## Working rules (behavioral — apply here too)
- **The user does ALL git** (commits, branches, pushes). Never commit or offer to. Prepare files;
  the user handles version control.
- **The user launches ALL GPU / training / real-robot-motion runs.** Prep, probe, validate, then
  wait for an explicit go. Never auto-launch.

## Memory
This repo's Claude memory was seeded (2026-08-06) by copying all 30 memory files from the
`Masterthesis-rl-train` slot. `memory/MEMORY.md` is the index loaded each session. Most relevant
here: `sim2real-deploy-checklist`, `d405-handeye-calib`, `e2e-deploy-run`, `e2e-weld-curric-run`,
`custom-gripper-no-tactile`, `iiwa-grasp-bug`, `iiwa-gripper-env-build`. Several entries
(`lr-cascade-instability`, `aux-head-build`, `weekend-squash-run`, `pb-screw-task-design`,
`*-rebaseline`) are training-loop internals kept for provenance — treat as background, not deploy guidance.
