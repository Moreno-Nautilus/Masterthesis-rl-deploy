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

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **Masterthesis-rl-deploy** (803 symbols, 1610 relationships, 59 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/Masterthesis-rl-deploy/context` | Codebase overview, check index freshness |
| `gitnexus://repo/Masterthesis-rl-deploy/clusters` | All functional areas |
| `gitnexus://repo/Masterthesis-rl-deploy/processes` | All execution flows |
| `gitnexus://repo/Masterthesis-rl-deploy/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
