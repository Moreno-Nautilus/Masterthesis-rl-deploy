# THESIS_CONTEXT_EXPORT.md

**Repository:** `Masterthesis-rl-deploy` — real-robot deployment of an end-to-end RL cooling-screw insertion policy
**Branch:** `moreno` (main branch: `master`) · **Export date:** 2026-09-14
**Author:** Moreno Wälchli (`moreno.waelchli@gmx.ch`, GitHub `Moreno-Nautilus`)

> Purpose: a single exhaustive context document for writing the thesis **methodology** and **results**
> sections. It reproduces in full every project document (`CLAUDE.md`, `README.md`, `docs/*.md`, the
> diagnosis/handoff notes, the pre-deploy runbooks), records the complete architecture as implemented,
> the chronological bug/fix history with technical reasoning, all quantitative experimental results,
> every design decision with its justifying evidence and rejected alternatives, and the current open
> issues. Where a number is quoted, it is the actual measured or configured value found in the repo.

---

## Table of Contents

1. [Scope, System Overview, and Provenance](#1-scope-system-overview-and-provenance)
2. [Full Architecture As Currently Implemented](#2-full-architecture-as-currently-implemented)
3. [Chronological Bug and Fix History](#3-chronological-bug-and-fix-history)
4. [Experiments and Quantitative Results](#4-experiments-and-quantitative-results)
5. [Design Decisions, Evidence, and Rejected Alternatives](#5-design-decisions-evidence-and-rejected-alternatives)
6. [Current TODOs and Known Open Issues](#6-current-todos-and-known-open-issues)
7. [Verbatim Reproduction of All Repository Documents](#7-verbatim-reproduction-of-all-repository-documents)
8. [Appendices: Configuration, Tests, File Inventory](#8-appendices-configuration-tests-file-inventory)

---

# 1. Scope, System Overview, and Provenance

## 1.1 What this repository is

`Masterthesis-rl-deploy` is a ROS 2 Humble `colcon` workspace that executes a policy trained in a
separate repository (`~/Masterthesis-rl-train`) on real hardware. It contains **no training code**.
Its entire job is to rebuild, on a physical robot, the exact observation the policy saw in
simulation, run the actor, and convert the resulting 5-DoF delta action into safe joint-position
commands streamed over KUKA FRI.

The one-line system description used throughout the project:

> iiwa7 + custom Y-gripper (later: pdz gripper), wrist D405 RGB-D + wrist F/T + proprioceptive state
> → 5-DoF policy delta action → damped-least-squares differential IK → FRI joint-position streaming
> @ 15 Hz.

## 1.2 Hardware and software environment

| Item | Value |
|---|---|
| Robot | KUKA LBR iiwa7 (7-DoF), dual-arm cell; deployment uses the **RIGHT** arm (`lbr_two`) |
| Left arm | `lbr_one` — in live use by another student; not brought up in single-arm mode |
| FRI | Right arm port **30201**, subnet 192.170.20.x, PC `192.170.20.1/24` on `enp3s0` |
| Left-arm FRI | port 30200, 192.170.10.x (alias kept, unused in single-arm mode) |
| End effector | Custom Y-gripper, later "pdz" gripper. **No tactile sensor** — wrist F/T only |
| Wrist camera | Intel RealSense **D405** RGB-D. Right arm = `realsense_2`, serial `260522275434`; left arm = `realsense_1`, serial `260322275185` |
| Scene camera | ZED2i (dual-arm mode only; not used in single-arm mode) |
| ROS | ROS 2 **Humble**, Ubuntu 22.04.5 |
| LBR stack | `lbr_fri_ros2_stack` humble branch **v2.2.3** (async `FTEstimator`, *not* the jazzy `WrenchEstimator`) |
| ros2_control | **2.54.0** (has `ChainableControllerInterface`, lacks `on_export_state_interfaces`) |
| Kernel | **6.8.0 generic** (PREEMPT_DYNAMIC, **not** PREEMPT_RT) |
| Torch | `torch 2.11.0+cu128` — an unusual build, **not on PyPI** (drives the env strategy, §5.2) |
| Policy framework | `rl_games` 1.6.5, `gymnasium` 1.2.1 |
| Control rate | **15 Hz** policy loop (66.7 ms/tick); FRI internal loop 10 ms |

## 1.3 Provenance: the training repository

The policy is trained in `~/Masterthesis-rl-train` for the Isaac Lab task
`Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0`. Checkpoints live under
`~/Masterthesis-rl-train/logs/rl_games/Forge/<run>/nn/`.

The deploy node loads the **rl_games actor only** and must rebuild the *exact* sim observation.
Sim-side source files referenced by deploy code and notes:

- `insertion_env_e2e_iiwa.py` — the E2E env; `_goal_delta_ee` (docstring L290) defines the reset
  geometry; L314 defines the raw, deadband-free contact force fed to the policy.
- `insertion_env.py::_get_camera_image` — the image normalization the deploy path mirrors.
- `InsertionEnv._stack_frames` — the oldest→newest temporal stack order.
- `factory_control.get_pose_error(..., jacobian_type='geometric')` — the IK error convention.
- `cooling_iiwa_tasks_cfg.py:574` — "Full 16:9 single-frame RGB, downscaled to 320x180 without cropping".
- `insertion_hybrid_network.py` — the custom rl_games network (registered at deploy without Isaac).

## 1.4 Checkpoint lineage (which policy is deployed, and why)

Deployment moved through several checkpoints. This lineage matters because each one changed the
observation/force contract, and several deploy bugs are direct consequences of those changes.

| Checkpoint | Date | Gripper | Obs contract | Key property |
|---|---|---|---|---|
| `e2e_weld_curric` | 2026-08-05 | Y-gripper | 21-D policy + 224×224×4 RGB-D | The **85.2 %** baseline. No estimator, **no gravity comp** |
| `w2_estimator_192` | 2026-08-10 | Y-gripper | 21-D + `aux_label`(4) + 224×224×4 | **Explicit-estimator winner**, 83.2 % @ full ±2.5 cm noise. Gravity comp **ON** |
| `we_A_resnet18_baseline` | ~2026-08-18 | Y-gripper | **15-D** + 180×320×3 ImageNet RGB, 6-D action | `use_gravity_comp: false` |
| `pdz_overnight_20260824` | 2026-08-24 | **pdz** | 15-D + 180×320×3, 6-D action | ResNet-18 `layer4` unfrozen; **broken** gravity comp (~0.75× weight baked in) |
| `pdz_v3_fulltilt_20260826` | 2026-08-26 | pdz | 15-D + 180×320×3, 6-D action | **CURRENT live profile.** Corrected gravity comp, halved action scale, full 25° tilt |

The README/CLAUDE.md default configuration still names `w2_estimator_192`; the *live* single-arm
deployment profile (`deploy_pdz_v3_fulltilt_single.yaml`) runs
`pdz_v3_fulltilt_20260826/nn/deploy_final_ep2000.pth`. Both paths are maintained in the codebase.

## 1.5 Hard invariants (from `CLAUDE.md`)

These are the project's stated non-negotiables; each is enforced somewhere in code.

- **Observation parity is non-negotiable.** The `policy` vector order and the `image` normalization
  (RGB `[0,1]` w/ per-image mean subtraction; depth inf→0, clamp `[0,far]`, /far; then temporal
  frame-stack oldest→newest) must match sim byte-for-byte. Use the `obs_parity` tool + `obs_dump_path`
  gate before enabling motion.
- **Socket = position anchor only.** Policy ignores socket orientation by design; only socket
  *position* feeds the action anchor. `socket_part_id: -1` holds motion closed.
- **Camera extrinsics = empirical at deploy.** The CAD `T_flange_cam` is locked in sim, but the CAD
  datum is the mount, not the COLOR optical frame (D405 is stereo → ~cm residual). Use
  `aligned_depth_to_color` and fix the residual empirically against a pose-matched real frame.
- **The gripper has NO tactile sensor.** Only wrist F/T is real. Force safeguards latch hold above
  `ft_force_cap_n`; no tactile/force-history fusion exists.
- **Control is 15 Hz.** Sim was trained at 15 Hz with no latency buffer — watch real latency.
- Start guarded: motion disabled by default, `max_joint_step_rad: 0.010`,
  `e2e_pos_action_scale: 0.01`, `freeze_on_seat: true`.

## 1.6 Working rules (process constraints on the project)

- **The user does ALL git** (commits, branches, pushes).
- **The user launches ALL GPU / training / real-robot-motion runs.** Prep, probe, validate, then wait
  for an explicit go. Never auto-launch.

These are recorded because they shape the evidence trail: the git history is coarse (8 commits) and
much of the engineering record lives in the markdown notes and in-code comments rather than in commits.

## 1.7 The physical task

Insertion of a cooling screw / peg into a socket on a "cooling base" part.

| Geometry | Value | Source |
|---|---|---|
| Socket opening (inner rim) diameter | **14.0 mm** | CAD `cooling_base.obj`, r ≈ 7.0 mm; equals sim `CoolingInsert.asset_size` |
| Outer counterbore rim diameter | ≈ **17.5 mm** | CAD (secondary ring, not used by default) |
| Socket depth | **17.5 mm** (`seat_socket_depth_m: 0.0175`) | sim cooling socket |
| Two sockets per base | local x = **±30 mm** → **60 mm** centre-to-centre | CAD; used as the pairing cue |
| Radial clearance | **~1 mm**, **chamfer-free** (3D-printed by design) | — |
| Seated tolerance | 25 % of depth ≈ **4.4 mm** above bottom | sim convention |

The chamfer-free ~1 mm clearance is the binding constraint: it makes precise **lateral** alignment the
dominant difficulty and is the original motivation for using vision at all.

---

# 2. Full Architecture As Currently Implemented

## 2.1 Workspace layout

```
Masterthesis-rl-deploy/
├── src/
│   ├── rl_deploy_inference/        ROS2 ament_python deployment package (all deploy logic)
│   ├── fp_debug_msgs/              FoundationPose debug/action/message definitions
│   └── lbr_fri_idl/                FRI ROS2 message definitions (from the local LBR stack)
├── docs/                           HOW_TO_DEPLOY, DEPLOY_PDZ, DEPLOY_WE_A, PREINSERT_MOVEIT
├── deploy_analysis/                Rosbag-derived CSVs, drop times, plot scripts, PNG plots
├── scripts/                        RT/CPU-pinning hardware bringup wrappers
├── deploy_env.sh                   Zero-install Python environment bridge
├── deploy_dds_no_fri_nic.xml       FastDDS profile keeping DDS off the FRI NIC
├── CLAUDE.md, README.md, AGENTS.md
├── DIAGNOSIS.md, FRI_CONTACT_DROP_{FIXES,HANDOFF}.md, FRI_RECEIVE_MULTIPLIER_FIX.md
├── CONTACT_SAFE_DEPLOYMENT_RATIONALE.md
└── PREDEPLOY_pdz_v3_fulltilt{,_SINGLE}.md
```

### Python module inventory (`src/rl_deploy_inference/rl_deploy_inference/`)

| File | LOC | Role |
|---|---:|---|
| `inference_node.py` | 1181 | **Main E2E node** (21-D obs, RGB-D, 5-D action) |
| `hole_align_planner.py` | 1018 | Vision-corrected MoveIt preinsert + corrected-anchor republisher |
| `preinsert_planner.py` | 802 | Headless MoveIt gross move to preinsert hover |
| `hole_detector.py` | 555 | Pure, ROS-free socket-opening localizer (contour + Hough) |
| `serl_backend.py` | 474 | SERL HTTP robot-server backend (real-robot RL fine-tuning path) |
| `motion_commander.py` | 457 | Reusable single-arm MoveIt `MoveGroup` action client |
| `inference_node_we_a.py` | 426 | **we_A / pdz node** (15-D obs, RGB-only, 6-D action) |
| `ik.py` | 251 | Isaac-free port of the Factory differential-IK controller |
| `guarded_joint_trajectory_bridge.py` | 232 | Guarded 15 Hz → JTC bridge (contact-safe path) |
| `obs_preprocessing.py` | 226 | E2E RGB-D preprocessing, frame stack, force EMA, policy vector |
| `policy.py` | 211 | rl_games actor adapter (Isaac-free network registration, aux_label) |
| `guarded_joint_smoother.py` | 179 | Stateful quintic q/dq/ddq planner with vel/acc/jerk limits |
| `command_upsampler.py` | 139 | Legacy 15 Hz → 200 Hz position bridge (adaptive ramp) |
| `kinematics.py` | 134 | KDL FK/Jacobian wrapper with fallback tip + offset |
| `obs_preprocessing_we_a.py` | 127 | we_A ImageNet RGB preprocessing, 15-D policy vector |
| `ik_we_a.py` | 117 | we_A TCP-relative action/goal-delta math |
| `kdl_parser.py` | 101 | **Vendored** `kdl_parser_py` (not packaged for Humble) |
| `policy_we_a.py` | 75 | we_A actor adapter |
| `deployment_guards.py` | 71 | Pure FRI-state check + action norm limiter |
| `parity_check.py` | 43 | `obs_parity` CLI: sim vs deploy npz comparison |

Console entry points (`setup.py`): `inference_node`, `inference_node_we_a`, `command_upsampler`,
`guarded_joint_trajectory_bridge`, `obs_parity`, `preinsert_planner`, `hole_align_planner`.

## 2.2 End-to-end runtime pipeline

The complete deployment is a sequence of five stages. Stages 1–3 run **once** before the policy; stage
4 is the 15 Hz closed loop; stage 5 is the command path to the robot.

```
 STAGE 1  GROSS MOVE           preinsert_planner  ──MoveIt/MoveGroup──▶ JTC ──▶ FRI
          (perception anchor, ~1 cm off; hover 0.15 m)

 STAGE 2  VISION CORRECTION    hole_align_planner ──hole_detector──▶ detected opening (base_link)
          └─ re-aims the hover at the DETECTED hole (hover 0.045–0.06 m)

 STAGE 3  ANCHOR REPUBLISH     hole_align_planner --republish-anchor
          └─▶ /rl_deploy/corrected_socket  @ 20 Hz  (frozen or live)

 STAGE 4  POLICY LOOP  @15 Hz  inference_node._control_tick()
          RGB+depth ─┐
          joint q  ──┼─▶ KDL FK/Jacobian ─▶ fingertip pose, ee vel
          τ / wrench ┼─▶ contact force (base frame)
          anchor ────┘
                     └─▶ obs {policy(21), image(224,224,4)} ─▶ rl_games actor ─▶ a(5)
                     └─▶ EMA ─▶ norm limits ─▶ action_to_target_pose ─▶ pose error
                     └─▶ DLS IK ─▶ dq ─▶ joint step clamp ─▶ q_cmd

 STAGE 5  COMMAND PATH         q_cmd ─▶ /rl_deploy/command_15hz
          ├─ LEGACY:  command_upsampler       ─▶ 200 Hz direct positions ─▶ LBR cmd controller
          └─ CURRENT: guarded_joint_trajectory_bridge ─▶ q/dq/ddq quintic horizons ─▶ JTC ─▶ FRI
```

## 2.3 Stage 1 — Gross preinsert (`preinsert_planner.py`, `motion_commander.py`)

Moves **one arm** to a safe preinsert pose hovering above the perceived socket, using **MoveIt** for
the gross motion — deliberately *not* the RL `reset_preinsert` IK servo, which "rattles from far away"
(a local damped-IK servo is unsuitable for large moves).

- Talks to an already-running `move_group` over the `moveit_msgs/action/MoveGroup` action; it is not a
  MoveIt process itself, so it needs no `robot_description`/SRDF/kinematics of its own.
- Arm selection: `--arm right` → group `arm_two`, tip `lbr_two_gripper_tcp`; `--arm left` → group
  `arm_one`, tip `lbr_one_gripper_tcp`. **Only the chosen arm's 7 joints are commanded.**
- **Planning frame == perception frame == `base_link`** (URDF root). `lbr_two_link_0` is a fixed child
  at `xyz = 0 0.42 0`. Perception publishes the socket in `base_link` and the hover is a global `+z`
  offset, so no frame conversion is needed.
- `orientation_mode`: `current_tcp` (default — the perceived *object* orientation is deliberately not
  trusted), `fixed`, or `down` (tool straight down; used by the live runbook).
- Conservative scaling: `velocity_scaling: 0.05`, `acceleration_scaling: 0.05` (well under the required 0.1).
- `target_mode: local_ik_joint` (see §3, bug B-11) computes a nearby joint target with KDL/DLS from the
  **current branch**, then plans in joint space, avoiding OMPL branch flips.
- Dry-run by default; `--execute` requires typing `MOVE`; `--yes` skips the prompt for scripts.
- Logs per run: current TCP pose, socket pose, computed target, plan success + `MoveItErrorCode` name,
  trajectory point count/duration, and **first and last joint targets** with joint names.

**Execution prerequisite:** the dual-arm `joint_trajectory_controller` spans **all 14 joints**, so
commanding one arm's 7 joints requires `allow_partial_joints_goal: true` on it, or the controller
rejects the trajectory ("Joints on incoming trajectory don't match the controller joints"). This flag
took single-arm execute from **0/14 → 13/14** successful plans. Plan-only does not need it.

## 2.4 Stage 2 — Vision correction (`hole_align_planner.py` + `hole_detector.py`)

Perception (FoundationPose) publishes the *tracked CAD centre* of `cooling_base`, which is **~1 cm off**
— enough to defeat the 1 mm-clearance socket. This stage localizes the actual socket **opening**
directly in the wrist D405 image.

Procedure:
1. Grab one time-synced RGB-D frame + `CameraInfo` (`img_sync_slop_s: 0.05`, queue 30).
2. Detect the opening (below), sampling opening depth from the aligned depth image.
3. Deproject the circle to a metric 3D point; transform to `base_link` via **live flange pose (TF) ∘
   calibrated camera-to-flange extrinsics** (`camera_extrinsics_realsense.yaml` — the same map the
   vision pipeline uses).
4. Re-aim the MoveIt preinsert hover at the **detected** hole, with the same `MotionCommander` +
   branch-safety.

Perception's socket pose is used **only** to disambiguate the base's two sockets and reject spurious
circles — never as the final target. The node prints the estimate-vs-detected delta.

### Detector (`hole_detector.py`) — pure numpy + OpenCV, no rclpy, unit-testable

Two methods:

- **`contour` (DEFAULT, robust)** — isolate the vividly-coloured part with an HSV **saturation mask**
  (the metal table and black gripper are gray → dropped), find dark blobs inside it, and keep only
  **round** ones by shape. The cooling **fins are thin elongated slots** → rejected by shape; only the
  circular sockets survive. Then a **CAD cross-check pairs the two sockets by their known 60 mm
  spacing**, which uniquely fixes them without perception or colour tuning.
- **`hough` (legacy)** — `cv2.HoughCircles` sized by the known hole diameter. **Over-triggers badly on
  the fin texture**, so it is not the default; kept only for comparison.

Tuned shape/segmentation parameters (`hole_align.yaml`):

| Parameter | Value | Rationale |
|---|---|---|
| `hole_diameter_m` | 0.014 | CAD inner rim; 0.0175 chases the counterbore instead |
| `radius_tol_frac` | 0.45 | radius window = predicted_px × (1 ± 0.45) |
| `socket_spacing_m` / tol | 0.060 / 0.012 | the CAD pairing cue |
| `sat_min` / `val_min` | 60 / 40 | HSV part segmentation; `val_min` drops the near-black gripper |
| `adaptive_block` / `adaptive_C` | 51 / 5 | local-darkness neighbourhood |
| `morph_open` | **false** | OFF: it *erodes real socket rims*; shape filters reject fins instead |
| `min_circularity` | 0.65 | 4πA/P²; fins are far from round |
| `max_aspect` | 1.8 | bbox long/short; fins are very elongated |
| `min_solidity` / `min_fill` | 0.80 / 0.55 | area/convex-hull, area/min-enclosing-circle |
| `require_part_surround` / `surround_part_frac` | true / 0.85 | a real socket is embedded in fins; a **mounting through-hole sits at the part edge** with table visible around it |
| `max_center_dist_px` | 60.0 | chosen circle must be near the projected perception estimate |
| `fallback_to_perception` | **false** | detection failure **aborts** rather than silently reverting to the ~1 cm-off pose |

**Debug-first:** debug is ON and motion OFF by default. Every run writes `holes_<ts>.png` (all circles
yellow, chosen one green, projected perception estimate as a red `+`) and `depth_<ts>.png` to
`/tmp/hole_align`.

## 2.5 Stage 3 — Corrected-anchor republisher

`hole_align_planner --republish-anchor` captures the CV hole and republishes it on
`/rl_deploy/corrected_socket` at **20 Hz** as a `DebugPoseItem` (`anchor_assembly_name: cooling_base`,
`anchor_part_id: 0`). The RL node consumes this as its socket anchor with
`socket_opening_offset_cad_xyz: [0,0,0]`.

`socket_anchor_mode: frozen` (default) captures the CV hole **once** and republishes it for the whole
trial; `live` tracks the latest perceived pose every tick. If the hole is occluded during the trial,
`anchor_use_fixed: true` + `anchor_xyz` pins a value taken from the hole-align dry run.

## 2.6 Stage 4 — The 15 Hz policy loop (`inference_node.py`)

`RLDeployInferenceNode` (`rclpy.Node`, name `rl_deploy_inference`). Timer period = `1/control_hz` =
66.7 ms. Three modes: `MODE_HOLD`, `MODE_POLICY`, `MODE_RESET_PREINSERT`.

### 2.6.1 Node construction order (`__init__`, L138–207)

1. `_declare_parameters()` — ~110 ROS parameters (full list in §8.1).
2. Allocate `TimedValue` slots for every input (rgb, depth, socket, flange, ft, joints, effort,
   external torque, LBR state), each carrying a monotonic wall time for freshness checks.
3. `_make_obs_config()` — build `ObsPreprocessConfig`, **auto-loading `env.yaml`** from the checkpoint's
   `params/` dir for image H/W/C, `frame_stack`, `ft_smoothing_factor`, depth far clip, and the sim
   pinhole intrinsics. Hard-fails unless `image_channels == 4`.
4. `_load_actor()` → `RlGamesActor`.
5. `_init_kinematics_once()` → KDL from `robot_description` (param, or fetched via the
   `robot_state_publisher` `GetParameters` service).
6. `_setup_ros_io()` — subscriptions, publishers, services.
7. **`_warmup_actor()`** — one throwaway inference (see §5.6).
8. Create the control timer.

### 2.6.2 Inputs and freshness gating (`_ready`, L797)

`_ready()` returns `(False, reason)` unless **all** required inputs are fresh:

| Input | Timeout | Notes |
|---|---|---|
| rgb, depth, joint_state | `input_timeout_s: 0.35` | RGB+aligned depth are **time-synced** on header stamp (`img_sync_slop_s: 0.02`, queue 30) so the obs is one coherent frame |
| socket pose | `socket_timeout_s: 0.75` | FRI state and socket use "latest", not sync |
| flange_pose | 0.35 | only if `fingertip_source: flange_pose` or `ft_frame: flange` |
| joint_effort / external_torque | 0.35 | depending on `force_source` |
| kinematics | — | must be constructed |
| FRI state | `fri_state_max_age_s` | only if `require_fri_state: true` |

It also refuses motion when `socket_part_id < 0` and `allow_wildcard_part_id` is false — the intentional
"socket anchor must be chosen explicitly" gate.

### 2.6.3 Observation construction (parity-critical)

**Fingertip pose** (`_compute_fingertip`, L728). Default `fingertip_source: fk` → KDL FK(q) to
`tip_link` (`lbr_two_gripper_tcp`), which **matches sim** (sim's fingertip is FK of joint state to the
gripper_tcp body) and keeps the observation, IK error, and Jacobian on **one kinematic source**. The
legacy `flange_pose` source uses `/right/ee_pose` + a `flange_to_fingertip` offset.

**End-effector velocities.** Finite differences on the FK pose between ticks:
`ee_linvel = (p_t − p_{t−1})/dt`; `ee_angvel` from the quaternion difference via
`axis_angle_from_quat_wxyz(q_t ⊗ q_{t−1}^*) / dt`, with hemisphere correction (`if qdiff[0] < 0: qdiff = −qdiff`).
On the first tick both are set to **zero**.

**Contact force** (`_contact_force_base`, L747) — 3-axis, **base frame** (sim `ft_force` is world-frame
contact force). Three selectable sources:

| `force_source` | Computation | Notes |
|---|---|---|
| `arm_external_torque` (default in code) | `sign · pinv(Jᵀ)·τ_ext [0:3]` | FRI `external_torque`; KUKA already gravity-compensates it |
| `arm_measured_torque` | `sign · pinv(Jᵀ)·τ_measured [0:3]` | **raw** effort — gravity removed only by the start baseline (see bug B-7) |
| `wrench_topic` | `WrenchStamped`, rotated base←flange if `ft_frame: flange` | the KUKA FTEstimator wrench; **ignores `ft_sign`** |

Then, in order: `nan_to_num` → optional **one-shot baseline capture** (`ft_baseline_on_start`, armed at
`start_policy`) → subtract baseline → **add `ft_bias_base_xyz`**.

The bias exists because sim's `force_sensor` sits *above* the gripper and therefore **includes the
constant distal-weight term** (gripper + screw, world frame). If the real force is gravity-compensated,
adding the bias back restores sim's DC. It is `[0,0,0]` for gravity-comp-trained checkpoints.

`wrench_from_external_torque` (`ik.py:94`) implements the static relation `τ = Jᵀ·W` inverted in
least-squares form `W = pinv(Jᵀ)·τ_ext`. Its docstring records that the **force part (0:3) is
reference-point invariant**, so the fingertip Jacobian is valid even though sim reads the wrist
`force_sensor` link — but warns that sim's `get_link_incoming_joint_force` carries the distal
gravity/inertia term while KUKA's `external_torque` is gravity-compensated, so sign/scale/DC must be
validated on hardware by a press test.

**Force smoothing.** `ForceSmoother` is an EMA with `alpha = ft_smoothing_factor = 0.25`, mirroring
Forge's `ft_smoothing_factor` path, and **initialized to zeros** (matching sim's reset). Critically,
smoothing is **observation-only**: safety evaluates the *unsmoothed* baseline-subtracted force.

**Image** (`preprocess_rgbd`, `obs_preprocessing.py`). Sim provenance: `insertion_env.py::_get_camera_image`
with deploy DR/noise off. Exact order:

1. **FOV-match crop** (`_fov_match_crop`) — the sim camera is a 224×224 pinhole with HFOV ≈ 57.7°; the
   real D405 color stream is ~88°. Center-crop to the window whose *post-resize* focal length equals
   sim's `f_out = (focal_length_mm / horizontal_aperture_mm) · out_width`, with
   `crop_w = fx·W_out/f_out`, `crop_h = fy·H_out/f_out`, centered on `(cx, cy)` and clamped to the image.
   `aligned_depth_to_color` shares the color intrinsics, so the **same pixel box applies to both**.
   No-op if `fov_match` is off or intrinsics are missing.
2. Optional `_clean_depth` (Telea inpaint + median blur) — **off by default**, an A/B test path (§5.9).
3. Resize: RGB `INTER_AREA`, depth `INTER_NEAREST`.
4. RGB → `[0,1]` (÷255 if max > 1.5), clip, then **per-image mean subtraction**
   (`rgb01 − rgb01.mean(axis=(0,1))`).
5. Depth: non-finite → 0, clamp `[0, depth_far_m]`, divide by `depth_far_m`.
6. Concatenate → `H×W×4` float32.

**Frame stacking.** `FrameStacker` concatenates along channels **oldest → newest**, matching
`InsertionEnv._stack_frames`. On the first push the history is filled with copies of the current frame.
`frame_stack: 1` for all current checkpoints, so this is a pass-through; the documented layout for
`N > 1` is `[old_rgb, old_depth, ..., newest_rgb, newest_depth]`.

**Policy vector** (`build_policy_vector`) — the 21-D E2E vector, asserted to be exactly `(21,)`:

| Slice | Contents | Dim |
|---|---|---|
| `[0:3]` | `fingertip_pos − socket_pos_estimate` | 3 |
| `[3:7]` | `fingertip_quat` (w, x, y, z) | 4 |
| `[7:10]` | `ee_linvel` | 3 |
| `[10:13]` | `ee_angvel` | 3 |
| `[13:16]` | `ft_force` (EMA-smoothed, base frame) | 3 |
| `[16:21]` | `prev_action` | 5 |

For the **we_A / pdz** path the contract is instead 15-D:
`[goal_delta_tcp(6), force_tcp(3), previous_action(6)]`, with force at indices **[6:9]** — the slice the
runbooks check.

### 2.6.4 Actor inference (`policy.py`)

`RlGamesActor` wraps the saved rl_games player and exposes only actions.

- **Isaac-free network registration** (`_register_insertion_network`). The custom `insertion_hybrid`
  network is registered by `exec`-ing `insertion_hybrid_network.py` **directly** via
  `importlib.util.spec_from_file_location`, because importing
  `insertion_policy.tasks.insertion.agents` pulls the package `__init__` chain → `isaaclab` → `pxr`/USD
  (the whole Isaac Sim runtime), which is exactly what deployment avoids. Falls back to the full package
  import if the file layout is unexpected.
- **`_patch_rl_games_dict_rms`** monkey-patches `torch.jit.script` to pass `RunningMeanStdObs` through
  unscripted (dict-observation RMS cannot be TorchScript-compiled here).
- **Obs space synthesis.** The env info is rebuilt by hand: `policy` Box(21), `image` Box(H,W,C·stack),
  action Box(5) in [−1,1], `num_actors: 1`, deterministic player, `use_vecenv: False`.
- **`aux_label` auto-detection** (`_detect_aux_label`). Active only when the network has an `aux_head`
  whose `label_key` is also listed in `env.obs_groups.obs`. The dim is the widest slice the aux heads
  reference (w2's `hole` target slice `[1,4]` → **dim 4**, matching the env's 4-D label
  `[grasp_angle, hole_gap_xyz]`). When present, the key is declared in the obs space so the saved
  input-RMS restores cleanly, and **dummy zeros `(1,4)`** are fed every step. The label is training-only
  — excluded from the policy input and its aux loss skipped at inference — so the zeros have **zero
  effect on the action**. A non-estimator checkpoint leaves the obs space unchanged.
- **`reset()`** zeroes the recurrent (LSTM) hidden state to match sim's per-episode reset, guarding the
  `batch_size`-not-yet-set case by forcing a clean re-init on the next `act()`.
- **`act()`** runs under `torch.no_grad()` with `is_deterministic=True`.

### 2.6.5 Action post-processing → joint command

1. **Clip** raw action to `[−1, 1]`.
2. **EMA**: `action = ema·raw + (1−ema)·prev_action`, `ema_factor: 0.0625`. Sim trained with per-episode
   DR `ema_factor_range [0.025, 0.1]`; 0.0625 is the **midpoint** of that trained range (§5.5).
3. **Norm limits** (`limit_action_step`, `deployment_guards.py`) — an independent *hardware* guard. The
   actor keeps its trained scales, but the applied action and `prev_action` are scaled down, **preserving
   direction**, so that the *physical* translation norm ≤ `max_policy_position_step_m` and rotation norm
   ≤ `max_policy_rotation_step_rad`. Disabled when the cap is ≤ 0.
4. **`action_to_target_pose`** (`ik.py`) — a direct port of `InsertionEnvE2EIiwa._pre_physics_step`:
   ```
   target_pos  = fingertip_pos + pos_scale · a[0:3]
   target_pos  = socket_pos + clip(target_pos − socket_pos, ±socket_action_bound)
   rotvec      = [a[3]·rot_scale, a[4]·rot_scale, 0]        # note: no yaw DoF
   target_quat = fingertip_quat ⊗ quat_from_rotvec(rotvec)
   ```
   The clip is the **socket-anchored action box** — a guardrail that bounds the target to a
   `±socket_action_bound` cube about the socket estimate.
5. **Software Z-compliance** (optional, `z_force_limit_enable`). The *downward* component of the target
   advance is scaled by `atten = clip(1 − |F|/z_force_limit_n, 0, 1)`; only `dz < 0` is attenuated, so
   lateral and retract commands are untouched. A cheap stand-in for real Cartesian impedance.
6. **Pose error** (`get_pose_error`) — matches `factory_control.get_pose_error(..., 'geometric')`:
   position difference plus the axis-angle of `q_target ⊗ q_current^*`, with a hemisphere fix
   (`if dot(q_target, q_current) < 0: q_target = −q_target`).
7. **Damped least-squares IK** (`get_delta_dof_pos`): `dq = Jᵀ (J Jᵀ + λ²I)^{-1} dx`, `λ = ik_damping = 0.1`.
8. **Joint step clamp + limits** (`_limit_command`): per-joint clip to `±max_joint_step_rad`, then clip
   into `[IIWA7_LOWER + margin, IIWA7_UPPER − margin]` with `joint_limit_margin_rad = 0.03`.
   iiwa7 limits used: `±[170, 120, 170, 120, 170, 120, 175]°`.
9. Publish `LBRJointPositionCommand` — **only** if not e-stopped/force-latched and `enable_motion` is true.

Two additional IK variants exist in `ik.py` (added in the uncommitted working tree, §3 B-14):
`get_delta_dof_pos_weighted` (weighted DLS, `Winv = diag(1/w)`, high weight = "expensive joint", to stop
the shoulder A1/A2/A3 dominating translation and dragging the TCP down) and
`get_delta_dof_pos_nullspace` (DLS + TCP-preserving nullspace posture task
`(I − J^dls J)·k·(q_rest − q)` to pin the redundant elbow DoF).

### 2.6.6 Safety, seating, and trial termination

**Layered safety:**

| Layer | Mechanism |
|---|---|
| Arming | `enable_motion` parameter (default **false**) |
| Runtime mode | `/rl_deploy/start_policy`, `/stop_policy`, `/reset_preinsert`, `/clear_latches` (all `std_srvs/Trigger`) |
| E-stop | `std_msgs/Bool` on `/rl_deploy/e_stop`, latched in software |
| Force cap | `ft_force_cap_n` → **latches** `force_cap_latched` and fails the trial |
| Force warning | `ft_warn_n` → status message only |
| Freshness | any stale input → hold |
| FRI health | `require_fri_state` + `fri_state_problem()` fail-closed check |
| Joint | per-tick step clamp + limit margin |
| Cartesian | `socket_action_bound` box + action norm caps |
| Z-compliance | force-proportional downward attenuation |

**FRI health check** (`deployment_guards.fri_state_problem`) requires, on a fresh `LBRState`:

| Field | Required |
|---|---|
| `session_state` | 4 `COMMANDING_ACTIVE` |
| `connection_quality` | 3 `EXCELLENT` |
| `safety_state` | 0 `NORMAL_OPERATION` |
| `drive_state` | 2 `ACTIVE` |
| `client_command_mode` | 1 `POSITION` |
| `overlay_type` | 1 `JOINT` |
| `control_mode` | the configured `expected_control_mode` (0 POSITION, 1 CART_IMP, 2 JOINT_IMP; −1 = any) |

It returns a concise joined mismatch string listing **all** bad fields (not just the first).

**Trial stop modes** (`_evaluate_trial_stop`, L1077). `trial_stop_mode` gates the geometry/seat checks:

- **`manual`** (bring-up default) — only the **hard** stops apply: force cap and timeout. The operator
  judges seating. This exists because the geometry checks are measured against the perceived socket
  **anchor**; a ~cm-off anchor otherwise trips `failed_overtravel` on the very first tick even though the
  fingertip is physically over the true hole.
- **`auto_seat`** — re-enables the full gate set, trustworthy only once the anchor is accurate (i.e. fed
  the vision-corrected hole):
  - **Overtravel failure**: `z_above_bottom < −trial_max_overtravel_m` (0.002).
  - **Geometry seat success**: `z_above_bottom ≤ seat_z_tolerance_m` (0.0045) **and**
    `xy_error ≤ seat_xy_tolerance_m` (0.004).
  - **Depth-ROI seat**: median of the valid central ROI (h/6 × w/6 box) ≤ `seat_depth_roi_max_m`.
  - **Seat force**: `|F| ≥ seat_force_n` (14 N). If `seat_force_requires_geometry` and geometry says not
    seated → **`failed_early_contact`** (a jam), not success.

`_seat_geometry` expresses the fingertip in the **perceived socket frame** (`Rᵀ·(p_tip − p_opening)`),
where opening is z=0 and bottom is `−seat_socket_depth_m`; it returns lateral `xy_error_m` and
`z_above_bottom_m`.

On seat: publish `/rl_deploy/seat_detected`, optionally open the gripper, and if `freeze_on_seat: true`
finish the trial into hold. Outcomes recorded: `succeeded_seated`, `failed_force_cap`,
`failed_overtravel`, `failed_early_contact`, `failed_timeout`, `stopped`.

**Hold behaviour** (`_hold`, L1138). The hold setpoint is **latched once** on entering hold and then
re-commanded every tick, rather than re-commanding the live measured position — see bug B-13.

### 2.6.7 Preinsert reset inside the node

Two modes, both running on the same 15 Hz timer and returning to hold on completion:
- **`joint`** — ramp to a fixed 7-joint pose, `preinsert_max_joint_step_rad: 0.006` per tick, done when
  `max|err| ≤ preinsert_tolerance_rad` (0.01). Needs only fresh joint state — no camera, socket, F/T, or
  policy readiness.
- **`socket_hover`** — Cartesian: move the fingertip to `socket_opening + [0,0,preinsert_hover_z_m]`,
  **holding the current fingertip orientation** (translation only). Needs a good socket estimate and a
  roughly upright start.

## 2.7 Stage 5 — Command path to the robot

Two mutually exclusive bridges; **only one may run**.

### Legacy: `command_upsampler.py`
Bridges `/rl_deploy/command_15hz` → the robot's 200 Hz joint-position command topic by ramp
interpolation or zero-order hold. The working tree adds an **adaptive input period** (median of the last
5 observed inter-command gaps, clamped to `[2/rate_hz, 0.5 s]`, reset to the configured period after a
> 0.5 s pause — "a paused source is a new stream, not a half-second interpolation") and a **hold topic**
that immediately republishes a hold setpoint and freezes the ramp.

### Current: `guarded_joint_trajectory_bridge.py` + `guarded_joint_smoother.py`
Keeps `joint_trajectory_controller` **active** and converts each 15 Hz policy target into a short
`JointTrajectory` carrying **position, velocity, and acceleration** samples, letting `ros2_control` own
the trajectory handoff instead of streaming piecewise-linear positions straight into FRI.

`GuardedJointSmoother` is a **receding-horizon quintic planner** that carries `q`, `dq`, `ddq` from one
policy tick into the next. Quintic coefficients from `(q0, v0, a0)` to `q_goal` over duration `T`:

```
c0=q0, c1=v0, c2=a0/2
c3=( 20Δ − 12 v0 T − 3 a0 T²)/(2T³)
c4=(−30Δ + 16 v0 T + 3 a0 T²)/(2T⁴)
c5=( 12Δ −  6 v0 T −   a0 T²)/(2T⁵)
```

It densely samples 2001 points, checks peak |velocity|, |acceleration|, |jerk| and joint limits, and if
any is violated **stretches the duration by ×1.25** and retries until `max_duration_s` (8 s), else raises.
It then emits `sample_count` samples at `robot_sample_period_s` and computes the **handoff state** at
`handoff_time_s` (= one input period), which is committed as the next plan's initial condition —
guaranteeing C² continuity across policy ticks.

Guard values:

| Guard | Value |
|---|---|
| Joint velocity | **3 deg/s** |
| Joint acceleration | **30 deg/s²** |
| Joint jerk | **300 deg/s³** |
| Max target lead from measured joints | **0.5 deg** |
| Max published-state-to-planned tracking error | **0.35 deg** |
| Joint-state max age | **50 ms** |
| Command horizon | 0.08 s, ≥ input period + sample dt; ≥ 3 samples |
| Sample period | 0.01 s |

Any violation **latches** the bridge: it logs the reason, publishes a measured hold, and refuses further
commands until `/rl_deploy/clear_bridge_latch` (which itself requires a fresh joint state).

## 2.8 Topics and services (E2E node)

**Subscribes:**
```
/realsense_2/camera/color/image_rect                  sensor_msgs/Image
/realsense_2/camera/aligned_depth_to_color/image_rect sensor_msgs/Image
/realsense_2/camera/color/camera_info                 sensor_msgs/CameraInfo
/perception/fp/pose_base/fused/assembly               fp_debug_msgs/DebugPoseItem
  (live profile: /rl_deploy/corrected_socket)
/lbr_dual_arm/joint_states                            sensor_msgs/JointState
state                                                 lbr_fri_idl/LBRState
/right/ee_pose                                        geometry_msgs/PoseStamped
/wrist_ft                                             geometry_msgs/WrenchStamped
/rl_deploy/e_stop                                     std_msgs/Bool
```
**Publishes:**
```
command/joint_position   lbr_fri_idl/LBRJointPositionCommand  (live: /rl_deploy/command_15hz)
/rl_deploy/status        std_msgs/String        (rate-limited to 1 Hz)
/rl_deploy/policy_obs    std_msgs/Float32MultiArray
/rl_deploy/seat_detected std_msgs/Bool
/gripper/open_cmd        std_msgs/Float64
```
**Services:** `/rl_deploy/{start_policy, stop_policy, reset_preinsert, clear_latches}`,
`/rl_deploy/clear_bridge_latch` (bridge node).

`start_policy` refuses to start when `socket_part_id` is still `-1`; it also resets `prev_action`,
clears the seat flag, arms the F/T baseline capture, and resets the LSTM state.

## 2.9 Alternative and auxiliary nodes

- **`inference_node_we_a.py`** — the we_A/pdz variant: 15-D policy vector, **RGB-only** (does not
  subscribe to or wait for depth), ImageNet normalization, 6-D TCP-relative action, plus its own
  calibration gates (`we_a_geometry_calibrated`, `we_a_force_bias_calibrated`), an axis guard
  (`we_a_axis_guard_deg: 60`), and `zero_force_obs`.
- **`visual_servo_insert.py`** — a deliberate **vision-only comparison baseline** that never loads the
  actor: detect hole → MoveIt corrected preinsert → switch to the LBR position controller → servo the
  TCP down with damped IK at 200 Hz keeping XY centered. Dry-run by default.
- **`serl_backend.py`** — a SERL HTTP robot-server backend reusing the same kinematics, force estimate,
  and command path, for real-robot RL (demo recording / actor-learner). Reads joint state and torques,
  FK via the same `kinematics.py`, force from measured joint torques via the Jacobian, commands through
  `/rl_deploy/command_15hz`, gripper via `/gripper/open_cmd`. Arm prefix is settable via
  `SERL_ARM_PREFIX` (default `lbr_one`).
- **`scene_collision_publisher.py`**, **`tune_hole_detection.py`**, **`preview_camera_crop.py`**,
  **`inspect_obs_dump.py`** — commissioning utilities.

---

# 3. Chronological Bug and Fix History

Dates are taken from commit dates, document headers, and in-code comments. Several bugs were found in
the *sim* side but are recorded here because they determined a deploy-side change. Each entry gives the
symptom, root cause, fix, and the technical reasoning that connects them.

Severity legend: **⛔ blocker** · **⚠ major** (silently wrong behaviour) · **▫ minor/robustness**

---

### B-1 ⛔ `kdl_parser_py` is not packaged for ROS 2 Humble
**When:** 2026-08-10 (deploy runtime bring-up, Steps 0–2)
**Symptom:** FK/Jacobian unavailable → the whole IK path (and therefore any motion) is blocked.
**Root cause:** `kdl_parser_py` — the URDF→KDL-tree bridge — has no Humble binary package. `PyKDL`
itself is available, but without the parser there is no way to build the chain from `robot_description`.
**Fix:** **Vendor** a minimal parser as `kdl_parser.py` (101 lines) exposing `tree_from_string`, and
import it as a fallback:
```python
try:    from kdl_parser_py.urdf import treeFromString
except ImportError: from .kdl_parser import tree_from_string as treeFromString
```
**Reasoning:** only `treeFromString` is needed; vendoring that one function avoids a source build of an
unmaintained package on the deploy PC. A regression test pins the behaviour:
`test_kdl_kinematics_uses_local_parser_when_kdl_parser_py_is_missing`.

---

### B-2 ⚠ PyKDL fixed-joint type reported inconsistently → false "joint not in FRI order" errors
**When:** during KDL chain bring-up
**Symptom:** `KinematicsUnavailable: KDL chain contains joints not present in measured FRI order: ...`
listing the **gripper's fixed joints**, even though those joints legitimately consume no measured `q`.
**Root cause:** the code filtered chain joints by comparing the joint **type name string** to `"Fixed"`.
**PyKDL reports the fixed type as `"Fixed"` in some builds and `"None"` in others**, so in the latter
the gripper's fixed mount/TCP frames passed the filter and were then demanded in the measured joint map.
**Fix:** compare the joint-type **enum** instead (`int(joint.getType()) != int(kdl.Joint.Fixed)`), in
`KdlKinematics._joint_names_in_chain`.
**Reasoning:** only movable joints consume a measured `q`; the enum is build-independent, the string is
not. This bug is why the deploy chain reaches `lbr_two_gripper_tcp` (through fixed gripper frames) at all.

---

### B-3 ⚠ Observation-parity defects vs. sim (the "deploy parity audit")
**When:** ~2026-08-10 → 2026-08-11, ongoing
**Symptom:** the real observation differed structurally from sim's, which would silently move the policy
out of distribution with no error anywhere.
**Root causes and fixes** (each is now the implemented behaviour):

| # | Defect | Fix |
|---|---|---|
| a | `ee_angvel` undefined on the first tick | zero both `ee_linvel`/`ee_angvel` when no previous pose exists |
| b | EMA smoothing factor wrong/absent | `ForceSmoother(alpha = ft_smoothing_factor = 0.25)`, **initialized to zeros** to match sim's reset |
| c | LSTM hidden state persisted across trials | `RlGamesActor.reset()` zeroes the recurrent state on every `start_policy`, matching sim's per-episode reset |
| d | Fingertip taken from the flange topic | default switched to **FK to `gripper_tcp`**, matching sim and unifying obs/IK/Jacobian on one kinematic source |
| e | Force not the external-torque estimate | `wrench_from_external_torque` = `pinv(Jᵀ)·τ_ext` |
| f | Real camera FOV (~88°) ≫ sim (~57.7°) | the `_fov_match_crop` center-crop before resize |
| g | Socket *centre* vs socket *opening* | `_socket_opening()` = `centre + R(perceived_quat) @ offset_cad`, mirroring sim's `fixed_pos_obs_frame = fixed_pos + R(fixed_quat) @ socket_offset_local` |
| h | EMA factor questioned ("H2") | resolved: the final run uses 0.01 |

**Reasoning for (g):** rotating the CAD-frame offset by the *perceived* orientation means a lateral hole
offset rotates with socket yaw, staying correct under rotation — exactly as sim rotates
`socket_offset_local` by `fixed_quat`. Perception reports the tracked CAD's **centre** (FoundationPose
uses the centred mesh) while sim's anchor is the **opening**; `cooling_base` is symmetric, with openings
at `(±0.030, 0, +0.0175)` m.

---

### B-4 ⚠ `torch.jit.script` cannot compile `RunningMeanStdObs` (dict observations)
**Symptom:** checkpoint restore crashed inside rl_games' JIT path.
**Fix:** `_patch_rl_games_dict_rms()` monkey-patches `torch.jit.script` to return `RunningMeanStdObs`
instances unscripted, guarded by a `torch.jit._insertion_rmsobs_patch` flag so it applies once.

---

### B-5 ⚠ Importing the training network pulls in the entire Isaac Sim runtime
**Symptom:** the deploy node could not load the actor without `isaaclab`/`pxr`/USD present.
**Root cause:** the `insertion_hybrid` network is registered at the bottom of
`insertion_hybrid_network.py`, but importing it via the package (`insertion_policy.tasks.insertion.agents`)
executes an `__init__` chain that imports `isaaclab` → `pxr`.
**Fix:** `exec` that single file directly with `importlib.util.spec_from_file_location`, which needs only
torch + rl_games; fall back to the package import if the layout is unexpected.
**Reasoning:** registration is one `model_builder.register_network(...)` call. This decouples deployment
from the simulator entirely — the central enabler of the lightweight deploy environment (§5.2).

---

### B-6 ⚠ Explicit-estimator checkpoint fails to restore (`w2_estimator_192`)
**When:** 2026-08-10 (commit `3c67529`, 2026-08-24 in git)
**Symptom:** restoring `w2_estimator_192` mismatched the saved network: its `agent.yaml` declares a
privileged `aux_label` obs group and an `aux_head` that the deploy obs space did not contain.
**Root cause:** with `normalize_input`, rl_games saves a **`RunningMeanStdObs` sub-module per obs key**.
Omitting `aux_label` from the obs space therefore breaks the state-dict restore — even though `aux_label`
is a *training-only* label (the true hole gap) that the network excludes from the policy input and whose
aux loss is skipped at inference.
**Fix:** auto-detect the group from the agent config (`_detect_aux_label`), declare `aux_label` (dim 4)
in the obs space, and feed **dummy zeros `(1,4)`** every step.
**Reasoning:** the fed-to-policy value is the head's own *image-derived prediction*, not the label, so
zeros have **zero effect on the action** — they only keep the obs dict shape identical to training. Dim 4
is derived as the widest aux-target slice (`hole` → `[1,4]`), matching the env's 4-D
`[grasp_angle, hole_gap_xyz]` and the saved input-RMS shape. Auto-detection means non-estimator
checkpoints are unaffected.

---

### B-7 ⛔ ~200 N phantom contact force from raw joint effort
**When:** 2026-08-27 (single-arm deploy)
**Symptom:** the policy force channel showed a **~200 N** force that does not physically exist.
**Root cause:** `force_source: arm_measured_torque` ran the **raw** joint effort — which *includes
gravity*, ~**77 Nm at J4** — through `pinv(Jᵀ)`. Gravity torque mapped through the Jacobian pseudo-inverse
produces a huge phantom end-effector force. The startup baseline could not rescue it because the residual
is strongly pose-dependent.
**Fix:** switch to `force_source: wrench_topic` — the FRI FTEstimator wrench, whose gravity is already
removed **by the KUKA controller**.
**Caveat recorded in the config:** `wrench_topic` **ignores `ft_sign`** (only the torque paths apply it).

---

### B-8 ⚠ +5 N Z-bias and a ~−15 N leaking phantom in the estimator wrench
**When:** 2026-08-27/28 (found in all 4 analysed bags)
**Symptom:** the wrench reads **Fz ≈ +5 N while the arm is HOVERING with no contact**, consistently
(+4.98, +4.94, +4.92, +5.09 N across the four bags). Worse, on the policy force-z channel the bag shows a
**~−15 N phantom** while the true physical force was only ~10 N.
**Root cause:** a **pose-dependent gravity residual in the KUKA's own external-torque estimate** — i.e. a
wrong Sunrise tool/payload model. `ft_baseline_on_start` captures and subtracts it at the start pose, but
the residual **"leaks" as the wrist tilts**, so the subtraction goes stale as the pose changes.
**Consequences:** seat detection and the force cap see a false +5 N floor, so true contact is a *deviation*
from +5 N — the 0 N dips are the tool pulling **up** and the 10 N spikes are only ~5–7 N of real contact.
**Fix (chosen):** `zero_force_obs: true` — feed the **policy** zero force (matching sim's pure-contact ≈ 0
at no contact), while safety and seat thresholds continue to use the real contact force. The rationale is
that the sim policy trained on pure-contact ≈ 0 force and physical forces here are tiny, so zeroing is
closer to the training distribution than a leaking phantom the policy never saw.
**Rejected alternative:** re-adding a static `ft_bias_base_xyz` z ≈ −5; rejected because the residual is
pose-dependent, so a constant cannot cancel it.
**Note:** the true fix is pendant-side (correct Sunrise tool/payload model). Recorded as open (§6).

---

### B-9 ⚠ Broken gravity compensation baked ~0.75 × gripper weight into training
**When:** discovered 2026-08-24/25 for `pdz_overnight_20260824`
**Symptom:** the deployed force could not match training even with a correct real F/T.
**Root cause (in the training env):** `env.yaml` sets `use_gravity_comp: true`, but the per-env gravity
reference was captured **from the EMA-smoothed force on the first post-reset step** (`alpha = 0.25`
starting from 0). Only **~25 %** of gravity was therefore subtracted, leaving **~0.75 × gripper_weight
baked into the training force observation**.
**Confirmed three independent ways:** (1) the committed (Aug 13) code froze `_grav_base` from
`force_sensor_smooth` — the bug itself; (2) the raw-reading fix is **uncommitted and postdates** the
Aug 24 run; (3) the saved policy's **force-slot running mean shows a large non-zero DC**, not the
near-zero of clean contact.
**Fix options implemented as togglable deploy paths:**
- **(A) Diagnostic ablation** — `zero_force_obs: true` zeros policy force indices `[6:9]` before each
  actor forward, isolating whether force helps or hurts on hardware. Safety/seat use `force_base`,
  unaffected. *The clean first test given the messy bias.*
- **(B) Parity** — reproduce the bias: deploy **adds** `ft_bias_base_xyz` into the policy force (rotated
  to TCP with the tool, exactly like sim's constant base-frame gravity residual) while **subtracting** it
  from the safety norm. Since the startup baseline removes the *full* real gravity, set
  `ft_bias_base_xyz = 0.75 × (measured real no-contact distal wrench in base)` — order **~7 N** if the pdz
  distal mass matches the y-gripper's **0.9675 kg**.
Both left at `[0,0,0]` and gated behind `we_a_force_bias_calibrated: false`.
**Resolution:** superseded by `pdz_v3_fulltilt_20260826`, which trains with **corrected** world-frame
gravity comp → pure contact force → `ft_bias_base_xyz` returns to `[0,0,0]`. The config warns explicitly:
injecting the old 5.3 N now would be *a phantom force the policy never saw* and would bias it to press.

---

### B-10 ⚠ 2 N force deadband blinded the policy to first contact
**When:** 2026-08-28
**Symptom:** sim2real force mismatch at the onset of contact.
**Root cause:** the KUKA `estimated_ft_sensor` applied `force_*_th: 2.0` / `torque_*_th: 0.5` deadbands,
whereas **sim feeds RAW baseline-subtracted contact force with NO deadband**
(`insertion_env_e2e_iiwa.py:314`). The deadband blinded the policy to the **first 2 N** of contact and
under-reported above it.
**Fix:** set `force_*_th: 0` / `torque_*_th: 0` in `lbr_two_system_config.yaml` / `lbr_one_system_config.yaml`.
**Status:** **KEEP** — explicitly flagged as a genuine parity fix, independent of the FRI-drop work
(for which it had no effect).

---

### B-11 ⚠ OMPL branch-flip: a ~1 cm correction became a ~70° joint swing
**Symptom:** the vision-correction move — a small Cartesian offset — planned as an enormous
redundant-arm reconfiguration.
**Root cause:** a plain Cartesian `pose_goal` lets OMPL (RRTConnect) choose **any** IK branch for the
7-DoF redundant arm; nothing ties the solution to the current elbow configuration.
**Fix:** `target_mode: local_ik_joint` (default) — compute a nearby joint target with **KDL/DLS from the
current branch**, then plan in **joint space**, keeping a small Cartesian correction a small in-branch
move. Backed by `constrain_branch: true`, `joint_near_current_tolerance_rad: 0.80`, and
`max_plan_joint_delta_rad: 0.85`. `pose_goal` is retained only for the case where KDL is unavailable.
**Operator-level mitigation** (safety checklist): "Eyeball the dry-run's first/last joint targets. A large
joint jump for a small Cartesian move = OMPL took a redundant-arm detour; re-run rather than execute it."

---

### B-12 ⚠ FOV-match center-crop **introduced** an off-center shift for we_A/pdz
**When:** 2026-08-31 — a reversal of an earlier decision
**Symptom:** the vision policy was steered wrong on hardware.
**Root cause:** the crop for the we_A path cropped to ≈ **831×467 centered on `(cx, cy)`** — the
*principal point*, which is **not** the image center. Sim, however, renders **native 320×180 16:9 with NO
crop** (`cooling_iiwa_tasks_cfg.py:574`: "Full 16:9 single-frame RGB, downscaled to 320x180 without
cropping"). The crop therefore injected an **off-center shift sim never had**.
**Additional finding:** the real D405 at 848×480 is **also 16:9** at ~88°, versus sim's ~87.2° — so a
**plain resize already matches sim byte-for-geometry**.
**Fix:** `WeAObsPreprocessConfig.fov_match` default → **`False`**, documented inline with the date and
the sim source line.
**Note:** the E2E path keeps `fov_match: True` — there sim is a **224×224 pinhole at ~57.7°** versus the
real ~88°, so the crop is genuinely required. The two paths differ because their *sim cameras* differ.

---

### B-13 ⚠ Hold chased its own gravity sag in Cartesian impedance
**When:** uncommitted working tree (after 2026-08-28)
**Symptom:** commanding "hold" made the arm progressively **droop**.
**Root cause:** in Cartesian impedance with an uncompensated tool payload the arm sags. The old `_hold()`
re-commanded the **live measured position** every tick, creating a positive feedback loop: measured drops
→ command that lower position → sag further → …
**Fix:** latch the hold setpoint **once** on entering hold (`self._hold_q`) and command that fixed target;
clear `_hold_q` whenever hold is left (`start_policy`, `stop_policy`, `reset_preinsert`) so the next hold
re-latches a fresh position.

---

### B-14 ▫ Shoulder dominance in DLS IK (weighted / nullspace IK added)
**When:** uncommitted working tree
**Symptom:** plain DLS let the shoulder joints (A1/A2/A3) dominate Cartesian translation, "dragging the
TCP down"; the redundant elbow DoF also drifted/collapsed.
**Fix:** two new solvers in `ik.py`: `get_delta_dof_pos_weighted` (per-joint cost `Winv = diag(1/w)`;
high weight = expensive joint) and `get_delta_dof_pos_nullspace` (TCP-preserving posture task
`(I − J^dls J)·k·(q_rest − q)`). Available but not yet wired into the main control tick.

---

### B-15 ▫ Stale `Forge.pth` "best" pointer would deploy an ~epoch-14 policy
**When:** 2026-08-26/27
**Symptom:** `nn/Forge.pth` is timestamped at launch (13:22) rather than at the end of training.
**Root cause:** rl_games keeps the "best" pointer at the epoch with the highest reward — which, under a
**hardening curriculum**, is an **early epoch on easy tilt** (~ep 14), not the best final policy.
**Fix:** deploy `last_Forge_ep_2000` (05:57), copied to `deploy_final_ep2000.pth`, with an explicit
config comment: "DEPLOY THIS ONE, not Forge.pth."
**Reasoning:** under a curriculum, **reward is not comparable across epochs** — later reward drops because
the task hardens, not because the policy is worse. The same note appears in `docs/DEPLOY_PDZ.md`.

---

### B-16 ▫ MultiThreadedExecutor made every tick ~4 s (0.27 Hz)
**Symptom:** the control loop ran at ~0.27 Hz, jerky.
**Root cause:** PyTorch/cuDNN pays a **~4 s lazy per-thread initialization**. A `MultiThreadedExecutor`
runs the timer callback on **rotating worker threads**, so that cost was re-paid on *every new thread*.
**Fix:** keep the executor **single-threaded** and run one throwaway inference at startup on the main
thread (`_warmup_actor`). First call ~4 s at init; every live tick ~**6 ms**.
**Reasoning:** the tick runs inference inline on the executor thread, so warming *that* thread is what
matters. Both the class docstring and `main()` carry explicit warnings not to reintroduce multithreading.

---

### B-17 ▫ `env.yaml` embeds Python objects that break `yaml.FullLoader`
**Symptom:** reading the training `env.yaml` for obs sizing crashed the node.
**Root cause:** the training `env.yaml` embeds python object tags (e.g. a `builtins.slice`) that
`FullLoader` cannot construct.
**Fix:** `_TolerantYamlLoader`, a `FullLoader` subclass with a multi-constructor mapping any
`tag:yaml.org,2002:python/` tag to `None`.
**Reasoning:** only plain scalars/lists are needed (image size, frame stack, clipping range), so ignoring
python tags is strictly better than failing to size the observation.

---

### B-18 ▫ Fused-assembly topic publishes `part_id = -1`
**Symptom:** `start_policy` / `_ready` blocked because `socket_part_id` looked unset.
**Root cause:** the fused assembly topic legitimately publishes `part_id = -1` (whole assembly, not a
numbered part), which collides with the sentinel meaning "not configured".
**Fix:** an explicit opt-in parameter `allow_wildcard_part_id` to accept `-1` as intentional, keeping the
fail-closed default.

---

### B-19 ⛔ THE DEPLOY BLOCKER — FRI session drops on first substantial contact
**When:** 2026-08-27 → 2026-08-28, unresolved
**Symptom (pendant), at the moment of contact:**
```
Quality change signalled POOR
Jitter 3.43 / Latency 1.6
Session State change MONITORING_WAIT
[Error] Wrong FRI session state 'Monitoring (Wait)' in active phase of FRI motion.
        Possible connection problem. (ERROR_FRI_CMD_WRONG_STATE_ACTIVE)
```
ROS side: `LBR left COMMANDING_ACTIVE. Please re-run lbr_bringup`. **Free-space motion is EXCELLENT /
jitter 0 — jitter and POOR appear ONLY at contact.**

This bug went through **three successive root-cause theories**; the diagnostic arc is itself a result.

**Theory 1 — 110 ms network freeze (SUPERSEDED).** Five of six bags show a repeatable ~110 ms gap in
joint/wrench publication at the transition, while the 200 Hz command topic keeps flowing. Initially read
as a read-side preemption from a generic (non-PREEMPT_RT) kernel.

**Correction — the 110 ms is a post-drop SHUTDOWN ARTIFACT.** Tracing Humble v2.2.3 showed:
`SystemInterface::read()` detects that FRI **has already left** `COMMANDING_ACTIVE`, then synchronously
calls `close_udp_socket()`, which waits for the FRI worker with a **fixed 100 ms polling sleep**. One
normal 10 ms update + that 100 ms sleep **exactly explains the observed ~110 ms**. So the gap locates the
drop *at contact* but says nothing about its cause or about any network outage duration. Explicit warning
in the docs: *do not shorten the 100 ms close sleep as a "fix" — it would only make the visible gap
smaller after FRI had already failed.*

Also invalidated: the continuing 200 Hz ROS command topic **does not prove the FRI UDP loop was healthy**,
because that topic is produced by a **separate upsampler process upstream** of `ros2_control` and the FRI
client. In bag `165249` the bag continues **212.7 s** after the last joint state and contains a further
**42,543** direct position commands — which cannot possibly have been delivered.

**Theory 2 — async FTEstimator contention (DISPROVEN BY TEST).** The `lbr_fri_ros2::FTEstimator` runs its
own thread at `update_rate = 100 Hz`, `rt_prio = 30`, next to the FRI send loop (`rt_prio = 80`) on the
same PC, and source inspection shows it computes a **Jacobian pseudo-inverse every cycle**, including in
free motion (crossing a force threshold does not gate the work). Reducing it to **30 Hz** (confirmed live:
26 Hz measured wrench-change rate in bag `pdz_single_30hz_20260828_140335`) — plus force thresholds → 0 —
**still dropped at contact, with the identical 110 ms signature.**

**Theory 3 (LEADING, unresolved) — rigid contact plus open-loop command progression.** The right-arm
profile combines `client_command_mode: position` with **`open_loop: true`**. In that mode
`AsyncClient::command()` calls `set_state_open_loop()`, which **replaces the measured joint position in
the published state interface with the filtered commanded position**. Consequences:
1. `/joint_states` is **not** evidence of physical tracking during position command mode;
2. the deploy node computes FK and its next target from a **virtual robot that appears to follow perfectly**;
3. the local command guard sees that substituted state and **cannot detect true command-to-robot
   separation** caused by a blocked tool;
4. **Sunrise still sees the actual encoders**, external torque, and internal motion/safety state.

At contact the real robot stops or deflects while the command path keeps advancing. With the old
requested changes of **5.6–7.3 mm and up to 6.8°/tick**, this creates a rapid physical following/load error
in stiff native position control, and Sunrise leaves the commanding phase. The ROS stack then observes the
transition and produces the 110 ms shutdown signature. This also explains why Julien's path survives:
Cartesian impedance absorbs contact displacement, **and the LBR client automatically overrides `open_loop`
to false in impedance control**.

**Competing hypothesis — FRI reply deadline.** With a 10 ms send period and `ReceiveMultiplier = 1`, a
reply is required **every 10 ms cycle**; a late/missing reply, missing controller packet, or packet loss
would degrade quality and drop the session. The pendant's `Quality POOR` + jitter supports this. It cannot
be tested with the existing bags — a **packet capture is required** to determine which direction becomes
late first.

**Fix ladder (attempted → outcome):** see §4.3. **Status: OPEN** (§6.1).

---

### B-20 ⚠ The policy never centers laterally — it drifts AWAY from its own anchor
**When:** 2026-08-27 bags, analysed 2026-08-28 (`DIAGNOSIS.md`)
**Symptom:** across **all four** bags, the peg tip never gets within 3.4 mm of the hole (clearance ~1 mm),
descends onto the cooling-base **top surface** beside the hole, cannot enter, hunts sideways, **drifts
further away**, then slips and hard-recontacts (0 → 10 N) — and that transient trips the FRI.
**Finding 1 — the deploy start-delta is NON-ZERO while sim's is ≈ 0 (the core distribution mismatch).**
Verified in sim code (`insertion_env_e2e_iiwa.py::_goal_delta_ee`, docstring L290): the peg is
**physically spawned above the NOISY goal** `G = true_socket + injected_goal_error`, so the policy's
observed delta (`fingertip − G`) is **≈ 0 at reset** — "the policy thinks it's at the goal" — while the
**true hole is up to `goal_anchor_lat_max` (±1.75 cm) away**. `hand_init_pos = [0, 0, 0.0675]` with noise
`[0.003, 0.003, 0.010]` makes the ±3 mm mere settling jitter about G; tip ≈ rim + 37.5 mm. The policy's
learned job is: *obs says delta ≈ 0, but VISION must reveal the true hole is offset and servo there.*
In deploy, `policy_obs[0:3]` at tick 0 is **5.9 / 8.6 / 4.7 / 5.1 mm** — non-zero. The policy is handed a
starting observation it **never saw in training**, and its learned response drives the delta to **grow**
(5 → 12–33 mm) rather than shrink. Height is fine (29–43 mm above socket-z ≈ sim's rim + 37.5 mm); the
problem is purely **lateral**.
*(This note explicitly corrects an earlier incorrect claim that "the peg spawns above the TRUE hole.")*
**Why it drifts away — the core open question.** Either (a) the hole **anchor XY is wrong** (hand-eye
residual) so "toward the true hole" per vision ≠ toward the anchor, or (b) the policy cannot localize the
real chamfer-free hole at this height and wanders. Anchor **detection is self-consistent** across runs
(spread dx 0.7 mm, dy 1.4 mm), so if the anchor is wrong it is a **consistent calibration offset**
(hand-eye), not noise.
**Status: OPEN** (§6.2). Moot until the FRI drop is fixed, since no run lasts long enough to judge.

---

### B-21 ▫ D405 hand-eye: CAD datum is the mount, not the color optical frame
**When:** 2026-08-05 (CAD arrived), open at deploy
**Root cause:** the CAD `T_flange_cam` is locked in sim, but the CAD datum is the **mount**, not the
**COLOR optical frame**. The D405 is a **stereo** camera, so the color optical center is offset from the
mount by a **~cm-class residual** — the same order as the entire insertion clearance budget.
**Mitigation:** always use `aligned_depth_to_color` (so color and depth share intrinsics and one frame),
and fix the residual **empirically** against a pose-matched real frame. The computed CAD_flipX offset ≈
prealign, and FOV matches (58°).
**Status: OPEN** — the prime suspect for B-20's consistent lateral offset.

---

### B-22 ▫ Infrastructure fixes that did **not** fix the drop but are kept
| Change | Effect | Keep? |
|---|---|---|
| DDS isolation to loopback (`deploy_dds_no_fri_nic.xml`) + `ROS_DOMAIN_ID=42` | fixed an unrelated cross-student TF interference issue | **yes** |
| Killed a stray 134 %-CPU Isaac process (load 6.3 → 0.16) | cleaned baseline jitter | yes |
| `ros2_control_node` → SCHED_FIFO 80 (`run_hardware_right_rt.sh`) | free-space jitter → 0 / EXCELLENT | **yes** |
| CPU pinning: FRI → cores 0–3, vision → 8–31 | `joint_states` std → 0.1 ms | **yes** |
| Shed CPU (dropped foxglove + pipeline viz) | lower load | situational |
| FTEstimator 100 → 30 Hz | none on the drop | harmless, keep |
| FTEstimator thresholds → 0 | none on the drop | **keep — real parity fix (B-10)** |

The CPU-pinning rationale is specific: the FRI read thread and the ~21 % RealSense **rectify container**
were landing on the same core during bursts; **RT priority alone does not help if two RT-ish threads share
a core**, so they must be pinned apart.

---

### B-23 ▫ Non-causes explicitly ruled out (recorded to prevent re-investigation)
- **Joint ordering — FINE.** `/joint_states` reports a scrambled order `[A1,A3,A5,A2,A6,A4,A7]`, but the
  node maps **by name** and commands A1..A7; tracking error ~2 mrad. Cosmetic, not a bug. **Caveat:** do
  not infer physical tracking from these bags — position mode used `open_loop: true` (see B-19).
- **Memory leak — NO.** 56 GiB free, swap ≈ 0; drops occur within 1–4 s, far too fast for a leak.
- **Whole-PC load —** improved but did not fix; and command-topic continuity proves nothing about the FRI
  thread.
- **FTEstimator async contention — DISPROVEN** by the 30 Hz test.

---

# 4. Experiments and Quantitative Results

All numbers below are the actual values recorded in the repository (documents, configs, CSVs) or
recomputed from the raw CSVs during this export. Where a number was **independently recomputed** from
`deploy_analysis/*.csv`, it is marked ✔.

## 4.1 Simulation results (training-side, quoted as the deployment baseline)

These are the sim success rates that justified each deploy checkpoint choice.

| Run | Success | Condition | Note |
|---|---:|---|---|
| `e2e_weld_curric` | **85.2 %** | eval at full 18° tilt | The baseline. Beats `res224` (65 %). Clean 1500 it from scratch, weld + curriculum + socket + proprio |
| `res224` (vision 224 px) | 65 % | — | Superseded by weld+curriculum |
| `w2_estimator_192` **with** estimator | **83.2 %** | **full ±2.5 cm socket-localization noise** | The **deployed winner** |
| `w2_estimator_192` **without** estimator | **67.8 %** | same ±2.5 cm noise | **The estimator ablation: +15.4 pp** |
| `e2e_deploy_g25t12` (v1) | **82 %** @128 px | soft-contact-propped | Degrades to **74.6 %** @192 px |
| 224 px vs 160 px (resolution ablation) | **+3.7 pp** @ ep1200 matched | despite a 48-vs-64-env handicap | high-tilt bin **+22.6 pp** |
| Vision vs no-vision (hardened env) | **75.6 %** vs **66.2 %** | rigid 45° wrist cam, grasp tilt, 10°/8 mm/25°, DR | vision **WON** |
| Corrected-env rebaseline | **94.5 / 95.5 %** | at 5° tilt | superseded by the hardened env |
| Aux-head screening (grasp head) | **−11 pp** | @160 px | **DEAD END** — unlearnable; line dropped |
| Aux-head screening (hole head) | neutral | — | dropped |
| Weld grid verification | **100/100** clean | grasp-weld fix | resolved the dominant confound |
| 2026-08-02 weekend tilt sweep | **0.4 – 5.5 %** (5 runs, all failed) | — | root cause was the broken-grasp bug, not obs stripping; post-fix seated at **24.6–33 %** |
| `pdz_overnight_20260824` ep500 / ep1000 / ep1500 | reward **183.7 / 151.5 / 144.4** | — | reward *drops* only because the curriculum hardens — later ≠ worse |
| `w2_estimator_192` final | reward **162.14** @ ep 2000 | — | `last_Forge_ep_2000_rew_162.13815.pth` |

**Sim contact regime (design targets used to set deploy limits):** mean contact force ≈ **5 N**, force
penalty threshold **12 N**, simulated bounce range **22–65 N** (explicitly *not* to be permitted on rigid
hardware). Episode length 13 s. Curriculum: full **25°** tilt, 128k steps done ≈ ep 1000, 2000 ep total.

**Sim reset geometry (for the deploy start-delta comparison, B-20):** `hand_init_pos = [0, 0, 0.0675]`,
noise `[0.003, 0.003, 0.010]`, `goal_anchor_lat_max = ±1.75 cm`, tip starts ≈ **rim + 37.5 mm**,
observed goal delta at reset ≈ **0**.

## 4.2 Hardware deployment runs — the four analysed bags (2026-08-27)

Config `deploy_pdz_v3_fulltilt_single`, checkpoint `pdz_v3_fulltilt_20260826`. Each bag contains **150
policy observations** (10 s at 15 Hz) and 3.6k–10.6k wrench samples.

### 4.2.1 Primary result table (✔ independently recomputed from the CSVs)

| Quantity | b162621 | b163807 | b165249 | b171512 |
|---|---:|---:|---:|---:|
| **FRI drop time (s)** ✔ | **1.111** | **3.592** | **1.820** | **2.675** |
| Start lateral XY miss (mm) ✔ | 5.87 | 8.56 | 4.68 | 5.09 |
| **Min** lateral XY miss (mm) ✔ | **5.87** | **3.42** | **4.68** | **3.69** |
| **End** lateral XY miss (mm) ✔ | **12.86** | **32.71** | **12.61** | **31.38** |
| Start height above socket-z (mm) ✔ | 28.74 | 37.39 | 42.99 | 31.35 |
| **Min** height above socket-z (mm) ✔ | **13.17** | **13.88** | **17.63** | **13.08** |
| End height above socket-z (mm) ✔ | 15.67 | 16.12 | 19.46 | 15.08 |
| Ever centered (XY < 3 mm)? | **no** | **no** | **no** | **no** |
| Ever near socket-z (z < 5 mm)? | **no** | **no** | **no** | **no** |
| Drop force cycle (N) | 5→0→10.2 | 5→0→10.8 | 5→0→10.0 | 5→0→12.1 |
| Peak \|F\| during trial (N) ✔ | 10.22 | 10.79 | 10.03 | 12.09 |
| Min \|F\| during trial (N) ✔ | 0.00 | 0.00 | 0.00 | 0.00 |
| **Hovering (no-contact) Fz (N)** ✔ | **+4.98** | **+4.94** | **+4.92** | **+5.09** |
| Force samples ✔ | 3873 | 7159 | 3598 | 10633 |
| Policy observations ✔ | 150 | 150 | 150 | 150 |

**Conclusions drawn from this table:**
1. **Lateral centering never succeeds.** The minimum XY miss over all four runs is **3.42 mm** against a
   **~1 mm** clearance, chamfer-free. Insertion was geometrically impossible in every run.
2. **The error grows.** XY miss goes from 4.7–8.6 mm at start to **12.6–32.7 mm** at the end — the policy
   servos *away* from its own socket estimate.
3. **Descent is blocked.** Minimum height never goes below **13.08 mm** above socket-z; the tip is resting
   on the base's top surface beside the hole.
4. **The force cycle is identical in all four runs** (5 → 0 → 10 N), i.e. the slip-and-recontact is
   systematic, not a one-off.
5. **A consistent +5 N Z-bias exists with no contact** (spread 4.92–5.09 N, i.e. ±0.09 N).
6. The drop **always coincides with the run's peak force** and always follows a **0 → 10 N re-contact
   transient after the tool unloads** — never during smooth motion.

**Anchor detection repeatability:** spread **dx 0.7 mm, dy 1.4 mm** across runs → any anchor error is a
**consistent calibration offset**, not detection noise.

### 4.2.2 Extended six-bag analysis (read directly from SQLite rosbag storage)

| Run | Policy obs | Largest joint-state gap | Force around gap | Direct cmds inside gap | Policy translation req. median / max | Policy rotation req. max |
|---|---:|---:|---:|---:|---:|---:|
| `20260827_165249` | 150 | **110.157 ms** | 9.118 → 10.029 N | 22 | **6.993 / 7.259 mm** | 2.740° |
| `20260827_171512` | 150 | **110.233 ms** | 11.059 → 12.094 N | 22 | 5.451 / 5.611 mm | **6.681°** |
| `20260828_134142` | **7** | 16.013 ms (**no drop**) | 6.251 N peak | 0 | **1.043 / 1.433 mm** | 1.080° |
| `20260828_134445` | 150 | **110.069 ms** | 10.175 → 10.175 N | 22 | 4.768 / 6.413 mm | **6.821°** |
| `20260828_135128` | 150 | **110.127 ms** | 10.775 → 12.054 N | 22 | **7.002 / 7.191 mm** | 2.418° |
| `20260828_140335` (30 Hz FTEstimator) | 150 | **110.234 ms** | 12.465 → 12.465 N | 22 | 7.089 / 7.223 mm | 2.506° |

Translation/rotation are reconstructed from the `prev_action` fields of `/rl_deploy/policy_obs`, after the
old EMA and before IK; they are **requested target changes**, not measured TCP travel.

**Key quantitative findings:**
- The **110 ms gap is extraordinarily repeatable**: 110.069–110.234 ms, a spread of **165 µs** across five
  independent drops. This tight reproducibility is what identified it as a **deterministic software path**
  (100 ms poll + one 10 ms update), not a stochastic network event.
- **`e2e_pos_action_scale: 0.005` limits each axis to 5 mm, not the 3-D norm** → a multi-axis action can
  request up to **√3 × 5 = 8.66 mm**. Failed runs actually requested up to **7.26 mm per 15 Hz tick**, and
  **97–98 % of active requests exceeded 1 mm**. This directly motivated the 1 mm **norm** cap.
- **Rotation requests reached 2.4–6.8°/tick**, making the new **0.5°** norm limit meaningful, not cosmetic.
- **Peak velocity** inferred from the old direct output: **4.75–7.79 deg/s**, versus Julien's guarded
  **3 deg/s** limit (with 30 deg/s², 300 deg/s³).
- Contact force is a **steep ramp**, not a literal step: `165249` rises **2.08 → 9.12 N over ~110 ms**,
  later recording 10.03 N. In two 30 Hz runs the force peak occurs **~30 ms before** the state gap.
- The old upsampler **keeps publishing after the FRI path is gone**: `165249` continues **212.7 s** past
  the final joint state with **42,543** further direct position commands.
- **A 15 N force cap would not have prevented any of these events** — all transitions occur at **9–12.5 N**.
- Physical |F| **never exceeds ~12.5 N** in any bag.

The short non-drop run `134142` is **suggestive but not a control**: only 7 policy observations, much
smaller requests (1.04 / 1.43 mm), peak only 6.25 N, and it terminated because **RGB went stale**.

### 4.2.3 Earlier single run (2026-08-27, `pdz_20260827_154207`)
First deploy run; dropped `COMMANDING_ACTIVE → MONITORING_READY` after a 110 ms FRI read stall.
Peak **|F| = 10.6 N**, peak torque **71 % of A4**. Joint states/wrench froze while the 200 Hz command
output kept flowing → originally read as read-side preemption (later corrected, B-19).

### 4.2.4 Partial-run policy behaviour before the drop
Recorded across bags: descends **~16–25 mm** toward the socket and closes goal distance from **~40 → 20 mm**
(one run: **32 → 21 mm in ~10 s**), but **never centers XY** (< ~3.4 mm) and jams on the base top surface.

## 4.3 The FRI-drop fix ladder — attempts and outcomes

| # | Attempt | Result |
|---|---|---|
| 1 | DDS isolation to loopback + `ROS_DOMAIN_ID=42` | Fixed an unrelated cross-student TF issue; **did NOT fix the drop** |
| 2 | Killed stray 134 % CPU Isaac process (load 6.3 → 0.16) | Cleaned baseline jitter; **did NOT fix** |
| 3 | `ros2_control_node` → **SCHED_FIFO 80** | Free-space jitter → **0 / EXCELLENT**; **still drops at contact** |
| 4 | CPU pinning FRI → 0–3, vision → 8–31 | `joint_states` std → **0.1 ms**; **still drops** |
| 5 | Shed CPU (drop foxglove + viz, camera-only) | Lower load; **still drops** |
| 6 | **FTEstimator 100 → 30 Hz** | **CONFIRMED live at 26 Hz**; **STILL drops, identical signature** → theory **disproven** |
| 7 | FTEstimator force/torque thresholds → 0 | **No effect on the drop**; **kept** as a genuine parity fix (B-10) |
| 8 | Guarded JTC path (1 mm / 0.5°, stateful quintic, fail-closed FRI checks) | **IMPLEMENTED** on branch `moreno`; **awaiting hardware test** |
| 9 | `ReceiveMultiplier` 1 → 3 (pendant) | **Blocked** — needs Sunrise Workbench access |
| 10 | PC-side estimation (`force_source: external_torque`) | Available, **untested**; bypasses the hardware FTEstimator with zero C++ surgery |
| 11 | Backport jazzy `WrenchEstimator` | **INVESTIGATED → NOT VIABLE** (see below) |

**Why the jazzy backport was rejected (scoped, then abandoned):** Jazzy's `EstimatedWrenchInterface`
exports the wrench as a **chained state interface** via `on_export_state_interfaces()`, which **Humble's
`ChainableControllerInterface` does not have** (even at ros2_control 2.54). It also uses `get_optional()`
(Humble: `get_value()`) and a 2-arg `update_reference_from_subscribers` (Humble: 0-arg). Decisively,
**our deploy reads a plain `/wrench` topic, not a chained interface — so the ported feature is not even
what we use.** No Humble-native "estimate wrench from external torque" node exists upstream. Scope had it
been needed: ~**486 lines across 4 new files** (`wrench_estimator.cpp` 58, `.hpp` 86,
`estimated_wrench_interface.cpp` 262, `.hpp` 80) plus modifications to `system_interface.cpp`, CMakeLists,
the plugin XML, and the bringup launch; **~half a day + debugging** on **shared robot infrastructure**
(the other student's left arm). A full Humble→Jazzy distro upgrade was judged **not viable**.

## 4.4 Cross-check against a surviving comparable deployment

Another student (Julien Casalini, repo `assembly_cell_ws`, commit `30b2083dbe10b5572346e074bbf7b481a6bac6ef`)
runs **force-based insertion on the same robot without these FRI drops**, using the **same** standard
Humble v2.2.3 LBR stack and FRI **POSITION** client command mode. His Sunrise Java **logs**
`getReceiveMultiplier()` but never **calls** `setReceiveMultiplier(...)`, so the repository does **not**
establish his live value.

**Material differences (the basis for the contact-safe redesign):**

| Aspect | Ours (old) | Julien's |
|---|---|---|
| Controller | switched to `LBRJointPositionCommandController` | keeps **`joint_trajectory_controller` active** |
| Command form | direct positions, linear 15→200 Hz bridge | short **q/dq/ddq horizons** from a stateful quintic smoother |
| Cartesian cap | none (up to 7.3 mm/tick observed) | **1 mm/tick** |
| Commissioning envelope | none | **3 mm** check |
| Force for safety | smoothed | **raw baseline-subtracted**, 10 N baseline-relative abort |
| FRI supervision | none during motion | continuous session/quality/safety/drive/command/overlay/control-mode checks |
| Joint velocity | 4.75–7.79 deg/s inferred | **3 deg/s** limit |
| Sunrise mode (real insertion) | native **position** (`control_mode=0`) | expects **CARTESIAN_IMPEDANCE** (`control_mode=1`), compliant lateral stiffness/damping |

His diagnostics record **successful contact while FRI remained COMMANDING_ACTIVE**.

**Caveat recorded in the rationale:** confidence in the guarded path is **moderate, not high**. It should
reduce the contact transient and stop safely when state/tracking is unhealthy, but it **cannot promise**
Sunrise will hold FRI through contact — especially if the decisive difference is **native position vs.
Cartesian impedance**. `expected_control_mode` is a **verifier, not a mode switch**; the mode must be
selected in the Sunrise application. Impedance stiffness/tool-load values must **not** be copied blindly.

## 4.5 Timing and performance measurements

| Quantity | Value | Source |
|---|---|---|
| Control loop | **15 Hz** (66.7 ms/tick) | `control_hz` |
| Actor inference, warm | **~6 ms/tick** | `_warmup_actor` docstring |
| Actor inference, cold (first call) | **~4 s** (cuDNN autotuning) | ditto |
| Per-thread cuDNN re-init penalty | **~4 s per new thread** → 0.27 Hz with a MultiThreadedExecutor | B-16 |
| FRI send period | **10 ms**; `ReceiveMultiplier = 1` → reply required every **10 ms** | pendant, seen at connect |
| Kernel latency spike (generic kernel) | isolated **~110 ms** ≈ every **30–60 s** | `FRI_RECEIVE_MULTIPLIER_FIX.md` (note: this framing predates the shutdown-artifact correction) |
| Post-drop shutdown gap | **110.069–110.234 ms** (5 samples, 165 µs spread) | six-bag analysis |
| `joint_states` jitter after pinning | std **0.1 ms** | handoff |
| Command upsampler output | **200 Hz** | legacy path |
| FTEstimator | 100 Hz → **30 Hz** (measured **26 Hz**) | drop test |
| Camera rate | ~15–30 Hz expected | predeploy checklist |
| Anchor republish | **20 Hz** | `anchor_publish_hz` |
| Max plan duration before failure | 8 s, growing by **×1.25** per retry | `GuardedJointSmoother` |
| MoveIt planning | 5 s, 10 attempts, vel/acc scaling 0.05 | `hole_align.yaml` |

## 4.6 Unit-test results (executed during this export)

```
$ source /opt/ros/humble/setup.bash && source install/setup.bash
$ cd src/rl_deploy_inference && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/ -q
46 passed in 0.36s
```

**All 46 tests pass.** Without the workspace overlay, 36 pass and 2 modules fail to *collect*
(`test_moveit_goal_construction.py`, `test_preinsert_target.py` need `rclpy`/`fp_debug_msgs`) — an
environment requirement, not a test failure. Note `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is required on this
machine: a system `anyio` pytest plugin is incompatible with the system pytest
(`ModuleNotFoundError: No module named '_pytest.scope'`).

| Test module | Tests | What it pins |
|---|---:|---|
| `test_hole_detector.py` | 12 | deproject/project roundtrip, radius scaling, cam→base roundtrip (identity + rotated flange), depth median sampling, synthetic circle detection, nearest-to-expected selection, **contour finds both sockets and rejects fins**, centre-nearest selection without perception, surround fraction high/low, **60 mm pair marking**, extrinsics loading |
| `test_preinsert_target.py` | 7 | hover adds to global z only; `current_tcp`/`fixed`/`down` orientation modes; required-input and unknown-mode errors |
| `test_obs_preprocessing.py` | 6 | **sim order and scaling**; **oldest→newest** stack order; **FOV crop hits sim focal length**; crop no-op without intrinsics; **force smoother starts from zero like sim reset**; **21-D policy vector order** |
| `test_ik.py` | 5 | DLS vs identity Jacobian with damping; pose error translation + tilt quaternion; quaternion↔rotmat roundtrip; **wrench recovery from applied torque**; **action clipping around socket with no yaw delta** |
| `test_deployment_guards.py` | 4 | FRI state accept/reject-all-mismatches; action norm caps; disabled caps preserve action |
| `test_guarded_joint_smoother.py` | 3 | **respects velocity/acceleration/jerk limits**; **replan starts from the committed q/dq/ddq handoff**; rejects out-of-limit goals |
| `test_ik_we_a.py` | 3 | goal delta targets socket bottom in TCP frame; **axis-sign guard rejects the old TCP +Z assumption**; TCP-relative action with bounded cumulative yaw |
| `test_moveit_goal_construction.py` | 3 | pose/joint goal fields; left arm targets `arm_one` |
| `test_obs_preprocessing_we_a.py` | 2 | ImageNet normalization **without** mean subtraction; exact we_A 15-D order |
| `test_kinematics.py` | 1 | **falls back to the vendored KDL parser** when `kdl_parser_py` is missing |

The test suite is deliberately concentrated on the two highest-risk areas: **observation parity** (both
contracts) and **geometry/safety math** — precisely the places where an error is silent.

## 4.7 Calibration and geometry measurements

| Quantity | Value | Note |
|---|---|---|
| Perception (FoundationPose) socket error | **~1 cm** | motivates the entire vision-correction stage |
| Vision-corrected anchor repeatability | dx **0.7 mm**, dy **1.4 mm** | consistent offset ⇒ calibration, not noise |
| D405 hand-eye residual (CAD vs color optical) | **~cm class** | stereo camera; CAD datum is the mount |
| D405 FOV | ~**88°** (848×480, 16:9) | vs sim E2E 224² pinhole ~**57.7°**; sim we_A ~**87.2°** |
| D405 depth dropout | **~45 %** structured no-return | vs sim ~**2 %** |
| Screw tip offset (we_A nominal) | **23 mm** along TCP +Z (5.5 mm shoulder + 17.5 mm shaft) | training randomized ±3 mm axial, ±2 mm lateral |
| Screw tip offset (pdz v3) | **30 mm** | matches sim after the fingerpad 0.005 trim |
| pdz vs y-gripper fingerpad | **3 mm** vs 12 mm; closed half-gap **6 mm** vs 2 mm | changes the grasp |
| Distal mass below the sim force sensor | **0.967506 kg** → **9.49 N** at 9.81 m/s² | masses randomized ±3 % in training |
| we_A ep2500 force-slot running mean | **[0.787, −0.426, −2.564] N** | includes contact samples; **not** a valid no-contact bias |
| Flange→fingertip fallback | **[0, 0, 0.1463] m** | used only when `fingertip_source: flange_pose` |
| Sim wrist-cam offset (y-gripper → pdz) | `[0.042, 0, −0.058]` → `[0.009, −0.05056, −0.07257]` | quat now null / look-at based → **hand-eye must be redone** |
| `lbr_two_link_0` in `base_link` | fixed child at `xyz = 0 0.42 0` | why planning/perception share `base_link` |
| Preinsert hover | 0.15 m (gross) → **0.045–0.06 m** (vision-corrected) | 0.042 m for pdz v3 (TCP 40 mm above opening + 2 mm margin) |

---

# 5. Design Decisions, Evidence, and Rejected Alternatives

Each decision below is stated with the alternative(s) considered, the evidence that settled it, and where
it is implemented.

## 5.1 Load the rl_games **actor only**, and rebuild the sim observation on the real robot
**Alternatives:** (a) export to TorchScript/ONNX; (b) run Isaac Lab alongside deployment; (c) reimplement
the network in plain torch.
**Decision:** load the saved rl_games player, use only its action output.
**Evidence/reasoning:** the image encoder is **not a separate deploy model** — the ResNet-18 branch is part
of the saved `insertion_hybrid` actor, so deploy feeds `{"policy", "image"}` and the loaded actor runs the
ResNet path internally. Re-exporting risks silent numerical drift in exactly the component whose parity
cannot be checked pixel-wise. Running Isaac on the deploy PC is heavy and was explicitly avoided (B-5).

## 5.2 Zero-install Python bridge instead of a fresh virtualenv
**Alternatives:** (a) fresh venv with pinned versions (documented as the *fallback*); (b) container.
**Decision:** `deploy_env.sh` points **system `python3.10`** at the **training conda env's site-packages**
via `PYTHONPATH`.
**Evidence:** the checkpoint was produced with **`torch 2.11.0+cu128`, which is not on PyPI** — reinstalling
torch risks load parity. System `python3.10` is **ABI-compatible with ROS Humble (both cp310)**, so one
interpreter can import both `rclpy`/workspace messages and `torch`/`rl_games`. No download; **byte-for-byte
the same torch/rl_games that saved the `.pth`**.
**Gate:** `python3 -c "import rclpy, torch, torchvision, gymnasium, rl_games, fp_debug_msgs, lbr_fri_idl"`.
**Fallback if unavoidable:** `python3 -m venv --system-site-packages`, `gymnasium==1.2.1`, `rl-games==1.6.5`.

## 5.3 Socket = **position anchor only**; orientation deliberately ignored
**Decision:** only socket *position* feeds the observation delta and the action box; orientation never
steers the policy.
**Evidence:** the policy was trained that way; and perceived object orientation is not trusted. The same
choice propagates upward: `preinsert_planner` defaults to `orientation_mode: current_tcp` ("the perceived
*object* orientation is deliberately **not** trusted yet"), and `socket_axis_mode: base_axis` with
`[0,0,1]` is used because "the fixture is vertical — use `pose` only after certifying FoundationPose tilt."
**Enforcement:** `socket_part_id: -1` holds motion closed; `start_policy` refuses to start.
**Consequence:** socket *yaw* still matters for the opening offset, which is why `_socket_opening()` rotates
the CAD offset by the perceived quaternion (B-3g).

## 5.4 Vision-corrected hole anchor rather than the perception pose
**Alternatives:** (a) use the FoundationPose socket pose directly; (b) improve FoundationPose; (c) detect
the hole directly in the wrist image.
**Decision:** (c), via `hole_align_planner` + `hole_detector`, republished as `/rl_deploy/corrected_socket`.
**Evidence:** the perception pose is **~1 cm off** — against a **~1 mm** clearance, chamfer-free, that is
decisive. Detection is **self-consistent to dx 0.7 / dy 1.4 mm**.
**Sub-decision — contour over Hough:** Hough **over-triggers badly on the fin texture**. The contour method
instead rejects fins **by shape** (circularity 0.65, aspect 1.8, solidity 0.80, fill 0.55) after an HSV
saturation mask isolates the coloured part from the gray table and black gripper.
**Sub-decision — CAD 60 mm pairing:** pairing the two sockets by their known **60 mm** spacing "uniquely
fixes them **without perception or colour tuning**" — a geometry-derived filter, far more robust than
threshold tuning.
**Sub-decision — `morph_open: false`:** morphological opening **erodes real socket rims**; shape filters do
the rejection instead.
**Sub-decision — `require_part_surround`:** a real insertion socket is **embedded in the fins**, whereas a
mounting through-hole **sits at the part edge** with table visible around it → require ≥ 85 % of the
surrounding ring to be part.
**Sub-decision — `fallback_to_perception: false`:** on detection failure, **abort** rather than silently
revert to the ~1 cm-off pose — "the whole point of this node is the vision correction."

## 5.5 EMA factor 0.0625, not 1.0
**Alternative:** no smoothing (`ema = 1.0`).
**Decision:** `ema_factor: 0.0625`.
**Evidence:** sim trained with **per-episode domain randomization `ema_factor_range [0.025, 0.1]`**
(forge_env). `1.0` is **10–40× outside that range**, which would make **both the applied action and the
`prev_action` observation out-of-distribution** — note that `prev_action` is part of the observation, so an
untrained EMA corrupts the input as well as the output. 0.0625 is the **midpoint** of the trained range.

## 5.6 Single-threaded executor + startup CUDA warmup
See B-16. **Rejected:** `MultiThreadedExecutor` — it moves the tick onto rotating worker threads and
re-pays PyTorch's ~4 s per-thread cuDNN init **every tick** (0.27 Hz). Warmup is done on the **main thread**
because the tick runs inference inline on that same thread. Warning comments are placed in both the class
docstring and `main()`.

## 5.7 FK-to-`gripper_tcp` as the fingertip source
**Alternative:** `/right/ee_pose` flange topic + a measured `flange_to_fingertip` offset (retained as a
fallback).
**Decision:** `fingertip_source: fk`.
**Evidence:** it **matches sim** (sim's fingertip is FK of the joint state to the `gripper_tcp` body) and
keeps the **observation, IK error, and Jacobian on one kinematic source**, eliminating a class of
inconsistency where the obs and the IK disagree about where the tool is.

## 5.8 Force source and the force "story"
The force path was revised three times (B-7 → B-8 → B-9) and ended at a deliberately **conservative**
position:
- `arm_external_torque` is the code default (gravity already removed by KUKA);
- `arm_measured_torque` is available but **dangerous with raw effort** (B-7: ~200 N phantom);
- `wrench_topic` is what the **live profile** uses;
- and on the live profile the **policy is fed zero force** (`zero_force_obs: true`) because the estimator
  wrench carries a **pose-dependent leaking residual** (B-8), while **safety and seat logic keep using the
  real contact force**.
**Reasoning:** sim trained on pure-contact ≈ 0 and real contact forces here are small, so zero is closer to
the training distribution than a phantom. The `w2_estimator` config additionally records that the KUKA
`estimated_ft_sensor` wrench was **verified DEAD on this cell — 0.0 N under a 22 Nm hand press** — which is
why the torque-based estimate exists at all.
**Convention:** `ft_sign: -1.0` because `arm_measured_torque` gives the force the **arm exerts**, while
sim's `force_sensor` reads the force **on the tip** (Newton's third law) — so the sign flip makes "socket
resisting insertion" read **+z**, as in sim. (`wrench_topic` ignores `ft_sign`.)
**Safety uses raw, not smoothed, force** — "safety should not wait for observation smoothing during a fast
contact ramp."

## 5.9 Depth cleaning kept OFF by default
**Decision:** `depth_clean: false`; the parity path is the default.
**Evidence/trade-off:** the real D405 has **~45 % structured dropout** vs sim's **~2 %**, so inpainting
(Telea, radius 3) + median denoise (k=5) makes the depth channel look more in-distribution — **but it
interpolates over the socket void too**, trading away the hole's depth cue for a smooth surface. The RGB
still carries the hole. Left as an explicit **A/B test** rather than a silent default.

## 5.10 FOV-match crop: ON for E2E, OFF for we_A/pdz
This looks inconsistent but is correct, because the two **sim cameras differ** (B-12): E2E sim is a
**224×224 pinhole at ~57.7°** vs a real ~88° stream → crop **required**; we_A sim renders **native 320×180
16:9 with no crop** and the real D405 is **also 16:9 at ~88° ≈ sim 87.2°** → a plain resize matches, and
cropping **introduces** an off-center shift because it centers on `(cx, cy)`, not the image center.

## 5.11 Guarded JTC bridge instead of direct 200 Hz position streaming
**Alternatives:** (a) keep the direct upsampler; (b) lower the action scale only; (c) change Sunrise first.
**Decision:** keep `joint_trajectory_controller` **active** and publish stateful quintic q/dq/ddq horizons.
**Evidence:** the bags show the old path requested **5.6–7.3 mm and 2.4–6.8°/tick** with **97–98 % of active
requests above 1 mm** and inferred peak **4.75–7.79 deg/s**, while the surviving comparable deployment caps
at **1 mm** and **3 deg/s** and keeps JTC. Letting `ros2_control` own the trajectory handoff replaces
piecewise-linear position streaming into FRI with a derivative-bounded, C²-continuous trajectory.
**Rejected (b):** lowering `e2e_pos_action_scale` 0.005 → 0.003 — "untested, low confidence (reduces but may
not eliminate the transient); **cannot go too low or the policy can't seat**."
**Norm-vs-axis insight:** the guard limits the **vector norm**, preserving direction; the trained per-axis
scale allowed **√3 ×** the nominal step in a diagonal move.
**Honest caveat:** confidence **moderate, not high** — the decisive difference may be Cartesian impedance.

## 5.12 Retraining deliberately deferred
**Question posed:** should training change before the next test?
**Decision:** **No, not for the next FRI-survival test.**
**Reasoning:** "retraining now would change two variables at once and make the result harder to interpret."
The 1 mm/0.5° guards **do** materially alter almost every active action in the failed runs, so the current
checkpoint may become slower or miss its timeout — **acceptable**, because the endpoint of that experiment
is *"FRI stayed healthy through repeated controlled contact,"* **not** insertion success.
**For the final policy:** retrain/fine-tune **with the deployment actuator model** — same vector-norm action
limits, 15 Hz action hold, bounded joint/Cartesian response, force-dependent downward attenuation, realistic
compliance/control mode, and the same force preprocessing — so success rate can be judged without a
train/deploy dynamics mismatch.

## 5.13 Manual seat judging for first hardware runs
**Decision:** `trial_stop_mode: manual`.
**Evidence:** the geometry gates are measured against the perceived socket **anchor**; a ~cm-off anchor trips
`failed_overtravel` on the **first tick** even when the fingertip is physically over the true hole. So the
policy still *receives* the perceived pose, but that pose **is not trusted to certify insertion**. Hard stops
(force cap, timeout, stale inputs, e-stop) remain active. `auto_seat` is re-enabled only once the anchor is
accurate.
**Related:** the deployed E2E actor is a **5-D action policy with no success-probability output** — an
explicit warning not to use an "80 % actor success" threshold unless a future checkpoint is trained with a
calibrated success head **and** the deploy adapter is updated to read it.

## 5.14 `seat_force_requires_geometry: true`
**Decision:** a seat-force hit **above** the bottom zone is a **jam/failure** (`failed_early_contact`), not
success.
**Reasoning:** with a chamfer-free 1 mm clearance, force alone is ambiguous — pressing on the base top
surface beside the hole produces the same force as seating. Bag evidence (B-20) confirms exactly this
failure mode. Legacy "force alone means seated" is recoverable by setting the flag false.

## 5.15 MoveIt for gross motion, RL for the final local insertion
**Rejected:** using the RL node's `reset_preinsert` IK servo for the gross move — "a local damped-IK servo
that **rattles from far away**."
**Decision:** MoveIt plans the collision-aware, jerk-limited gross move; the RL policy does **only** the
final local insertion. Controller handoff is explicit and ordered (JTC for preinsert, then switch), because
both controllers share the position command interface and **only one may be active**.

## 5.16 Fail-closed everywhere
A consistent theme, implemented as: motion disabled by default; `policy_active_on_start: false`;
`socket_part_id: -1` blocks start; calibration gates (`we_a_geometry_calibrated`,
`we_a_force_bias_calibrated`) block motion on a new end effector; `fallback_to_perception: false`;
detection failure aborts; the bridge **latches** and requires an explicit service call to clear; FRI health
is required **before and during** motion; dry-run defaults with typed `MOVE` confirmation on every planner.

## 5.17 Pure, ROS-free core modules for testability
`hole_detector.py`, `obs_preprocessing.py`, `ik.py`, `deployment_guards.py`, and `guarded_joint_smoother.py`
contain **no rclpy**, so the parity and geometry math runs in plain pytest (46 tests, §4.6). The ROS glue is
isolated in the node files. This is what makes observation parity testable at all.

## 5.18 Single-arm right-only bringup
**Decision:** bring up only the right arm (`lbr_two`, one FRI session, port 30201).
**Reasoning:** the left arm is **in live use by another student**. MoveIt keeps the **dual** robot_description
(left arm = zeros placeholder) plus a workspace-separated **keep-out box at y < −0.20** by design.
**Consequence:** the FoundationPose pipeline runner is **hardcoded to exactly 3 cameras**
(`choices=[3]`), so in single-camera mode the perception pipeline is **not run at all** and the socket anchor
comes entirely from the **geometric hole-align** path — which is precisely why `use_socket_pair` (the 60 mm
CAD cue) matters: it needs no perception.

## 5.19 Not chasing the 110 ms number
An explicit methodological decision after the shutdown-artifact discovery: *"Do not use the 110 ms
publication gap itself as the cause"*, and *"Do not modify the 100 ms socket-close polling sleep as a
proposed fix — shortening it would only make the visible ROS publication gap smaller after FRI had already
failed."* The objective was restated as identifying the event **before** the shutdown wait, which requires
packet capture and `LBRState`/`rosout` recording that the old bags simply did not contain.

---

# 6. Current TODOs and Known Open Issues

## 6.1 ⛔ OPEN — FRI drops on first substantial contact (THE deploy blocker)
**Why unresolved:** the cause has not been *localized to a side*. The 110 ms signature is a post-drop
shutdown artifact (B-19), the FTEstimator theory is disproven by test, and the old bags recorded **neither
`LBRState`, nor `/rosout`, nor FRI UDP packets**, so they cannot say whether a PC reply was late, a
controller packet was missing, or Sunrise itself initiated the transition on a motion/safety condition.
The leading hypothesis (rigid contact + `open_loop: true` command progression) and the competing one (reply
deadline with `ReceiveMultiplier = 1`) **make the same observable** in the available data.
**Blocked on:**
- **Sunrise Workbench access** (not on the deploy PC) to change `ReceiveMultiplier` 1 → 3 and to read the
  live startup line (control mode, send period, receive multiplier). The deploy engineer does not have it.
- **A packet capture** on UDP 30201 synchronized with `LBRState`/`rosout`.
- **Julien's live Sunrise startup values** (his repo logs but never sets the multiplier).
**Next actions (ordered):**
1. Deploy through the guarded JTC bridge (1 mm, 0.5°, raw-force 15 N abort, 6 N downward attenuation,
   fail-closed FRI checks).
2. Record `/lbr_dual_arm_y_gripper/state`, `/rosout`, the JTC trajectory topic, joint states, raw wrench,
   policy targets **plus a `tcpdump` of UDP 30201**; confirm the actual Sunrise `control_mode` before
   setting `expected_control_mode`.
3. Ask Julien for his live startup line; test multiplier 3 separately if packet timing shows a late reply.
**Staged test plan (A–D) with an explicit decision rule:**
- **A** no-motion instrumentation (≥ 30 s healthy, one JTC publisher, both packet directions captured).
- **B** contact **without** policy advance — external loads at ~2, 4, 6 N, stopping before the historical
  9–12.5 N range. *Pass = force ramps through 4–6 N with no state gap > 30 ms, no quality/state transition,
  no latch.*
- **C** very small policy contact — **0.25 mm** cap, 0.25° rotation, 4 N attenuation, **10 N** abort, short
  timeout, **≥ 5 controlled contacts**; then 0.5 mm, then 1 mm.
- **D** control-mode comparison — repeat under correctly commissioned Cartesian impedance with identical ROS
  limits.
- **Decision rule:** B drops ⇒ command shaping is insufficient, focus on Sunrise mode/`open_loop`/tool-load/
  multiplier/network. B survives but C drops ⇒ the commanded approach/force ramp is implicated. Position
  drops but impedance survives ⇒ **native contact compliance is the decisive difference**. All survive ⇒
  extend the timeout and only then evaluate insertion performance.
**Also flagged:** `open_loop: true` vs `false` must be tested **deliberately** in T1 — it requires hardware
restart/reconfiguration and is **not** a runtime observation parameter. Until then, **neither the old bag
tracking metric nor the bridge's 0.35° tracking guard proves physical tracking**, because the published
state is the filtered command. Cartesian impedance automatically forces closed-loop state reporting.

## 6.2 ⛔ OPEN — The policy never centers laterally (the insertion blocker proper)
**Why unresolved:** two candidate causes remain unseparated (B-20): a **consistent hand-eye calibration
offset** in the anchor, versus the **policy being unable to localize** the chamfer-free hole at that height.
It is currently **untestable** because no run survives long enough (drops at 1.1–3.6 s).
**Fix order proposed in `DIAGNOSIS.md`:**
1. **+5 N Z-bias** — recapture the F/T baseline at the true hover pose, or set `ft_bias_base_xyz` z ≈ −5.
   Removes a known confound from seat/safety. *Cheap.*
2. **Anchor XY vs true hole (hand-eye)** — place the tip over the hole by hand and compare against the
   republished anchor; if offset, fix hand-eye. **This is the seating blocker.**
3. **Start centered** — ensure the peg is placed **at** the anchor XY at policy start (obs XY delta ≈ 0 as in
   sim), not 4–9 mm off; tighten preinsert/hole-align XY convergence before START.
4. **FRI robustness** so contact transients do not kill runs while tuning.

## 6.3 ⚠ OPEN — Sim/deploy start-distribution mismatch
The deploy start-delta is **5–9 mm** where sim's is **≈ 0** (B-20, Finding 1). This is a genuine
train/deploy **distribution** mismatch, not a bug in either system: sim *places* the peg at the noisy
anchor G so the observed delta starts at zero, whereas deploy places the tip at a hover pose while the
observation is measured against a separately republished anchor, and the two do not coincide at t=0.
**Unresolved because** fixing it properly means either changing the deploy start convention (item 3 above)
or retraining with the deploy convention — and retraining is deliberately deferred (§5.12).

## 6.4 ⚠ OPEN — D405 hand-eye residual
CAD `T_flange_cam` uses the **mount** datum, not the **color optical frame**; the D405 is stereo → ~cm
residual (B-21). Must be fixed **empirically** against a pose-matched real frame. Compounded by the pdz
gripper move: the sim wrist-cam offset changed from `[0.042, 0, −0.058]` to `[0.009, −0.05056, −0.07257]`
with a now-null/look-at quaternion, and **there is no camera-extrinsic parameter in the deploy config** —
the match is physical plus the image itself, so the hand-eye must be redone for the new mount.

## 6.5 ⚠ OPEN — Pose-dependent gravity residual in the KUKA wrench
The ~+5 N hover bias "leaks" as the wrist tilts (B-8), producing a ~−15 N phantom on policy force-z. The
**true fix is pendant-side** — a corrected Sunrise tool/payload model. Currently worked around with
`zero_force_obs: true`. Consequently **force is not contributing to the deployed policy at all**, which is
a meaningful limitation for a contact-rich task: the handoff note records "revisit a proper force story only
if vision+pose alone underperforms."

## 6.6 ⚠ OPEN — pdz gripper geometry not fully certified
`screw_tip_offset_tcp_xyz` was carried over from we_A as a **placeholder** `[0,0,0.023]`, later set to
`[0,0,0.030]` for pdz v3 as *nominal*. The config comments still instruct: re-measure the real pdz grasp
(tip offset and shaft axis), confirm `tip_link` names the pdz TCP link in the **real** `robot_description`,
and re-check the `preinsert_hover_z_m` margin. The live profile sets `we_a_geometry_calibrated: true`, but
the surrounding comments were not updated to record *what* was measured — a documentation gap worth closing
before the thesis writes a number down.

## 6.7 ▫ OPEN — Configuration portability
`deploy_pdz_v3_fulltilt.yaml` and `deploy_pdz_v3_fulltilt_single.yaml` are **ignored by the repository's
broad `*.yaml` rule**. They hold the live guard values on this machine and the installed symlink workspace
sees them, but those edits **will not travel with a normal commit/clone** until the profiles are
force-tracked or the ignore rule is narrowed. Mitigation in the meantime: verify live values with
`ros2 param get` during commissioning.

## 6.8 ▫ OPEN — Uncommitted work on branch `moreno`
12 modified files (+678/−113) plus untracked files are **not committed** (the user handles all git):
`CONTACT_SAFE_DEPLOYMENT_RATIONALE.md` (+156), `FRI_CONTACT_DROP_{FIXES,HANDOFF}.md`,
`deploy_inference_pdz.launch.py` (+56), `command_upsampler.py` (+61, adaptive period + hold topic),
`deployment_guards.py` (+15), `ik.py` (+75, weighted/nullspace IK), `inference_node.py` (+20, hold latch),
`inference_node_we_a.py` (+6), `obs_preprocessing_we_a.py` (+7, `fov_match` default flip),
`serl_backend.py` (+306), `setup.py` (+1); untracked: `.vscode/`, `dump_policy_image.py`,
`scripts/run_hardware_left_rt.sh`, `sim_wrist_rollout.mp4`.

## 6.9 ▫ OPEN — `preinsert_mode: socket_hover` untested on hardware
Marked **"UNTESTED ON HARDWARE → walk it guarded"** in both the parameter comment and the config. It needs
a good socket estimate and a roughly upright start pose.

## 6.10 ▫ OPEN — `auto_seat` gates never exercised
All hardware runs used `trial_stop_mode: manual`, so the geometry/depth-ROI/seat-force success logic
(`seat_z_tolerance_m`, `seat_xy_tolerance_m`, `seat_depth_roi_max_m`, overtravel) has **never run on real
hardware** — it is unit-tested only.

## 6.11 ▫ OPEN — Weighted/nullspace IK not wired in
`get_delta_dof_pos_weighted` and `get_delta_dof_pos_nullspace` exist and address a real observed problem
(shoulder dominance / elbow collapse, B-14) but the control tick still calls plain `get_delta_dof_pos`.
They are also **not covered by the test suite**.

## 6.12 ▫ OPEN — Offline preinsert-planner backlog
Recorded as a backlog for when hardware is unavailable: hardening, mock tests, handoff documentation, docs.

## 6.13 ▫ Environment note — pytest plugin conflict
Running the tests requires `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` on this machine: the system `anyio` pytest
plugin is incompatible with the system pytest (`ModuleNotFoundError: No module named '_pytest.scope'`).
Two test modules additionally require the built workspace to be sourced.

## 6.14 Deferred / explicitly abandoned lines
| Line | Status | Reason |
|---|---|---|
| Jazzy `WrenchEstimator` backport | **abandoned** | Humble lacks `on_export_state_interfaces`; and we read a topic, not a chained interface (§4.3) |
| Full Humble → Jazzy distro upgrade | **not viable** | shared robot infrastructure |
| Aux grasp head | **dead end** | −11 pp @160 px, unlearnable |
| Aux hole head | dropped | neutral |
| Tactile / force-history fusion | **dead** | the gripper has **no tactile sensor**; force history (LSTM) also dropped |
| Shortening the 100 ms socket-close sleep | **explicitly forbidden** | would only shrink the visible gap *after* FRI already failed |
| Lowering action scale 0.005 → 0.003 | deprioritized | untested, low confidence, risks inability to seat |

---

# 7. Verbatim Reproduction of All Repository Documents

This section reproduces every markdown document in the repository **in full and unmodified**. Each is
enclosed in a five-backtick fence so that its own markdown (including triple-backtick code fences)
survives intact. Where a document's content is also summarized earlier in this export, this section is
the authoritative text.

## 7.1 `CLAUDE.md`

`````markdown
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
`````

## 7.2 `README.md`

`````markdown
# Masterthesis-rl-deploy

Real-robot deployment workspace for the end-to-end RL cooling-screw insertion policy:
iiwa7 + custom Y-gripper, wrist D405 RGB-D + wrist F/T + proprioceptive state, 5-DoF policy
delta action, damped least-squares differential IK, and FRI joint-position streaming at 15 Hz.

The trained policy is produced in `~/Masterthesis-rl-train` for
`Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0`. The deployment node loads the rl_games
actor only and builds the same dict observation as sim:

`policy`: `[fingertip_pos - socket_pos_estimate, fingertip_quat, ee_linvel, ee_angvel, ft_force, prev_action]`

`image`: wrist RGB-D, RGB in `[0, 1]` with per-image mean subtraction, depth inf-to-zero,
clamped to `[0, far]`, divided by `far`, then temporal frame-stacked oldest-to-newest.

## Workspace

- `src/rl_deploy_inference`: ROS2 `ament_python` deployment package.
- `src/fp_debug_msgs`: FoundationPose debug/action/message package.
- `src/lbr_fri_idl`: FRI ROS2 message definitions extracted from the local LBR stack.

## Build

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --symlink-install
source install/setup.bash
```

The inference node must be run from a Python environment that can import `torch`, `gymnasium`,
`rl_games`, and the training repo's `insertion_policy` package.

## Run

For the practical operator guide, use [docs/HOW_TO_DEPLOY.md](docs/HOW_TO_DEPLOY.md).

Start the vision stack and the LBR/FRI stack first, then:

```bash
ros2 launch rl_deploy_inference deploy_inference.launch.py
```

The default config is installed at:

`src/rl_deploy_inference/config/deploy.yaml`

Motion is disabled by default. The node will publish hold commands while disabled if joint state is
fresh, but it will not publish policy-generated commands until `enable_motion:=true`.

Policy execution is controlled through services:

```bash
ros2 service call /rl_deploy/start_policy std_srvs/srv/Trigger {}
ros2 service call /rl_deploy/stop_policy std_srvs/srv/Trigger {}
ros2 service call /rl_deploy/reset_preinsert std_srvs/srv/Trigger {}
```

Before enabling motion, set `socket_part_id` to the fused FoundationPose part ID for the socket.
The node ignores socket orientation by design and only uses socket position as the action anchor.
With `socket_part_id: -1`, motion is held closed even if `enable_motion` is set.

## Key Defaults

- Wrist camera: `/realsense_2/camera/color/image_rect`,
  `/realsense_2/camera/aligned_depth_to_color/image_rect`
- Socket pose: `/perception/fp/pose_base/fused/assembly`
- Joint state: `/lbr_dual_arm/joint_states` or `state`
- Flange pose: `/right/ee_pose`
- FRI command: `command/joint_position`
- E-stop: publish `std_msgs/Bool(true)` on `/rl_deploy/e_stop`
- Deploy checkpoint: `~/Masterthesis-rl-train/logs/rl_games/Forge/w2_estimator_192/nn/last_Forge_ep_2000_rew_162.13815.pth`

`w2_estimator_192` is the explicit-estimator policy (83.2% success at the full ±2.5 cm socket-localization
noise, vs 67.8% without the estimator). It is the same E2E visuomotor policy as `e2e_weld_curric` plus an
internal vision estimator head that localizes the hole from the wrist image and feeds it into the policy.
Two deploy-relevant consequences, both already handled in code:

- Its `agent.yaml` lists a privileged `aux_label` obs group and an `aux_head`. `aux_label` is a
  training-only label (the true hole gap) that the network excludes from the policy input and whose aux
  loss is skipped at inference, so it has **zero effect on the action**. The loader auto-declares
  `aux_label` in the obs space (so the saved input-RMS restores cleanly) and feeds dummy zeros
  `(1, 4)` every step.
- It was trained with **gravity compensation**: the `ft_force` obs is pure contact force. Keep
  `ft_bias_base_xyz: [0, 0, 0]` and feed the robot's gravity/payload-compensated F/T directly.

## Safety Gates

1. Build and import-test the workspace.
2. Run the node with `enable_motion: false` and verify all inputs are fresh.
3. Dump one real observation by setting `obs_dump_path` and compare it against a matching sim rollout
   with `ros2 run rl_deploy_inference obs_parity`.
4. Confirm socket part ID, F/T sign/frame, flange-to-fingertip offset, and joint order.
5. Enable low-power/low-PD robot settings externally.
6. Start with `max_joint_step_rad: 0.010`, `e2e_pos_action_scale: 0.01`, and `trial_stop_mode: manual`.
7. Enable motion only after the dry run is boring in the best possible way.

First hardware runs use manual success stopping: the policy still receives the perceived socket pose,
but that pose is not trusted to certify insertion. Force cap, timeout, stale inputs, and e-stop still
stop/hold the robot. Set `trial_stop_mode: auto_seat` later to re-enable geometry/depth/seat-force
success gates.
`````

## 7.3 `AGENTS.md`

`````markdown
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
`````

## 7.4 `DIAGNOSIS.md`

`````markdown
# Deploy failure diagnosis — 4 bags (2026-08-27), analysed 2026-08-28

Bags: `pdz_single_20260827_{162621,163807,165249,171512}` (config `deploy_pdz_v3_fulltilt_single`,
checkpoint `pdz_v3_fulltilt_20260826`). Plots + CSVs in this folder.

## TL;DR
The FRI drop is a **symptom**. Root cause: **the peg tip never centers on the hole (never < 3.4 mm
XY; hole clearance ~1 mm), descends onto the cooling-base TOP surface next to the hole, can't enter,
hunts sideways, DRIFTS FURTHER AWAY, then slips + hard-recontacts (0→10 N) — and that force
transient trips the ReceiveMultiplier=1 FRI.** Plus a **+5 N Z-bias** in the wrench (reads +5 N while
hovering, no contact).

## Consistent across ALL 4 bags (not one-off)
| Quantity | b162621 | b163807 | b165249 | b171512 |
|---|---|---|---|---|
| min lateral XY miss (mm) | 5.9 | 3.4 | 4.7 | 3.7 |
| XY miss at END (mm) | 12.9 | 32.7 | 12.6 | 31.4 |  ← drifts AWAY
| min height above socket-z (mm) | 13.2 | 13.9 | 17.6 | 13.1 |  ← blocked, never < 13 mm
| ever centered (XY<3mm)? | no | no | no | no |
| ever near socket-z (z<5mm)? | no | no | no | no |
| drop force cycle | 5→0→10.2 | 5→0→10.8 | 5→0→10.0 | 5→0→12.1 | ← identical
| hovering (no-contact) Fz | +5.0 | +4.9 | +4.9 | +5.1 | ← Z-BIAS

The drop always coincides with the run's peak force, which is always a 0→10 N re-contact transient
after the tool unloads — never during smooth motion. The slip is CONSISTENT, every bag.

## Finding 1 — the deploy start-delta is NON-ZERO; sim's is ~0 (THE core distribution mismatch)
Sim design (VERIFIED in code — `insertion_env_e2e_iiwa.py::_goal_delta_ee`, docstring L290):
- The peg is **physically spawned above the NOISY goal G = true_socket + injected_goal_error**.
- So the policy's obs delta (`fingertip − G`) is **≈ 0 at reset** — "the policy thinks it's at the goal."
- The **TRUE hole is up to `goal_anchor_lat_max` (±1.75 cm) away** from where the peg starts.
- `hand_init_pos=[0,0,0.0675]`, noise `[0.003,0.003,0.010]` → the ±3 mm is jitter about G (settling),
  tip ~rim+37.5 mm. The policy's job: obs says delta≈0, but VISION must reveal the true hole is offset
  and servo there. (Corrects an earlier wrong note that said "peg spawns above the TRUE hole.")

Deploy reality (from bags, `policy_obs[0:3]` = fingertip − socket_estimate at tick 0):
- b162621 |xy|=5.9, b163807 |xy|=8.6, b165249 |xy|=4.7, b171512 |xy|=5.1 mm — **NON-ZERO.**
- In sim this delta is ~0 at reset; in deploy it starts 5–9 mm. **The policy is handed a starting obs
  it never saw in training.** In sim, a nonzero delta means "estimate error to close via vision, from
  a ~0 baseline"; here we start it already nonzero from a different geometric convention, and the
  learned response drives the delta to GROW (5 → 12–33 mm) instead of shrinking.
- Height: tip starts 29–43 mm above socket-z ≈ sim rim+37.5 mm — **height is fine**, problem is LATERAL.

Root of the mismatch: in sim the peg is PLACED at the anchor G so obs-delta starts 0. In deploy the
preinsert/hover places the tip at the hover pose, but obs-delta is vs the republished anchor, and the
two do NOT coincide at t=0 (they're 5–9 mm apart). That nonzero start is out-of-distribution and the
policy's learned servo diverges from it.

## Finding 2 — the +5 N Z-bias is real and unhandled
The wrench reads **Fz ≈ +5 N with the arm HOVERING (no contact)** in all 4 bags. `ft_baseline_on_start`
did not zero it (or captured at a different pose). This config sets `ft_bias_base_xyz → 0` on the
assumption gravity comp is "now correct" — but the data says there is still a ~5 N residual.
Consequences: seat-detection and force-cap safety see a false +5 N floor; true contact = deviation
from +5 N (so the 0 N dips are the tool pulling UP, the 10 N spikes are ~5–7 N real contact).
`zero_force_obs=true` hides it from the policy but not from the safety/seat logic.

## Why it drifts AWAY (the core open question)
Fingertip−anchor XY grows 5 → 12–33 mm over the run: the policy servos the tip AWAY from its own
socket estimate. Either (a) the hole ANCHOR XY is wrong (hand-eye residual) so "toward true hole" per
vision ≠ toward anchor, or (b) the policy can't localize the real hole at this height without a
chamfer and wanders. Anchor DETECTION is self-consistent across runs (spread dx 0.7 mm, dy 1.4 mm),
so if the anchor is wrong it's a consistent CALIBRATION offset (hand-eye), not noise.

## Fix order (proposed)
1. **+5 N Z-bias**: recapture ft baseline at the true hover pose, or set `ft_bias_base_xyz` z≈−5.
   Removes a known confound from seat/safety. Cheap.
2. **Anchor XY vs true hole (hand-eye)**: verify the republished anchor lands on the real hole center
   (place tip over hole by hand, compare). If offset, fix hand-eye (see memory `d405-handeye-calib`).
   This is the seating blocker.
3. **Start centered**: ensure the peg is placed AT the anchor XY at policy start (obs XY delta ~0 like
   sim), not 4–9 mm off. Tighten preinsert/hole-align XY convergence before START.
4. FRI robustness (`ReceiveMultiplier`≥3, pendant) so contact transients don't kill runs while tuning.
`````

## 7.5 `FRI_CONTACT_DROP_FIXES.md`

`````markdown
# FRI drops on contact — diagnosis + fix ladder (2026-08-28)

## Confirmed problem and 110 ms correction
The FRI session drops (`COMMANDING_ACTIVE -> MONITORING`, `ERROR_FRI_CMD_WRONG_STATE_ACTIVE`) during
the first substantial contact ramp. Five of six available raw bags show a repeatable ~110 ms gap in
joint/wrench publication at the transition.

That 110 ms interval is almost certainly a **post-drop shutdown artifact**, not the cause or measured
duration of a robot-to-PC network freeze. Humble v2.2.3 detects that FRI already left
`COMMANDING_ACTIVE`, calls `close_udp_socket()` synchronously from `SystemInterface::read()`, and waits
for the worker with a fixed 100 ms polling sleep. One normal 10 ms update plus that sleep explains the
observed ~110 ms. The independent 200 Hz ROS command topic continuing does not prove the FRI UDP send
loop was healthy. The old bags did not record `LBRState`, `/rosout`, or packets, so they cannot tell
which UDP direction or Sunrise condition initiated the transition.

## ⚠️ UPDATE 2026-08-28 — FTEstimator theory DISPROVEN by test
Ran the deploy with FTEstimator at 30 Hz (CONFIRMED live: wrench value-change rate measured 26 Hz in
bag `pdz_single_30hz_20260828_140335`). **It STILL dropped at contact**, with the same post-transition
110 ms shutdown signature. Force-threshold=0 also applied, no effect. This disproves estimator rate as
a sufficient fix, but it does not prove the actual FRI UDP thread met every deadline. Julien's surviving
deployment differs materially in command shaping, safety supervision, and intended Sunrise control mode.
The primary next test is the guarded JTC path plus packet capture, not more estimator tuning.

## Root cause (ORIGINAL THEORY — now disproven, see UPDATE above)
The async `lbr_fri_ros2::FTEstimator` runs its own thread at update_rate=100 Hz / rt_prio=30, next to
the FRI send loop (rt_prio=80) on the same PC. Source inspection shows it computes the Jacobian
pseudo-inverse every cycle, including free motion; crossing a force threshold does not turn that work on.
Upstream later removed the async worker and replaced it with a synchronous WrenchEstimator controller,
but the release note does not establish our contact-drop mechanism.
Our stack: humble-v2.2.3, still the async FTEstimator. ros2_control = 2.54.0 (recent; HAS
ChainableControllerInterface).

## Constraint: the policy USES force
3 of 4 bags show non-zero policy force obs; sim trains WITH force in the 15-D vector
[goal_delta(6), force(3), prev_action(6)]. So we CANNOT disable the estimator — force must keep
flowing. The policy reads force at 15 Hz, so any estimator rate >= ~20 Hz feeds it fully.

## Fix ladder (cheap -> expensive) — try in order
1. **FTEstimator 100 -> 30 Hz** [DONE 2026-08-28, DID NOT FIX DROP]. Edited both
   lbr_two_system_config.yaml (right) and lbr_one_system_config.yaml (left). 30 Hz >> 15 Hz policy,
   < 100 Hz controller_manager. Keeps force + safety + seat-detect. Further rate tuning is low priority.
2. **Guard the deployment command path** [IMPLEMENTED on branch `moreno`]: keep JTC active; publish
   stateful q/dq/ddq quintic horizons; cap applied TCP change to 1 mm/tick and 0.5 deg/tick; use raw
   baseline-subtracted force for the 15 N abort; attenuate downward command toward 6 N; require healthy
   live FRI state before and during policy motion.
3. **ReceiveMultiplier 1 -> 3 or higher** on the pendant remains a useful tolerance experiment. At a
   10 ms send period, multiplier 3 permits the normal reply cadence to be 30 ms. The observed 110 ms is
   post-drop shutdown and must not be used as the required tolerance. Capture packets, ask Julien for his
   live startup value, and test this independently. See `FRI_RECEIVE_MULTIPLIER_FIX.md`.
3b. **PC-SIDE ESTIMATION (the ace — likely better than the backport)**: the deploy node ALREADY
   computes the wrench from raw external torque PC-side: `wrench_from_external_torque(J, tau_ext)` =
   `pinv(J^T)*tau_ext` (ik.py:94), selectable via `force_source: external_torque` (uses the FRI
   `external_torque` state interface directly, published by the existing lbr_state_broadcaster). This
   BYPASSES the hardware async FTEstimator entirely and removes that worker as one variable,
   with ZERO C++/config surgery on the kuka repo. Caveat: validate sign/scale/DC on hardware
   (press-test) and it's NOT gravity-comp'd the same way as the estimator wrench, so re-check the
   force story. Try this if 30/20Hz insufficient, BEFORE any backport.

3. **Backport the synchronous WrenchEstimator from jazzy** — INVESTIGATED 2026-08-28, NOT VIABLE as a
   clean port: Jazzy's EstimatedWrenchInterface exports the wrench as a CHAINED STATE INTERFACE via
   `on_export_state_interfaces()`, which Humble's ChainableControllerInterface (even ros2_control 2.54)
   does NOT have. Also uses `get_optional()` (Humble = `get_value()`) and a 2-arg
   `update_reference_from_subscribers` (Humble = 0-arg). AND our deploy reads a plain /wrench TOPIC, not
   a chained interface -- so the ported feature isn't even what we use. No ready-made Humble-native
   "estimate wrench from external torque" node exists upstream (searched). A from-scratch Humble-native
   async-free broadcaster is possible but UNNECESSARY given 2b. SCOPE (if ever needed):
   - New files from origin/jazzy (~486 lines, 4 files):
     lbr_fri_ros2/{src/wrench_estimator.cpp (58), include/lbr_fri_ros2/wrench_estimator.hpp (86)}
     lbr_ros2_control/{src/controllers/estimated_wrench_interface.cpp (262),
                       include/lbr_ros2_control/controllers/estimated_wrench_interface.hpp (80)}
   - Modify: system_interface.cpp (drop async FTEstimator, expose state ifaces), CMakeLists, plugin xml,
     bringup launch (spawn estimated_wrench_interface chainable controller).
   - EstimatedWrenchInterface : public controller_interface::ChainableControllerInterface -> available
     in our ros2_control 2.54.0.
   - ~half-day + debugging. Shared robot infra (also the other student's left arm) -> do deliberately,
     not rushed. Reversible via git. Full-distro Humble->Jazzy upgrade = NOT viable; backport only.

## Non-causes ruled out (don't re-chase)
- Joint ordering: FINE (node maps joint_states by name; cmd is A1..A7; tracking err ~2 mrad). The
  "scrambled joint_states order" is cosmetic; NOT a bug.
- Memory leak: NO (56 Gi free, swap ~0; drops happen in 1-4 s, far too fast for a leak).
- General PC load was reduced (SCHED_FIFO 80 + core pinning + killed a stray Isaac hog), and free-motion
  behavior improved. This does not prove the actual FRI `ClientApplication::step()` thread met every
  10 ms reply deadline at contact.
- A long whole-PC stall is unlikely, but continuity of the separate ROS command publisher does not rule
  out a short FRI-thread, UDP, NIC, or controller-side timing failure.
`````

## 7.6 `FRI_CONTACT_DROP_HANDOFF.md`

`````markdown
# FRI Contact-Drop — Handoff / Problem Summary (2026-08-28)

**Author context:** real-robot deployment of the E2E RL cooling-screw insertion policy
(iiwa7 + custom Y/pdz gripper, wrist D405 RGB-D + wrist F/T → 5-DoF delta → IK → FRI joint-position
@ 15 Hz). Single-arm RIGHT (`lbr_two`, FRI port 30201). This file summarizes THE deploy blocker, what
we proved, what we tried, and what's left. Detailed data + plots live in `deploy_analysis/`.

---

## TL;DR
**The FRI session drops on the first substantial contact ramp, killing insertion before the policy can
search/seat.** The repeated ~110 ms publication gap is now traced to the Humble stack's post-transition
shutdown path: after FRI has already left `COMMANDING_ACTIVE`, `close_udp_socket()` polls the worker with
a fixed 100 ms sleep. It is not evidence that a 110 ms network freeze caused the drop. The old bags do
not identify which side missed the exchange. The strongest actionable difference remains our command
behavior: direct joint-position streaming, multi-axis requests up to ~7.3 mm, a 42 N cap, smoothed-force
safety, and no FRI-state/tracking preflight versus Julien's guarded 1 mm q/dq/ddq JTC path.

---

## The exact symptom
Pendant, at the moment of contact:
```
Quality change signalled POOR
Jitter 3.43 / Latency 1.6
Session State change MONITORING_WAIT
[Error] Wrong FRI session state 'Monitoring (Wait)' in active phase of FRI motion.
        Possible connection problem. (ERROR_FRI_CMD_WRONG_STATE_ACTIVE)
```
ROS side: `LBR left COMMANDING_ACTIVE. Please re-run lbr_bringup`, FRI session disposed.

Crucially: **jitter/POOR appears ONLY at contact** — free-space motion is EXCELLENT / jitter 0.

---

## What the data proves (analyzed across 6 deploy rosbags, `deploy_analysis/`)
1. **Contact-correlated, not simply elapsed-time periodic.** Drops happen at different elapsed times,
   during steep force ramps into roughly 10-12 N. The bags establish correlation, not whether Sunrise
   motion/safety or an FRI reply deadline initiates the transition.
2. **The 110 ms gap is shutdown, not root cause.** The FRI system interface blocks its controller-manager
   read cycle in a fixed 100 ms close poll after detecting the state transition. The 200 Hz command topic
   is produced by another process upstream of the real FRI UDP loop, so its continuity does not prove the
   client replied on time or that the robot stopped sending.
3. **Moderate recorded force.** Physical |F| never exceeds roughly 12.5 N. The old open-loop joint
   state does not support a trustworthy physical tracking/velocity conclusion.
4. **`ReceiveMultiplier = 1`** (from the pendant FRI config, seen at connect: `SendPeriod 10ms |
   ReceiveMultiplier 1`) requires a reply every 10 ms cycle. Because 110 ms is a shutdown artifact,
   multiplier 3 may provide useful tolerance for an actual one- or two-cycle client delay; packet capture
   is required before assigning a missed-cycle count.
5. **Kernel is generic (PREEMPT_DYNAMIC, not PREEMPT_RT).** No long whole-PC stall was logged, but the
   separate ROS command topic cannot rule out a short FRI-thread, UDP, or NIC deadline failure.

---

## What we TRIED (all on the PC side) — and the result
| Attempt | Result |
|---|---|
| DDS isolation to loopback (`deploy_dds_no_fri_nic.xml`) + ROS_DOMAIN_ID=42 | Fixed unrelated cross-student TF issue; did NOT fix the drop |
| Killed a stray 134%-CPU Isaac process; load 6.3→0.16 | Cleaned baseline jitter; did NOT fix the drop |
| RT priority: `ros2_control_node` → SCHED_FIFO 80 (`scripts/run_hardware_right_rt.sh`) | Free-space jitter → 0 / EXCELLENT; still drops at contact |
| CPU core pinning: FRI node → cores 0-3, vision → 8-31 | joint_states std 0.1 ms; still drops at contact |
| Shed CPU: dropped foxglove + pipeline viz, camera-only | Lower load; still drops at contact |
| **FTEstimator `update_rate` 100 → 30 Hz** (`lbr_two/one_system_config.yaml`) | CONFIRMED live (26 Hz measured); **STILL drops at contact, identical** |
| **FTEstimator force/torque thresholds → 0** (parity, sim has no deadband) | No effect on the drop (kept for force-obs parity, see below) |

**Conclusion: lowering the async FTEstimator rate is not a sufficient fix.** The existing evidence does
not identify which side of the real FRI exchange initiates the transition.

---

## Pendant tolerance experiment (still useful, not a confirmed root fix)
### ReceiveMultiplier 1 → 3 or higher on the Sunrise pendant
In the Sunrise project, `LBRServer_select.java` (or the FRI app config):
```java
friConfiguration.setSendPeriodMilliSec(10);
friConfiguration.setReceiveMultiplier(1);   // -> change to 3
```
Then **Synchronize** to the controller (Sunrise Workbench, Ubuntu-24.04 / Windows laptop — NOT the
deploy PC). This changes the expected reply cadence from 10 ms to 30 ms. The observed 110 ms is a
post-drop shutdown delay, not a measured pre-drop packet outage, so it no longer argues that multiplier
3 is too small. Record packets and the runtime value, test deliberately, and do not treat this setting
as proof of the cause. It does not change speed, limits, or control mode. Full note:
`FRI_RECEIVE_MULTIPLIER_FIX.md`.

*Blocker:* needs Sunrise Workbench access (whoever set up the FRI app). The deploy engineer does not
have easy access to change it.

### Secondary / long-shots (if pendant change is impossible)
- **Lower `e2e_pos_action_scale` 0.005 → 0.003** (deploy config): smaller per-tick motion → gentler
  contact → smaller force step. Untested, low confidence (reduces but may not eliminate the transient).
  NOTE: cannot go too low or the policy can't seat.
- **PC-side force estimation** (avoid the hardware FTEstimator entirely): the deploy node already has
  `wrench_from_external_torque` = pinv(J^T)·τ_ext (`src/rl_deploy_inference/rl_deploy_inference/ik.py`).
  Set `force_source: external_torque` → reads the FRI `external_torque` state interface, computes the
  wrench PC-side. Does NOT touch the FRI thread. (Won't help if the drop is the robot's contact
  reaction rather than the estimator — but removes one variable.) Validate sign/scale on a press-test.
- **Backport jazzy synchronous `WrenchEstimator`** — INVESTIGATED, NOT viable as a clean port: Humble's
  `ChainableControllerInterface` (even ros2_control 2.54) lacks `on_export_state_interfaces`; and our
  deploy reads a `/wrench` TOPIC not a chained interface. Skip.

---

## Cross-check completed against Julien's surviving deployment
Another student (Julien Casalini, repo `assembly_cell_ws`, private — Moreno-Nautilus has access) runs
FORCE-based insertion on the SAME robot **without** these FRI drops. Commit
`30b2083dbe10b5572346e074bbf7b481a6bac6ef` uses the standard Humble v2.2.3 LBR stack and FRI POSITION
client command mode. Its Sunrise Java logs `getReceiveMultiplier()` but never calls
`setReceiveMultiplier(...)`, so the repository does not establish his live value.

The material differences are in deployment behavior: Julien keeps `joint_trajectory_controller` active,
publishes short q/dq/ddq horizons from a stateful quintic smoother, caps policy Cartesian change to 1 mm
per tick, checks a 3 mm commissioning envelope, checks raw baseline-subtracted force, and continuously
requires healthy FRI session/quality/safety/drive/command/overlay/control-mode fields. His diagnostics record
successful contact while FRI remained COMMANDING_ACTIVE. Our previous procedure switched away from JTC to
the direct LBR command controller and used a linear 15-to-200 Hz position bridge.

---

## Config state left in the repos (for the next person)
- `~/kuka_fri_omar_ws/.../ros2_control/lbr_two_system_config.yaml` and `lbr_one_system_config.yaml`
  (symlink install — live, no rebuild):
  - `estimated_ft_sensor.update_rate: 30` (was 100) — ineffective for the drop; harmless; still feeds
    the 15 Hz policy fine. Revert to 100 if desired.
  - `estimated_ft_sensor.force_*_th: 0` / `torque_*_th: 0` (was 2.0/0.5) — **keep**: this is a real
    sim2real PARITY fix (sim feeds RAW baseline-subtracted contact force with NO deadband;
    `insertion_env_e2e_iiwa.py:314`). The 2 N deadband was blinding the policy to the first 2 N of
    contact and under-reporting above.
- `~/Masterthesis-rl-deploy/scripts/run_hardware_right_rt.sh` — RT + CPU-pinning bringup wrapper (keep;
  it cleaned free-space jitter even though it didn't fix the contact drop).
- Deploy config `src/rl_deploy_inference/config/deploy_pdz_v3_fulltilt_single.yaml`:
  `force_source: wrench_topic`, `ft_frame: flange`, `ft_baseline_on_start: true`, `ft_bias_base_xyz 0`.

---

## Force-obs facts (verified, so no one re-investigates)
- Raw KUKA FTEstimator wrench reads **+5 N on Z while HOVERING (no contact)** — a pose-dependent gravity
  residual in the KUKA's own external-torque estimate (a wrong Sunrise tool/payload model would be the
  true-fix, pendant-side). It "leaks" as the wrist tilts.
- BUT `ft_baseline_on_start` captures + subtracts it at start, so the **POLICY sees ~0 force before
  contact in every bag** (verified). No gravity-comp recalibration is needed for the policy input for
  now; the tilt-leak is a small second-order effect.

---

## System facts
- ROS 2 **Humble** on Ubuntu 22.04.5; lbr_fri_ros2_stack **humble branch, v2.2.3** (has the old async
  `FTEstimator`, NOT the jazzy `WrenchEstimator`). ros2_control 2.54.0.
- Kernel 6.8.0 generic (PREEMPT_DYNAMIC, not RT).
- FRI: right arm port 30201, subnet 192.170.20.x (PC `192.170.20.1/24` on `enp3s0`).

---

## Ruled out — do NOT re-chase
- **Joint ordering** — FINE. `/joint_states` reports a scrambled order `[A1,A3,A5,A2,A6,A4,A7]` but the
  deploy node maps BY NAME and command goes out A1..A7. Do not infer physical tracking error from these
  bags: position mode used `open_loop: true`, which substitutes the filtered command for measured joint
  position in the published state interface.
- **Memory leak** — NO (56 Gi free, swap ~0; drops happen in seconds).
- **Whole-PC load** — improved with SCHED_FIFO 80, pinning, and removal of the CPU hog. This does not
  prove the actual FRI UDP thread replied on every 10 ms cycle.
- **FTEstimator async contention** — DISPROVEN by the 30 Hz test.

---

## Recommended next actions (in order)
1. Deploy through the new guarded JTC bridge with 1 mm applied steps, 0.5 degree rotation steps,
   raw-force 15 N abort, 6 N downward attenuation, and fail-closed FRI checks.
2. Record `/lbr_dual_arm_y_gripper/state`, `/rosout`, the JTC trajectory topic, joint states, raw wrench,
   and policy targets, plus a packet capture of UDP port 30201. Confirm the actual Sunrise `control_mode`
   before setting `expected_control_mode` to 0 or 1.
3. Ask Julien for his live Sunrise startup line containing control mode, SendPeriod, and ReceiveMultiplier.
   Test multiplier 3 separately if packet timing indicates a late client reply.

Once the FRI holds through contact, the policy can finally be judged. Prior partial-run behavior
(from bags, before the drop): descends ~16–25 mm toward the socket, closes goal distance ~40→20 mm,
but never centers XY (< ~3.4 mm; hole clearance ~1 mm, no chamfer) and jams on the base top surface —
a separate alignment/hand-eye question addressed in `deploy_analysis/DIAGNOSIS.md`, moot until the
drop is fixed.
`````

## 7.7 `FRI_RECEIVE_MULTIPLIER_FIX.md`

`````markdown
# FRI ReceiveMultiplier fix — hand this to whoever has Sunrise Workbench

## Why
The deploy PC runs a generic (non-PREEMPT_RT) Linux kernel. Even with the ROS control node at
SCHED_FIFO priority 80 and pinned to dedicated CPU cores, the kernel produces a rare, isolated
~110 ms latency spike roughly once every 30-60 s (kernel housekeeping / memory / IRQ — unavoidable
without an RT kernel). With the FRI `ReceiveMultiplier = 1`, the robot requires a fresh command
EVERY 10 ms cycle, so a single 110 ms spike (~11 missed cycles) instantly drops the session:

    LBR switched from 'COMMANDING_ACTIVE' to 'MONITORING_READY'
    ERROR_FRI_CMD_WRONG_STATE_ACTIVE

Verified from deploy rosbags: physical forces are tiny (<=10 N), the 200 Hz command stream never
stalls, and the drop always coincides with a single ~110 ms freeze of joint_states (the FRI read),
not force/torque. So the fix is to let the FRI TOLERATE a late command.

## The change (one line)
In the Sunrise project, open the FRI application `LBRServer_select.java`. Find the FRI configuration:

    FRIConfiguration friConfiguration =
        FRIConfiguration.createRemoteConfiguration(lbr, clientName);
    friConfiguration.setSendPeriodMilliSec(10);
    friConfiguration.setReceiveMultiplier(1);     // <-- change this

Change to:

    friConfiguration.setReceiveMultiplier(3);     // tolerate late commands (30 ms window)

Then **Synchronize** the project to the controller (Workbench -> Synchronize).

## Notes
- SendPeriod stays 10 ms; the robot still runs its internal loop at 10 ms. ReceiveMultiplier only
  relaxes how often the external ROS client must reply: with 3, a command is required every 30 ms.
- 3 covers the common small jitter. The rare 110 ms spike is still > 30 ms, so if drops persist,
  bump to 5 (50 ms) or higher. Trade-off: a larger multiplier means the arm gets a fresh command
  less often, i.e. coarser control. Start at 3, raise only if needed.
- This is safe: it does not change robot speed, limits, or the control mode (still POSITION).
- The real robust fix would be a PREEMPT_RT kernel on the deploy PC (eliminates the spike entirely),
  but ReceiveMultiplier is the small, low-risk change that unblocks deploy runs now.

## After the change
Deploy runs should hold COMMANDING_ACTIVE for the full trial, so the policy can be judged/tuned
(current status: in ~10 s it descended ~16 mm and closed goal distance 32 -> 21 mm but did not seat
before the FRI dropped — need uninterrupted runs to tune convergence).
`````

## 7.8 `CONTACT_SAFE_DEPLOYMENT_RATIONALE.md`

`````markdown
# Contact-Safe Deployment Changes and Test Plan

Date: 2026-08-28  
Branch: `moreno`  
Reference implementation: [Julien Casalini's `assembly_cell_ws`](https://github.com/JulienCasalini/assembly_cell_ws), inspected at commit `30b2083dbe10b5572346e074bbf7b481a6bac6ef`.

## Short conclusion

The guarded deployment changes are justified by both Julien's surviving contact implementation and our raw rosbags. The old deployment asked for large Cartesian changes, converted them to discontinuous 15 Hz joint targets, and linearly streamed positions directly at 200 Hz. Five of the six available raw bags contain the same approximately 110 ms robot-state and wrench interruption at contact, while the separate PC upsampler continues publishing 22 direct joint-command messages during the interruption.

**Correction after tracing Humble v2.2.3:** the 110 ms gap is almost certainly a shutdown artifact *after* FRI has already left `COMMANDING_ACTIVE`, not the cause of the FRI drop. `SystemInterface::read()` sees the session transition and synchronously calls `App::close_udp_socket()`. That function waits for the FRI worker using a fixed 100 ms polling sleep. One normal 10 ms update plus this 100 ms wait explains the repeatable 110 ms interval. The bags therefore locate the drop at contact, but do not show whether it was initiated by a missed PC reply, missing controller packet, Sunrise motion/safety transition, or another controller-side condition.

The regular 200 Hz ROS command topic does **not** prove that the actual FRI UDP loop remained healthy: it is produced by a separate upsampler process upstream of `ros2_control` and the FRI client. It does show that our old path was aggressive and unsupervised compared with Julien's, and that changing this path is still the strongest low-risk motion experiment before modifying the Sunrise application or FRI stack.

Confidence is therefore **moderate, not high**. I expect the new path to reduce the contact transient and to stop safely when state or tracking is unhealthy. I cannot promise that Sunrise will keep FRI active through contact, especially if the remaining material difference is native position control versus Cartesian impedance control.

## What the raw bags add

I read the SQLite rosbag storage directly rather than relying only on the extracted CSV plots. The six available raw bags are under `~/deploy_bags`.

| Run | Policy observations | Largest joint-state gap | Force around gap | Direct commands inside gap | Policy translation request, median / max | Policy rotation request, max |
|---|---:|---:|---:|---:|---:|---:|
| `20260827_165249` | 150 | 110.157 ms | 9.118 -> 10.029 N | 22 | 6.993 / 7.259 mm | 2.740 deg |
| `20260827_171512` | 150 | 110.233 ms | 11.059 -> 12.094 N | 22 | 5.451 / 5.611 mm | 6.681 deg |
| `20260828_134142` | 7 | 16.013 ms max, no drop signature | 6.251 N peak | 0 in a large gap | 1.043 / 1.433 mm | 1.080 deg |
| `20260828_134445` | 150 | 110.069 ms | 10.175 -> 10.175 N | 22 | 4.768 / 6.413 mm | 6.821 deg |
| `20260828_135128` | 150 | 110.127 ms | 10.775 -> 12.054 N | 22 | 7.002 / 7.191 mm | 2.418 deg |
| `20260828_140335` | 150 | 110.234 ms | 12.465 -> 12.465 N | 22 | 7.089 / 7.223 mm | 2.506 deg |

The translation and rotation values are reconstructed from the previous-action fields in `/rl_deploy/policy_obs`, after the old EMA and before IK. They describe requested target changes, not measured TCP travel.

### Findings beyond the previous CSV analysis

1. The repeatable 110 ms interval matches the stack's post-transition shutdown path: 100 ms polling sleep plus the normal approximately 10 ms controller update. Treat it as evidence that the session transition occurred, not as the network outage duration or its cause.
2. The contact force is a steep ramp into the transition, not literally an instantaneous 0-to-10 N discontinuity. For example, `165249` rises from 2.08 N to 9.12 N over about 110 ms and later records 10.03 N. In two of the 30 Hz runs, the recorded force peak occurs about 30 ms before the state gap.
3. `e2e_pos_action_scale: 0.005` limits each axis to 5 mm, not the 3D translation norm to 5 mm. A multi-axis action can request up to `sqrt(3) * 5 mm = 8.66 mm`. The failed runs actually request as much as 7.26 mm per 15 Hz tick, and 97-98% of their active requests exceed 1 mm.
4. Rotation requests are also substantial: maxima are 2.4-6.8 degrees per tick in the failed runs. The new 0.5 degree norm limit is therefore meaningful, not cosmetic.
5. Peak velocity inferred from the old direct output is 4.75-7.79 deg/s in the failed runs. Julien's guarded path uses a 3 deg/s joint limit, with 30 deg/s2 acceleration and 300 deg/s3 jerk limits.
6. The old upsampler keeps publishing after the FRI/ros2_control path is gone. In `165249`, the bag continues for 212.7 seconds after the final joint state and contains another **42,543** direct position commands. This cannot restore the FRI session and does not demonstrate actual UDP command delivery.
7. None of the old bags records `/lbr_dual_arm_y_gripper/state`, `/rosout`, or FRI UDP packets. Consequently, the bags cannot establish which side missed the deadline or the exact state/quality/safety ordering.
8. A 15 N force cap alone would not have prevented these recorded events: the transitions occur around 9-12.5 N. The more relevant protections are smaller applied steps, force-dependent downward attenuation, a smoother JTC trajectory, tracking/state checks, and possibly Cartesian impedance on Sunrise.

The short non-drop run `134142` is suggestive but not a control experiment. It contains only seven policy observations, requested much smaller translations, reached only 6.25 N, and then stopped because RGB became stale.

## Why each change was made

### 1. Keep `joint_trajectory_controller` active

The old procedure deactivated the standard JTC, activated `LBRJointPositionCommandController`, and ran `command_upsampler.py --interpolate` to publish direct positions at 200 Hz.

The new procedure leaves JTC active and starts:

```bash
ros2 run rl_deploy_inference guarded_joint_trajectory_bridge
```

The bridge converts each 15 Hz policy target into a short `JointTrajectory` containing position, velocity, and acceleration samples. This matches the architecture used by Julien and lets ros2_control own the trajectory handoff instead of sending piecewise-linear positions directly to FRI.

### 2. Make trajectory generation stateful and derivative-bounded

The bridge carries its planned `q`, `dq`, and `ddq` state from one policy tick into the next and uses quintic trajectories. Its current limits are:

| Guard | Value |
|---|---:|
| Joint velocity | 3 deg/s |
| Joint acceleration | 30 deg/s2 |
| Joint jerk | 300 deg/s3 |
| Maximum target lead from measured joints | 0.5 deg |
| Maximum published-state-to-planned tracking error | 0.35 deg |
| Joint-state maximum age | 50 ms |

The 0.35 degree threshold is close to the largest old direct-command lead, 0.349 degrees. However, the current position-control hardware profile uses `open_loop: true`; in that mode the LBR stack substitutes its filtered command for measured joint position in the published state interface. Therefore neither the old bag metric nor this bridge check proves physical tracking. Cartesian impedance automatically forces the stack to closed-loop state reporting. If position control is retained, test `open_loop: false` deliberately in T1 before relying on the tracking guard.

### 3. Bound the physical Cartesian step by vector norm

The actor still runs with its trained scales, but the applied action and `prev_action` are rescaled to at most:

```yaml
max_policy_position_step_m: 0.001
max_policy_rotation_step_rad: 0.00872665  # 0.5 deg
```

This preserves direction while limiting the total translation/rotation norm. It directly addresses the 5.6-7.3 mm and 2.4-6.8 degree requests visible in the failed bags.

### 4. Require healthy, fresh FRI state

Both inference nodes can now refuse motion unless fresh `LBRState` reports:

| Field | Required value |
|---|---|
| `session_state` | 4, `COMMANDING_ACTIVE` |
| `connection_quality` | 3, `EXCELLENT` |
| `safety_state` | 0, `NORMAL_OPERATION` |
| `drive_state` | 2, `ACTIVE` |
| `client_command_mode` | 1, `POSITION` |
| `overlay_type` | 1, `JOINT` |
| `control_mode` | The configured expected native mode |

This is a fail-closed guard and a diagnostic improvement. It cannot stop Sunrise from dropping FRI, but it prevents the policy from continuing as though the session were healthy.

### 5. Use raw contact force for safety

The policy observation may still use EMA-smoothed force, but the force warning and abort evaluate the current baseline-subtracted force. Safety should not wait for observation smoothing during a fast contact ramp.

Current commissioning settings attenuate only downward target motion as force approaches 6 N and latch a force abort at 15 N:

```yaml
z_force_limit_enable: true
z_force_limit_n: 6.0
ft_warn_n: 8.0
ft_force_cap_n: 15.0
```

Because the historical gap occurs below 15 N, use 10 N as the first commissioning cap if the objective is diagnosis rather than completing an insertion. Julien's launch also defaults to a 10 N baseline-relative abort. The 6 N attenuation is expected to matter more than either hard cap because it acts before the old drop range.

### 6. Record the missing evidence

The runbook now records `LBRState`, `/rosout`, and the JTC trajectory topic, and it calls for a simultaneous packet capture on UDP port 30201. `LBRState` may still miss the final bad state because the hardware `read()` returns an error immediately after detecting the transition. The FRI `onStateChange` log and packet direction/timing are therefore essential.

## Important remaining Julien difference

Julien uses the same standard Humble-era LBR/FRI stack, but his real-insertion launch expects native Sunrise `CARTESIAN_IMPEDANCE_CONTROL` (`control_mode=1`) for contact. His documented Sunrise server uses compliant lateral stiffness and damping for peg-in-hole work. Our current local fulltilt deployment profiles expect native position control (`control_mode=0`).

That may be as important as the ROS command bridge. The ROS parameter `expected_control_mode` only verifies the live mode; changing it does **not** switch Sunrise into impedance control. The mode must be selected/configured in the Sunrise application, then confirmed from the live `LBRState`. Before drawing a conclusion from the guarded test, record Julien's actual startup line and our own, including control mode, send period, and receive multiplier.

Do not copy impedance stiffness or tool-load settings blindly. They are robot/tool-specific and belong to Sunrise commissioning.

## Why FRI probably shuts down at contact

The ROS system interface is not initiating the failure. It first observes that Sunrise/FRI has already left `COMMANDING_ACTIVE`, then deliberately shuts down its local FRI worker. There are two credible ways the earlier state transition can happen.

### Leading hypothesis: rigid contact plus open-loop command progression

The current right-arm hardware profile combines:

```yaml
client_command_mode: position
open_loop: true
```

In this mode, `AsyncClient::command()` calls `set_state_open_loop()`, which replaces the state interface's measured joint position with the filtered commanded joint position. This has several consequences:

1. `/joint_states` is not reliable evidence of physical tracking during active position command mode.
2. The old deployment computes FK and its next target from a virtual robot that appears to follow perfectly.
3. The local LBR command guard also sees that substituted state, so it cannot detect the true command-to-robot separation caused by a blocked tool.
4. Sunrise still sees the actual encoders, external torque, and internal motion/safety state.

At contact, the real robot can stop or deflect while the command path continues advancing. With old requested changes of 5.6-7.3 mm and up to 6.8 degrees per policy tick, this can create a rapid physical following/load error in stiff native position control. Sunrise may then leave the commanding phase because of its internal motion/safety handling. The ROS stack observes that transition and produces the 110 ms shutdown signature.

This mechanism explains why the event is repeatably contact-correlated even though ordinary PC load, inference, and force-estimator work also exist in free space. It also explains why Julien's path is materially different: Cartesian impedance absorbs some contact displacement and the LBR client automatically overrides `open_loop` to false in impedance control.

This remains a hypothesis until the final pre-transition `LBRState`, Sunrise message, and packet timing are captured. The exact internal Sunrise condition might be following error, a safety/load condition, or the FRI motion application reacting to an unexpected state.

### Competing hypothesis: FRI reply deadline

With a 10 ms send period and `ReceiveMultiplier=1`, the KUKA client SDK sends a reply for every monitoring packet. A late/missing PC reply, missing controller packet, or packet loss can make Sunrise degrade connection quality and leave `COMMANDING_ACTIVE`. The pendant's `Quality POOR` and jitter report support this possibility.

The old 200 Hz ROS command topic cannot test this hypothesis because it is produced by a separate process before `ros2_control` and `ClientApplication::step()`. Likewise, the 110 ms shutdown delay is not the missing-packet duration. A packet capture is needed to determine which direction becomes late first.

Pure contact-triggered PC computation is less convincing than before. The old asynchronous force estimator computes its Jacobian inverse continuously, not only when force crosses a threshold, and reducing its rate to 30 Hz did not prevent the transition. That does not completely exclude a client-side deadline miss, but it removes the proposed contact-only estimator branch.

## What changes for deployment

1. Build and source this branch before starting the deployment node.
2. Leave `joint_trajectory_controller` active. Do not activate `lbr_joint_position_command_controller` for the policy run.
3. Run exactly one bridge: `guarded_joint_trajectory_bridge`. Do not run the legacy `command_upsampler` at the same time.
4. Echo `/lbr_dual_arm_y_gripper/state` before enabling motion. Set `expected_control_mode` to the value intentionally selected on Sunrise, not the value that merely makes the check pass.
5. Start the rosbag before policy motion and include `LBRState`, JTC trajectory, policy target, measured joints, raw wrench, and status.
6. After any bridge latch, inspect the reason before calling `/rl_deploy/clear_bridge_latch`.
7. Keep an operator at the enabling device/e-stop and begin below the old contact-force range.

The exact terminal sequence is maintained in `PREDEPLOY_pdz_v3_fulltilt_SINGLE.md`.

### Configuration portability warning

The edited local profiles

```text
src/rl_deploy_inference/config/deploy_pdz_v3_fulltilt.yaml
src/rl_deploy_inference/config/deploy_pdz_v3_fulltilt_single.yaml
```

are currently ignored by the repository's broad `*.yaml` rule. They contain the guard values in this document on this machine, and the installed symlink workspace sees them, but those edits will not automatically travel with a normal commit/clone until the profiles are deliberately force-tracked or the ignore rule is narrowed. Verify the live parameters with `ros2 param get` during commissioning.

## Does training need to change?

**Not for the next FRI-survival test.** The present checkpoint is suitable for testing whether safer deployment survives contact. Retraining now would change two variables at once and make the result harder to interpret.

The 1 mm/0.5 degree guards do materially alter almost every active action in the failed runs, so the current checkpoint may become slower or fail to finish the task within its old timeout. That is acceptable for the first experiment: the endpoint is "FRI stayed healthy through repeated controlled contact," not insertion success.

For the final policy, retrain or fine-tune with the deployment actuator model after the FRI issue is isolated. Training should include the same vector-norm action limits, 15 Hz action hold, bounded joint/Cartesian response, force-dependent downward attenuation, realistic compliance/control mode, and the same force observation preprocessing used on hardware. Then success rate can be judged without a train/deploy dynamics mismatch.

## How to debug this on Monday

The objective is to identify the event that occurs **before** the ROS stack's 100 ms shutdown wait. Do not use the 110 ms publication gap itself as the cause.

### 1. Record the exact runtime configuration

Before moving, save the Sunrise startup line and check the live state:

```bash
ros2 topic echo /lbr_dual_arm_y_gripper/state --once
ros2 control list_controllers -c /lbr_dual_arm_y_gripper/controller_manager
```

Record these values in the trial note:

- Sunrise native control mode: position or Cartesian impedance.
- FRI client command mode and overlay type.
- Send period and receive multiplier.
- Selected tool/load model and impedance stiffness/damping, if applicable.
- `open_loop` value from `lbr_two_system_config.yaml`.
- Active controller and the publishers on its command topic.

Do not change only `expected_control_mode` to make the ROS check pass. It is a verifier, not a Sunrise mode switch.

### 2. Start packet and ROS evidence before contact

In the rosbag terminal, use the command in `PREDEPLOY_pdz_v3_fulltilt_SINGLE.md`; it now includes `/rosout`, `LBRState`, the JTC trajectory, policy command, joint state, and wrench.

In a separate terminal:

```bash
mkdir -p ~/deploy_bags
sudo tcpdump -i enp3s0 -s 0 -B 4096 \
  -w ~/deploy_bags/fri_$(date +%Y%m%d_%H%M%S).pcap 'udp port 30201'
```

Also save NIC counters before and after the trial:

```bash
ip -s link show enp3s0
sudo ethtool -S enp3s0
```

Capture the complete terminal output from hardware bringup. The useful line is the first FRI state/quality/safety message, not only the later generic `LBR left COMMANDING_ACTIVE` error.

### 3. Baseline with no contact

Run JTC plus the guarded bridge for at least 30 seconds with policy motion disabled. Verify:

- FRI remains `COMMANDING_ACTIVE` and `EXCELLENT`.
- Controller-to-PC monitoring packets arrive at approximately 10 ms intervals.
- PC replies follow the configured receive multiplier.
- No bridge latch or unexpected controller switch occurs.

This establishes the packet timing and thread behavior immediately before adding the contact variable.

### 4. Contact while policy motion is disabled

Using the lab-approved controlled-force procedure, apply repeatable external loads while JTC holds the arm and the policy cannot advance. Start around 2 N and step through approximately 4 N and 6 N. Stop before the historical 9-12.5 N transition range on the first attempt.

- If FRI drops here, the learned action and IK are not required for the failure. Focus on Sunrise mode, open-loop behavior, tool/load configuration, safety state, and FRI packet timing.
- If FRI survives, commanded approach and command-to-contact interaction become much more likely.

### 5. Conservative policy-contact ladder

For the first moving test, set:

```bash
ros2 param set /rl_deploy_inference max_policy_position_step_m 0.00025
ros2 param set /rl_deploy_inference max_policy_rotation_step_rad 0.004363323
ros2 param set /rl_deploy_inference z_force_limit_n 4.0
ros2 param set /rl_deploy_inference ft_force_cap_n 10.0
ros2 param set /rl_deploy_inference trial_timeout_s 3.0
```

Run at least five controlled contacts at 0.25 mm. If all survive, repeat at 0.5 mm and then 1 mm while keeping every other setting unchanged. Inspect any bridge latch before clearing it.

### 6. Classify the packet trace

Use the rosbag/`rosout` transition time to inspect the same instant in Wireshark or:

```bash
tcpdump -tttt -nn -r <capture.pcap> 'udp port 30201'
```

Interpret the first abnormal event:

| First abnormal evidence | Most likely area |
|---|---|
| Controller monitoring packets continue, but the PC reply is late/missing | FRI client thread, scheduling, command callback, or PC transmit path |
| Controller monitoring packets stop first | Sunrise controller, controller transmit path, physical link, or controller-side session handling |
| Both directions remain timely, then `onStateChange` reports monitoring/safety change | Sunrise motion, control mode, tool/load, or safety reaction rather than network timing |
| Packet loss/errors appear in NIC counters | NIC, cable, driver, interrupt/coalescing, or network configuration |
| Only the ROS command topic remains regular | No conclusion about FRI; this is expected from the separate upsampler/bridge process |

### 7. Run one-variable comparisons

After the guarded position-control baseline, compare separately:

1. `ReceiveMultiplier=1` versus `3`, keeping motion and Sunrise control mode unchanged.
2. Native position control versus the correctly commissioned Cartesian impedance mode, keeping policy limits unchanged.
3. If position control must remain, `open_loop: true` versus `false` in T1 with conservative motion. This requires hardware restart/reconfiguration and can change controller behavior; do not treat it as a runtime observation parameter.

Cartesian impedance is the most relevant parity test with Julien. Confirm its actual stiffness, damping, tool load, and `control_mode=1` from Sunrise/`LBRState`; do not copy values blindly.

### 8. Evidence required for a conclusion

For each attempt retain:

- Rosbag, pcap, full bringup terminal log, and pendant/Sunrise first error.
- Exact parameters and control mode.
- First contact time, force rise, and peak force.
- First bad FRI packet interval/direction.
- First FRI state, quality, safety, or drive-state transition.
- Bridge latch reason and JTC command around contact.

Do not modify the 100 ms socket-close polling sleep as a proposed fix. Shortening it would only make the visible ROS publication gap smaller after FRI had already failed.

## Next evidence: staged test

### Stage A: no-motion instrumentation check

1. Start hardware, JTC, guarded bridge, inference node, and rosbag.
2. Keep policy motion disabled or set `max_joint_step_rad` to zero for the observation check.
3. Confirm fresh joints/wrench and continuously healthy `LBRState` for at least 30 seconds.
4. Confirm there is one JTC trajectory publisher and no direct-position upsampler.
5. Capture both directions of FRI UDP traffic on port 30201.

### Stage B: contact without policy advance

With the robot holding through JTC and under the lab's normal contact-safety procedure, introduce a small controlled external load without policy motion. This separates "contact alone breaks the Sunrise/FRI session" from "our commanded approach creates the failure."

Pass evidence: repeated force ramps through approximately 4-6 N with no state gap above 30 ms, no FRI quality/state transition, and no bridge latch.

### Stage C: very small policy contact

Use a 0.25 mm position cap, 0.25 degree rotation cap, a 4 N downward attenuation point, a 10 N raw-force abort, and a short trial timeout. Run at least five controlled contacts. Then repeat at 0.5 mm and finally at the current 1 mm cap only if the earlier stages remain healthy.

Record for every attempt:

- First-contact force and force rise rate.
- Maximum joint-state and wrench inter-arrival gap.
- FRI session, quality, control mode, overlay, and drive-state transitions.
- Bridge target lead, tracking error, and latch reason.
- JTC trajectory position/velocity/acceleration continuity.
- Whether the arm retracted/held normally after the contact.
- First missing or late UDP direction: controller-to-PC monitoring packet or PC-to-controller reply.

### Stage D: control-mode comparison

If position-control contact still reproduces the 110 ms interruption, repeat the same conservative test with the correctly commissioned Sunrise Cartesian impedance mode used for contact work. Keep the ROS trajectory and action limits identical so control mode is the only material variable.

### Decision rule

- If Stage B drops FRI with no policy motion, command shaping is not sufficient; focus on Sunrise control mode, FRI application behavior, tool/load configuration, receive multiplier, and network/session logs.
- If Stage B survives but Stage C drops, the commanded approach or force ramp is implicated; keep the guarded path and lower motion/force limits based on the trajectory trace.
- If position mode drops but impedance mode survives, native contact compliance is the decisive difference.
- If all guarded stages survive, only then extend the timeout and evaluate policy insertion performance.

The next step is therefore testing, but in controlled stages. One successful insertion is encouraging; repeated contact with synchronized `LBRState`, `/rosout`, force, measured-joint, trajectory, and packet-timing evidence is what will tell us whether the deployment fix actually solved the FRI failure mode.
`````

## 7.9 `PREDEPLOY_pdz_v3_fulltilt.md`

`````markdown
# Pre-deploy checklist — pdz_v3_fulltilt (2026-08-27)

Deploying the `pdz_v3_fulltilt_20260826` run (ram-fix + corrected gravity comp + 30mm geometry,
full 25° tilt curriculum). Config already prepped: `config/deploy_pdz_v3_fulltilt.yaml`.

Work top-to-bottom. **Do NOT skip the no-contact force check (step 4)** — it's the one that catches
the gravity-comp mismatch that burned the last deploy.

---

## 0. Confirm training finished cleanly (train box)
- [ ] Run is done / at target: `grep "MAX EPOCHS\|epoch: 2000" /tmp/pdz_v3_fulltilt_20260826.log`
- [ ] Final checkpoint exists: `ls -la logs/rl_games/Forge/pdz_v3_fulltilt_20260826/nn/Forge.pth`
- [ ] No NaN/obs-guard in the log:
      `grep -iE "obs-guard|traceback|[^a-z]nan[^a-z]" /tmp/pdz_v3_fulltilt_20260826.log | grep -iv futurewarning`
- [ ] Sanity numbers (want success high @ FULL tilt, force low):
      `~/miniconda3/envs/isaaclab51/bin/python scripts/plot_e2e_progress.py --full --runs pdz_v3_fulltilt_20260826 --out diagnostics/plots/predeploy_check.png`
      → success last-50 should be strong; contact_force mean well under the 12N penalty threshold.
- [ ] (optional but recommended) render 5 full-tilt clips, eyeball the descent/bounce:
      see the render command in the run memory note.

## 1. Build the deploy package (deploy box)
- [ ] `cd ~/Masterthesis-rl-deploy`
- [ ] `colcon build --packages-select rl_deploy_inference`   ← REQUIRED: installs the new yaml
- [ ] `source install/setup.bash`
- [ ] Confirm the yaml shipped:
      `ls install/rl_deploy_inference/share/rl_deploy_inference/config/deploy_pdz_v3_fulltilt.yaml`

## 2. Robot + perception up
- [ ] Arm driver / FRI up, robot in the pre-insert home, E-stop in reach.
- [ ] RealSense streaming: `ros2 topic hz /realsense_1/camera/color/image_raw` (expect ~15-30 Hz)
- [ ] Hole aligner ready to publish the corrected anchor (see hole_align / --republish-anchor).

## 3. Launch inference (GATED — no motion yet)
- [ ] `ros2 launch rl_deploy_inference deploy_inference_pdz.launch.py \`
      `    params_file:=$(ros2 pkg prefix rl_deploy_inference)/share/rl_deploy_inference/config/deploy_pdz_v3_fulltilt.yaml`
- [ ] Node comes up, subscribes, streams the held pose. `policy_active_on_start:=false` so it will NOT move.

## 4. ⚠️ NO-CONTACT FORCE CHECK (the critical gate — do BEFORE any motion)
The new run trained on PURE contact force (corrected gravity comp) and `ft_bias_base_xyz=0`. So at a
free-space hover the policy force channels MUST read ~0.
- [ ] Hover the tool in free space (no screw touching anything), tool pointing down.
- [ ] Read the debug obs (publish_debug_obs:=true is set):
      `ros2 topic echo /rl_deploy/debug_obs`   → force is indices **[6:9]** of the 15-D vector.
- [ ] **PASS = all three force values ≈ 0** (a few tenths of N is fine).
- [ ] **FAIL = a large DC offset (esp. a negative TCP-z ~5N).** If so: STOP. The real gravity comp
      isn't matching sim. Do NOT set ft_bias back to 5.3 (that was the OLD broken-sim workaround).
      Recheck: ft_baseline_on_start captured at a clean no-contact pose? force_source? ft_sign?
- [ ] Also confirm the force SIGN: press the tip gently by hand → TCP-z force should go POSITIVE.
      If negative, flip `ft_sign` (currently -1.0).

## 5. Geometry spot-check (quick, avoids a 12mm-class error)
- [ ] `screw_tip_offset_tcp_xyz = [0,0,0.030]` matches the real tip-to-TCP you measured (~30mm). ✓ nominal.
- [ ] `socket_depth_m = 0.0175` matches the real socket depth.
- [ ] `tip_link` (lbr_two_gripper_tcp) resolves in the real robot_description (FK gives a sane TCP pose).
- [ ] Anchor: the hole-aligner publishes the detected OPENING; the node subtracts socket_depth → bottom.
      One echo of `/rl_deploy/corrected_socket` should sit at the real hole opening.

## 6. First trial (armed)
- [ ] Screw grasped, arm at pre-insert hover (~tip 3-4cm above the hole, near-vertical if possible —
      the policy is strongest near the tilt it saw; extreme tilt is the untested edge).
- [ ] Hand on E-stop.
- [ ] Arm motion: `ros2 service call /rl_deploy/start_policy std_srvs/srv/Trigger {}`
- [ ] WATCH: does it descend gently and seat, or ram? `ft_force_cap_n=25` will abort on a hard jam.
      Expected (from sim): mean contact ~5N, gentle approach. A hard ram = investigate before retrying.
- [ ] Log the outcome + |F|max + whether it seated, exactly like the last 3-trial table.

## If it rams anyway (contingency — matches the sim watch-list)
The sim levers to soften contact, in order of preference:
1. Soften arm PD: `e2e_arm_stiffness_scale < 1.0` (retrain) — makes contact compliant not rigid.
2. Halve action scale again: 0.005 → 0.0025 (retrain) — gentler max command.
Neither is a same-day fix; note it and fall back to reviewing the descent dynamics.

---
Prepped 2026-08-26. Config diff vs ep1500 = checkpoint paths + ft_bias 5.3→0 + action_scale 0.01→0.005.
No deploy-code changes were needed (force pipeline was already pure-contact).
`````

## 7.10 `PREDEPLOY_pdz_v3_fulltilt_SINGLE.md`

`````markdown
# Deploy runbook — pdz_v3_fulltilt SINGLE-ARM (right only, one RealSense) (2026-08-27)

Single-robot / single-camera variant of `PREDEPLOY_pdz_v3_fulltilt.md`. Differences vs the
dual runbook are marked **[SINGLE]**. Everything else is identical.

**What "single" means here**
- **Robot:** right arm only (`lbr_two`, one FRI session, port 30201) via
  `hardware_right.launch.py`. The left arm (`lbr_one`) is NOT brought up.
- **Camera:** the right-wrist D405 only, published as **`realsense_2`** (serial
  `260522275434`) via `mv_launch realsense_single.launch.py`. The static ZED2i scene
  camera and the left-arm `realsense_1` are NOT brought up.
- **Consequence — no FoundationPose pipeline:** `run_pipeline_track_multicam_realsense.py`
  is hardcoded to exactly 3 cameras (`choices=[3]`), so **T4 and T-SCENE below are NOT run**
  in this mode. The socket anchor comes from the **geometric hole-align** path
  (`use_socket_pair: true`, the known 60 mm socket spacing), which does not need perception.
  `/perception/fp/pose_base/fused/assembly` will NOT be published — that's expected; keep
  `fallback_to_perception: false` so a missed detection aborts instead of using a stale/absent
  perception pose.
- **Deploy config:** use the new **`deploy_pdz_v3_fulltilt_single.yaml`** (identical to
  `deploy_pdz_v3_fulltilt.yaml` except the policy RGB-D reads `/realsense_2/...`).

**One-time build (user runs git/builds):**
- `~/zed_ros2_ws` — rebuild `mv_launch` so `realsense_single.launch.py` is installed
  (`colcon build --packages-select mv_launch`).
- `~/Masterthesis-rl-deploy` — rebuild `rl_deploy_inference` so
  `deploy_pdz_v3_fulltilt_single.yaml` is installed
  (`colcon build --packages-select rl_deploy_inference`).
- `launch_host_single.sh` is already executable; no build needed for the vision scripts.

# ══════════════════════════════════════════════════════════════════════
#  SOURCING — paste at the top of EVERY new terminal
# ══════════════════════════════════════════════════════════════════════
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/kuka_fri_omar_ws/install/setup.bash
source ~/Masterthesis-rl-deploy/install/setup.bash

── T1 ROBOT BRINGUP **[SINGLE: right arm only]** ──

sudo ip addr add 192.170.10.1/24 dev enp3s0
sudo ip addr add 192.170.20.1/24 dev enp3s0
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/kuka_fri_omar_ws/install/setup.bash
ros2 launch lbr_dual_arm_y_gripper_bringup hardware_right.launch.py
#   ^ single FRI session (lbr_two, port 30201). Namespace stays /lbr_dual_arm_y_gripper and every
#     lbr_two_* frame is unchanged, so the deploy stack, MoveIt, and topics below need no edits.
#   The 192.170.10.1 alias (left arm / port 30200) is not strictly needed but is harmless to keep.

── T2 MOVEIT (headless) ──

export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/kuka_fri_omar_ws/install/setup.bash
ros2 launch lbr_dual_arm_y_gripper_bringup move_group.launch.py mode:=hardware rviz:=false
#   MoveIt keeps the DUAL robot_description (left arm = zeros placeholder) + a keep-out box at
#   y<-0.20 by design (see memory single-arm-right-bringup). Right-arm planning is unaffected.

── T3 CAMERA **[SINGLE: right-wrist RealSense only, no ZED]** ──

cd ~/Masterthesis-vision
scripts/launch_host_single.sh
#   ^ brings up ONLY /realsense_2/... (right wrist D405) + /right/ee_pose. No ZED, no realsense_1.
#     Override serial: RS2_SERIAL=<serial> scripts/launch_host_single.sh

── T4 VISION PIPELINE **[SINGLE: SKIP]** ──

#   NOT RUN in single-camera mode. The FoundationPose pipeline runner is fixed at 3 cameras.
#   The socket anchor comes from the geometric hole-align below (no perception needed).

── T-SCENE CAMERA → MOVEIT PLANNING SCENE **[SINGLE: SKIP]** ──

#   NOT RUN — depends on the ZED scene camera, which is not brought up here. MoveIt plans without
#   the live scene collision objects. If you need scene collision avoidance, run the dual runbook.

── T5 F/T BROADCASTER (logging only — policy force is torque-estimated) ──

export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/kuka_fri_omar_ws/install/setup.bash
cat > /tmp/ft_broadcaster.yaml <<'EOF'
/**/force_torque_broadcaster:
  ros__parameters:
    frame_id: lbr_two_link_ee
    sensor_name: estimated_ft_sensor
EOF
ros2 run controller_manager spawner force_torque_broadcaster \
  -c /lbr_dual_arm_y_gripper/controller_manager \
  -t force_torque_sensor_broadcaster/ForceTorqueSensorBroadcaster \
  -p /tmp/ft_broadcaster.yaml
ros2 topic hz /lbr_dual_arm_y_gripper/force_torque_broadcaster/wrench

── T6 CHECKS (all must return data) **[SINGLE: fused assembly line dropped]** ──

source /opt/ros/humble/setup.bash
source ~/Masterthesis-vision/install/setup.bash
source ~/Masterthesis-rl-deploy/install/setup.bash
ros2 topic echo /lbr_dual_arm_y_gripper/joint_states --once
ros2 topic echo /realsense_2/camera/color/image_rect --once   # was: perception fused assembly (no pipeline now)
ros2 topic echo /lbr_dual_arm_y_gripper/force_torque_broadcaster/wrench --once
ros2 action list | grep move_action

── T-PRE PREINSERT (right, local_ik_joint) ──

export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/kuka_fri_omar_ws/install/setup.bash
source ~/Masterthesis-rl-deploy/install/setup.bash
# DRY-RUN:
ros2 run rl_deploy_inference preinsert_planner --arm right --ros-args \
  -p orientation_mode:=down -p socket_timeout_s:=30.0 -p target_mode:=local_ik_joint
# EXECUTE (type MOVE):
ros2 run rl_deploy_inference preinsert_planner --arm right --execute --ros-args \
  -p orientation_mode:=down -p socket_timeout_s:=30.0 -p target_mode:=local_ik_joint

── T-HOLE HOLE ALIGN (right, 4.5 cm) **[SINGLE: camera topics -> realsense_2]** ──

# The hole_align.yaml file still names /realsense_1/... for the wrist camera; override the three
# camera topics to realsense_2 on the command line (arm=right already maps camera_id -> realsense_2).
# DRY-RUN (check /tmp/hole_align/holes_*.png):
ros2 run rl_deploy_inference hole_align_planner --arm right --ros-args \
  --params-file ~/Masterthesis-rl-deploy/src/rl_deploy_inference/config/hole_align.yaml \
  -p rgb_topic:=/realsense_2/camera/color/image_rect \
  -p depth_topic:=/realsense_2/camera/aligned_depth_to_color/image_rect \
  -p camera_info_topic:=/realsense_2/camera/color/camera_info \
  -p hover_z_m:=0.045
# EXECUTE (type MOVE; NOTE the printed "DETECTED hole (base_link)" value):
ros2 run rl_deploy_inference hole_align_planner --arm right --execute --ros-args \
  --params-file ~/Masterthesis-rl-deploy/src/rl_deploy_inference/config/hole_align.yaml \
  -p rgb_topic:=/realsense_2/camera/color/image_rect \
  -p depth_topic:=/realsense_2/camera/aligned_depth_to_color/image_rect \
  -p camera_info_topic:=/realsense_2/camera/color/camera_info \
  -p hover_z_m:=0.045

── T-ANCHOR CORRECTED-ANCHOR REPUBLISHER (leave RUNNING) **[SINGLE: camera topics -> realsense_2]** ──

export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/kuka_fri_omar_ws/install/setup.bash
source ~/Masterthesis-rl-deploy/install/setup.bash
ros2 run rl_deploy_inference hole_align_planner --arm right --republish-anchor --ros-args \
  --params-file $(ros2 pkg prefix rl_deploy_inference)/share/rl_deploy_inference/config/hole_align.yaml \
  -p rgb_topic:=/realsense_2/camera/color/image_rect \
  -p depth_topic:=/realsense_2/camera/aligned_depth_to_color/image_rect \
  -p camera_info_topic:=/realsense_2/camera/color/camera_info \
  -p socket_timeout_s:=300.0
# If occluded (no hole): add  -p anchor_use_fixed:=true -p anchor_xyz:="[<x>,<y>,<z>]"  (use T-HOLE's DETECTED value)
ros2 topic echo /rl_deploy/corrected_socket --once

── T-CONTROLLER: KEEP JTC ACTIVE ──

export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/kuka_fri_omar_ws/install/setup.bash
ros2 control list_controllers -c /lbr_dual_arm_y_gripper/controller_manager
# REQUIRED: joint_trajectory_controller active; lbr_joint_position_command_controller inactive.
# The dual-arm JTC must have allow_partial_joints_goal: true so the bridge can command lbr_two only.
# If needed (ROBOT MAY MOVE — hand on e-stop):
ros2 control switch_controllers -c /lbr_dual_arm_y_gripper/controller_manager \
  --activate joint_trajectory_controller \
  --deactivate lbr_joint_position_command_controller --strict
ros2 topic info /lbr_dual_arm_y_gripper/joint_trajectory_controller/joint_trajectory -v
ros2 topic echo /lbr_dual_arm_y_gripper/state --once
# Before start_policy: session_state=4, connection_quality=3, safety_state=0, drive_state=2,
# client_command_mode=1, overlay_type=1. This profile expects control_mode=0; if the pendant reports
# control_mode=1 Cartesian impedance, set expected_control_mode:=1 deliberately before starting.

── T-GUARDED 15 Hz → JTC BRIDGE (only ONE) ──

export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/kuka_fri_omar_ws/install/setup.bash
source ~/Masterthesis-rl-deploy/install/setup.bash
ros2 run rl_deploy_inference guarded_joint_trajectory_bridge
# Enforces: fresh measured joints, 0.5deg max target lead, 0.35deg tracking error,
# and stateful 3/30/300 deg/s(/s2,/s3) q/dq/ddq quintic horizons.
# Do NOT run command_upsampler at the same time.

── T-BAG ROSBAG ──

export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/Masterthesis-vision/install/setup.bash
source ~/Masterthesis-rl-deploy/install/setup.bash
mkdir -p ~/deploy_bags
ros2 bag record -o ~/deploy_bags/pdz_single_$(date +%Y%m%d_%H%M%S) \
  /rl_deploy/policy_obs /rl_deploy/command_15hz \
  /lbr_dual_arm_y_gripper/joint_trajectory_controller/joint_trajectory \
  /rl_deploy/corrected_socket /rl_deploy/status /rl_deploy/seat_detected /rosout \
  /lbr_dual_arm_y_gripper/state /lbr_dual_arm_y_gripper/joint_states \
  /lbr_dual_arm_y_gripper/force_torque_broadcaster/wrench \
  /realsense_2/camera/color/image_rect /realsense_2/camera/aligned_depth_to_color/image_rect
# In one additional terminal, capture actual FRI packets. This distinguishes a missing controller packet
# from a late/missing client reply; the 200Hz ROS command topic cannot do that. Ctrl-C after the attempt.
sudo tcpdump -i enp3s0 -s 0 -B 4096 -w ~/deploy_bags/fri_$(date +%Y%m%d_%H%M%S).pcap 'udp port 30201'

── T-RL DEPLOY NODE — **[SINGLE: single-arm YAML]** (motion gated) ──

export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source ~/Masterthesis-rl-deploy/deploy_env.sh
source ~/kuka_fri_omar_ws/install/setup.bash
source ~/Masterthesis-vision/install/setup.bash
source ~/Masterthesis-rl-deploy/install/setup.bash
ros2 launch rl_deploy_inference deploy_inference_pdz.launch.py \
  params_file:=$(ros2 pkg prefix rl_deploy_inference)/share/rl_deploy_inference/config/deploy_pdz_v3_fulltilt_single.yaml
#   ^ GATE: no "stale/missing". Policy RGB-D reads /realsense_2/... (single-arm). pdz v3: gravity
#     comp FIXED -> at rest, policy force [6:9] ≈ 0 (all axes).

── T-PARITY **[SINGLE]** NO-CONTACT FORCE CHECK (run in T6; arm stays still) ──

ros2 param set /rl_deploy_inference obs_dump_path /tmp/deploy_pdz_obs.npz
ros2 param set /rl_deploy_inference max_joint_step_rad 0.0
ros2 service call /rl_deploy/start_policy std_srvs/srv/Trigger "{}"   # starts with q-step clamped to zero
sleep 1
ros2 service call /rl_deploy/stop_policy  std_srvs/srv/Trigger "{}"
python3 -c "import numpy as np; d=np.load('/tmp/deploy_pdz_obs.npz'); p=d['policy']; print('shape',p.shape); print('force[6:9]',p[...,6:9])"
#   (live obs also on topic /rl_deploy/policy_obs — force is indices [6:9] of the 15-D vector)
#   ✅ PASS: force[6:9] ≈ 0 on ALL axes (a few tenths N ok).
#   ❌ FAIL: large DC (esp. negative-z ~5N) -> STOP. Do NOT re-add ft_bias. Recheck ft_baseline_on_start / force_source / ft_sign.
#   Then gently press the socket by hand -> CONTACT should read POSITIVE TCP z; if opposite, flip ft_sign (NOT the bias).
ros2 param set /rl_deploy_inference max_joint_step_rad 0.0045

── ARM + START (robot moves under RL) ──

ros2 param set /rl_deploy_inference we_a_geometry_calibrated true     # after screw_tip_offset(30mm)/tip_link verified
ros2 param set /rl_deploy_inference we_a_force_bias_calibrated true   # after T-PARITY: force[6:9]≈0 + contact sign correct
ros2 param set /rl_deploy_inference enable_motion true
ros2 service call /rl_deploy/start_policy std_srvs/srv/Trigger "{}"
#   WATCH: applied TCP step is capped at 1mm/tick, downward motion attenuates toward 6N,
#   raw contact force aborts at 15N, and the trajectory bridge must remain unlatched.

── STOP / CLEAR / RESTORE MOVEIT ──

ros2 service call /rl_deploy/stop_policy std_srvs/srv/Trigger "{}"
ros2 service call /rl_deploy/clear_latches std_srvs/srv/Trigger "{}"
ros2 service call /rl_deploy/clear_bridge_latch std_srvs/srv/Trigger "{}"
# MoveIt already owns the same JTC. Ctrl-C T-RL, T-GUARDED, and T-ANCHOR, then verify JTC remains active:
ros2 control list_controllers -c /lbr_dual_arm_y_gripper/controller_manager

── WHEN FULLY DONE ──

sudo ip addr del 192.170.20.1/24 dev enp3s0
sudo ip addr del 192.170.10.1/24 dev enp3s0
`````

## 7.11 `docs/HOW_TO_DEPLOY.md`

`````markdown
# How To Deploy The RL Insertion Policy

This is the practical bring-up guide for `Masterthesis-rl-deploy`.

The deploy stack is a normal ROS2 Humble colcon workspace. The inference node is also a Python
policy runner, so the shell that starts it must be able to import both ROS2 Python packages and the
RL packages (`torch`, `gymnasium`, `rl_games`, `torchvision`, and the training repo).

## 1. Python Environment

The node is both a ROS2 node and a torch/rl_games policy runner, so one interpreter must import both.

**Recommended (zero-install bridge — `source deploy_env.sh`).** The checkpoint was produced with an
unusual torch build (`torch 2.11.0+cu128`, not on PyPI), so reinstalling torch risks load parity.
System `python3.10` is ABI-compatible with ROS Humble (both cp310), so we point it at the SAME
packages the training/eval env uses (the isaaclab conda env's site-packages) via `PYTHONPATH` — no
download, byte-for-byte the same torch/rl_games that saved the `.pth`. Build once, then source the
env each shell:

```bash
cd ~/Masterthesis-rl-deploy
source /opt/ros/humble/setup.bash
colcon build --symlink-install        # first time / after message changes
source deploy_env.sh                    # sources ROS + this ws + conda site-packages on PYTHONPATH
```

`deploy_env.sh` uses the isaaclab env by default; override with
`RL_DEPLOY_PY_SITE=/path/to/env/lib/python3.10/site-packages` before sourcing. The launch/`ros2 run`
commands then use plain `python3`.

**Fallback (fresh venv).** Only if you cannot reuse the training env's packages. Reinstall torch etc.
into a system-visible venv, matching the training versions as closely as possible:

```bash
python3 -m venv --system-site-packages ~/venvs/rl-deploy
source ~/venvs/rl-deploy/bin/activate
pip install "gymnasium==1.2.1" "rl-games==1.6.5" opencv-python pyyaml   # + a torch that loads the .pth
source /opt/ros/humble/setup.bash && source install/setup.bash
```

Either way, the final shell must pass all of these imports (the gate):

```bash
python3 -c "import rclpy, torch, torchvision, gymnasium, rl_games, fp_debug_msgs, lbr_fri_idl"
```

## 2. Checkpoint Files

The `.pth` file does not need to live in this repo. The node reads absolute paths from
`src/rl_deploy_inference/config/deploy.yaml`.

Current deploy defaults (the `w2_estimator_192` explicit-estimator winner):

```yaml
policy_checkpoint: /home/moreno/Masterthesis-rl-train/logs/rl_games/Forge/w2_estimator_192/nn/last_Forge_ep_2000_rew_162.13815.pth
agent_config: /home/moreno/Masterthesis-rl-train/logs/rl_games/Forge/w2_estimator_192/params/agent.yaml
env_config: ""
auto_obs_config_from_env_yaml: true
```

`w2_estimator_192` differs from the earlier `e2e_weld_curric` checkpoint in two deploy-relevant ways,
both handled automatically by the loader — no operator action needed:

- Its `agent.yaml` declares a privileged `aux_label` obs group + an `aux_head` (the internal vision
  estimator). `aux_label` is a training-only label, excluded from the policy input and with its aux
  loss skipped at inference, so it has **zero effect on the action**. `policy.py` detects the group
  from the agent config, declares `aux_label` (dim 4) in the obs space so the saved input-RMS restores
  cleanly, and feeds dummy zeros `(1, 4)` every inference step. Startup logs
  `Explicit-estimator checkpoint: feeding dummy 'aux_label' zeros (dim=4) ...`.
- It was trained with **gravity compensation**, so its `ft_force` obs is pure contact force. Keep
  `ft_bias_base_xyz: [0.0, 0.0, 0.0]` and feed the robot's gravity/payload-compensated F/T directly;
  do not re-add a gravity DC offset.

To run a different checkpoint, switch `policy_checkpoint` and `agent_config` together; the loader
re-detects `aux_label`/obs sizing from the paired config so non-estimator checkpoints (no `aux_head`)
just leave the obs space unchanged.

With `auto_obs_config_from_env_yaml: true`, the node automatically looks for `env.yaml` next to
`agent.yaml`. That is where `image_height`, `image_width`, `image_channels`, `frame_stack`,
`ft_smoothing_factor`, and camera depth far clip come from. This prevents a checkpoint trained with a
different frame stack from accidentally receiving the wrong tensor.

The image encoder is not a separate deploy model. The ResNet-18 branch is part of the saved
`insertion_hybrid` rl_games actor. Deploy feeds the dict obs as `{"policy": ..., "image": ...}`;
the loaded actor runs the ResNet path internally.

## 3. One Arm / One RealSense Mapping

The camera launch is in:

```text
~/zed_ros2_ws/src/mv_launch/launch/zed_realsense_trio.launch.py
```

That launch starts one static ZED plus two wrist D405s:

| Arm | Camera | Serial | Flange Pose |
| --- | --- | --- | --- |
| left | `realsense_1` | `260322275185` | `/left/ee_pose` |
| right | `realsense_2` | `260522275434` | `/right/ee_pose` |

This deploy package defaults to the right arm and `realsense_2`:

```yaml
rgb_topic: /realsense_2/camera/color/image_rect
depth_topic: /realsense_2/camera/aligned_depth_to_color/image_rect
flange_pose_topic: /right/ee_pose
robot_prefix: lbr_two
base_link: lbr_two_link_0
tip_link: lbr_two_gripper_tcp
fallback_tip_link: lbr_two_link_ee
```

For the left arm later, change those to:

```yaml
rgb_topic: /realsense_1/camera/color/image_rect
depth_topic: /realsense_1/camera/aligned_depth_to_color/image_rect
flange_pose_topic: /left/ee_pose
robot_prefix: lbr_one
base_link: lbr_one_link_0
tip_link: lbr_one_gripper_tcp
fallback_tip_link: lbr_one_link_ee
```

Only run one inference node for one arm for now.

## 4. Topics

The node subscribes to:

```text
/realsense_2/camera/color/image_rect                  sensor_msgs/Image
/realsense_2/camera/aligned_depth_to_color/image_rect sensor_msgs/Image
/perception/fp/pose_base/fused/assembly               fp_debug_msgs/DebugPoseItem
/lbr_dual_arm/joint_states                            sensor_msgs/JointState
state                                                  lbr_fri_idl/LBRState
/right/ee_pose                                        geometry_msgs/PoseStamped
/wrist_ft                                             geometry_msgs/WrenchStamped
/rl_deploy/e_stop                                     std_msgs/Bool
```

The node publishes:

```text
command/joint_position        lbr_fri_idl/LBRJointPositionCommand
/rl_deploy/status             std_msgs/String
/rl_deploy/policy_obs         std_msgs/Float32MultiArray
/rl_deploy/seat_detected      std_msgs/Bool
/gripper/open_cmd             std_msgs/Float64
```

`state` and `command/joint_position` are relative by default. If your FRI stack uses a namespace,
launch the node inside the same namespace or make these absolute in the YAML.

## 5. Control Frequency

The policy loop runs at:

```yaml
control_hz: 15.0
```

So the node publishes at 15 Hz, about one command every 66.7 ms.

Freshness gates:

```yaml
input_timeout_s: 0.35
socket_timeout_s: 0.75
```

Policy motion only runs when RGB, depth, socket position, F/T, joint state, flange pose, and
kinematics are all fresh.

## 6. Safety Arm, Start, Stop, Reset

There are two layers:

1. `enable_motion`: hard software arming parameter.
2. Runtime mode: controlled by ROS services.

The node starts in hold mode:

```yaml
enable_motion: false
policy_active_on_start: false
```

Arm the node only after the FRI side is in low-power/low-PD safe mode:

```bash
ros2 param set /rl_deploy_inference enable_motion true
```

Start policy:

```bash
ros2 service call /rl_deploy/start_policy std_srvs/srv/Trigger {}
```

Stop policy and hold:

```bash
ros2 service call /rl_deploy/stop_policy std_srvs/srv/Trigger {}
```

Manual e-stop:

```bash
ros2 topic pub /rl_deploy/e_stop std_msgs/msg/Bool "{data: true}" --once
```

Clear manual e-stop:

```bash
ros2 topic pub /rl_deploy/e_stop std_msgs/msg/Bool "{data: false}" --once
```

Clear force/seat latches:

```bash
ros2 service call /rl_deploy/clear_latches std_srvs/srv/Trigger {}
```

These services can be used as buttons in `rqt_service_caller`, Foxglove, or a tiny custom operator
panel.

For first hardware runs, policy trials are configured for manual success stopping:

```yaml
trial_stop_mode: manual
ft_force_cap_n: 20.0
trial_timeout_s: 10.0
```

In this mode the policy still receives the perceived socket pose, but that pose is not trusted to
certify insertion. The operator stops on observed insertion; force cap, timeout, stale inputs, and
e-stop still stop/hold the robot.

Later, set `trial_stop_mode: auto_seat` to re-enable automatic seating gates:

```yaml
seat_socket_depth_m: 0.0175
seat_z_tolerance_m: 0.0045
seat_xy_tolerance_m: 0.004
seat_force_requires_geometry: true
trial_max_overtravel_m: 0.002
```

The geometry gate uses the perceived socket frame: socket opening is `z=0`, socket bottom is
`-seat_socket_depth_m`. A trial succeeds when the fingertip/screw-tip estimate is close to the bottom
and laterally close to the socket axis. A force hit before that bottom-zone geometry is treated as an
early-contact failure and the node holds instead of pushing. Force cap, overtravel, and timeout also
stop the trial into hold.

The deployed E2E actor is a 5-D action policy (`[dx, dy, dz, rot_a, rot_b]`). It does not expose a
success-probability output, so do not use an "80% actor success" threshold unless a future checkpoint is
explicitly trained with a calibrated success head and the deploy adapter is updated to read it.

## 7. Preinsert Pose

There are two valid workflows.

Option A, safest first deployment: let the other student's MoveIt/teleop stack move the robot to
the preinsert pose. Keep this RL node in hold mode with `enable_motion: false`. Once the robot is
already at preinsert, arm and start the policy.

Option B: use this node's joint-space reset. First record or choose a 7-DoF joint target for the
active arm:

```bash
ros2 topic echo /lbr_dual_arm/joint_states --once
```

Copy the active arm's seven joints into YAML:

```yaml
preinsert_joint_position: [q1, q2, q3, q4, q5, q6, q7]
preinsert_joint_position_set: true
preinsert_max_joint_step_rad: 0.006
preinsert_tolerance_rad: 0.01
```

Then run:

```bash
ros2 param set /rl_deploy_inference enable_motion true
ros2 service call /rl_deploy/reset_preinsert std_srvs/srv/Trigger {}
```

Reset-to-preinsert only needs fresh joint state. It does not require camera, socket, F/T, or policy
readiness. It streams a simple joint-position ramp at the same 15 Hz timer and then returns to hold.

## 8. Socket Pose

The socket anchor comes from:

```yaml
socket_pose_topic: /perception/fp/pose_base/fused/assembly
socket_pose_type: debug_pose_item
socket_assembly_name: cooling_manifold
socket_part_id: -1
```

Set `socket_part_id` to the actual socket part before starting the policy:

```bash
ros2 param set /rl_deploy_inference socket_part_id <SOCKET_PART_ID>
```

If `socket_part_id` is still `-1`, `/rl_deploy/start_policy` refuses to start. This is intentional,
because the policy ignores socket orientation and uses only socket position as the action anchor.

## 9. Observation Parity

Dump one **sim** observation from the training repo (GPU/Isaac; `dump_sim_obs.py` writes the same
npz format as the deploy dump — `policy` (21,) + `image` (H,W,C)):

```bash
cd ~/Masterthesis-rl-train
OMNI_KIT_ACCEPT_EULA=YES TORCHDYNAMO_DISABLE=1 PYTHONUNBUFFERED=1 \
  /home/moreno/miniconda3/envs/isaaclab/bin/python scripts/dump_sim_obs.py \
    --task Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0 \
    --headless --enable_cameras \
    --experience /home/moreno/Masterthesis-rl-train/apps/isaaclab.python.headless.rendering.physx1065.kit \
    --num_envs 1 --warmup 2 --out /tmp/sim_obs.npz
```

(No checkpoint needed — the dump is a pure env-obs snapshot; it forces the deploy obs contract
`e2e_use_proprio_obs=True` + `e2e_keep_aux_label=True` so the sim `policy` is 21-d.)

Dump one **deploy** observation (captures the first obs once the policy is streaming):

```bash
ros2 param set /rl_deploy_inference obs_dump_path /tmp/deploy_obs.npz
```

Compare:

```bash
ros2 run rl_deploy_inference obs_parity --sim /tmp/sim_obs.npz --deploy /tmp/deploy_obs.npz
```

Sim and real scenes can't be matched pixel-for-pixel, so this checks **structure/units/normalization/
ranges** (per-channel image means, depth scaling, the 21-d `policy` layout) — not exact pixel
equality. The `aux_label` is training-only and is not part of the parity check. `dump_sim_obs.py`
also prints the policy vector and per-channel image stats so you can eyeball ranges directly.

For `w2_estimator_192`, `params/env.yaml` says:

```yaml
image_height: 224
image_width: 224
image_channels: 4
frame_stack: 1
```

Frame stacking, if a future checkpoint uses `N > 1`, is oldest frame to newest frame, with RGB-D
channels kept adjacent inside each timestep:

```text
[old_rgb, old_depth, ..., newest_rgb, newest_depth]
```

## 10. Minimal Bring-Up Checklist

1. Start dual-arm FRI/hardware stack.
2. Start `zed_realsense_trio.launch.py` or the equivalent RealSense-only launch.
3. Start FoundationPose/fusion.
4. Source the Python env, ROS, and this workspace.
5. Verify imports.
6. Launch `rl_deploy_inference` with `enable_motion: false`.
7. Watch `/rl_deploy/status`.
8. Set the correct checkpoint pair.
9. Set `socket_part_id`.
10. Move to preinsert by external stack or `/rl_deploy/reset_preinsert`.
11. Verify observation parity.
12. Set `enable_motion true`.
13. Call `/rl_deploy/start_policy`.
14. Keep `/rl_deploy/stop_policy` and `/rl_deploy/e_stop` ready.
`````

## 7.12 `docs/DEPLOY_PDZ.md`

`````markdown
# Deploy pdz_overnight_20260824

Deploy profiles for the `pdz_overnight_20260824` run: the same end-to-end policy as
`we_A_resnet18_baseline`, retrained on the **pdz gripper** with the ResNet-18 **layer4 fine-tune tail
unfrozen** (`finetune_tail: true`, `finetune_grad_scale: 0.1`) and **gravity compensation on**.

Three profiles cover the requested spread (reward drops over epochs only because the dynamics/tilt
curriculum hardens — later is not worse):

| Config | Checkpoint | Reward |
|--------|-----------|--------|
| `config/deploy_pdz_ep500.yaml`  | `nn/last_Forge_ep_500_rew_183.74542.pth`  | 183.7 |
| `config/deploy_pdz_ep1000.yaml` | `nn/last_Forge_ep_1000_rew_151.45567.pth` | 151.5 |
| `config/deploy_pdz_ep1500.yaml` | `nn/last_Forge_ep_1500_rew_144.36847.pth` | 144.4 |

## Why the deploy code did not change

The observation/action contract is byte-identical to `we_A_resnet18_baseline`: 15-D policy vector
(`obs_order` unchanged), ImageNet-normalized 180x320 RGB, `frame_stack: 1`, no `aux_head`/`aux_label`,
6-D TCP-relative action. So `inference_node_we_a`, `policy_we_a.py`, and `obs_preprocessing_we_a.py`
are reused unchanged — these profiles only swap the parameter file.

The **unfrozen layer4 is inert at inference**. `finetune_tail` only sets `requires_grad=True` on
`layer4` and registers a gradient-scale hook; both do nothing under the deploy `torch.no_grad()`. The
frozen-stem/trainable-tail forward is numerically identical (BatchNorm is forced to `eval()` on the
ImageNet running stats regardless), and the saved `state_dict` is the standard torchvision ResNet-18
layout. The loader rebuilds the network from this run's own `agent.yaml`, so the fine-tuned weights
restore correctly with no code path change.

## Build and launch

```bash
cd /home/moreno/Masterthesis-rl-deploy
colcon build --symlink-install --packages-select rl_deploy_inference
source deploy_env.sh
# Default = episode 1500:
ros2 launch rl_deploy_inference deploy_inference_pdz.launch.py
# Or pick another checkpoint:
ros2 launch rl_deploy_inference deploy_inference_pdz.launch.py \
    params_file:=$(ros2 pkg prefix rl_deploy_inference)/share/rl_deploy_inference/config/deploy_pdz_ep500.yaml
```

The launch also starts `command_upsampler`. The policy starts in HOLD
(`policy_active_on_start: false`) and cannot enter policy mode until `/rl_deploy/start_policy`.

## What changed vs deploy_we_a.yaml

Motion is **gated in all three profiles** (`we_a_geometry_calibrated: false` and
`we_a_force_bias_calibrated: false`) because the pdz gripper is a new physical end effector. Certify
the items below on the bench, set the flags to `true`, then follow the same live preflight as
[DEPLOY_WE_A.md](DEPLOY_WE_A.md).

### 1. Force contract — broken gravity comp baked a ~0.75×weight bias into training

`pdz_overnight/params/env.yaml` sets `use_gravity_comp: true`, but the run trained with a **broken
gravity-comp baseline**. The per-env gravity reference was captured from the EMA-smoothed force on the
first post-reset step (`alpha=0.25` starting from 0), so only ~25% of gravity was subtracted and
**~0.75×gripper_weight stayed baked into the training force obs**. This is confirmed three ways: the
committed (Aug 13) code froze `_grav_base` from `force_sensor_smooth` (the bug); the raw-reading fix is
still *uncommitted* and postdates this Aug 24 run; and the saved policy force-slot running-mean shows a
large non-zero DC (not the near-zero of clean contact). The real F/T is properly gravity-compensated,
so the deployed force will **not** match training unless corrected. Two options, toggled per run:

- **(A) Diagnostic ablation — `zero_force_obs: true`.** Zeros the policy force channels (indices
  `[6:9]`) before every actor forward, isolating whether force helps or hurts on hardware. The
  real-contact safety/seat thresholds use `force_base` and are **unaffected**. This is the clean first
  test given the messy baked-in bias.
- **(B) Parity — reproduce the bias via `ft_bias_base_xyz`.** The deploy already **adds**
  `ft_bias_base_xyz` into the policy force (rotated to TCP with the tool, exactly like sim's constant
  base-frame gravity residual) while **subtracting** it from the safety norm. The startup baseline
  removes the *full* real gravity, so set `ft_bias_base_xyz = 0.75 × (measured real no-contact distal
  wrench in base)`. Tool-down hover ≈ `[0, 0, 0.75×distal_weight_N]` base +Z (order ~7 N if the pdz
  distal mass matches the y-gripper's 0.9675 kg — **measure it on the pdz gripper**).

Both leave `ft_bias_base_xyz: [0,0,0]` and `zero_force_obs: false` by default, gated by
`we_a_force_bias_calibrated: false`. Capture the startup baseline with the screw clear of contact,
re-verify `ft_sign: -1.0` (pressing → positive TCP Z) on the pdz mount, choose (A) or (B), then set
`we_a_force_bias_calibrated: true`.

### 2. Held-screw geometry — new gripper grasp

The pdz gripper changes the grasp (`franka_fingerpad_length: 0.003` vs `0.012`, closed half-gap
`0.006` vs `0.002`), so the held screw sits differently in the TCP. `screw_tip_offset_tcp_xyz` is a
**placeholder** carried over from we_A (`[0, 0, 0.023]`). Measure the real pdz grasp (tip offset and
shaft axis), confirm `tip_link` names the pdz TCP link in the **real** `robot_description` (with
`fingertip_source: fk` the FK chain to that link supplies the fingertip pose;
`flange_to_fingertip_*` is only a fallback), re-check the `preinsert_hover_z_m` margin against the new
tip offset, then set `we_a_geometry_calibrated: true`.

### 3. Camera hand-eye — new wrist mount

The sim wrist-cam pose moved (`wrist_cam_offset_pos [0.042, 0, -0.058] -> [0.009, -0.05056, -0.07257]`,
`wrist_cam_offset_quat` now `null`/look-at based). Observation parity requires the real D405 mount to
match this new viewpoint. There is no camera-extrinsic parameter in the config — the match is physical
plus the image itself — so redo the empirical hand-eye against a pose-matched real frame
(`aligned_depth_to_color` color topic, matched joints) before trusting the image branch. See memory
`d405-handeye-calib`.

## Observation check

Identical to we_A: set `max_joint_step_rad: 0.0` to run preprocessing + inference with the published
target pinned to the measured pose, dump the obs, and confirm `policy=(15,)`, `image=(180,320,3)`,
order `[goal_delta_tcp(6), force_tcp(3), previous_action(6)]`. See
[DEPLOY_WE_A.md](DEPLOY_WE_A.md#observation-check) for the exact commands and the live preflight.
`````

## 7.13 `docs/DEPLOY_WE_A.md`

`````markdown
# Deploy we_A_resnet18_baseline

This profile is independent from the existing W2 deployment path. It fixes the saved policy contract
to a 15-D vector, ImageNet-normalized 180x320 RGB, and a 6-D TCP-relative action.

## Build and launch

```bash
cd /home/moreno/Masterthesis-rl-deploy
colcon build --symlink-install --packages-select rl_deploy_inference
source deploy_env.sh
ros2 launch rl_deploy_inference deploy_inference_we_a.launch.py
```

The profile uses episode 2500. The launch also starts `command_upsampler`, which bridges
`/rl_deploy/command_15hz` to the robot's 200 Hz joint-position command topic. Do not start a second
upsampler. The policy starts in HOLD (`policy_active_on_start: false`) and cannot enter policy mode
until `/rl_deploy/start_policy` is called. The node consumes RGB directly and does not wait for depth.

## Held-part geometry gate

The saved training transform flips the held screw by 180 degrees about gripper Y: screw local `+Z`
is therefore TCP `-Z`. At a tool-down seated pose, TCP `+Z` points down, so TCP `-Z` and socket/base
`+Z` point the same way. Keep `socket_axis_base_xyz: [0,0,1]` and use
`shaft_axis_tcp_xyz: [0,0,-1]`.

The nominal training TCP-to-tip offset is `23 mm` along TCP `+Z`: the screw shoulder is `5.5 mm`
outside the TCP and the shaft tip is another `17.5 mm` beyond it. Training randomized the grasp by
up to `+/-3 mm` axially and `+/-2 mm` laterally, so measure the real held screw and replace
`screw_tip_offset_tcp_xyz` if needed. The automatic seat/overtravel checks use this tip position.

The real transform has been confirmed and `we_a_geometry_calibrated` is enabled in this profile.

The runtime `we_a_axis_guard_deg` defaults to `60` degrees. The saved task starts within about
`33` degrees; an axis-sign error approaches `180` degrees and stops the trial before a policy command.

## Force contract gate

The run artifact at `we_A_resnet18_baseline/params/env.yaml` records `use_gravity_comp: false`.
Training therefore did not subtract the pre-contact force-sensor baseline: its observation retained
the distal gripper/fingers/screw weight and then rotated that force into the TCP frame. The real
`arm_measured_torque` path still needs its startup baseline to remove the static arm/payload estimate;
`ft_bias_base_xyz` then restores the sim-equivalent distal DC for the policy observation.

The nominal saved asset mass below the simulated sensor is `0.967506 kg`; at `9.81 m/s^2`, this is
`9.49 N` in base `+Z`. Training randomized robot masses by `+/-3%`, so this value is the nominal run
contract, not a measured real-payload weight. The synthetic term is excluded from the real-contact
force cap, warning, and seat checks in the we_A policy loop.

For reference, episode 2500's saved policy normalizer has force-slot running mean
`[0.787, -0.426, -2.564] N`. That confirms a non-zero training distribution, but it includes contact
samples and is not itself a valid no-contact bias calibration.

Hardware pressing was verified as positive TCP Z with `ft_sign: -1`. The startup baseline must be
captured while the screw is clear of contact. The nominal bias and calibration gate are enabled in
this profile; an exact no-contact sim capture remains a useful parity check, but is not a hardware
payload measurement.

## Observation check

For the first live parity check, set the joint-step limit to zero before entering policy mode. This
still runs preprocessing and actor inference, but every published joint target remains at the measured
pose:

```bash
ros2 param set /rl_deploy_inference obs_dump_path /tmp/deploy_we_a_obs.npz
ros2 param set /rl_deploy_inference max_joint_step_rad 0.0
ros2 service call /rl_deploy/start_policy std_srvs/srv/Trigger {}
ros2 service call /rl_deploy/stop_policy std_srvs/srv/Trigger {}
python3 -c "import numpy as np; d=np.load('/tmp/deploy_we_a_obs.npz'); print({k:d[k].shape for k in d})"
ros2 param set /rl_deploy_inference max_joint_step_rad 0.0045
```

Expected shapes are `policy=(15,)` and `image=(180,320,3)`. Policy order is
`[goal_delta_tcp(6), force_tcp(3), previous_action(6)]`. Inspect the dump before restoring motion.

## Live preflight

1. Complete the vision-corrected preinsert and keep a fresh `cooling_base`, part `0` anchor publishing
   on `/rl_deploy/corrected_socket` for the whole trial.
2. Launch this profile and confirm `/rl_deploy/status` says HOLD. Check that the upsampler is the only
   publisher to `/lbr_dual_arm_y_gripper/command/joint_position`.
3. With the arm stationary, switch from `joint_trajectory_controller` to
   `lbr_joint_position_command_controller` and confirm the arm continues holding its measured pose.
4. Perform the zero-step observation check above. Keep the screw out of contact so the measured-torque
   baseline is captured cleanly.
5. Restore `max_joint_step_rad: 0.0045`, keep the physical E-stop in hand, then call start for the
   motion trial:

```bash
ros2 service call /rl_deploy/start_policy std_srvs/srv/Trigger {}
```

Keep the physical E-stop in hand. Stop from another shell with:

```bash
ros2 service call /rl_deploy/stop_policy std_srvs/srv/Trigger {}
```

Before the first motion trial, verify the corrected opening, policy observation shape `(15,)`, finite
actor output, axis guard, force direction, and no-contact hold. The remaining gates are live hardware
and ROS topic checks; they cannot be certified offline.
`````

## 7.14 `docs/PREINSERT_MOVEIT.md`

`````markdown
# Headless MoveIt preinsert planner (single arm, right or left)

Move **one arm** to a safe **preinsert** pose hovering above the perceived socket, using **MoveIt**
for the gross motion — *not* the RL `reset_preinsert` IK servo (a local damped-IK servo that rattles
from far away). The RL policy then only does the final local insertion from this hover.

Pick the arm with `--arm right` (default: group `arm_two`, tip `lbr_two_gripper_tcp`) or `--arm left`
(group `arm_one`, tip `lbr_one_gripper_tcp`). Only the chosen arm moves.

- Package: `rl_deploy_inference`
- Code: [`motion_commander.py`](../src/rl_deploy_inference/rl_deploy_inference/motion_commander.py)
  (reusable MoveIt client), [`preinsert_planner.py`](../src/rl_deploy_inference/rl_deploy_inference/preinsert_planner.py)
  (node/CLI), config [`config/preinsert.yaml`](../src/rl_deploy_inference/config/preinsert.yaml).
- Approach mirrors the supervisor's proven `Masterthesis-vision/src/calibration/moveit_dual_arm.py`
  (`DualArmMoveitClient`): plan and execute through the `moveit_msgs/action/MoveGroup` action, single
  arm only. We reuse that pattern deliberately.

---

## Architecture

1. **MoveIt** handles the gross motion to the preinsert hover (collision-aware, jerk-limited).
2. **RL policy** handles only the final local insertion.

The planner talks to an already-running `move_group` over the `MoveGroup` action; it is **not** a
MoveIt process itself, so it needs no robot_description/SRDF/kinematics of its own.

- **Only the inserting arm moves.** The goal targets planning group `arm_two` (the right arm's 7
  joints); `arm_one` is never commanded.
- **Planning frame == perception frame == `base_link`.** `base_link` is the URDF root; `lbr_two_link_0`
  is a fixed child at `xyz = 0 0.42 0`. Perception already publishes the socket in `base_link`, and the
  `+0.15 m` hover is a global `+z` offset — no frame conversion.
- **Target orientation is the current TCP orientation** by default (`orientation_mode: current_tcp`),
  read from TF. The perceived *object* orientation is deliberately **not** trusted yet. Switch to
  `orientation_mode: fixed` + `fixed_orientation_xyzw` for a configured fixed insertion orientation.
- **Conservative** velocity/acceleration scaling (`0.05`, well under the required `0.1`).

---

## ⚠️ Prerequisite for EXECUTION: `allow_partial_joints_goal`

The dual-arm `joint_trajectory_controller` spans **all 14 joints**. Commanding one arm (7 joints)
requires `allow_partial_joints_goal: true` on it, or the controller rejects the trajectory with
*"Joints on incoming trajectory don't match the controller joints"* (the supervisor hit this exact
gap; the flag took single-arm execute from 0/14 → 13/14).

This has been added to the deploy kuka repo:
`~/kuka_fri_omar_ws/.../lbr_dual_arm_description/ros2_control/dual_arm_controllers.yaml`.
**It only takes effect after you rebuild + relaunch hardware:**

```bash
cd ~/kuka_fri_omar_ws
colcon build --packages-select lbr_dual_arm_description
# then relaunch hardware.launch.py
```

**Plan-only (dry-run) does NOT need this flag** — you can validate the whole pipeline first.

---

## Requirements at runtime

- **MoveIt must be installed/sourced** on the machine running `move_group` + this node
  (`moveit_msgs` importable, e.g. `sudo apt install ros-humble-moveit`). Without it, `plan`/`move`
  fail with a clear error (the module still imports; TF/readback still work).
- `joint_trajectory_controller` must be the **active** controller on the position interface — the RL
  `lbr_joint_position_command_controller` must be **inactive** (they share the interface).

---

## Where it fits in the full deploy sequence

The preinsert planner runs **once**, after the vision pipeline is publishing a fused socket pose and
**before** you hand control to the RL policy: it does the gross move to the hover, then the RL node
does the final insertion. Concretely it slots between the checks (T6) and RL inference (T7).

```bash
# ── T1  ROBOT BRINGUP ────────────────────────────────────────────────────────
sudo ip addr add 192.170.10.1/24 dev enp3s0
sudo ip addr add 192.170.20.1/24 dev enp3s0
cd ~/kuka_fri_omar_ws
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/kuka_fri_omar_ws/install/setup.bash
ros2 launch lbr_dual_arm_y_gripper_bringup hardware.launch.py
#   ^ spawns joint_state_broadcaster + joint_trajectory_controller (JTC) active. JTC is what
#     preinsert/MoveIt drives, so no extra controller setup is needed for the preinsert step.

# ── T2  MOVEIT (headless) ────────────────────────────────────────────────────
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/kuka_fri_omar_ws/install/setup.bash
ros2 launch lbr_dual_arm_y_gripper_bringup move_group.launch.py mode:=hardware rviz:=false

# ── T3  HOST STACK (cameras / foxglove) ──────────────────────────────────────
cd ~/Masterthesis-vision
scripts/launch_host_realsense.sh

# ── T4  VISION PIPELINE ──────────────────────────────────────────────────────
cd ~/Masterthesis-vision
scripts/launch_pipeline_realsense.sh init-only

# ── T5  F/T BROADCASTER ──────────────────────────────────────────────────────
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/kuka_fri_omar_ws/install/setup.bash
cat > /tmp/ft_broadcaster.yaml <<'EOF'
/**/force_torque_broadcaster:
  ros__parameters:
    frame_id: lbr_two_link_ee
    sensor_name: estimated_ft_sensor
EOF
ros2 run controller_manager spawner force_torque_broadcaster \
  -c /lbr_dual_arm_y_gripper/controller_manager \
  -t force_torque_sensor_broadcaster/ForceTorqueSensorBroadcaster \
  -p /tmp/ft_broadcaster.yaml

# ── T6  CHECKS ───────────────────────────────────────────────────────────────
source /opt/ros/humble/setup.bash
source ~/Masterthesis-vision/install/setup.bash
source ~/Masterthesis-rl-deploy/install/setup.bash
ros2 topic echo /lbr_dual_arm_y_gripper/joint_states --once
ros2 topic echo /perception/fp/pose_base/fused/assembly --once
ros2 topic echo /lbr_dual_arm_y_gripper/force_torque_broadcaster/wrench --once
ros2 action list | grep move_action     # -> /lbr_dual_arm_y_gripper/move_action (move_group up)

# ── T-PRE  PREINSERT (this tool) — run once, THEN start the RL policy ─────────
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/kuka_fri_omar_ws/install/setup.bash          # move_group / tf / joint_states
source ~/Masterthesis-rl-deploy/install/setup.bash    # this package
# DRY-RUN first (never moves):
ros2 run rl_deploy_inference preinsert_planner --arm right
# Then EXECUTE (prompts you to type MOVE). Use --arm left for the left arm:
ros2 run rl_deploy_inference preinsert_planner --arm right --execute

# ── T7  RL INFERENCE ─────────────────────────────────────────────────────────
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source ~/Masterthesis-rl-deploy/deploy_env.sh
source ~/kuka_fri_omar_ws/install/setup.bash
source ~/Masterthesis-vision/install/setup.bash
source ~/Masterthesis-rl-deploy/install/setup.bash
ros2 launch rl_deploy_inference deploy_inference.launch.py params_file:=/tmp/rl_dryrun.yaml
```

### ⚠️ Controller handoff (preinsert → RL policy)

Preinsert drives **`joint_trajectory_controller`** (active out of T1). The RL node commands
**`lbr_joint_position_command_controller`** (`command/joint_position`). They share the position
command interface, so **only one is active at a time.** Order matters: do the preinsert move FIRST
(JTC active), and only switch to the RL command controller before `/rl_deploy/start_policy`. If your
RL bring-up doesn't switch automatically:

```bash
ros2 control list_controllers                                          # see what's active
ros2 control switch_controllers \
  --activate lbr_joint_position_command_controller \
  --deactivate joint_trajectory_controller
```

If you need to preinsert again after this switch, switch JTC back on first.

---

## Usage

Build + source this repo first:

```bash
cd ~/Masterthesis-rl-deploy
colcon build --symlink-install --packages-select rl_deploy_inference
source install/setup.bash
```

### One-shot CLI (recommended for bring-up)

```bash
# DRY-RUN (default): waits for a socket pose, plans, prints a full report, exits. NEVER moves.
ros2 run rl_deploy_inference preinsert_planner                 # right arm (default)
ros2 run rl_deploy_inference preinsert_planner --arm left      # left arm

# EXECUTE: plans, prints the report, then asks you to type MOVE before it moves the arm.
ros2 run rl_deploy_inference preinsert_planner --arm right --execute

# Non-interactive execute (scripts): skip the prompt. Use only when you are sure.
ros2 run rl_deploy_inference preinsert_planner --arm right --execute --yes
```

**Choosing the arm:** `--arm right` (default) plans group `arm_two` / tip `lbr_two_gripper_tcp`;
`--arm left` plans group `arm_one` / tip `lbr_one_gripper_tcp`. Only that arm's 7 joints move; both
arms plan through the same `move_group`. (Service form: `arm:=left` launch arg, or `arm` param.)

Override any parameter inline, e.g. a lower hover or a specific socket part:

```bash
ros2 run rl_deploy_inference preinsert_planner --ros-args \
  -p hover_z_m:=0.12 -p socket_part_id:=0
```

### Service form (reusable / long-lived)

```bash
ros2 launch rl_deploy_inference preinsert_planner.launch.py                 # execute service disabled
ros2 launch rl_deploy_inference preinsert_planner.launch.py allow_execute:=true

# dry-run:
ros2 service call /lbr_dual_arm_y_gripper/preinsert_planner/plan_preinsert std_srvs/srv/Trigger
# MOVES the arm (needs allow_execute:=true):
ros2 service call /lbr_dual_arm_y_gripper/preinsert_planner/move_preinsert std_srvs/srv/Trigger
```

---

## What gets logged (requirement 7)

Each run prints: current TCP pose (base_link), socket pose (base_link, orientation ignored), the
computed target pose, plan success/failure + MoveItErrorCode name, trajectory point count/duration,
and the **first and last joint targets** (with joint names). Execute logs the same for the executed
motion.

---

## Safety checklist

- [ ] Pendant **A1 mastered**? If it shows unmastered, **remaster before moving** — do not execute.
- [ ] Physical **e-stop** in hand.
- [ ] `joint_trajectory_controller` active, RL command controller inactive.
- [ ] For execute: `allow_partial_joints_goal: true` built + relaunched (see above).
- [ ] Run **dry-run first**; confirm the target/plan report looks right.
- [ ] **Eyeball the dry-run's first/last joint targets.** A large joint jump for a small Cartesian
      move = OMPL (RRTConnect) took a redundant-arm detour; re-run/re-plan rather than execute it.
- [ ] `socket_part_id` set to the real part (not `-1`) before executing.
- [ ] Hover starts **high** (`0.15 m`) because the socket pose is ~1 cm off; the RL policy closes it.

The planner prints `ROBOT MAY MOVE NOW. Be ready on the physical E-STOP.` before any motion.

---

## Vision-corrected variant: `hole_align_planner` (hover above the DETECTED hole)

`preinsert_planner` hovers above the **perception** socket pose (tracked `cooling_base` CAD centre),
which is ~1 cm off — enough to defeat the 1 mm-clearance insertion. `hole_align_planner` instead
**localizes the socket opening directly in the wrist D405 image** and hovers above *that*:

1. Grabs one time-synced RGB-D frame + `CameraInfo`.
2. Detects the opening with a **Hough-circle** detector sized by the **known 14 mm hole** (measured
   from CAD `cooling_base.obj` inner rim ≈ 7.0 mm radius; equals sim `CoolingInsert.asset_size`).
   The opening depth is sampled from the aligned depth image.
3. Deprojects the circle to a metric 3D point and transforms it to `base_link` via the **live flange
   pose (TF) ∘ calibrated camera-to-flange extrinsics** (`camera_extrinsics_realsense.yaml`, the same
   map the vision pipeline uses).
4. Re-aims the MoveIt preinsert hover at the **detected** hole, planned with the same `MotionCommander`
   + branch-safety as above.

Perception's socket pose (when available) is used **only** to disambiguate the base's two sockets and
to reject spurious circles — never as the final target. It also prints the estimate-vs-detected delta.

**Debug-first / dry-run:** debug is **ON** and motion is **OFF** by default. Every run writes
`holes_<ts>.png` (all circles yellow, the chosen one green, the projected perception estimate as a red
`+`) and `depth_<ts>.png` to `debug_dir` (default `/tmp/hole_align`), and logs the full geometry, so
you can inspect a dry run before anything moves.

```bash
# DRY-RUN (default): detect, plan, dump debug images + report, exit. NEVER moves.
ros2 run rl_deploy_inference hole_align_planner --arm right \
  --ros-args --params-file $(ros2 pkg prefix rl_deploy_inference)/share/rl_deploy_inference/config/hole_align.yaml

# EXECUTE: same, then asks you to type MOVE before it moves the arm.
ros2 run rl_deploy_inference hole_align_planner --arm right --execute --ros-args --params-file <hole_align.yaml>

# Tune detection inline, e.g. looser accumulator / chase the outer counterbore rim:
ros2 run rl_deploy_inference hole_align_planner --ros-args -p hough_param2:=14.0 -p hole_diameter_m:=0.0175
```

Same execution prerequisites as `preinsert_planner` (`allow_partial_joints_goal`, active JTC, MoveIt
sourced). Additional requirements: **OpenCV** (`cv2`, already in the deploy env), the wrist RGB-D +
`CameraInfo` topics publishing, and the **`extrinsics_yaml`** path valid on the machine you run it on.
Inspect the dry-run `holes_*.png` and the printed **correction delta** before trusting the target — a
large delta or a circle on the wrong socket means re-tune the Hough params / `max_center_dist_px`.

---

## Reuse beyond preinsert

`MotionCommander` (in `motion_commander.py`) is a general right-arm MoveIt client:
`plan_to_pose` / `move_to_pose` (Cartesian) and `plan_to_joint` / `move_to_joint` (joint-space, e.g.
a named retract/transport config), plus `get_link_pose` (TF) and `wait_until_plannable` (absorbs
move_group's startup dirty-state window). Point it at a different group/tip/frame via
`MotionCommanderConfig` to drive other chains.
`````

---

# 8. Appendices

## 8.1 Complete parameter reference — E2E `inference_node`

Defaults as declared in `inference_node.py::_declare_parameters`. "Live" = the value in
`deploy_pdz_v3_fulltilt_single.yaml` where it differs meaningfully.

### Arming and loop
| Parameter | Default | Meaning |
|---|---|---|
| `enable_motion` | `false` | Hard software arming gate |
| `policy_active_on_start` | `false` | Start in HOLD |
| `stream_hold_when_disabled` | `true` | Publish the latched hold setpoint while disabled |
| `control_hz` | `15.0` | Policy loop rate (matches sim) |

### Policy / checkpoint
| Parameter | Default |
|---|---|
| `policy_checkpoint` | `.../w2_estimator_192/nn/last_Forge_ep_2000_rew_162.13815.pth` |
| `agent_config` | `.../w2_estimator_192/params/agent.yaml` |
| `train_repo` | `/home/moreno/Masterthesis-rl-train` |
| `env_config` | `""` (auto: `env.yaml` next to `agent.yaml`) |
| `auto_obs_config_from_env_yaml` | `true` |
| `policy_device` | `cuda:0` |
| `deterministic_policy` | `true` |

### Perception and camera
| Parameter | Default | Note |
|---|---|---|
| `rgb_topic` | `/realsense_2/camera/color/image_rect` | right arm |
| `depth_topic` | `/realsense_2/camera/aligned_depth_to_color/image_rect` | shares color intrinsics |
| `rgb_camera_info_topic` | `/realsense_2/camera/color/camera_info` | |
| `img_sync_slop_s` / `img_sync_queue` | `0.02` / `30` | header-stamp sync of the RGB-D pair |
| `fov_match` | `true` | crop to sim FOV before resize (E2E only) |
| `image_height` / `image_width` | `224` / `224` | live pdz: `180` / `320` |
| `frame_stack` | `1` | oldest→newest if > 1 |
| `depth_far_m` | `0.3` | from `clipping_range[1]` |
| `depth_clean` (+ `_inpaint_radius` 3, `_median_ksize` 5) | `false` | A/B test only |

### Socket anchor
| Parameter | Default | Note |
|---|---|---|
| `socket_pose_topic` | `/perception/fp/pose_base/fused/assembly` | live: `/rl_deploy/corrected_socket` |
| `socket_pose_type` | `debug_pose_item` | |
| `socket_assembly_name` | `cooling_manifold` | live: `cooling_base` |
| `socket_part_id` | `-1` | **blocks start until set** |
| `allow_wildcard_part_id` | `false` | accept `-1` deliberately |
| `socket_opening_offset_cad_xyz` | `[0,0,0]` | rotated by the perceived quaternion |

### Robot / kinematics / FRI
| Parameter | Default |
|---|---|
| `lbr_state_topic` | `state` |
| `require_fri_state` | `false` (live: `true`) |
| `fri_state_max_age_s` | `0.2` (live: `0.05`) |
| `expected_control_mode` | `-1` (any); live: `1` (Cartesian impedance) |
| `joint_state_topic` | `/lbr_dual_arm/joint_states` |
| `lbr_command_topic` | `command/joint_position` (live: `/rl_deploy/command_15hz`) |
| `flange_pose_topic` | `/right/ee_pose` |
| `fingertip_source` | `fk` |
| `robot_prefix` / `base_link` / `tip_link` | `lbr_two` / `lbr_two_link_0` (live `base_link`) / `lbr_two_gripper_tcp` |
| `fallback_tip_link` | `lbr_two_link_ee` |
| `flange_to_fingertip_xyz` / `_quat_wxyz` | `[0,0,0.1463]` / `[1,0,0,0]` |
| `robot_description` / `_service` | `""` / `/lbr_dual_arm/robot_state_publisher/get_parameters` |

### Force
| Parameter | Default | Note |
|---|---|---|
| `force_source` | `arm_external_torque` | live: `wrench_topic` |
| `ft_sign` | `1.0` | live `-1.0`; ignored by `wrench_topic` |
| `ft_bias_base_xyz` | `[0,0,0]` | sim gravity-inclusive DC restoration |
| `ft_baseline_on_start` | `true` | one-shot capture at `start_policy` |
| `ft_topic` / `ft_frame` | `/wrist_ft` / `base` | live: broadcaster wrench / `flange` |
| `ft_smoothing_factor` | `0.25` | observation-only EMA |
| `force_noise_std` | `0.0` | |
| `zero_force_obs` *(we_a path)* | `false` | live: `true` |

### Action and IK
| Parameter | Default | Note |
|---|---|---|
| `ema_factor` | `0.0625` | mid of trained DR range `[0.025, 0.1]` |
| `e2e_pos_action_scale` | `0.01` | live: `0.005` (must match the run) |
| `e2e_rot_action_scale` | `0.1` | |
| `max_policy_position_step_m` | `0.0` (off) | live: `0.001` |
| `max_policy_rotation_step_rad` | `0.0` (off) | live: `0.00872665` (0.5°) |
| `socket_action_bound` | `0.08` | live: `0.05` |
| `ik_damping` | `0.1` | DLS λ |
| `max_joint_step_rad` | `0.010` | live: `0.0045` |
| `joint_limit_margin_rad` | `0.03` | |
| `z_force_limit_enable` / `_n` | `false` / `12.0` | live: `true` / `6.0` |

### Preinsert
| Parameter | Default |
|---|---|
| `preinsert_mode` | `joint` (live: `socket_hover`) |
| `preinsert_hover_z_m` | `0.06` (live: `0.042`) |
| `preinsert_pos_tol_m` | `0.005` |
| `preinsert_joint_position` / `_set` | `[0]*7` / `false` |
| `preinsert_tolerance_rad` | `0.01` |
| `preinsert_max_joint_step_rad` | `0.006` |

### Safety, seating, trial
| Parameter | Default | Note |
|---|---|---|
| `input_timeout_s` / `socket_timeout_s` | `0.35` / `0.75` | |
| `ft_force_cap_n` | `20.0` | w2: `25.0`; live: **`15.0`** |
| `ft_warn_n` | `10.0` | live: `8.0` |
| `seat_force_n` | `14.0` | |
| `trial_stop_mode` | `manual` | `auto_seat` re-enables geometry gates |
| `seat_socket_depth_m` | `0.0175` | opening z=0, bottom −depth |
| `seat_z_tolerance_m` / `seat_xy_tolerance_m` | `0.0045` / `0.004` | ~25 % of depth |
| `seat_force_requires_geometry` | `true` | force above the bottom zone ⇒ jam |
| `trial_timeout_s` | `10.0` | |
| `trial_max_overtravel_m` | `0.002` | |
| `seat_depth_roi_max_m` | `0.0` (off) | central-ROI median depth trigger |
| `freeze_on_seat` | `true` | |
| `open_gripper_on_seat` / `gripper_open_topic` / `_value` | `false` / `/gripper/open_cmd` / `0.04` | |
| `publish_debug_obs` / `obs_dump_path` | `false` / `""` | live: `true` / `""` |

### we_A / pdz-only parameters
`socket_depth_m` `0.0175` · `socket_axis_mode` `base_axis` · `socket_axis_base_xyz` `[0,0,1]` ·
`screw_tip_offset_tcp_xyz` `[0,0,0.030]` · `shaft_axis_tcp_xyz` `[0,0,-1]` ·
`we_a_geometry_calibrated` · `we_a_force_bias_calibrated` · `we_a_axis_guard_deg` `60.0` ·
`e2e_action_safety_box` `0.05` · `e2e_yaw_action_bound` `1.5708` · `zero_force_obs`.

## 8.2 Configuration profile inventory

| File | Purpose |
|---|---|
| `deploy.yaml` | Base E2E profile |
| `deploy_w2_estimator.yaml` | Explicit-estimator (`w2_estimator_192`) — README default |
| `deploy_we_a.yaml` | we_A ResNet-18 baseline (ep 2500), 15-D/RGB |
| `deploy_pdz_ep500/1000/1500.yaml` | pdz overnight, three checkpoints (reward 183.7 / 151.5 / 144.4) |
| `deploy_pdz_v3_fulltilt.yaml` | pdz v3, dual-arm bringup |
| **`deploy_pdz_v3_fulltilt_single.yaml`** | **LIVE single-arm right-only profile** |
| `deploy_pdz_v3_fulltilt_single_LEFT.yaml` | left-arm variant |
| `preinsert.yaml` | MoveIt gross preinsert |
| `hole_align.yaml` | Hole detector + vision-corrected preinsert + anchor republisher |
| `scene_collisions.yaml` | MoveIt planning-scene collision objects |

Launch files: `deploy_inference.launch.py`, `..._w2_estimator`, `..._we_a`, `..._pdz`,
`preinsert_planner.launch.py`, `hole_align_planner.launch.py`.

## 8.3 Experimental data assets in the repository

`deploy_analysis/`:
- `pdz_single_20260827_{162621,163807,165249,171512}_force.csv` — `t, fx, fy, fz, fmag`
  (3875 / 7161 / 3600 / 10635 lines incl. header)
- `..._policy.csv` — `t, gdx, gdy, gdz, a0..a5, is_drop` (151 lines each = 150 policy ticks)
- `..._drop.txt` — drop time in seconds (1.111 / 3.592 / 1.820 / 2.675)
- `PLOT_*.png`, `ZOOM1_*.png`, `PLOT_SUMMARY_aligned.png`, `PLOT_ZOOM_grid.png`
- `make_plots.py`, `make_zoom.py`, `make_single_zoom.py`

Root-level imagery: `compare_real_vs_sim.png`, `force_compare.png`, `force_compare_all3.png`,
`trial3.png`, `policy_image_320x180.png`, `policy_image_raw_full.png`, `sim_frame_{2,20,38}.png`,
`sim_wrist_at_pose.png`, `sim_wrist_rollout.mp4`, `raw_rgb_*.png`, `depth_*.png`, `holes_*.png`.

`outputs/`: `actor_rgb.png`, `actor_depth.png`, `sim_actor_rgb{,2}.png`, `sim_actor_depth.png`,
`deploy_wrist{,_derot}.png`, `hole_estimation.png`, `hole_align_20260813_082424.png`,
`real_detected.png`, `explicit_estimator_bag_plot.png`, `sim_wrist_{closeup,production,ymount}.png`,
`sim_kuka_ref.png`, `cam_crop_sample_crop.png`, `dep_fincrop.png`, `depth_{before,after}.png`.

These are the raw assets for thesis figures: real-vs-sim wrist views, actor input channels, hole
detection overlays, force comparisons, and the deploy-run plots.

## 8.4 Git history

Branch `moreno` (main: `master`), 8 commits — the engineering record lives largely in the markdown
notes and in-code comments rather than in commit granularity.

| Hash | Date | Subject |
|---|---|---|
| `69814b1` | 2026-08-11 | RL insertion-policy inference + MoveIt preinsert planner (right arm) |
| `1343ec8` | 2026-08-13 | Vision correction for preinsert + corrected base estimation for more precise hover above hole before starting policy |
| `49be7d7` | 2026-08-18 | Visual servo script and F/T correction |
| `3c67529` | 2026-08-24 | add W2 estimator deploy config and preinsert targeting |
| `63bdf87` | 2026-08-25 | Deploy adjustments to newest checkpoint version |
| `09f495f` | 2026-08-27 | Right robot only deployment code |
| `773dbb8` | 2026-08-28 | Md files for diagnosis and fri drop problems |
| `96804e8` | 2026-08-28 | Julien deploy adaption gates |

Uncommitted: 12 modified files (+678/−113) + 4 untracked paths (§6.8).

## 8.5 Project chronology (deployment phase)

| Date | Event |
|---|---|
| 2026-08-05 | `e2e_weld_curric` done — **85.2 %**, beats res224 (65 %), verified at full 18° tilt. D405 CAD `T_flange_cam` arrives |
| 2026-08-10 | Deploy runtime bring-up Steps 0–2: `deploy_env.sh`, node smoke PASS, `dump_sim_obs.py`. **Blocker: `kdl_parser_py`** (B-1). `w2_estimator_192` becomes the deploy checkpoint |
| 2026-08-11 | First commit: inference node + MoveIt preinsert planner |
| 2026-08-13 | Vision correction for preinsert + corrected base estimation |
| 2026-08-18 | Visual-servo comparison script + F/T correction |
| 2026-08-24 | W2 estimator deploy config; `pdz_overnight_20260824` trained (broken gravity comp, B-9) |
| 2026-08-25 | Deploy adjustments to the newest checkpoint; single-arm right bringup |
| 2026-08-26 | `pdz_v3_fulltilt_20260826` trained (corrected gravity comp, halved action scale, 25° tilt). Predeploy checklist written |
| **2026-08-27** | **First hardware deploy runs.** 4 bags; FRI drops at 1.1–3.6 s every time. Single-arm code committed |
| **2026-08-28** | Diagnosis: 110 ms = shutdown artifact; FTEstimator theory disproven at 30 Hz; guarded JTC bridge implemented; Julien cross-check; contact-safe rationale + staged test plan written |
| 2026-08-31 | `fov_match` reversal for we_A (B-12); sim-vs-real image comparison assets produced |
| 2026-09-02 | Last working-tree activity (hold-latch fix, weighted/nullspace IK, SERL backend) |
| **2026-09-14** | This export |

## 8.6 How to reproduce the key checks

```bash
# Build + environment
cd ~/Masterthesis-rl-deploy
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source deploy_env.sh
python3 -c "import rclpy, torch, torchvision, gymnasium, rl_games, fp_debug_msgs, lbr_fri_idl"

# Unit tests (46 pass)
source install/setup.bash
cd src/rl_deploy_inference
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/ -q

# Observation parity: sim dump (on the GPU/Isaac box)
cd ~/Masterthesis-rl-train
OMNI_KIT_ACCEPT_EULA=YES TORCHDYNAMO_DISABLE=1 PYTHONUNBUFFERED=1 \
  ~/miniconda3/envs/isaaclab/bin/python scripts/dump_sim_obs.py \
    --task Isaac-Insertion-CoolingPeg-Iiwa-E2E-Vision-Direct-v0 \
    --headless --enable_cameras --num_envs 1 --warmup 2 --out /tmp/sim_obs.npz

# Observation parity: deploy dump + compare
ros2 param set /rl_deploy_inference obs_dump_path /tmp/deploy_obs.npz
ros2 run rl_deploy_inference obs_parity --sim /tmp/sim_obs.npz --deploy /tmp/deploy_obs.npz

# No-contact force check (the critical gate; arm must be still and clear of contact)
ros2 param set /rl_deploy_inference max_joint_step_rad 0.0
ros2 service call /rl_deploy/start_policy std_srvs/srv/Trigger "{}"
ros2 service call /rl_deploy/stop_policy  std_srvs/srv/Trigger "{}"
python3 -c "import numpy as np; d=np.load('/tmp/deploy_pdz_obs.npz'); print(d['policy'][...,6:9])"
# PASS = force[6:9] ~ 0 on all axes. FAIL = large DC -> STOP, do NOT re-add ft_bias.
ros2 param set /rl_deploy_inference max_joint_step_rad 0.0045
```

**Note on parity scope:** sim and real scenes cannot be matched pixel-for-pixel, so `obs_parity` checks
**structure / units / normalization / ranges** (per-channel image means, depth scaling, the 21-D layout) —
**not exact pixel equality**. `aux_label` is training-only and is excluded from the check.

## 8.7 Terminology

| Term | Meaning |
|---|---|
| **E2E** | End-to-end visuomotor policy: RGB-D + force → 5-DoF delta, no proprio/plan/residual decomposition |
| **Anchor** | The socket-position estimate used both for the observation delta and as the action-box centre |
| **Seat / seated** | The screw tip reaching the socket bottom zone within lateral tolerance |
| **Weld** | Rigid grasp fix in sim (RigidObject + per-env FixedJoint) that removed the dominant grasp confound |
| **Curriculum** | Progressive hardening (tilt, noise) during training — makes reward non-comparable across epochs |
| **`aux_label`** | Privileged training-only label (true hole gap); excluded from the policy input at inference |
| **Explicit estimator** | An aux head predicting the hole gap from the image, whose *prediction* is fed back into the policy |
| **FRI** | KUKA Fast Robot Interface — the real-time UDP link to the controller |
| **`ReceiveMultiplier`** | How many FRI send periods may pass before a client reply is required (1 → every 10 ms) |
| **`open_loop: true`** | LBR position mode where the *published* joint state is the filtered command, not the measurement |
| **JTC** | `joint_trajectory_controller` |
| **Latch** | A sticky safety state requiring an explicit service call to clear |
| **pdz** | The second-generation gripper (vs. the original custom Y-gripper) |

---

*End of export. Generated 2026-09-14 from the `moreno` branch working tree (including uncommitted
changes). All quantitative values are taken from repository documents, configuration files, or were
recomputed directly from `deploy_analysis/*.csv` during generation; Section 7 reproduces all 14 project
documents byte-exactly (verified programmatically).*
