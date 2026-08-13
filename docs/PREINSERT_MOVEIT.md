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
