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
