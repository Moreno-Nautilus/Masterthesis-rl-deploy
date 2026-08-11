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
