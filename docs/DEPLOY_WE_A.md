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
