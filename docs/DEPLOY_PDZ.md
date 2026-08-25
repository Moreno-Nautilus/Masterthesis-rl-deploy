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
