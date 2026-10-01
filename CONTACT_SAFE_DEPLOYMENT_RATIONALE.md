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
