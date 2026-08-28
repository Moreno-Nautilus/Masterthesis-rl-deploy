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
1. **Contact-triggered, not time/CPU/periodic.** Drops happen at wildly different times
   (25 s, 39 s, 59 s, 75 s…) but ALWAYS at the exact tick contact force steps 0 → 10–12 N. In every
   bag the drop coincides with the run's peak force. Force profile into every drop: ~5 N rest → dips
   ~0 → slams 10–12 N → 110 ms freeze.
2. **The 110 ms gap is shutdown, not root cause.** The FRI system interface blocks its controller-manager
   read cycle in a fixed 100 ms close poll after detecting the state transition. The 200 Hz command topic
   is produced by another process upstream of the real FRI UDP loop, so its continuity does not prove the
   client replied on time or that the robot stopped sending.
3. **Slow motion, gentle force.** Joint velocity into contact ~0.05 rad/s; physical |F| never exceeds
   ~12 N. It is the force STEP (0→12 N in ~1 tick), not speed or magnitude, that triggers it.
4. **`ReceiveMultiplier = 1`** (from the pendant FRI config, seen at connect: `SendPeriod 10ms |
   ReceiveMultiplier 1`) requires a reply every 10 ms cycle. Because 110 ms is a shutdown artifact,
   multiplier 3 may provide useful tolerance for an actual one- or two-cycle client delay; packet capture
   is required before assigning a missed-cycle count.
5. **Kernel is generic (PREEMPT_DYNAMIC, not PREEMPT_RT)** — but no hung-task/stall/OOM events logged,
   and the PC command stream never gaps, so this is NOT a PC kernel latency spike.

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

**Conclusion: the async-FTEstimator-contention theory is DISPROVEN.** The drop is independent of PC
RT/pinning/CPU/estimator-rate. It is the robot's own reaction to the contact force transient.

---

## Pendant tolerance experiment (still useful, not a confirmed root fix)
### ReceiveMultiplier 1 → 3 or higher on the Sunrise pendant
In the Sunrise project, `LBRServer_select.java` (or the FRI app config):
```java
friConfiguration.setSendPeriodMilliSec(10);
friConfiguration.setReceiveMultiplier(1);   // -> change to 3
```
Then **Synchronize** to the controller (Sunrise Workbench, Ubuntu-24.04 / Windows laptop — NOT the
deploy PC). This increases tolerance for late client commands. At a 10 ms send period, multiplier 3
represents only about 30 ms, so it does **not** arithmetically cover the observed 110 ms gap by itself.
Record the runtime value, test deliberately, and do not treat this setting as proof of the cause. It does
not change speed, limits, or control mode. Full note: `FRI_RECEIVE_MULTIPLIER_FIX.md`.

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
  deploy node maps BY NAME; command goes out A1..A7 which the controller expects; tracking error ~2 mrad.
  The scramble is cosmetic.
- **Memory leak** — NO (56 Gi free, swap ~0; drops happen in seconds).
- **PC RT / CPU / jitter** — addressed (SCHED_FIFO 80 + pinning + killed hog); free-space jitter ~0.
- **Kernel latency spike** — no hung-task/stall events; PC command stream never gaps.
- **FTEstimator async contention** — DISPROVEN by the 30 Hz test.

---

## Recommended next actions (in order)
1. Deploy through the new guarded JTC bridge with 1 mm applied steps, 0.5 degree rotation steps,
   raw-force 15 N abort, 6 N downward attenuation, and fail-closed FRI checks.
2. Record `/lbr_dual_arm_y_gripper/state`, the JTC trajectory topic, measured joints, raw wrench, and policy
   targets. Confirm the actual Sunrise `control_mode` before setting `expected_control_mode` to 0 or 1.
3. Ask Julien for his live Sunrise startup line containing SendPeriod and ReceiveMultiplier. Test a larger
   multiplier as a separate tolerance experiment if the guarded path still produces the 110 ms gap.

Once the FRI holds through contact, the policy can finally be judged. Prior partial-run behavior
(from bags, before the drop): descends ~16–25 mm toward the socket, closes goal distance ~40→20 mm,
but never centers XY (< ~3.4 mm; hole clearance ~1 mm, no chamfer) and jams on the base top surface —
a separate alignment/hand-eye question addressed in `deploy_analysis/DIAGNOSIS.md`, moot until the
drop is fixed.
