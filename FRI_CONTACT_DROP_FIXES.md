# FRI drops on contact — diagnosis + fix ladder (2026-08-28)

## Confirmed problem (4 bags, 100% consistent)
The FRI session drops (COMMANDING_ACTIVE -> MONITORING, ERROR_FRI_CMD_WRONG_STATE_ACTIVE) with a
single ~110 ms robot->PC joint_states freeze, landing EXACTLY on the first contact force step
(0 -> 10-12 N) in every run. PC keeps publishing commands at 200 Hz through the gap (PC side fine).
Tracking error stays ~2 mrad (robot tracks fine). Motion is slow (~0.05 rad/s) so it's the FORCE
event, not speed. Pendant shows Quality POOR / Jitter 3.4 ONLY at contact. ReceiveMultiplier=1 =
zero tolerance for the 110 ms gap -> instant drop.

## ⚠️ UPDATE 2026-08-28 — FTEstimator theory DISPROVEN by test
Ran the deploy with FTEstimator at 30 Hz (CONFIRMED live: wrench value-change rate measured 26 Hz in
bag `pdz_single_30hz_20260828_140335`). **It STILL dropped at contact**, identical signature: 110 ms
stall at policy tick 18/150, |F| went 4→1→6→12 N into the drop, command stream fine (5 ms) = PC fine,
robot-side freeze. Force-threshold=0 also applied, no effect. => The async-FTEstimator-contention root
cause below is WRONG. The drop is a ROBOT-SIDE reaction to the contact force transient, independent of
PC RT/pinning/CPU/estimator-rate (all tried, none fixed it). The ONLY remaining fix that addresses the
actual mechanism is **ReceiveMultiplier 1→3 on the pendant** (fix #2). Do NOT spend more on PC-side
estimator tuning. Optional PC-side long-shot: lower e2e_pos_action_scale 0.005→0.003 to shrink the
0→12 N transient (untested, low confidence).

## Root cause (ORIGINAL THEORY — now disproven, see UPDATE above)
The async `lbr_fri_ros2::FTEstimator` runs its own thread at update_rate=100 Hz / rt_prio=30, next to
the FRI send loop (rt_prio=80) on the same PC. It's idle in free motion (forces < 2 N thresholds); at
contact it computes a damped Jacobian pseudo-inverse EVERY 100 Hz cycle -> contends with the FRI loop
-> the 110 ms hiccup. Upstream (lbr-stack) REMOVED this exact async worker in jazzy-v2.4.0 and replaced
it with a SYNCHRONOUS WrenchEstimator + EstimatedWrenchInterface chainable controller — for this reason.
Our stack: humble-v2.2.3, still the async FTEstimator. ros2_control = 2.54.0 (recent; HAS
ChainableControllerInterface).

## Constraint: the policy USES force
3 of 4 bags show non-zero policy force obs; sim trains WITH force in the 15-D vector
[goal_delta(6), force(3), prev_action(6)]. So we CANNOT disable the estimator — force must keep
flowing. The policy reads force at 15 Hz, so any estimator rate >= ~20 Hz feeds it fully.

## Fix ladder (cheap -> expensive) — try in order
1. **FTEstimator 100 -> 30 Hz** [DONE 2026-08-28, applied, no rebuild — symlink install]. Edited both
   lbr_two_system_config.yaml (right) and lbr_one_system_config.yaml (left). 30 Hz >> 15 Hz policy,
   < 100 Hz controller_manager. Keeps force + safety + seat-detect. TEST THIS FIRST.
   - If still drops: try 20 Hz (still > 15 Hz policy).
2. **ReceiveMultiplier 1 -> 3** on the pendant (Sunrise LBRServer_select.java, setReceiveMultiplier(3)).
   One line, needs Sunrise Workbench (another person). Lets the session survive the 110 ms gap
   regardless of cause. See ../FRI_RECEIVE_MULTIPLIER_FIX.md.
2b. **PC-SIDE ESTIMATION (the ace — likely better than the backport)**: the deploy node ALREADY
   computes the wrench from raw external torque PC-side: `wrench_from_external_torque(J, tau_ext)` =
   `pinv(J^T)*tau_ext` (ik.py:94), selectable via `force_source: external_torque` (uses the FRI
   `external_torque` state interface directly, published by the existing lbr_state_broadcaster). This
   BYPASSES the hardware async FTEstimator entirely -> no in-FRI-thread contention -> no contact drop,
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
- PC RT/jitter: already fixed (SCHED_FIFO 80 + core pinning + killed a stray Isaac hog); free-motion
  jitter is ~0. The remaining drop is contact-triggered robot-side, not PC.
- Kernel latency spike: no hung-task/stall events; PC command stream never gaps.
