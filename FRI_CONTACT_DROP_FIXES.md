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
