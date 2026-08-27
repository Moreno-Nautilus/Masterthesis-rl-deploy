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

── T-SWITCH JTC → RL CONTROLLER ──

export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/kuka_fri_omar_ws/install/setup.bash
ros2 topic info /lbr_dual_arm_y_gripper/command/joint_position -v   # Publisher count 0
cat > /tmp/lbr_two_position_controller.yaml <<'EOF'
/**/lbr_joint_position_command_controller:
  ros__parameters:
    robot_name: lbr_two
EOF
ros2 run controller_manager spawner lbr_joint_position_command_controller \
  -c /lbr_dual_arm_y_gripper/controller_manager \
  -t lbr_ros2_control/LBRJointPositionCommandController \
  -p /tmp/lbr_two_position_controller.yaml
# ROBOT MAY MOVE — hand on e-stop
ros2 control switch_controllers -c /lbr_dual_arm_y_gripper/controller_manager \
  --activate lbr_joint_position_command_controller \
  --deactivate joint_trajectory_controller --strict
ros2 control list_controllers -c /lbr_dual_arm_y_gripper/controller_manager

── T-UPSAMPLE 15 Hz → 200 Hz bridge (only ONE) ──

export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/kuka_fri_omar_ws/install/setup.bash
source ~/Masterthesis-rl-deploy/install/setup.bash
python3 ~/Masterthesis-rl-deploy/src/rl_deploy_inference/scripts/command_upsampler.py --interpolate

── T-BAG ROSBAG ──

export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/Masterthesis-vision/install/setup.bash
source ~/Masterthesis-rl-deploy/install/setup.bash
mkdir -p ~/deploy_bags
ros2 bag record -o ~/deploy_bags/pdz_single_$(date +%Y%m%d_%H%M%S) \
  /rl_deploy/policy_obs /rl_deploy/command_15hz /lbr_dual_arm_y_gripper/command/joint_position \
  /rl_deploy/corrected_socket /rl_deploy/status /rl_deploy/seat_detected \
  /lbr_dual_arm_y_gripper/joint_states /lbr_dual_arm_y_gripper/force_torque_broadcaster/wrench \
  /realsense_2/camera/color/image_rect /realsense_2/camera/aligned_depth_to_color/image_rect

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
ros2 service call /rl_deploy/start_policy std_srvs/srv/Trigger "{}"   # REFUSED (gated) — fine; obs still dumps
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
#   WATCH: sim shows IMPACT SPIKES 22-65N + a 40-60mm bounce, then recover-and-seat. ft_force_cap_n=42 (raised
#   from 25) allows the bounce through so you can see if the real FRI recovers like sim. A hard stuck-jam still
#   trips at 42. Log outcome + |F|max + did it seat. WEAK LINK = the 3D-printed part (watch for marring).

── STOP / CLEAR / RESTORE MOVEIT ──

ros2 service call /rl_deploy/stop_policy std_srvs/srv/Trigger "{}"
ros2 service call /rl_deploy/clear_latches std_srvs/srv/Trigger "{}"
# back to MoveIt (Ctrl-C T-RL, T-UPSAMPLE, T-ANCHOR first):
ros2 control switch_controllers -c /lbr_dual_arm_y_gripper/controller_manager \
  --activate joint_trajectory_controller \
  --deactivate lbr_joint_position_command_controller --strict

── WHEN FULLY DONE ──

sudo ip addr del 192.170.20.1/24 dev enp3s0
sudo ip addr del 192.170.10.1/24 dev enp3s0
