#!/usr/bin/env bash
# Combined GROSS + VISION-CORRECTED preinsert in one go. Prompts you to type MOVE TWICE:
#   1) the gross MoveIt move to hover over the PERCEIVED socket (preinsert_planner, local-IK joint)
#   2) the vision-corrected move to hover EXACTLY over the DETECTED hole (hole_align_planner)
#
# Both run on the joint_trajectory_controller / MoveIt (no controller switch between them). Run this
# BEFORE the MoveIt->RL controller switch. Each step shows its plan + report first, then waits for you
# to type MOVE (Ctrl-C or anything else aborts and nothing moves).
#
# Usage:
#   ARM=left  ./preinsert_and_correct.sh      # left arm  (default) -> realsense_1 / lbr_one
#   ARM=right ./preinsert_and_correct.sh      # right arm           -> realsense_2 / lbr_two
# Override the hover height (matches the gross preinsert so step 2 is a lateral re-centering):
#   HOVER_Z=0.063 ARM=left ./preinsert_and_correct.sh
#
# Requires (already up from the runbook): hardware.launch.py, move_group.launch.py, the wrist D405,
# and (optionally) the perception pipeline. SAFETY: A1 mastered, physical e-stop in hand.

# NOTE: no `-u` -- the ROS/ament setup.*.sh scripts reference unbound vars (AMENT_PYTHON_EXECUTABLE).
set -eo pipefail

ARM="${ARM:-left}"
HOVER_Z="${HOVER_Z:-0.063}"            # gross preinsert hover above the PERCEIVED socket
CORR_HOVER_Z="${CORR_HOVER_Z:-0.05}"  # corrected preinsert hover above the DETECTED hole (5 cm)
# Orientation of the tool at preinsert. "fixed" + FIXED_QUAT pins the wrist YAW so the wrist image
# matches sim (the policy's hole estimator is not yaw-invariant). [1,0,0,0] xyzw = sim nominal
# (tool straight down, tool_x along base +x). Rotate the yaw here to match the wrist image to sim.
ORIENT_MODE="${ORIENT_MODE:-fixed}"
FIXED_QUAT="${FIXED_QUAT:-[1.0, 0.0, 0.0, 0.0]}"
# Allow a large wrist turn on the GROSS step (pinning yaw can be a ~2 rad A7 rotation).
MAX_DELTA="${MAX_DELTA:-2.5}"
# Orientation tolerance: A7's joint limit can leave the achievable yaw ~10 deg off sim-nominal -- still
# well inside the policy's +/-45 deg trained yaw band, so accept it rather than failing LOCAL_IK.
ORI_TOL="${ORI_TOL:-0.30}"
if [ "$ARM" = "right" ]; then RS=realsense_2; else RS=realsense_1; fi

export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
source /opt/ros/humble/setup.bash
source ~/kuka_fri_omar_ws/install/setup.bash
source ~/Masterthesis-vision/install/setup.bash 2>/dev/null || true
source ~/Masterthesis-rl-deploy/install/setup.bash

PARAMS="$(ros2 pkg prefix rl_deploy_inference)/share/rl_deploy_inference/config/hole_align.yaml"

echo
echo "=================================================================="
echo " STEP 1/2 - GROSS preinsert (${ARM} arm) : hover over PERCEIVED socket"
echo "            You will be prompted to type MOVE before it moves."
echo "=================================================================="
ros2 run rl_deploy_inference preinsert_planner --arm "$ARM" --execute --ros-args \
  -p socket_timeout_s:=300.0 \
  -p orientation_mode:="$ORIENT_MODE" \
  -p fixed_orientation_xyzw:="$FIXED_QUAT" \
  -p orientation_tolerance_rad:="$ORI_TOL" \
  -p hover_z_m:="$HOVER_Z" \
  -p target_mode:=local_ik_joint \
  -p local_ik_max_total_delta_rad:="$MAX_DELTA" \
  -p max_plan_joint_delta_rad:="$MAX_DELTA"

echo
echo "=================================================================="
echo " STEP 2/2 - VISION-CORRECTED preinsert (${ARM} arm, ${RS})"
echo "            Localizes the hole in the wrist cam and re-centers on it."
echo "            You will be prompted to type MOVE again."
echo "=================================================================="
ros2 run rl_deploy_inference hole_align_planner --arm "$ARM" --execute --ros-args \
  --params-file "$PARAMS" \
  -p rgb_topic:=/${RS}/camera/color/image_rect \
  -p depth_topic:=/${RS}/camera/aligned_depth_to_color/image_rect \
  -p camera_info_topic:=/${RS}/camera/color/camera_info \
  -p socket_timeout_s:=300.0 \
  -p orientation_mode:="$ORIENT_MODE" \
  -p fixed_orientation_xyzw:="$FIXED_QUAT" \
  -p orientation_tolerance_rad:="$ORI_TOL" \
  -p max_plan_joint_delta_rad:="$MAX_DELTA" \
  -p local_ik_max_total_delta_rad:="$MAX_DELTA" \
  -p hover_z_m:="$CORR_HOVER_Z"

echo
echo "Both preinsert steps done. Next: switch MoveIt -> RL controller, then start the policy."
