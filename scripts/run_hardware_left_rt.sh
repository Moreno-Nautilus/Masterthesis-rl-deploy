#!/usr/bin/env bash
#
# Launch the right-arm FRI bringup and give the ros2_control_node REAL-TIME priority so a
# CPU/GPU/rosbag burst cannot preempt its FRI packet-read thread.
#
# WHY: deploy run #1 (pdz_20260827_154207) dropped COMMANDING_ACTIVE -> MONITORING_READY after the
# FRI packet-read stalled 110 ms (joint_states/wrench froze while the 200 Hz command output kept
# flowing -> it was a READ-side preemption, not force/torque: peak |F|=10.6 N, peak torque 71% A4).
# SCHED_FIFO on ros2_control_node lets its read loop preempt normal-priority bursts.
#
# This does NOT edit the launch file or any controller yaml. It just re-schedules the process that
# hardware_left.launch.py already starts, once it appears.
#
# Pair it with DDS-off-the-FRI-NIC + domain isolation (the primary fix):
#   export ROS_DOMAIN_ID=42
#   export FASTRTPS_DEFAULT_PROFILES_FILE=~/Masterthesis-rl-deploy/deploy_dds_no_fri_nic.xml
#
# Usage:
#   scripts/run_hardware_left_rt.sh              # launch + auto-RT the control node
#   RT_PRIO=90 scripts/run_hardware_left_rt.sh   # override SCHED_FIFO priority (default 80)
#
# chrt needs privilege. If you cannot run rtprio unprivileged (ulimit -r), this falls back to the
# highest nice it can and prints a warning; run once with sudo -v beforehand, or add an rtprio limit
# in /etc/security/limits.conf for your user to make it work without sudo.

set -euo pipefail

RT_PRIO="${RT_PRIO:-80}"

export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
# ROS setup.bash references unbound vars (AMENT_TRACE_SETUP_FILES etc.); relax nounset while sourcing.
set +u
source /opt/ros/humble/setup.bash
source "$HOME/kuka_fri_omar_ws/install/setup.bash"
set -u

# CPU layout: give the FRI control node dedicated cores with nothing else on them, and push the
# heavy vision load (RealSense driver + rectify container) onto the OTHER cores. The 110 ms FRI-read
# stall that drops COMMANDING_ACTIVE happens when the FRI read thread and the ~21% RealSense rectify
# container land on the same core during a burst. Pinning them apart removes that collision, which
# RT priority alone did not (SCHED_FIFO doesn't help if they share a core and the busy one is also RT).
FRI_CORES="${FRI_CORES:-0-3}"          # dedicated to the FRI control node
VISION_CORES="${VISION_CORES:-8-31}"   # everything heavy pushed here
RUN() { if ! "$@" 2>/dev/null; then sudo -n "$@" 2>/dev/null; fi; }  # try unprivileged, fall back to sudo -n

# Background watcher: as soon as the nodes exist, RT-prioritize + CPU-pin them.
(
  echo "[rt] waiting for ros2_control_node to start..."
  DONE_CTRL=0; DONE_VIS=0
  for _ in $(seq 1 240); do
    PID=$(pgrep -f "ros2_control_node" | head -1 || true)
    if [ -n "${PID:-}" ] && [ "$DONE_CTRL" = 0 ]; then
      # pin ALL threads of the control node to the dedicated FRI cores
      RUN taskset -a -cp "$FRI_CORES" "$PID" && echo "[rt] ros2_control_node (pid $PID) pinned to cores $FRI_CORES"
      # SCHED_FIFO on every thread (the FRI read + FTEstimator run in separate threads)
      if RUN chrt -a -f -p "$RT_PRIO" "$PID"; then
        echo "[rt] ros2_control_node -> SCHED_FIFO prio $RT_PRIO (all threads)"
      else
        RUN renice -n -20 -p "$PID"
        echo "[rt] WARN: no rtprio privilege; nice -20 fallback. For true RT: 'sudo -v' first."
      fi
      DONE_CTRL=1
    fi
    # push the RealSense driver + rectify container off the FRI cores
    if [ "$DONE_VIS" = 0 ]; then
      for vp in $(pgrep -f "realsense2_camera_node|component_container" 2>/dev/null); do
        RUN taskset -a -cp "$VISION_CORES" "$vp" && echo "[rt] vision pid $vp -> cores $VISION_CORES"
        DONE_VIS=1
      done
    fi
    [ "$DONE_CTRL" = 1 ] && [ "$DONE_VIS" = 1 ] && break
    sleep 0.5
  done
  echo "[rt] setup done (ctrl=$DONE_CTRL vision=$DONE_VIS)."
) &

echo "[rt] launching hardware_left.launch.py ..."
exec ros2 launch lbr_dual_arm_y_gripper_bringup hardware_left.launch.py
