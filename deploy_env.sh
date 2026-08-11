# shellcheck shell=bash
# Runtime shell for the RL deploy node: ONE interpreter that imports both ROS2 (rclpy + the built
# workspace messages) and the RL stack (torch / rl_games / gymnasium / cv2).
#
# Why this instead of a fresh venv: the checkpoint was produced with an unusual torch build
# (torch 2.11.0+cu128) that is not on PyPI, so reinstalling torch risks load parity. System
# python3.10 is ABI-compatible with ROS Humble (both cp310), so we just point it at the SAME
# packages the training/eval env uses (the isaaclab conda env's site-packages) via PYTHONPATH.
# No download; guaranteed byte-for-byte the same torch/rl_games that saved the .pth.
#
# Usage:
#   source deploy_env.sh          # sets up the shell
#   ros2 launch rl_deploy_inference deploy_inference.launch.py
#
# Override the RL site-packages (e.g. a different conda env) by exporting RL_DEPLOY_PY_SITE first.

_ros_distro="${ROS_DISTRO:-humble}"
_repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_rl_site="${RL_DEPLOY_PY_SITE:-/home/moreno/miniconda3/envs/isaaclab/lib/python3.10/site-packages}"

# 1) ROS base + this colcon workspace (built messages: fp_debug_msgs, lbr_fri_idl).
# shellcheck disable=SC1090
source "/opt/ros/${_ros_distro}/setup.bash"
if [ -f "${_repo_dir}/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "${_repo_dir}/install/setup.bash"
else
  echo "[deploy_env] WARNING: ${_repo_dir}/install/setup.bash not found — run 'colcon build --symlink-install' first." >&2
fi

# 2) RL stack: prepend the training env's site-packages so system python3.10 sees the exact
#    torch/rl_games/gymnasium/cv2 that produced the checkpoint.
if [ -d "${_rl_site}" ]; then
  export PYTHONPATH="${_rl_site}:${PYTHONPATH}"
else
  echo "[deploy_env] WARNING: RL site-packages not found: ${_rl_site} (set RL_DEPLOY_PY_SITE)." >&2
fi

# 3) The node is launched with plain 'python3' (ABI-matched to ROS). Make that explicit.
export RL_DEPLOY_PYTHON="python3"

echo "[deploy_env] ROS=${_ros_distro} | ws=${_repo_dir} | rl_site=${_rl_site}"
echo "[deploy_env] verify: python3 -c 'import rclpy, torch, rl_games, fp_debug_msgs, lbr_fri_idl; print(\"OK\")'"
