#!/usr/bin/env python3
"""Compatibility wrapper for the installed command_upsampler executable.

Prefer ``ros2 run rl_deploy_inference command_upsampler``. The source-tree path remains usable for
older operator commands.
"""
from rl_deploy_inference.command_upsampler import main


if __name__ == "__main__":
    main()
