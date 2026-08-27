# FRI ReceiveMultiplier fix — hand this to whoever has Sunrise Workbench

## Why
The deploy PC runs a generic (non-PREEMPT_RT) Linux kernel. Even with the ROS control node at
SCHED_FIFO priority 80 and pinned to dedicated CPU cores, the kernel produces a rare, isolated
~110 ms latency spike roughly once every 30-60 s (kernel housekeeping / memory / IRQ — unavoidable
without an RT kernel). With the FRI `ReceiveMultiplier = 1`, the robot requires a fresh command
EVERY 10 ms cycle, so a single 110 ms spike (~11 missed cycles) instantly drops the session:

    LBR switched from 'COMMANDING_ACTIVE' to 'MONITORING_READY'
    ERROR_FRI_CMD_WRONG_STATE_ACTIVE

Verified from deploy rosbags: physical forces are tiny (<=10 N), the 200 Hz command stream never
stalls, and the drop always coincides with a single ~110 ms freeze of joint_states (the FRI read),
not force/torque. So the fix is to let the FRI TOLERATE a late command.

## The change (one line)
In the Sunrise project, open the FRI application `LBRServer_select.java`. Find the FRI configuration:

    FRIConfiguration friConfiguration =
        FRIConfiguration.createRemoteConfiguration(lbr, clientName);
    friConfiguration.setSendPeriodMilliSec(10);
    friConfiguration.setReceiveMultiplier(1);     // <-- change this

Change to:

    friConfiguration.setReceiveMultiplier(3);     // tolerate late commands (30 ms window)

Then **Synchronize** the project to the controller (Workbench -> Synchronize).

## Notes
- SendPeriod stays 10 ms; the robot still runs its internal loop at 10 ms. ReceiveMultiplier only
  relaxes how often the external ROS client must reply: with 3, a command is required every 30 ms.
- 3 covers the common small jitter. The rare 110 ms spike is still > 30 ms, so if drops persist,
  bump to 5 (50 ms) or higher. Trade-off: a larger multiplier means the arm gets a fresh command
  less often, i.e. coarser control. Start at 3, raise only if needed.
- This is safe: it does not change robot speed, limits, or the control mode (still POSITION).
- The real robust fix would be a PREEMPT_RT kernel on the deploy PC (eliminates the spike entirely),
  but ReceiveMultiplier is the small, low-risk change that unblocks deploy runs now.

## After the change
Deploy runs should hold COMMANDING_ACTIVE for the full trial, so the policy can be judged/tuned
(current status: in ~10 s it descended ~16 mm and closed goal distance 32 -> 21 mm but did not seat
before the FRI dropped — need uninterrupted runs to tune convergence).
