# G1 hardware notes

## Open issues on the G1

1. **Link-loss watchdog: added 2026-11-25, not yet run on the robot.**
   - On 2026-11-15 the ethernet link dropped twice (a loose cable), and once the robot fell mid-pivot.
     When `rt/lowstate` stopped arriving, `hardware.py` kept publishing the last state as if it were new,
     and on **reconnection** it resumed sending policy commands (the robot "came back to life").
   - Now: if no new `lowstate` arrives for `LOW_LEVEL_COMM_TIMEOUT` (50 ms), in any FSM state,
     `hardware.py` sets the same latch as the tilt safety, prints a SAFETY banner and writes damp until
     it is restarted, even after the link returns. In 156 min of logs the state never froze for more
     than 25 ms, so 50 ms should not trip on a healthy link.
   - While the link is down, the robot can't receive commands either, so the PC still can't prevent
     the fall itself.
   - Still open: port the Go2's command timeout (damp if the controller stops sending).
2. **A stale command is written on `control` re-entry.**
   - `hardware.py` applies the last stored command on the first tick of `control`/`track`. After
     `control → damp → home → control`, the first tick wrote a **14.3 s old** command (measured on the
     robot).
   - Harmless if you left `control` while standing, but **not if you pressed damp mid-stride**, which is
     the usual emergency case. Fix: reset the stored command to the current pose on each FSM transition.
3. **`control_29dof_vel.py` runs the policy before `control`.**
   - It has no FSM subscription, so it steps in every state and feeds back its own unapplied actions. On
     the robot its first `control` command was **2.67 rad** off the measured pose (right ankle pitch).
     Every other G1 controller is gated and zeroes its action outside `control`/`track`.
   - Fix: gate on FSM state, as the other controllers do.
   - (Separately, that policy alternates its waist-roll command at about 22 Hz on hardware. That's a
     policy problem, not timing.)
4. **Mimic `track` entry can still read a stale `fsm_time`, but now rarely.**
   - The controller gets the FSM state from the joystick, while `fsm_time` comes from `hardware.py`.
     With the old free-running timer, the first `track` step often used the time spent in `control`
     (frame 174 of 213 in 6 of 7 bench launches).
   - With the `fsm_time` trigger, the window is sub-millisecond, and all 4 entries on the robot were clean.
     The full fix is to have `hardware.py` publish the FSM state it acted on.
5. **The logger misses messages; don't use its rows to measure rates.** `deploy/logger/log.py` samples
   the *latest* message of each topic on its own unsynchronized timer, so it can miss or duplicate
   messages. (On the Go2, logged `hardware_time` suggested 157 Hz when the loop was really at 200 Hz.)
6. **A power loss leaves an empty log** (seen on the Go2: h5 at 0 bytes). The fix is `os.fsync` after each
   periodic dump in `utils/logger.py`. On the G1 this matters if the onboard mini PC can lose power with
   the robot.
7. **Delay randomization.** Every G1 mjlab task randomizes actuator delay over 0–40 ms (`min_delay_sec=0`),
   so the new ~2 ms latency (plus the motor response) is inside the trained range. On the Go2, a
   delay-randomized policy and the latency fix together reduced sway the most.
8. **TODO: full-body joint calibration on the current G1** (the unit in use since 2026-09-28 ~21:00).
   - **Symptom:** the right foot doesn't sit flat while standing: it is rolled +3 to +5° (forward
     kinematics of the joints plus the pelvis IMU), while the left is at about −0.5°.
   - **Cause:** the right ankle roll reads about +0.05 rad high. The policy "corrects" a tilt that isn't
     there and pushes the real foot onto its edge with 2–5 Nm. The left ankle holds about 0 Nm.
   - **Not the policy:** this happens on every run on this unit, with every policy (walk v0/v1, backpack,
     jog), from the first second after the policy engages. Sim is symmetric. The previous unit's logs
     (up to 2026_09_28__20_51) are flat.
   - **This unit's waist pitch reads with the opposite sign**, which is why `--flip-waist-pitch` defaults to on.
     Re-check that flag after calibrating.
   - **To check after calibrating:** hang the robot in `home` and compare both soles with an inclinometer.
     In the logs, the standing right/left sole roll should match within about 1°.

## Pelvis IMU backup

When a pelvis IMU packet is lost (zero quaternion), `hardware.py` fills in the pelvis orientation and
angular velocity from the torso IMU and the three waist joints, using `utils/sensor_guard_utils.py`. Nothing is
fitted to data. The accelerometer is not backed up and keeps its last value. This is part of
`--imu-guard` (on by default): a lost torso IMU packet holds its last values, a lost pelvis IMU packet
gets this estimate. If the torso IMU is lost at the same time, the pelvis IMU holds too.
`--no-imu-guard` passes lost packets through as they arrive.

### Frames and inputs
- $R_p$, $R_t$: pelvis and torso orientation from their IMUs (body frame to that IMU's world frame).
- $q_y, q_r, q_p$: waist yaw, roll, pitch (model sign convention), after the joint dropout guard.
- $\omega_t$: torso gyro, in the torso frame.

The waist chain is yaw (z), then roll (x), then pitch (y), so the rotation taking torso-frame vectors
into the pelvis frame is

$$R_{pt} = R_z(q_y)\,R_x(q_r)\,R_y(q_p)$$

### Orientation
The pelvis pose implied by the torso IMU is

$$R_{fk} = R_t\,R_{pt}^{-1}$$

It differs from the real pelvis IMU in two ways: the two IMUs have unrelated headings, and there is a
small tilt error (about 2 deg while moving). While the pelvis IMU is healthy, both are measured every
tick. With $D = R_p\,R_{fk}^{-1}$ and its quaternion $(w, x, y, z)$:

$$\psi = 2\,\mathrm{atan2}(z, w) \qquad C = R_{fk}^{-1}\,R_z(-\psi)\,R_p$$

$\psi$ is the heading offset (the part of $D$ about the vertical) and $C$ is the leftover tilt, in the
pelvis frame. During a dropout both stay frozen at their last healthy values $\psi_0$, $C_0$ and are
applied to the live torso-implied pose:

$$\hat R_p = R_z(\psi_0)\,R_{fk}\,C_0$$

At the last healthy tick this equals $R_p$ exactly, so the estimate starts with zero error and only
drifts as far as the torso-to-pelvis relation changes during the dropout. Roll, pitch, yaw are taken
from $\hat R_p$ with `quat_to_rpy`.

### Angular velocity
The torso's angular velocity is the pelvis's plus what the waist joints add. Solving for the pelvis,
in the pelvis frame:

$$\hat\omega_p = R_{pt}\,\omega_t \;-\; \left( \dot q_y\,\hat z \;+\; R_z(q_y)\,\dot q_r\,\hat x \;+\; R_z(q_y)\,R_x(q_r)\,\dot q_p\,\hat y \right)$$

The waist rates $\dot q$ are position differences between consecutive ticks, which gave a lower error
than the reported joint velocities.

### How well it works (measured on logs, 2026-11-19)
Against 557 real dropouts in five jog/run/sprint logs, comparing each method with the first fresh pelvis
sample when the dropout ends:

| Dropout length | Tilt error, backup / hold | Gyro error, backup / hold |
|---|---|---|
| 10 ms | 0.35 / 1.02 deg | 0.55 / 0.85 rad/s |
| 20 ms | 0.46 / 1.44 | 0.75 / 1.39 |
| 30-50 ms | 0.90 / 3.18 | 0.85 / 2.10 |
| 60-90 ms | 1.34 / 8.21 | 2.36 / 4.59 |
| 100 ms and up | 3.27 / 14.63 | 1.24 / 4.99 |

Not yet run on the robot.

## Joint dropout guard

When the robot blanks a motor record, `hardware.py` fills in that joint's position and velocity with
`JointDropoutGuard` in `utils/sensor_guard_utils.py`. Nothing is tuned or fitted. Torque and temperature
pass through untouched, so a dropout stays visible in the log. Turn it off with `--no-joint-guard`.

### Detecting a lost record
A lost record reads exactly 0 on position, velocity, torque and temperature (the ankles read a tiny
non-zero position). The test is **both temperature sensors read 0 degC**, which no working motor does.
A motor is only guarded once it has reported a real temperature, so a dead sensor never triggers it.
Across 155,510 ticks of logs this flagged exactly the lost samples and nothing else.

### Filling it in
With $q_0$, $\dot q_0$ the last good reading of that joint, taken at time $t_0$:

$$\hat q(t) = \mathrm{clip}\big(q_0 + \dot q_0\,(t - t_0),\; q_{min},\; q_{max}\big) \qquad \hat{\dot q}(t) = \dot q_0$$

The limits come from the model named by the config's `xml_path`.

### How well it works (measured on logs, 2026-11-20)
110 real leg and waist-yaw dropouts, compared with the first fresh reading when each ends:

| | Position error (rms) | Velocity error (rms) |
|---|---|---|
| No guard (zeros reach the policy) | 0.537 rad | 5.54 rad/s |
| Hold the last reading | 0.190 | 4.46 |
| This guard (constant velocity) | 0.134 | 4.46 |

It beats holding up to about 100 ms and is worse beyond that (the joint has reversed by then); the
longest joint dropout seen so far is 90 ms. There is no time limit on the extrapolation other than
the joint-limit clamp. Not yet run on the robot.
