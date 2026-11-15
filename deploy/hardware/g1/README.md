# G1 hardware notes

> **⚠️ Two open timing issues, both fixed on the Go2 (`foxy` branch) but NOT on the G1. Fix both before
> trusting G1 hardware results:**
> 1. [`fsm_time` can run slower than real time](#️-open-issue-hardware_time--fsm_time-can-run-slower-than-real-time)
>    (tick-counting clock). Go2 fix: `3edaad8`.
> 2. [Control loop latency: up to 30 ms of avoidable delay](#️-open-issue-control-loop-latency-up-to-30-ms-of-avoidable-delay)
>    (three unsynchronized timers). Go2 fix: `5265d6c`. **On the Go2 this visibly reduced swaying.**
>
> Fix #1 first: the #2 fix steps the policy off `fsm_time`, so it needs a correct clock.

## ⚠️ Open issue: `hardware_time` / `fsm_time` can run slower than real time

**Status (2026-11-14):** We found and fixed this bug on the **Go2** (`foxy` branch, commit `3edaad8`).
The **G1** `hardware.py` on this branch uses the **same tick-counting clock**, so it can have the same bug.
**Nobody has measured this on the G1 yet, and the fix is not applied here.** Before you trust any G1 gait or
motion timing, run the check in [How to check the G1](#how-to-check-the-g1).

### Summary

`hardware.py` keeps time by adding a fixed `LOW_LEVEL_CONTROL_DT` on every tick of the control loop:

```python
# deploy/hardware/g1/hardware.py, LowCmdWrite()
self.time_ += LOW_LEVEL_CONTROL_DT          # LOW_LEVEL_CONTROL_DT = 0.01 (100 Hz)
...
self.fsm_time = self.time_ - self.fsm_start_time
```

The loop runs in the Unitree SDK's `RecurrentThread`. That timer **drops ticks it misses; it does not catch
up.** Any tick where the loop body runs late (too slow, or waiting for the Python GIL held by another thread)
is skipped. `time_` only advances by the ticks that actually ran, so it falls behind the wall clock. Every
consumer of `fsm_time` slows down by the same factor, with no error or warning.

### What we saw on the Go2

- The Go2 `hardware.py` ran a 500 Hz loop (`dt = 0.002`) on the Jetson Orin NX and decoded `rt/lowstate`
  in an SDK callback.
- Measured on the Orin:
  - decoding one `LowState` message: about 0.65 ms (at 500 Hz in the callback thread)
  - encoding `LowCmd`: about 0.43 ms
  - CRC: about 0.24 ms
  
  The callback thread and the control thread fight for the GIL.
- The loop actually ran at about **150–280 Hz instead of 500 Hz**, with 20–30 ms gaps.
- `fsm_time` therefore ran at about **0.3–0.56× real time**.
- The gait clock in `control_vel.py` is `phase = (fsm_time % gait_period) / gait_period` with
  `gait_period = 0.6 s`, i.e. 1.67 Hz trained. On hardware it stepped at about **0.94 Hz**. The robot looked
  "underdamped" and sluggish, because the policy was being fed a phase that ran too slowly.
- The home ramp (`home_pos_duration`) also took longer in real seconds than configured.

### The SDK timer drops missed ticks (verified)

`unitree_sdk2py/utils/thread.py`, `RecurrentThread.__LoopFunc`:

```python
tfd = timerfd_create(1, 0)                      # CLOCK_MONOTONIC
timerfd_settime(tfd, 0, <interval, interval>)
while not quit:
    target()                                    # one loop body
    os.read(tfd, 8)                             # blocks until next expiry; returns the NUMBER of
                                                # expirations since last read -- which is ignored
```

If the body overruns, several timer expirations pile up. One `read` consumes all of them, and the body runs
**once**. So the number of ticks is always ≤ the number of periods that passed.

To reproduce, run a `RecurrentThread` at `interval=0.01` with a 15 ms target for 3 s of wall time:

```
wall 3.00 s, ticks 200, tick-counted time 2.00 s, ratio 0.67
```

The loop quantizes to 2 periods (20 ms), and tick-counted time runs at 0.67× real time.

### Where the G1 code has the same pattern

All references are to `deploy/hardware/g1/hardware.py` on `dev`, as of `13ab7ea`.

| What | Where | Note |
|---|---|---|
| Loop period | `LOW_LEVEL_CONTROL_DT = 0.01` | 100 Hz, so a 10 ms budget per tick (Go2 had 2 ms) |
| Loop thread | `Start()`: `RecurrentThread(interval=LOW_LEVEL_CONTROL_DT, target=self.LowCmdWrite)` | drops missed ticks |
| Clock | `LowCmdWrite()`: `self.time_ += LOW_LEVEL_CONTROL_DT` | **the bug** |
| `fsm_time` | `LowCmdWrite()`: `self.fsm_time = self.time_ - self.fsm_start_time` | inherits the drift |
| State decode | `LowStateHandler` callback on `rt/lowstate` (+ `TorsoIMUHandler` on `rt/secondary_imu`) | decoded in SDK callback threads, which compete for the GIL |
| Published | `deploy_robot/hardware_time`, `deploy_robot/fsm_time` (ROS timer, 100 Hz) | what the controllers and logger see |

`g1_low_level_example.py` counts ticks the same way (`self.time_ += self.control_dt_`).

**Will the G1 actually drift?** We don't know until it's measured.
- **Less likely than the Go2:** the G1 loop has 5× more time per tick (10 ms vs 2 ms).
- **More likely than the Go2:**
  - The `hg` `LowState_` carries **35** `MotorState_` entries vs 20 for the Go2, so each decode is
    heavier.
  - There's a second subscriber (`secondary_imu`).
  - The loop writes 29 motors of `LowCmd` in a Python `for` loop.
  - `rclpy` spinning runs in the same process.
- **Depends on the computer:** it matters which computer runs `hardware.py`, and what else is running on it
  (controller, joystick, logger).
- **Even small drift matters:** dropping 1 tick in 20 is a 5% slow gait clock.

### Which G1 controllers are affected

| Controller | Uses `fsm_time` for | Effect of a slow `fsm_time` |
|---|---|---|
| `control_29dof_vel.py` | gait phase: `phase = (fsm_time % gait_period) / gait_period` | gait clock runs slow (same failure as the Go2) |
| `control_29dof_mimic.py` | ramp to frame 0: `fsm_time / frame_pos_duration`; motion frame: `int(fsm_time / ctrl_dt)` | ramp takes longer; **reference motion plays in slow motion** while the policy steps at 50 Hz |
| `hardware.py` itself (all controllers) | `home` ramp: `fsm_time / home_pos_duration` | stand-up takes longer than `home_pos_duration` |
| `control_29dof_walk_unicycle*.py`, `control_29dof_jog_unicycle*.py`, `control_29dof_crawl_*.py` | nothing time-critical: the gait clock is `phase_step`, +1 per policy step | **not** affected by the `hardware.py` clock (see the related issue below) |

**Related issue (separate, same class).** The `phase_step` controllers advance their gait one frame per call
of their own `rclpy` timer (`control_dt = 0.02`). rcl timers also skip missed periods instead of catching up.
If a policy step (e.g. an LSTM or history model) takes more than 20 ms, that controller's gait clock slows
down too.

This is not verified on hardware. To check it, measure the controller's real callback rate: log
`time.monotonic()` in `control_callback`, or `ros2 topic hz deploy_robot/command`. It should be 50 Hz.

### How to check the G1

This needs no motion. The robot can hang in `init`/`damp`.

1. Run `hardware.py` as usual, plus the controller you plan to use, the joystick and the logger: the full
   normal process load.
2. In another terminal with the same ROS env and `ROS_DOMAIN_ID`, run this script. It compares
   `hardware_time` against the wall clock every 5 s:

   ```python
   # save as e.g. clock_check.py and run: python clock_check.py
   import time, rclpy
   from std_msgs.msg import Float64
   rclpy.init(); node = rclpy.create_node("clock_check"); s = {}
   def cb(msg):
       now = time.monotonic()
       if "hw0" not in s:
           s["hw0"], s["wall0"], s["last"] = msg.data, now, now
       elif now - s["last"] >= 5.0:
           s["last"] = now
           hw, wall = msg.data - s["hw0"], now - s["wall0"]
           print(f"hardware_time {hw:7.2f} s   wall {wall:7.2f} s   ratio {hw / wall:.3f}", flush=True)
   node.create_subscription(Float64, "deploy_robot/hardware_time", cb, 10)
   rclpy.spin(node)
   ```

   The script was tested against a fake publisher running at 0.5× speed, and it printed `ratio 0.500`.

3. Read the ratio:
   - **≈ 1.000 (within about 0.005):** fine.
   - **< 1:** `fsm_time` runs slow by that factor. Apply the fix below.
4. Also check while a policy is running (`control` state), not only while idle. The load is different.

### The fix (as done on the Go2, commit `3edaad8` on `foxy`)

**1. Required: take time from the wall clock, not from ticks.**

```python
# __init__
self.time_ = 0.0          # [sec] wall-clock time since the control loop started
self.time_start = None    # time.monotonic() at the first control loop tick

# LowCmdWrite(), replacing `self.time_ += LOW_LEVEL_CONTROL_DT`
now = time.monotonic()
if self.time_start is None:
    self.time_start = now
self.time_ = now - self.time_start
```

`time_` stays monotonic and starts at 0, and the logger's `hardware_time` keeps the same meaning. Ticks can
still be dropped, but time no longer drifts.

**2. Recommended: warn when the loop runs slow.** Copy `monitor_loop_rate()` and the `LOOP_*` constants from
the Go2 `hardware.py`. It prints a warning when the measured rate is below 90% of the target, or a gap is
longer than 3× dt (checked every 5 s).

**3. Only if the loop rate is actually too low:** stop decoding every `LowState` in a callback. Instead, take
the latest sample once per tick:

```python
from unitree_sdk2py.core.channel import ChannelFactory
from cyclonedds.qos import Qos, Policy
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic

participant = ChannelFactory()._ChannelFactory__participant   # private attr: re-check after SDK upgrades
self.lowstate_reader = DataReader(participant, Topic(participant, "rt/lowstate", LowState_),
                                  Qos(Policy.History.KeepLast(1)))

def read_low_state(self):
    samples = self.lowstate_reader.take(N=1)
    if samples:
        self.LowStateHandler(samples[0])
```

Call `read_low_state()` at the top of `LowCmdWrite()`.

G1-specific differences from the Go2 version:
- `Start()` waits on `update_mode_machine_`, which `LowStateHandler` sets. The wait loop must call
  `read_low_state()` itself, otherwise it never exits.
- `rt/secondary_imu` (torso IMU) has its own callback. Either leave it as is (it's lighter), or poll it the
  same way.

On the Go2, step 3 (together with dropping the loop from 500 to 200 Hz) gave a steady 200 Hz with ≤ 5 ms
gaps.

### ⚠️ Behavior changes after the fix (read before running on the robot)

If the G1 was drifting, everything timed off `fsm_time` gets **faster in real seconds** after the fix, by
1 / ratio. It now matches the configured values:

- **The `home` stand-up becomes faster.** `home_pos_duration: 3.0` now really takes 3.0 s. If the robot used
  to take e.g. 5 s and you liked that feel, raise `home_pos_duration` (we did this on the Go2: 1.0 → 3.0).
  `hardware.py` asserts `home_pos_duration >= 3.0`.
- **The `mimic` ramp and motion playback** run at the recorded speed. If the policy step is late, frames get
  skipped (`frame = int(fsm_time / ctrl_dt)`) instead of the motion slowing down. That is the correct
  behavior for a time-indexed reference.
- **The `vel` gait clock** runs at the trained `gait_period`.

Anything that was tuned on hardware *while drifting* was tuned against the wrong clock. Re-check it.

---

## ⚠️ Open issue: control loop latency: up to 30 ms of avoidable delay

**Status (2026-11-15):** Found and fixed on the **Go2** (`foxy` branch, commit `5265d6c`). On the Go2,
the robot visibly swayed less afterwards. The **G1** `hardware.py` and every G1 hardware controller on
this branch use the **same structure**, and the G1 loop is 2× slower (100 Hz vs 200 Hz), so the
avoidable delay on the G1 is about **twice as large**. **Not measured or fixed on the G1 yet.**

### Summary

A sensor reading has to pass through three timers before the motors respond, and the timers are **not
synchronized** with each other. At each one, the data waits a random 0 to 1 period for the next tick:

```
robot state ──► [hardware.py control loop, reads state]  (Unitree SDK RecurrentThread)
            ──► [hardware.py ROS timer, publishes sensors]   ← wait 0–1 period (timer 1)
            ──► ROS ──► [controller ROS timer, runs policy]   ← wait 0–1 sensor period (timer 2)
            ──► ROS ──► [hardware.py control loop, writes LowCmd] ← wait 0–1 period (timer 3)
            ──► motor
```

This hurts in two ways:
- **Average delay:** each timer adds half a period on average.
- **Jitter:** the three timer offsets drift against each other, so the total delay varies from step to
  step. Delay randomization in training usually samples one fixed delay per episode, so a delay that
  changes every step is out of distribution even when the average is inside the trained range.

The transport and compute themselves are cheap. Measured on the Go2's Jetson Orin NX (Foxy, Python 3.8):

| Step | Cost |
|---|---|
| One ROS2 hop between processes (`Float32MultiArray`, 60 floats) | about 0.6 ms (1.25 ms round trip, p99 2.1 ms) |
| Building the `Float32MultiArray` | 0.05 ms |
| ONNX policy inference (MLP, CPU) | 0.05 ms |

Nearly all of the avoidable delay is timer waits.

### What we saw on the Go2

- **Symptom:** clear swaying and shaking while stepping, with the body roll rate peaking at about 8–15 Hz.
  The policy's joint targets contained about 5× more 8–15 Hz motion than the joints actually made, which
  points to the policy fighting a delayed loop.
- **How we measured the delay:** we fit the measured motor torque to the PD law,
  `tau_est ≈ Kp·(q_des(t − d) − q) − Kd·dq`, scanning the delay `d`. The torque-fit error has a sharp
  minimum at the true delay. As a cross-check, cross-correlating `Δq_des` with `Δtau_est` peaks at the same
  delay. This works on any existing log with `command` and `joint_state` (logger at 200 Hz, 5 ms
  resolution).
- **Before the fix:** command → motor torque response was **15 ms**, identical on all 12 joints and on two
  runs with two different policies. Adding the observation side, the estimated sensor → motor delay was
  about 20–25 ms, of which about 7.5 ms average (up to 15 ms) was timer waits.
- **After the fix:** command → motor was **5–10 ms** (best fit 10 ms on two runs, cross-correlation peak
  5–10 ms per joint). Bench test with a fake hardware node: sensor tick → command back at the hardware node
  in **2.5 ms** median (3.2 ms max), with the policy still at **50.00 Hz**.
- **On the robot**, with the same delay-randomized policy and similar driving, the 8–15 Hz roll-rate
  shake while moving dropped from 0.133 to 0.112 rad/s RMS. The user reported the swaying was visibly much
  better. (A delay-randomized policy had already cut it from 0.188; the two help each other.)

### Where the G1 code has the same pattern

All references are to `dev` as of `1c7e9bd`.

| Timer | Where | G1 period | Avoidable wait (avg / worst) |
|---|---|---|---|
| 1. Sensor publishing on its own ROS timer, not synchronized with the control loop | `hardware.py`: `self.pub_timer = self.create_timer(ROS_SENSOR_PUBLISH_DT, self.publish_sensor_data)` | 10 ms | 5 / 10 ms |
| 2. Policy on a free-running `create_timer(ctrl_dt)`, not synchronized with sensor arrival | every `control_29dof_*.py`: `self.control_timer = self.create_timer(self.ctrl_dt, self.control_callback)` | sensors every 10 ms | 5 / 10 ms |
| 3. New command waits for the next control tick before it's written | `hardware.py`: `command_callback` only stores the command; `LowCmdWrite` writes it on the next 100 Hz tick | 10 ms | 5 / 10 ms |
| **Total** | | | **about 15 ms avg, up to 30 ms** |

The Go2 had the same three timers at 5 ms periods, so it had about 7.5 ms average and up to 15 ms. Also:
- The G1 controllers subscribe to sensors with queue depth 10. If a controller falls behind, it works
  through stale messages in order. Depth 1 means it always acts on the newest.
- Order inside a tick: the G1 loop decodes and processes state before building and writing `LowCmd` for 29
  motors in a Python `for` loop. It's worth timing a tick on the G1 computer.

### The fix (as done on the Go2, commit `5265d6c` on `foxy`)

The fix keeps the policy at exactly 50 Hz and keeps one action per 20 ms. It only removes the waits.

**1. Publish sensors from the control loop, right after reading the state.** Delete `pub_timer` and call
`publish_sensor_data()` inside `LowCmdWrite()`, after the state read and the FSM / `fsm_time` update, and
before building the command. Keep `fsm_time` as the **last** publish (it already is on the G1:
`hardware_time`, then `fsm_time`). The policy triggers on it, so the joint and IMU messages from the same
tick must already be out.

```python
# LowCmdWrite(), Go2 version
self.read_low_state()
...                                   # FSM transition, self.fsm_time = self.time_ - self.fsm_start_time
self.publish_sensor_data()            # fresh state out now; fsm_time is published last
with self.write_lock:
    ...                               # safety checks, build LowCmd for the current FSM state, CRC, Write
```

On the G1 the state arrives through SDK callbacks (`LowStateHandler`, `TorsoIMUHandler`), not by polling.
Publishing from the loop still removes the publish-timer wait. The state is then up to one callback period
old (`rt/lowstate` is about 500 Hz, so ≤ 2 ms).

**2. Step the policy when `fsm_time` arrives, not on a free timer.** In each controller, remove
`create_timer(ctrl_dt, ...)` and step from `time_callback`:

```python
# __init__: subscriptions -- keep fsm_time LAST. rclpy's executor handles ready subscriptions in
# creation order, so this tick's IMU and joint messages are taken before the policy steps.
# Sensor queue depth 1.
self.last_step = None

def time_callback(self, msg):
    prev_time = self.fsm_time
    self.fsm_time = msg.data
    step = int(self.fsm_time // self.ctrl_dt)
    # new 20 ms step, or fsm_time restarted (FSM transition)
    if self.last_step is None or step != self.last_step or self.fsm_time < prev_time:
        self.last_step = step
        self.control_callback()
```

- **Rate:** exactly `1/ctrl_dt` steps per second of the hardware clock, with no drift. A dropped sensor
  tick delays only that one step by one tick.
- **Needs the clock fix:** with a tick-counting `fsm_time`, the policy rate would inherit the drift.
- **G1 `phase_step` controllers** (`walk/jog_unicycle*`, `crawl_*`): this also fixes the related rcl-timer
  issue in the first section, because steps are now tied to the hardware clock. If a step can be skipped
  (inference longer than `ctrl_dt`), advance `phase_step` by `step - last_step` instead of +1.
- **`control_29dof_mimic.py`** already indexes frames by `int(fsm_time / ctrl_dt)`. This change lines the
  policy steps up with those frame boundaries.
- **Safety side effect:** if `hardware.py` stops publishing, the policy stops, and the command timeout
  (if enabled) puts the robot into damp. Before, the controller kept stepping on stale data.

**3. Write new commands immediately.** At the end of `command_callback`, after storing the command:

```python
with self.write_lock:
    if (self.prev_fsm_state in ("control", "track") and not self.safety_triggered
            and not self.exit_requested):
        self.write_policy_cmd()     # set motor_cmd from q/dq/Kp/Kd/tau_ff, CRC, Write
```

Put the whole build-CRC-Write section of `LowCmdWrite()` under the same `write_lock`. The callback (ROS
thread) and the loop (SDK thread) share one `low_cmd` object.

G1-specific details for `write_policy_cmd()`:
- It must apply `self.joint_sign[i]` to `q`, `dq` and `tau`, exactly as `LowCmdWrite` does.
- It must set `mode_pr` and `mode_machine`, exactly as `LowCmdWrite` does.
- Use `prev_fsm_state` (the loop's view, set after it handles a transition), not `fsm_state` from the
  joystick. Then a command can't be written in `control` before the loop has entered `control`.
- The extra write costs one CRC and DDS write per policy step (50 Hz). That's negligible next to the 100 Hz
  loop.

### What to expect on the G1

- **Policy rate:** still 50 Hz on average. With a real hardware loop, the interval between steps now
  varies by ±1 sensor tick. We saw 15–25 ms on the Go2 (5 ms ticks), so expect about 10–30 ms on the G1
  (10 ms ticks). The average stays exactly 20 ms. If that's a problem, publish sensors at a finer rate than
  the policy, or move the policy into the hardware loop.
- **Delay:** expect about 15 ms less on average and much less jitter. Measure before and after with the
  torque-fit method above. On the G1 use `Kp`/`Kd` from the config, `command[:, :29]` as `q_des`, and
  `joint_state` = `[q, dq, ddq, tau_est]`. The logged state and command are both in the model frame
  (`hardware.py` applies `joint_sign` on the way in and out), so they can be fitted directly.
- **Policies trained with a minimum delay:** if a policy was trained with delay randomization whose
  *minimum* is above the new real delay, the robot is now outside that range. Check the training config.

### Further option (not done on the Go2)

Run the policy inside `hardware.py`'s control loop, every `ctrl_dt / LOW_LEVEL_CONTROL_DT` ticks. This
removes both ROS hops and all three timer waits, so the delay is the robot round trip only, with no jitter.
The cost is losing the separate controller node that mirrors the sim setup.
