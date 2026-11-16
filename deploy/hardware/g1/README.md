# G1 hardware notes

> **✅ The three timing issues below are fixed on the G1 (2026-11-15)** and verified on the robot.
> See [G1 timing fixes and measurements](#g1-timing-fixes-and-measurements-2026-11-15) for what changed
> and what was measured. Robot state → motor command went from **about 20 ms to about 1.8 ms**.
> 1. [`fsm_time` could run slower than real time](#fixed-hardware_time--fsm_time-could-run-slower-than-real-time)
>    (tick-counting clock). G1: `f508a05`.
> 2. [Control loop latency: avoidable delay from unsynchronized timers](#fixed-control-loop-latency-avoidable-delay-from-unsynchronized-timers).
>    G1: `f508a05` (`hardware.py`), `5993747` + `0cc5a6d` (controllers).
> 3. [Policy steps 1 tick early or late](#fixed-policy-steps-land-1-tick-early-or-late) (a bug in the
>    first Go2 version of fix #2). The G1 uses the corrected half-tick trigger from the start.
>
> **Still open on the G1:** see [Open issues](#open-issues-on-the-g1). These include no link-loss
> watchdog, a stale command on `control` re-entry, and the vel controller running before `control`.

## G1 timing fixes and measurements (2026-11-15)

### What changed
**`hardware.py`** (`f508a05`), the same changes as the Go2 on `foxy`:
- **Wall-clock time:** `time_ = time.monotonic() - time_start`, instead of adding `LOW_LEVEL_CONTROL_DT`
  per tick.
- **Sensors published from the control loop**, right after the FSM / `fsm_time` update, with `fsm_time`
  last. The separate 10 ms ROS publish timer is gone.
- **New commands written immediately** in `command_callback`: only in `control`/`track` as seen by the loop
  (`prev_fsm_state`), and never after the tilt safety trips. The loop and the callback share `write_lock`.
- **State is polled once per tick:** `rt/lowstate` and `rt/secondary_imu` are read through `KeepLast(1)`
  DataReaders at the top of `LowCmdWrite()`, instead of decoding every message in SDK callbacks.
  `Start()` polls while it waits for the first state.

**Every hardware controller** (`5993747`, `0cc5a6d`):
- **No more `create_timer`.** The policy steps in `time_callback` once per `control_dt` slot of the
  hardware clock, using the half-tick trigger `step = floor((fsm_time + 0.005) / control_dt)`.
- **Sensor subscriptions have depth 1**, and `fsm_time` is subscribed last.
- **`phase_step` controllers** (walk, jog, crawl, history, LSTM) advance `phase_step` by the number of
  slots elapsed.
- **vel** builds its gait phase from `slot_time = step * control_dt`.
- **mimic** indexes frames with `min(last_step, num_frames - 1)`.

### Measured on the robot
On the desktop (Ryzen 9 7950X) over `enp8s0`, with the walk LSTM. The before and after runs used the same
procedure, measured with an external passive DDS + ROS probe.

| | Before | After |
|---|---|---|
| Robot state → `lowcmd` written | **~20 ms** (0.8 state → publish, 9.2 waiting for the controller timer, 10.3 waiting for a tick) | **~1.8 ms** (0.8 + 0.3 + 0.6) |
| Policy step interval | 20 ms, but the wait above depended on the timer phase at launch (3–20 ms) and drifted during a run | 99.8% within 20 ± 0.5 ms, max 21–22 ms |
| Clock ratio (`hardware_time` / wall) | 1.0000 (no drift on this desktop, see below) | 1.0000 |
| Ticks with fresh robot state | n/a | 100% |
| Walking, 5–25 Hz leg torque | 1.95 Nm | 1.57–1.69 Nm (different driving, indicative only) |

Mimic (squat) after the fix: 0 off-slot steps in 2262, and all four `control`/`track` entries started
from the reset `fsm_time`.

### Measured facts about the G1 that differ from earlier assumptions
- **`rt/lowstate` and `rt/secondary_imu` each arrive at ~1052 Hz, not 500 Hz.** About 5% of `lowstate`
  messages repeat the previous `tick`.
- **The robot zeroes the pelvis IMU in `lowstate` while walking.** In one 82 s walking run, 38 of ~87k packets
  had the whole IMU block zeroed (quaternion and gyro) while the joint data in the same packet was normal.
  These came in 1–2 packet bursts, sometimes clustered within 100 ms. Standing, hanging and squatting
  showed none, and the torso IMU (`secondary_imu`) showed none. The IMU guard catches them. With
  polling, `hardware.py` only sees about 1 in 8 of them (it uses 100 of ~1050 packets/s), so the
  `pelvis IMU packet lost` warnings now undercount. The skipped packets never reach the policy.

### Weaker computers (the onboard mini PC)
The desktop hid the clock drift: the old code used about 1/30 of its 10 ms budget. Decoding every
`lowstate` + `secondary_imu` message in callbacks costs about **33% of a core of GIL time on the desktop**,
so it saturates on a CPU 3–4× slower. A bench with a fake 1052 Hz robot, with `hardware.py` pinned to a
fraction of one core, compared the two versions:

| `hardware.py` CPU share | Original code: state age / end-to-end p50, max / clock | New code |
|---|---|---|
| 1/3 of a core | 8.5 ms / 19, 48 ms / 0.9999 | 0.5 ms / 2.4, 6.5 ms / 1.0000 |
| 1/4 of a core | 22.7 ms / 40, 98 ms / **0.9952 (drifting)** | 0.5 ms / 2.6, 8.5 ms / 1.0000 |

**Not yet measured on the mini PC itself.** Do that before trusting outdoor runs.

---

## Fixed: `hardware_time` / `fsm_time` could run slower than real time

**Status (2026-11-15): fixed on the G1 in `f508a05`** (wall clock), as on the Go2 (`3edaad8`). The rest of
this section is the original analysis, and the code references describe the G1 *before* the fix. On the
desktop the drift never actually showed (ratio 1.0000 in every run). It appeared only in the CPU-starved
bench above.

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

## Fixed: control loop latency: avoidable delay from unsynchronized timers

**Status (2026-11-15): fixed on the G1** in `f508a05` (`hardware.py`) and `5993747` + `0cc5a6d` (all
controllers). Measured on the robot: about 20 ms → 1.8 ms (see
[the measurements](#measured-on-the-robot)). The rest of this section is the original analysis, and the
code references describe the G1 *before* the fix.

**Correction to the estimate below:** the G1 decodes state in a ~1 kHz callback, so the sensor publish
timer didn't add a third independent wait. The controller's wait for sensors and `hardware.py`'s wait to
write the command added up to a total fixed by the phase of `hardware.py`'s own two timers. That gave
about 12–14 ms on the bench and about 20 ms in one robot launch, not 15 ms average / 30 ms worst.

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

### The fix (as done on the Go2: `5265d6c` on `foxy`, trigger corrected in `ea09935`)

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
`create_timer(ctrl_dt, ...)` and step from `time_callback`.

> ⚠️ The first Go2 version (`5265d6c`) used `step = int(fsm_time // ctrl_dt)` with no offset. That
> version has a bug: a third of policy steps landed one tick early or late. See
> [the bug section](#fixed-policy-steps-land-1-tick-early-or-late). Use this
> corrected version (`ea09935`):

```python
# hardware.py ticks every LOW_LEVEL_CONTROL_DT, and fsm_time restarts exactly on a tick, so later
# ticks land exactly on the ctrl_dt slot boundaries. Shift the slot grid by half a tick, so a
# boundary falls BETWEEN two ticks and tick jitter can never move a step to another tick.
TRIGGER_OFFSET = 0.005   # [sec] half of the G1's LOW_LEVEL_CONTROL_DT (0.01). On the Go2: 0.0025.

# __init__: subscriptions -- keep fsm_time LAST. rclpy's executor handles ready subscriptions in
# creation order, so this tick's IMU and joint messages are taken before the policy steps.
# Sensor queue depth 1.
self.last_step = None
self.slot_time = 0.0     # [sec] step * ctrl_dt: the ideal time of the current policy step

def time_callback(self, msg):
    prev_time = self.fsm_time
    self.fsm_time = msg.data
    step = int((self.fsm_time + TRIGGER_OFFSET) // self.ctrl_dt)
    # new step, or fsm_time restarted (FSM transition)
    if self.last_step is None or step != self.last_step or self.fsm_time < prev_time:
        self.last_step = step
        self.slot_time = step * self.ctrl_dt
        self.control_callback()

# anything that depends on time uses the slot time (or the step index), NOT the raw fsm_time:
phase = (self.slot_time % self.gait_period) / self.gait_period      # e.g. control_29dof_vel.py
```

- **Rate:** exactly `1/ctrl_dt` steps per second of the hardware clock, with no drift, every
  `ctrl_dt / LOW_LEVEL_CONTROL_DT` ticks (2 on the G1, 4 on the Go2). A dropped tick delays only that one
  step.
- **Use the slot time, not `fsm_time`, for anything timed:**
  - **Gait phase:** the phase then advances exactly `ctrl_dt` per step, as in training.
  - **Mimic frame index:** in `control_29dof_mimic.py`, `frame = int(self.fsm_time / self.ctrl_dt)` has
    the same boundary problem. With event-driven steps, `fsm_time` sits within microseconds of a frame
    boundary, so some steps would get the previous frame: frames repeat and then skip. Use
    `frame = min(self.last_step, self.motion_num_frames - 1)`.
  - **Ramps:** for the mimic `control` ramp (`fsm_time / frame_pos_duration`), raw `fsm_time` is fine. It
    only needs to be smooth.
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

- **Policy rate:** exactly 50 Hz. With the corrected trigger, every step is exactly 2 ticks apart, with
  only scheduling jitter. On the Go2: 19.3–20.7 ms (1st–99th percentile). With the buggy trigger it
  would be 10/20/30 ms on the G1, which is worse than the Go2's 15/20/25 ms.
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

---

## Fixed: policy steps land 1 tick early or late

**Status (2026-11-15):** Found on the **Go2** with the timing measurements below, and fixed in `ea09935` on
`foxy`. It was a bug in the first version of the latency fix (`5265d6c`). **The G1 controllers use the
corrected half-tick trigger from the start** (`TRIGGER_OFFSET = 0.005`). On the robot, 99.8% of steps
were within 20 ± 0.5 ms, with no 10 or 30 ms steps.

### Symptom
After the latency fix, a third of the Go2's policy steps were not 20 ms apart. In a 106 s run:

| Step interval | 15 ms | 20 ms | 25 ms |
|---|---|---|---|
| Count | 849 | 3617 | 849 |

The offset from the ideal step time was always either ~0 ms or ~5 ms (one tick), never in between. The
average rate was still exactly 50 Hz, so a rate check didn't show it. Only the per-step intervals did.

### Cause
1. `hardware.py` ticks every 5 ms (10 ms on the G1).
2. On every FSM transition it sets `fsm_start_time = time_` *at a tick*. So `fsm_time` is 0 exactly at a
   tick, and every later tick lands within microseconds of a multiple of 5 ms.
3. The policy slot boundaries are multiples of 20 ms, so every 4th tick sits *exactly* on a boundary.
4. The trigger was `floor(fsm_time / ctrl_dt)`. Whether the boundary tick reads 19.99 ms or 20.01 ms is
   decided by microseconds of scheduling jitter. At 19.99 ms the step runs on the *next* tick, 5 ms late.
5. As the jitter flips sign, consecutive steps come 15 or 25 ms apart.

Before the first FSM transition, the boundaries happened to fall mid-tick, because `hardware_time`'s
origin isn't on a tick. That's why the controller's first timing lines showed a steady 3.2 ms offset and
only switched to 15/25 ms intervals after `init -> damp`.

### Fix (`ea09935`)
1. **Half-tick offset:** `step = floor((fsm_time + LOW_LEVEL_CONTROL_DT/2) / ctrl_dt)`. Each slot boundary
   now falls *halfway between* two ticks. Jitter under half a tick (2.5 ms on the Go2, 5 ms on the G1)
   can no longer move a step to another tick.
2. **Slot time for anything timed:** the gait phase uses `slot_time = step * ctrl_dt`, not the raw
   `fsm_time`. The phase input then advances exactly `ctrl_dt` per step, with no timing jitter, as in
   training. On the G1, the mimic frame index needs the same treatment (see fix #2).

### Result on the Go2 (109 s run, same robot and policy)

| | Before (`5265d6c`) | After (`ea09935`) |
|---|---|---|
| Step intervals 15 / 20 / 25 ms | 849 / 3617 / 849 | **0 / 5442 / 0** |
| Step interval, 1st–99th percentile | 14.8–25.2 ms | **19.3–20.7 ms** |
| Offset from ideal step time, 1st–99th percentile | 0–5.0 ms | **−0.01 to 0.03 ms** |

Before going on the robot, a bench test reproduced the bug. A fake hardware node published tick-aligned
`fsm_time` with ±0.1 ms jitter. The old trigger gave 251/496/251 intervals of 15/20/25 ms; the new
trigger gave 0/548/0.

**Behavior:** we couldn't measure a difference in roll shake between the two runs, because they were
driven differently. The second run had about 30% larger stick commands and 4× more pose input. Comparing
behavior needs identical commands, for example a scripted joystick replay. The timing fix is correct
either way: the policy now gets the timing it was trained with.

---

## Timing measurements (Go2 only, not ported to the G1)

Added on the Go2 in `ea09935`. They measure everything above directly on the robot, without the torque
fit. **Not ported to the G1.** The G1 numbers in this README came from an external passive probe
(separate DDS + ROS subscribers, not in the repo). Porting these would make the mini PC easy to check on
the robot.

### What was added
**`hardware.py`**
- **`deploy_robot/loop_timing`:** one `Float32MultiArray` per tick, published at the end of the tick:
  `[seq, tick interval, read, publish, cmd, tick total, new state (0/1), last callback write, callback writes since last tick]`.
  Durations are in ms; `cmd` includes waiting for `write_lock`.
- **`--print-timing`:** prints one summary line every 5 s, with the loop rate, longest gap, mean/max per
  part, % of ticks with no new state, and the callback write count and cost.
- **Read stamps:** `imu_state` and `joint_state` get one extra value at the end, the `hardware_time` at
  which that state was read. Readers slice the leading fields, so this is backward compatible. The logger
  stores the wider rows, and `plot.py` infers the joint count with `width // 4`.
- **Clock origin:** `time_start` (the `time.monotonic()` origin of `hardware_time`) is set in `Init()` and
  published as `time_origin_monotonic` in the latched `hardware_info` message. `time.monotonic()` is
  system-wide, so a controller on the same computer can convert its clock into `hardware_time`.

**Controllers:** `utils/policy_timing.py` (`PolicyTiming`), shared by all controllers.
- **`deploy_robot/policy_timing`:** one row per step: `[seq, step interval, slot offset, imu age, joint
  age, trigger age, same tick (0/1), inference, step total]`, in ms.
- **Subscription order:** it subscribes to `hardware_info` and `hardware_time`. Create it *before* the
  `fsm_time` subscription, so `hardware_time` is processed first within a tick.
- **`--print-timing`:** prints a summary every 5 s.

**Logger:** `loop_timing` and `policy_timing` are optional datasets. Add them to the G1's `optional` list
in `deploy/logger/log.py`.

### What it showed on the Go2 (Jetson Orin NX, 200 Hz loop, pose policy)

| Quantity | Value |
|---|---|
| Loop rate | 200 Hz in every FSM state; longest gap 6.7 ms in 109 s; **no dropped ticks** |
| Tick cost (mean / max) | 2.1–2.3 / 4.9 ms of a 5 ms budget: read + decode 0.83, publish 0.36, command build + CRC + write 1.0–1.1 |
| Immediate writes from `command_callback` | 50/s, about 1 ms each; stretch the longest gap from about 5.3 ms to 6–7 ms in `control` (harmless) |
| Observation age at the policy step | 1.2 ms mean, 2.6 ms max |
| IMU and joint data from the triggering tick | 100% (confirms the ordering assumption) |
| Trigger age (tick start → policy step starts) | about 2.0 ms mean, 3.5 ms max |
| Inference / whole step | 0.3 / 0.7 ms (MLP, ONNX on CPU) |
| Command → motor (torque fit) | 10 ms (15 ms before the latency fix) |

**G1 differences to expect:**
- The `hg` `LowState_` carries 35 motor entries (vs 20), and the loop writes 29 motors. So read and command
  build will cost more per tick, but the G1 has a 10 ms budget.
- The G1 also reads `rt/secondary_imu`.
- The G1 decodes state in SDK callbacks, so the `read` column will be small. Watch `tick interval` and the
  longest gap instead: callback decoding at 500 Hz competes for the GIL.

---

## Open issues on the G1

1. **No link-loss watchdog.**
   - On 2026-11-15 the ethernet link dropped twice (a loose cable), and once the robot fell mid-pivot.
     When `rt/lowstate` stops arriving, `hardware.py` keeps publishing the last state as if it were new, and
     the policy keeps stepping on it. Nothing is printed.
   - While the link is down, the robot can't receive commands either, so the PC can't prevent that fall.
     The dangerous part is **reconnection**: `hardware.py` immediately resumes sending policy commands
     computed from stale state.
   - Fix: if no new `lowstate` arrives for ~50 ms in `control`/`track`, latch damp until restart (like
     the tilt safety). Port the Go2's command timeout too (damp if the controller stops sending).
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
