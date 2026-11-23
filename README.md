# Deploy Robot in Simulation and on Hardware
<p align="center"><img src="utils/img/masih_kcf.png" width="50%"/></p>

Asynchronous policy deployment for Unitree G1 and Go2. Walking KCF-MPC is
self-contained in this repository and currently validated in G1 simulation.

- `deploy`: command, policy, simulation and hardware nodes.
- `kcf_mpc`: optimized MPC runtime, gains and trajectory references.
- `models`, `policy`, `motions`: robot and controller assets.
- `utils`: shared deployment utilities.

# Installation

## ROS2

KCF uses **ROS Humble, Python 3.11 / NumPy 2 and an NVIDIA GPU**.
Install [ROS Humble](https://docs.ros.org/en/humble/Installation.html)
and Conda first. Jazzy/Python 3.12 is not qualified for KCF.

From the repository root:

```bash
sudo apt install build-essential cmake git-lfs python3-colcon-common-extensions
git lfs install
git lfs pull
make install
conda activate deploy
```

`make install` creates the environment, builds matching ROS Python bindings
and sets `DEPLOY_ROOT_DIR`. ROS loads automatically with `conda activate deploy`

```bash
make update      # update dependencies
make kcf-setup   # repair/rebuild ROS bindings and activation hooks
make uninstall   # remove the environment after conda deactivate
```

## Unitree SDK2 (hardware only)

Install the pinned official [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2_python)
and its native Cyclone DDS dependency into the deployment Conda environment:

```bash
make hardware-setup
```

## Policy and MPC configuration

`deploy/configs/g1/g1_29dof_kcf.yaml` is the single `walking` profile for both
simulation and hardware. It selects `policy/g1/g1_29dof_vel.onnx`, the 29-joint XML,
and the epoch-3240 model/normalization in `policy/g1/kcf/walking/` with asset checksums.
The model uses 103 states, all 29 joints and previous raw policy actions, three
physical body commands, and a 256-dimensional lift.
MPC gains, standing hold/release, and trajectory defaults are in `kcf_mpc/config/mpc_config.py`.

# Deployment

Run every command from this repository root, with `conda activate deploy`
in each terminal. KCF uses keyboard controls; no joystick node or pygame is needed.

## Simulation

Terminal 1 — KCF command node (wait for `ready`):

```bash
python deploy/kcf/kcf_ros.py
```

Terminal 2 — walking policy:

```bash
python deploy/simulation/g1/control_29dof_kcf.py --config g1_29dof_kcf.yaml
```

Terminal 3 — visible MuJoCo simulation:

```bash
python deploy/simulation/g1/simulation.py --config g1_29dof_kcf.yaml
```

In **Terminal 1**, press `d` → `h` → `c` → `t` to enter KCF tracking.
`c` leaves tracking with zero velocity; **Space** or `x` stops tracking;
`q` or Ctrl+C exits. Keep this terminal in the foreground.
Loss of terminal control, stale feedback or a long loop pause disarms tracking;
repeat `h` → `c` → `t` after recovery. These keys do not provide a physical emergency stop.

To select a trajectory, add e.g. `--reference-mode lissajous` to Terminal 1.
Use `--shadow` for prediction without actuation or FSM commands.

Terminal 4 (optional logging to `logs/simulation/`):

```bash
python deploy/logger/log.py --mode sim
```

### Mocap input

The OptiTrack adapter is `deploy/mocap/estimator.py`. It publishes the
`nav_msgs/Odometry` input on `/deploy_kcf/mocap` (contract in
`deploy/kcf/kcf_interface.py`): world-to-pelvis pose and body-frame velocity,
stamped on the ROS clock.

```bash
python deploy/mocap/estimator.py               # robot mode: IMU orientation/gyro, yaw aligned to mocap
python deploy/mocap/estimator.py --mode mocap  # everything from mocap (no robot needed)
python deploy/mocap/visualizer.py              # optional: view the published pose
```

On hardware, `deploy/hardware/g1/control_29dof_kcf.py` subscribes to it and
merges the pose into the KCF feedback, converting the ROS-clock stamp onto the
policy clock; keep `kcf.state_source: simulation` (feedback used as published).
Simulation supplies its atomic MuJoCo snapshot directly; it does not need the mocap estimator.

## Hardware

**Torques are live.** `deploy/hardware/g1/hardware.py` has `ZERO_TORQUE = False`;
set it to `True` for a dry run (every motor gets Kp = Kd = tau = 0, the robot is limp,
sensors and the KCF pipeline still run). Do not use the simulation launcher on a physical robot.

The hardware controller `deploy/hardware/g1/control_29dof_kcf.py` walks with the
29-joint velocity policy `g1_29dof_vel.onnx` (mechanical waist lock OFF) and takes
its twist from KCF. All 29 joint positions, velocities and previous raw actions
are passed directly to the model. Pelvis pose comes from `deploy/mocap/estimator.py`.
Policy, model and command limits are selected by the same `g1_29dof_kcf.yaml`.

Before a hardware session, check imports, model/policy compatibility, ONNX inference
and the complete 145-value SDK motor-command/CRC path without opening a Unitree channel:

```bash
python deploy/hardware/g1/hardware.py --config g1_29dof_kcf --preflight
```

This check does not start a robot connection or send motor commands.

KCF keyboard: `h` home, `c` control (zero twist: stand in place), `t` track (KCF twist),
`d` damp, Space/`x` stop.

Hardware deployment on this branch is G1 only.

### Network Configuration

Connect Ethernet and set IPv4 address `192.168.123.99`, netmask `255.255.255.0`.
Use the name of the Ethernet interface connected to the robot, e.g. `enp8s0`,
where `<network_interface_name>` appears below.

### Sanity Check

The existing Unitree low-level example checks communication by moving the G1's
arms and ankles.

```bash
python deploy/hardware/g1/g1_low_level_example.py <network_interface_name>
```

This checks the SDK connection, not the KCF/OptiTrack integration.

### Launching

Use separate terminals, each with `conda activate deploy`.

Terminal 1 — KCF command node:

```bash
python deploy/kcf/kcf_ros.py                       # MPC with the learned KCF model
python deploy/kcf/kcf_ros.py --single-integrator   # MPC with the ideal single integrator (same guards and safety)
```

Terminal 2 — hardware policy controller:

```bash
python deploy/hardware/g1/control_29dof_kcf.py --config g1_29dof_kcf
```

Terminal 3 — hardware SDK node:

```bash
python deploy/hardware/g1/hardware.py --config g1_29dof_kcf --network <network_interface_name>
```

Terminal 4 — mocap pelvis estimator (Motive streaming the `kcf` rigid body):

```bash
python deploy/mocap/estimator.py
```

Terminal 5 (optional, logging to `logs/hardware/`):

```bash
python deploy/logger/log.py --mode hw              # G1
python deploy/logger/log.py --mode hw --robot go2  # Go2 logger
```

# Plotting

```bash
python logs/plot.py                                   # latest log
python logs/plot_kcf.py --trace KCF_TRACE.npz --compare SI_TRACE.npz --output-dir /logs/plots
```
Omit `--compare` if there is only one run you want to visualize and add `--no-playback` for the static plots.  
