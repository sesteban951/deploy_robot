# Deploy Robot in Simulation and on Hardware
<p align="center"><img src="utils/img/menhir.png" width="50%"/></p>

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

Inside `conda activate deploy`, follow the
[Unitree SDK2 installation instructions](https://github.com/unitreerobotics/unitree_sdk2_python).

## Policy and MPC configuration

The walking policy, KCF checkpoint and normalization are under `policy/g1/kcf/`.
`deploy/configs/g1/g1_23dof_kcf.yaml` contains the robot/policy settings and the
`kcf:` model bundle. Set `kcf.state_source` to `simulation` or `mocap` there;
an optional `--state-source` flag overrides it for one run.
MPC gains and trajectory defaults are in `kcf_mpc/config/mpc_config.py`.

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
python deploy/simulation/g1/control_23dof_vel.py --config g1_23dof_kcf.yaml
```

Terminal 3 — visible MuJoCo simulation:

```bash
python deploy/simulation/g1/simulation.py --config g1_23dof_kcf.yaml
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

Start `python deploy/kcf/dummy_mocap_ros.py` in another activated terminal and
set `kcf.state_source: mocap` in the deployment YAML. The dummy is a simulation-backed
placeholder for the future OptiTrack adapter, not a separate controller path.

Both publishers use the same `nav_msgs/Odometry` input on `/deploy_kcf/mocap`:
calibrated world-to-pelvis pose, body-frame velocity and measurement timestamps
aligned with the policy clock. The contract is in `deploy/kcf/kcf_interface.py`.
Later, stop the dummy and launch your OptiTrack adapter on that interface;
the KCF and policy commands stay unchanged. If needed, remap the KCF input using
`--ros-args -r /deploy_kcf/mocap:=/your/calibrated_pelvis` in Terminal 1.

## Hardware

**KCF hardware control is not connected yet.** OptiTrack feedback, the physical
motor mapping and the hardware command hookup still require implementation and
validation. `--mode hardware` currently fails closed; do not use the simulation
launcher to control a physical robot.

Hardware deployment on this branch is G1 only.

### Network Configuration

Connect Ethernet and set IPv4 address `192.168.123.99`, netmask `255.255.255.0`.
Use the name of the Ethernet interface connected to the robot, e.g. `enp8s0`,
where `<network_interface_name>` appears below.

### Sanity Check

The existing Unitree low-level example checks communication by moving the G1's
arms and ankles. Follow the robot's hardware safety procedure and run it by
itself, not alongside another controller:

```bash
python deploy/hardware/g1/g1_low_level_example.py <network_interface_name>
```

This checks the SDK connection, not the KCF/OptiTrack integration.

### Launching

Use separate terminals, each with `conda activate deploy`. The KCF variant keeps
the original controller/SDK layout and replaces only the joystick command node
with the keyboard-enabled KCF node. Its hardware mode remains blocked until the
integration above is complete.

Terminal 1 — KCF command node:

```bash
python deploy/kcf/kcf_ros.py
```

Terminal 2 — hardware policy controller:

```bash
python deploy/hardware/g1/<control_script>.py --config <your-config-file>.yaml
```

Terminal 3 — hardware SDK node:

```bash
python deploy/hardware/g1/hardware.py --config <your-config-file>.yaml --network <network_interface_name>
```

Terminal 4 (optional, logging to `logs/hardware/`):

```bash
python deploy/logger/log.py --mode hw              # G1
python deploy/logger/log.py --mode hw --robot go2  # Go2 logger
```

# Plotting

```bash
python logs/plot.py                                   # latest log
python logs/plot.py --filename logs/hardware/<run>.h5  # selected log
```
