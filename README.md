# Deploy Robot in Simulation and on Hardware
<p align="center"><img src="utils/img/menhir.png" width="50%"/></p>

This repo is for deploying asynchronous sim and real robot code for the Unitree G1 robot. 

A brief overview of the repo structure is as follows:
- `deploy`: main deployment code for both sim and real robot. The `simulation` folder contains code for sim deployment, and the `hardware` folder contains code for real robot deployment.
- `models`: contains the Mujoco XML files and mesh files for the robot.
- `motions`: contains motion files for the robot.
- `policy`: contains the policies to use for controlling the robot.
- `utils`: contains utility code.

---

# Installation
## ROS2
Use `ROS2 Humble` to communicate across different pieces of code. Install instructions are here: https://docs.ros.org/en/humble/Installation.html

If your scripts use the ROS joystick, you can install `joy` package via:
```bash
sudo apt update
sudo apt install ros-humble-joy
```

## Conda Environment
Use `make` to create the `deploy` conda environment from `environment.yml`:
```bash
make install
```
This creates the env **and** sets the `DEPLOY_ROOT_DIR` environment variable (scoped to the env) to this repo. Then, activate the env:
```bash
conda activate deploy
```
Other `make` targets:
```bash
make update      # update the env from environment.yml (after editing dependencies)
make uninstall   # remove the deploy env (run `conda deactivate` first)
```

## Unitree SDK2 for Python
After the `conda` environment is set up, install the Unitree SDK inside the conda environment.
Make sure you are in the `deploy` conda environment by using `conda activate deploy`, and then follow the Unitree SDK installation instructions here: https://github.com/unitreerobotics/unitree_sdk2_python

## Policy
To download a policy from WandB into the `policy/` folder, you can for example run:
```bash
python policy/get_wandb_policy.py sesteban-california-institute-of-technology-caltech/mjlab/bysdsnbu
```

---

# Deployment
## Simulation
### Launching
You will open three terminals (plus an optional fourth to log data), each in the `deploy` conda environment. In the first terminal, you will launch the joystick node (if you use it). In the second terminal, you will launch the controller. In the third terminal, you will launch the Mujoco simulation.

Terminal 1 (if you use joystick):
```bash
python deploy/joystick/joystick_ros.py 
```
Terminal 2:
```bash
python deploy/simulation/<control_script>.py --config <your-config-file>.yaml
```
Terminal 3:
```bash
python deploy/simulation/simulation.py --config <your-config-file>.yaml
```
Terminal 4 (optional, to log data to `logs/simulation/`):
```bash
python deploy/logger/log.py --mode sim
```

## Hardware
### Turning the robot on
Press the power button twice and hold on the second press. You should see the robot's eyes light up and hear fan sounds from the motors. After a while of setup, the robot will say "zero torque mode".

This means the robot is on and ready to receive commands.

### Network Configuration
First ensure that you are somehow connected to the robot via an ethernet cable. Then go to Ubuntu settings and configure the IPv4 Method to `Manual`, set the Address to `192.168.123.99`, and the Netmask to `255.255.255.0`. 

You can find the network interface name (e.g. `enp8s0`) via `ifconfig` command in a terminal. You will need it to run hardware deployment code.

### Sanity Check
To make sure everything is working and you can communicate with the robot, you can run the example low level control script where the robot moves its arms and ankles. Run:
```bash
python deploy/hardware/g1_low_level_example.py <network_interface_name>
```
where `<network_interface_name>` is the name of your network interface (e.g. `enp8s0`).

### Launching
You will open three terminals (plus an optional fourth to log data), each in the `deploy` conda environment. In the first terminal, you will launch the joystick node (if you use it). In the second terminal, you will launch the controller. In the third terminal, you will launch the hardware SDK node.

Terminal 1 (if you use joystick):
```bash
python deploy/joystick/joystick_ros.py 
```
Terminal 2:
```bash
python deploy/hardware/<control_script>.py --config <your-config-file>.yaml
```
Terminal 3:
```bash
python deploy/hardware/hardware.py --config <your-config-file>.yaml
```
Terminal 4 (optional, to log data to `logs/hardware/`):
```bash
python deploy/logger/log.py --mode hw
```

---

# Plotting
To visualize a logged run, use `logs/plot.py`. With no arguments it plots the most recent log found under `logs/simulation/` or `logs/hardware/`:
```bash
python logs/plot.py
```
To plot a specific log, pass its path:
```bash
python logs/plot.py --filename logs/hardware/<your-log>.h5
```
This brings up joint positions and velocities (measured vs commanded), estimated joint torques, the pelvis and torso IMUs, joystick commands, and motor temperatures — whichever of these datasets the log contains.

---

# Small details
If you want VS Code to recognize the Unitree SDK source code, just add the following to your `.vscode/settings.json` that is located in your root directory:
```json
{
  "python.analysis.extraPaths": ["<path-to>/unitree_sdk2_python"]
}
```
To be clear, this is only so you can do the `ctrl + click` to jump to the source code of the Unitree SDK. You don't need this for the code to run.
