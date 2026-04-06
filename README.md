# Deploy Robot in Simulation and on Hardware
<p align="center"><img src="utils/img/menhir.png" width="50%"/></p>

This repo is for deploying asynchronous sim and real robot code for the Unitree G1 robot. 

A brief overview of the repo structure is as follows:
- `deploy`: main deployment code for both sim and real robot. The `simulation` folder contains code for sim deployment, and the `hardware` folder contains code for real robot deployment.
- `models`: contains the Mujoco XML files and mesh files for the robot.
- `motions`: contains motion files for the robot.
- `policy`: contains the policies to use for controlling the robot.
- `utils`: contains utility code.

Download a motion from WandB:
```bash
python motions/get_wandb_motion.py wandb-registry-Motions/walk1_subject1:latest
```
Download a policy from WandB:
```bash
python policy/get_wandb_policy.py sesteban-california-institute-of-technology-caltech/mjlab/bysdsnbu
```

---

# Installation
## Setting the environment variable
Set the directory of this repo as a environment variable in your `~/.bashrc` file, for example:
```bash
export DEPLOY_ROOT_DIR="/home/sergio/projects/deploy_robot"
```

## ROS2
Use `ROS2 Humble` to communicate across different pieces of code. Install instructions are here: https://docs.ros.org/en/humble/Installation.html

If your scripts use the joystick, you can install `joy` package via:
```bash
sudo apt update
sudo apt install ros-humble-joy
```

## Conda Environment
Use `conda` and install via:
```bash
conda env create -f environment.yml
conda activate env_deploy
```

## Unitree SDK2 for Python
After the `conda` environment is set up, install the Unitree SDK inside the conda environment.
Make sure you are in the `env_deploy` conda environment by using `conda activate env_deploy`, and then follow the Unitree SDK installation instructions here: https://github.com/unitreerobotics/unitree_sdk2_python

---

# Deployment
## Simulation
### Launching
You will open three terminals, each with in the `env_deploy` conda environment. In the first terminal, you will launch the joystick node (if you use it). In the second terminal, you will launch the controller. In the third terminal, you will launch the Mujoco simulation.

Terminal 1:
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
You will open three terminals, each with in the `env_deploy` conda environment. In the first terminal, you will launch the joystick node (if you use it). In the second terminal, you will launch the controller. In the third terminal, you will launch the hardware SDK node.

Terminal 1:
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

---

# Small details
If you want VS Code to recognize the Unitree SDK source code, just add the following to your `.vscode/settings.json` that is located in your root directory:
```json
{
  "python.analysis.extraPaths": ["<path-to>/unitree_sdk2_python"]
}
```
To be clear, this is only so you can do the `ctrl + click` to jump to the source code of the Unitree SDK. You don't need this for the code to run.
