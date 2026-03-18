# Hardware Deploy
Repo for deploying on hardware. 

## Installation

### Setting the environment variable
Set the directory of this repo as a environment variable in your `~/.bashrc` file, for example:
```bash
export DEPLOY_ROOT_DIR="/home/sergio/projects/deploy_robot"
```

### ROS2
Use `ROS2 Humble` to communicate across different pieces of code. Install instructions are here: https://docs.ros.org/en/humble/Installation.html

### Conda Environment
Use `conda` and install via:
```bash
conda env create -f environment.yml
conda activate env_deploy
```

### Unitree SDK2 for Python
After the `conda` environment is set up, install the Unitree SDK inside the conda environment.
Make sure you are in the `env_deploy` conda environment by using `conda activate env_deploy`, and then follow the Unitree SDK installation instructions here: https://github.com/unitreerobotics/unitree_sdk2_python

## Deployment
### Hardware

#### Network Configuration
First ensure that you are somehow connected to the robot via an ethernet cable. Then go to Ubuntu settings and configure the IPv4 Method to be `Manual`, set the Address to `192.168.123.99`, and the Netmask to `255.255.255.0`. 

You can find the network interface name (e.g. `enp8s0`) via `ifconfig`. You will need it to run hardware deployment code.

#### Mode Config
Use the Unitree remote to make sure that the robot is in "development mode" (I think it's "debug" mode).

#### Sanity Check
To make sure everything is working and you can communicate with the robot, you can run the example low level control script where the robot moves its arms and ankles. Run:
```bash
python deploy/deploy_real/g1_low_level_example.py <network_interface_name>
```
where `<network_interface_name>` is the name of your network interface (e.g. `enp8s0`).