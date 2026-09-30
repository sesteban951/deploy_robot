##
#
# Use Mujoco's Viewer to show the model.
#
##

# standard imports
import argparse
import numpy as np

# mujoco imports
import mujoco
from mujoco.viewer import launch

# directory imports
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")

###########################################################
# PARSE THE MODEL TO LOAD
###########################################################

parser = argparse.ArgumentParser(
    description="Visualize a MuJoCo model."
)
parser.add_argument(
     "model", 
     help="Name of the XML model file to load, with or without the .xml extension (must be in the 'models' directory)."
)
args = parser.parse_args()

###########################################################
# MODEL INFO
###########################################################

# accept the name with or without the .xml extension
model_name = args.model if args.model.endswith(".xml") else args.model + ".xml"
xml_file = ROOT_DIR + "/models/" + model_name

# load and launch the model
model = mujoco.MjModel.from_xml_path(xml_file)
data = mujoco.MjData(model)

# set the print precision
np.set_printoptions(precision=4, suppress=True)

# print some info about the model
print("\n#####################  INFO  #####################")

# map sensor enum values to readable names
sensor_type_dict = {int(v): k for k, v in mujoco.mjtSensor.__members__.items()}

# file name
print("Model file name:", xml_file)

# basic info 
nq = model.nq
nv = model.nv
nu = model.nu
print(f"\nNumber of generalized positions (nq): {nq}")
print(f"Number of generalized velocities (nv): {nv}")
print(f"Number of control inputs (nu): {nu}")

# joints
joint_type_dict = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
print("\nNumber of joints:", model.njnt)
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jtype = model.jnt_type[i]
    lower, upper = model.jnt_range[i]
    print(f"    Joint {i} name:", name)
    print(f"    Joint {i} type:", joint_type_dict[jtype])
    if jtype in (2, 3):  # slide or hinge
        print(f"    Limits: [{lower:.4f}, {upper:.4f}]")
    else:
        print("    Limits: N/A")

# actuators
print("\nNumber of actuators:", model.nu)
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    atype = model.actuator_trntype[i]  # transmission type
    lower, upper = model.actuator_ctrlrange[i]
    gear = model.actuator_gear[i, 0]
    print(f"    Actuator {i} name:", name)
    print(f"    Actuator {i} transmission type:", atype)
    print(f"    Control limits: [{lower:.4f}, {upper:.4f}]")
    print(f"    Gear ratio: {gear}")

# bodies
print("\nNumber of bodies:", model.nbody)
total_mass = 0.0
for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        mass = model.body_mass[i]
        inertia = model.body_inertia[i]
        total_mass += mass
        print(f"    Body {i} name: {name}")
        print(f"    Body {i} mass: {mass:.4f}")
        print(f"    Body {i} inertia: {inertia}")

print(f"\n    Total mass: {total_mass:.4f}")

# sensors
print("\nNumber of sensors:", model.nsensor)
for i in range(model.nsensor):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    stype = int(model.sensor_type[i])
    stype_name = sensor_type_dict.get(stype, f"UNKNOWN({stype})")
    sdim = model.sensor_dim[i]
    sadr = model.sensor_adr[i]
    print(f"    Sensor {i} name: {name}")
    print(f"    Sensor {i} type: {stype_name}")
    print(f"    Sensor {i} dim: {sdim}")
    print(f"    Sensor {i} adr: {sadr}")


print("\n##################################################")


###########################################################
# SIMULATION
###########################################################

# launch the viewer
launch(model)
