##
#
# Replay a motion trajecotry.
#
##

# standard imports
import argparse
import time
import numpy as np

# mujoco imports
import mujoco
import mujoco.viewer

# directory imports
import os
import xml.etree.ElementTree as ET
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")


# Isaac/ONNX joint order (same as joint_pos in the NPZ)
ISAAC_JOINT_NAMES = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint',
    'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint',
    'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint',
    'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
    'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint',
    'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint'
]


def get_isaac_to_mujoco_indices(xml_path):
    """
    Parse MuJoCo XML joint order and return reorder indices
    so that isaac_arr[indices] gives MuJoCo-ordered array.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_joint_names = []
    for joint in root.iter("joint"):
        name = joint.attrib.get("name", None)
        if name is not None:
            xml_joint_names.append(name)
    xml_joint_names.pop(0)  # remove floating base joint

    indices = []
    for name in xml_joint_names:
        indices.append(ISAAC_JOINT_NAMES.index(name))
    return indices


#####################################################################
# MAIN
#####################################################################

if __name__ == "__main__":

    # parser for command line arguments
    parser = argparse.ArgumentParser(description="Replay a motion trajectory.")
    parser.add_argument(
        "--motion", 
        required=True,
        type=str, 
        help="Path to the motion file."
    )
    args = parser.parse_args()

    # load the motion trajectory
    motion_path = args.motion
    motion_path_full = ROOT_DIR + "/motions/" + motion_path

    # load the npz motion trajectory
    motion_traj = np.load(motion_path_full)
    print(f"Loaded motion trajectory from: [{motion_path_full}].")
    for key in motion_traj.keys():
        print(f"  - {key}")

    # extract some data
    fps = motion_traj["fps"]
    qpos = motion_traj["joint_pos"]
    qvel = motion_traj["joint_vel"]
    body_pos_w = motion_traj["body_pos_w"]
    body_quat_w = motion_traj["body_quat_w"]
    body_lin_vel_w = motion_traj["body_lin_vel_w"]
    body_ang_vel_w = motion_traj["body_ang_vel_w"]

    # print shapes 
    print(f"fps: {fps}")
    print(f"qpos shape: {qpos.shape}")
    print(f"qvel shape: {qvel.shape}")
    print(f"body_pos_w shape: {body_pos_w.shape}")
    print(f"body_quat_w shape: {body_quat_w.shape}")
    print(f"body_lin_vel_w shape: {body_lin_vel_w.shape}")
    print(f"body_ang_vel_w shape: {body_ang_vel_w.shape}")

    # create time array
    n_frames = qpos.shape[0]
    times = np.arange(n_frames) / fps

    # load the G1 mujoco model
    xml_path = ROOT_DIR + "/models/g1_29dof_rev_1_0.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    # build isaac -> mujoco joint reorder indices
    reorder = get_isaac_to_mujoco_indices(xml_path)

    # launch the viewer
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    # run the visualization
    try:
        t0 = time.time()
        while True:

            if viewer.is_running() == False:
                break

            i = np.searchsorted(times, time.time() - t0)
            i = min(i, len(times) - 1)  # Clamp to valid range

            print(f"Time: {time.time() - t0:.2f}, Index: {i}\r", end="")

            # base pose from motion (convert quat from xyzw to MuJoCo wxyz)
            base_pos = body_pos_w[i, 0, :]
            base_quat = body_quat_w[i, 0, :]  # already wxyz

            mj_data.qpos[:] = np.concatenate([base_pos, base_quat, qpos[i, reorder]])
            mj_data.qvel[:] = np.concatenate([body_lin_vel_w[i, 0, :], body_ang_vel_w[i, 0, :], qvel[i, reorder]])

            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            if time.time() - t0 > times[-1]:
                time.sleep(1.0)
                t0 = time.time()

    except KeyboardInterrupt:
        print("\nClosed visualization.")

    viewer.close()

    # save the motion trajectory as a new npz file with MuJoCo joint ordering
    save_path = motion_path_full.replace(".npz", "_mujoco.npz")
    np.savez(
        save_path,
        fps=fps,
        joint_pos=qpos[:, reorder],
        joint_vel=qvel[:, reorder],
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    print(f"Saved MuJoCo-ordered motion to: {save_path}")