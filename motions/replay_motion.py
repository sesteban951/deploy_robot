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
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")


#####################################################################
# MAIN
#####################################################################

if __name__ == "__main__":

    # parser for command line arguments
    parser = argparse.ArgumentParser(description="Replay a motion trajectory.")
    parser.add_argument("motion", help="Path to the motion file.")
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

    # total motion time
    total_time = times[-1]
    print(f"Total motion time: {total_time:.2f} seconds.")

    # load the G1 mujoco model
    xml_path = ROOT_DIR + "/models/g1_29dof_mjlab.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    # launch the viewer
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data,
        show_left_ui=False,
        show_right_ui=False,
    )

    # viewer settings
    viewer_font_scale = getattr(
        mujoco.mjtFontScale,
        'mjFONTSCALE_250',
        getattr(mujoco.mjtFontScale, 'mjFONTSCALE_200', mujoco.mjtFontScale.mjFONTSCALE_150),
    )

    # camera settings
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20
    viewer.cam.distance = 3.0
    viewer.cam.lookat[:] = body_pos_w[0, 0, :]

    # run the visualization
    try:
        t0 = time.time()
        while True:

            if viewer.is_running() == False:
                break

            elapsed = time.time() - t0
            i = np.searchsorted(times, elapsed)
            i = min(i, len(times) - 1)

            # display playback time and speed
            playback_speed = elapsed / times[i] if times[i] > 0 else 0.0
            progress = (i / (len(times) - 1)) * 100.0
            viewer.set_texts((
                viewer_font_scale,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                f"Motion time: {times[i]:.2f}s\nReal time:   {elapsed:.2f}s\nSpeed:       {playback_speed:.2f}x\nProgress:    {progress:.1f}%",
                "",
            ))

            # base pose from motion (convert quat from xyzw to MuJoCo wxyz)
            base_pos = body_pos_w[i, 0, :]
            base_quat = body_quat_w[i, 0, :]  # already wxyz

            mj_data.qpos[:] = np.concatenate([base_pos, base_quat, qpos[i]])
            mj_data.qvel[:] = np.concatenate([body_lin_vel_w[i, 0, :], body_ang_vel_w[i, 0, :], qvel[i]])

            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            if time.time() - t0 > times[-1]:
                time.sleep(1.0)
                t0 = time.time()

    except KeyboardInterrupt:
        print("\nClosed visualization.")

    viewer.close()