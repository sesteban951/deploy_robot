##
#
# Kinematic MuJoCo replay of an HDF5 deploy log. Not a script: run it with
#   python logs/plot.py --replay
#
##

# standard imports
import os
import time
import numpy as np
import yaml

# mujoco imports
import mujoco
import mujoco.viewer


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# fallback scene per joint count, for logs whose config snapshot has no xml_path
DEFAULT_XML = {29: "g1/g1_29dof_mode11_scene.xml", 23: "g1/g1_23dof_scene.xml", 12: "go2/go2_scene.xml"}

SPEEDS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)    # playback speeds stepped by [-] / [=]
GHOST_RGBA = np.array([0.3, 0.6, 1.0, 0.25])           # tint of the q_des ghost
TARGET_FPS = 50.0                                      # display rate; log frames in between are skipped

HELP = ("[Space] pause   [Left/Right] -/+1   [Up/Down] -/+50   [PgUp/PgDn] -/+500\n"
        "[0-9] seek 0-90%   [Home/End] first/last   [-/=] slower/faster   [\\] 1x\n"
        "[G] Reference Ghost   [C] Camera Lock   [R] Restart")


########################################################################
# HELPERS
########################################################################

def resolve_model_path(meta: dict, N: int) -> str:
    """Scene xml for the replay: the config snapshot's xml_path, else picked by joint count."""
    rel = None
    if meta.get("config_yaml"):
        try:
            rel = (yaml.safe_load(str(meta["config_yaml"])) or {}).get("xml_path")
        except Exception:
            rel = None
    if rel is None:
        rel = DEFAULT_XML.get(N)
    if rel is None:
        raise ValueError(f"No scene known for {N} joints.")
    return os.path.join(MODELS_DIR, rel)


def base_quats(imu: np.ndarray) -> np.ndarray:
    """(T, 4) wxyz base orientation from an IMU block [rpy, quat, gyro, acc], yaw zeroed at the start.

    Lost IMU packets (near-zero quaternion) hold the previous valid orientation.
    """
    q = imu[:, 3:7].astype(np.float64)
    norm = np.linalg.norm(q, axis=1)
    valid = norm > 0.5
    q[valid] /= norm[valid, None]
    q[~valid] = [1.0, 0.0, 0.0, 0.0]
    q = q[np.maximum.accumulate(np.where(valid, np.arange(len(q)), 0))]

    # left-multiply by the yaw-only quaternion that undoes the first valid frame's heading
    w, x, y, z = q[np.argmax(valid)]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = np.cos(-yaw / 2.0), np.sin(-yaw / 2.0)
    w, x, y, z = q.T
    return np.stack([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w], axis=1)


def robot_collision_geoms(model) -> np.ndarray:
    """Ids of the robot's colliding geoms (all robot geoms if none collide); world geoms excluded."""
    robot = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0]
    colliding = [g for g in robot if model.geom_contype[g] or model.geom_conaffinity[g]]
    return np.array(colliding or robot)


def lowest_point(model, data, geoms: np.ndarray) -> float:
    """World z of the lowest point over the given geoms: exact for primitives, bounding box otherwise."""
    G = mujoco.mjtGeom
    Rz = data.geom_xmat[geoms, 6:9]           # world-z row of each geom's rotation
    size = model.geom_size[geoms]
    aabb = model.geom_aabb[geoms]             # [center(3), half-size(3)] in the geom frame
    drop = np.sum(np.abs(Rz) * aabb[:, 3:], axis=1) - np.sum(Rz * aabb[:, :3], axis=1)
    gtype = model.geom_type[geoms]
    tilt = np.sqrt(np.clip(1.0 - Rz[:, 2] ** 2, 0.0, 1.0))
    drop = np.where(gtype == G.mjGEOM_SPHERE, size[:, 0], drop)
    drop = np.where(gtype == G.mjGEOM_CAPSULE, size[:, 0] + np.abs(Rz[:, 2]) * size[:, 1], drop)
    drop = np.where(gtype == G.mjGEOM_CYLINDER, size[:, 0] * tilt + np.abs(Rz[:, 2]) * size[:, 1], drop)
    drop = np.where(gtype == G.mjGEOM_BOX, np.sum(np.abs(Rz) * size, axis=1), drop)
    return float(np.min(data.geom_xpos[geoms, 2] - drop))


def floating_body(model) -> int:
    """Body carrying the free joint (the base), or -1 for a fixed-base model."""
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_bodyid[j])
    return -1


def set_pose(model, data, quat, q, geoms):
    """Pose the robot: joints q, base orientation quat, base height so its lowest point is on the floor."""
    data.qpos[0:3] = 0.0
    data.qpos[3:7] = quat
    data.qpos[7:] = q
    mujoco.mj_kinematics(model, data)
    data.qpos[2] = -lowest_point(model, data, geoms)
    mujoco.mj_forward(model, data)


def add_ghost(scn, m, d, opt, pert, qpos):
    """Append the robot at qpos to the user scene as a translucent ghost (dynamic geoms only)."""
    d.qpos[:] = qpos
    mujoco.mj_kinematics(m, d)
    start = scn.ngeom
    mujoco.mjv_addGeoms(m, d, opt, pert, int(mujoco.mjtCatBit.mjCAT_DYNAMIC), scn)
    for i in range(start, scn.ngeom):
        scn.geoms[i].rgba[:] = GHOST_RGBA


########################################################################
# REPLAY
########################################################################

def replay_log(data: dict, meta: dict, name: str, speed: float = 1.0):
    """Replay a loaded log (see plot.load_log) in a passive MuJoCo viewer until the window closes."""

    # log arrays; the logger records no base position, so the feet are put on the floor each frame
    t = data["time"][:, 0].astype(np.float64)
    T = len(t)
    N = data["joint_state"].shape[1] // 4
    q = data["joint_state"][:, :N].astype(np.float64)
    q_des = data["command"][:, :N].astype(np.float64) if "command" in data else None
    js = data.get("joystick")
    imu_key = next((k for k in ("pelvis_imu", "imu") if k in data), None)
    quat = base_quats(data[imu_key]) if imu_key else np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))
    if imu_key is None:
        print("[WARN] no IMU in log; base orientation stays upright.")

    # model (plus a second copy that draws the ghost)
    path = resolve_model_path(meta, N)
    print(f"Model: {path}")
    model = mujoco.MjModel.from_xml_path(path)
    if model.nq != 7 + N:
        raise ValueError(f"{os.path.basename(path)} has nq={model.nq}, expected 7 + {N} joints.")
    mdata = mujoco.MjData(model)
    ghost = None
    if q_des is not None:
        gm = mujoco.MjModel.from_xml_path(path)
        ghost = (gm, mujoco.MjData(gm), mujoco.MjvOption(), mujoco.MjvPerturb())
    geoms = robot_collision_geoms(model)
    base_bid = floating_body(model)

    print(f"Replaying {name}: {T} frames, {t[-1]:.2f} s, speed {speed:g}x")
    print("controls: " + HELP.replace("\n", "\n          "))

    # playback state shared with the key callback (runs on the viewer thread)
    state = {"paused": False, "k": 0, "sim_t": 0.0, "speed": float(speed),
             "ghost": ghost is not None, "track": False, "cam_dirty": False}

    def seek(k, pause=True):
        state["k"] = int(min(max(k, 0), T - 1))
        state["sim_t"] = float(t[state["k"]])
        if pause:
            state["paused"] = True

    def step_speed(direction):
        s = state["speed"]
        if direction > 0:
            state["speed"] = next((v for v in SPEEDS if v > s + 1e-9), SPEEDS[-1])
        else:
            state["speed"] = next((v for v in reversed(SPEEDS) if v < s - 1e-9), SPEEDS[0])

    # frame jumps that pause: right/left, up/down, page up/down
    jumps = {262: 1, 263: -1, 265: 50, 264: -50, 266: 500, 267: -500}

    def key_cb(keycode):
        if keycode == 32:                                  # space: pause / resume
            state["paused"] = not state["paused"]
        elif keycode in jumps:
            seek(state["k"] + jumps[keycode])
        elif 48 <= keycode <= 57:                          # 0-9: seek to 0-90%, keep playing state
            seek(int((keycode - 48) / 10.0 * (T - 1)), pause=False)
        elif keycode == 268:                               # home
            seek(0)
        elif keycode == 269:                               # end
            seek(T - 1)
        elif keycode == 45:                                # '-'
            step_speed(-1)
        elif keycode == 61:                                # '='
            step_speed(+1)
        elif keycode == 92:                                # '\'
            state["speed"] = 1.0
        elif keycode == 82:                                # R: restart
            seek(0, pause=False)
        elif keycode == 71 and ghost is not None:          # G: ghost on / off
            state["ghost"] = not state["ghost"]
        elif keycode == 67 and base_bid >= 0:              # C: camera lock
            state["track"] = not state["track"]
            state["cam_dirty"] = True

    with mujoco.viewer.launch_passive(model, mdata, key_callback=key_cb,
                                      show_left_ui=False, show_right_ui=False) as viewer:

        # the viewer's own shortcuts also fire: digits toggle geom groups, C toggles contact points
        geomgroup = viewer.opt.geomgroup.copy()
        set_pose(model, mdata, quat[0], q[0], geoms)
        viewer.cam.lookat[:] = mdata.xpos[max(base_bid, 0)]
        viewer.cam.distance = 3.0

        try:
            last = time.perf_counter()
            while viewer.is_running():
                tick = time.perf_counter()
                if not state["paused"]:
                    state["sim_t"] += (tick - last) * state["speed"]
                    if state["sim_t"] > t[-1]:
                        state["sim_t"] = state["sim_t"] % t[-1] if t[-1] > 0 else 0.0
                    state["k"] = min(max(int(np.searchsorted(t, state["sim_t"], side="right")) - 1, 0), T - 1)
                last = tick
                k = state["k"]

                set_pose(model, mdata, quat[k], q[k], geoms)

                # user scene: ghost at the logged base with q_des joints
                viewer.user_scn.ngeom = 0
                if state["ghost"]:
                    add_ghost(viewer.user_scn, *ghost, np.concatenate([mdata.qpos[:7], q_des[k]]))

                if state["cam_dirty"]:
                    state["cam_dirty"] = False
                    if state["track"]:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                        viewer.cam.trackbodyid = base_bid
                    else:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.opt.geomgroup[:] = geomgroup
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False

                # overlays: status top-left, key help bottom-left
                flags = "".join([
                    "   [PAUSED]" if state["paused"] else "",
                    "   [GHOST]" if state["ghost"] else "",
                    "   [CAM LOCK]" if state["track"] else "",
                ])
                status = (f"{name}\n"
                          f"time = {t[k]:.2f} / {t[-1]:.2f} s   frame {k}/{T - 1}\n"
                          f"speed = {state['speed']:g}x{flags}")
                if js is not None:
                    status += f"\njoystick  vx {js[k, 1]:+.2f}   vy {js[k, 2]:+.2f}   wz {js[k, 3]:+.2f}"
                viewer.set_texts([
                    (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, status, None),
                    (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, HELP, None),
                ])

                viewer.sync()

                sleep_for = 1.0 / TARGET_FPS - (time.perf_counter() - tick)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            pass
