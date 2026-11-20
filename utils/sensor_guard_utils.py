##
#
# Sensor guards: stand-ins for sensor data the robot fails to deliver.
# Methods and measurements: deploy/hardware/g1/README.md ("Pelvis IMU backup", "Joint dropout guard").
#
##

# other imports
import numpy as np
from scipy.spatial.transform import Rotation as R


############################################################################
# PELVIS IMU BACKUP
############################################################################

class PelvisImuBackup:
    """
    Stand-in for the pelvis IMU, built from the torso IMU and the waist joints (no fitted values).

    Call step() once per hardware tick. While the pelvis IMU is healthy, pass its quaternion so
    the estimate stays locked to it; while it is lost, pass None and use the returned values.
    Quaternions are [w, x, y, z]; waist_q is [yaw, roll, pitch] in the model's sign convention.
    """

    # longest gap [sec] between ticks over which waist rates are still differenced
    MAX_RATE_DT = 0.05

    def __init__(self):
        self.reset()

    # forget the latched correction and the previous waist sample
    def reset(self):
        self.heading = 0.0            # yaw offset between the two IMUs' world frames [rad]
        self.residual = R.identity()  # small pelvis-frame rotation left after the heading offset
        self.latched = False
        self.prev_t = None
        self.prev_waist_q = None

    # advance one tick; returns (pelvis quaternion [w, x, y, z], pelvis angular velocity)
    def step(self, t, waist_q, torso_quat, torso_gyro, pelvis_quat=None):
        waist_q = np.asarray(waist_q, dtype=np.float64)
        yaw, roll, pitch = waist_q
        Rz, Rx, Ry = R.from_euler("z", yaw), R.from_euler("x", roll), R.from_euler("y", pitch)

        # pelvis-from-torso rotation through the waist, and the pelvis pose the torso implies
        R_pt = Rz * Rx * Ry
        R_fk = _from_wxyz(torso_quat) * R_pt.inv()

        # waist joint rates from position differences (cleaner than the reported velocities)
        rates = np.zeros(3)
        if self.prev_t is not None and 0.0 < t - self.prev_t <= self.MAX_RATE_DT:
            rates = (waist_q - self.prev_waist_q) / (t - self.prev_t)
        self.prev_t, self.prev_waist_q = t, waist_q

        # torso gyro in the pelvis frame, minus the rotation the waist joints add on top
        waist_rate = (np.array([0.0, 0.0, rates[0]])
                      + Rz.apply([rates[1], 0.0, 0.0])
                      + (Rz * Rx).apply([0.0, rates[2], 0.0]))
        gyro = R_pt.apply(torso_gyro) - waist_rate

        # healthy pelvis IMU: latch how it differs from the torso-implied pose
        if pelvis_quat is not None:
            R_p = _from_wxyz(pelvis_quat)
            x, y, z, w = (R_p * R_fk.inv()).as_quat()
            self.heading = 2.0 * np.arctan2(z, w)
            self.residual = R_fk.inv() * R.from_euler("z", -self.heading) * R_p
            self.latched = True

        # carry the latched difference forward on the live torso-implied pose
        R_est = R.from_euler("z", self.heading) * R_fk * self.residual if self.latched else R_fk
        return _to_wxyz(R_est), gyro


############################################################################
# JOINT DROPOUT GUARD
############################################################################

class JointDropoutGuard:
    """
    Fills in joint readings the robot failed to deliver (no tuned or fitted values).

    A motor record counts as lost when both its temperature sensors read 0 degC, once that motor
    has reported a real temperature (so a dead sensor never triggers it). A lost joint keeps its
    last good velocity, and its position advances at that velocity, clamped to the joint limits.
    Call step() once per hardware tick with every joint's raw reading.
    """

    def __init__(self, num_joints, q_min=None, q_max=None):
        self.q_min = None if q_min is None else np.asarray(q_min, dtype=np.float64)
        self.q_max = None if q_max is None else np.asarray(q_max, dtype=np.float64)
        self.temp_seen = np.zeros(num_joints, dtype=bool)
        self.last_q = np.zeros(num_joints)    # last good position, velocity and its time, per joint
        self.last_dq = np.zeros(num_joints)
        self.last_t = np.zeros(num_joints)

    # returns (q, dq, lost): the readings with lost joints filled in, and which joints were lost
    def step(self, t, q, dq, temps):
        q = np.array(q, dtype=np.float64)
        dq = np.array(dq, dtype=np.float64)

        # which motor records are lost this tick
        temp_zero = (np.asarray(temps) == 0).all(axis=1)
        lost = temp_zero & self.temp_seen
        self.temp_seen |= ~temp_zero

        # remember the good readings
        good = ~lost
        self.last_q[good], self.last_dq[good], self.last_t[good] = q[good], dq[good], t

        # lost joints: constant velocity from the last good reading
        if lost.any():
            q[lost] = self.last_q[lost] + self.last_dq[lost] * (t - self.last_t[lost])
            if self.q_min is not None:
                q[lost] = np.clip(q[lost], self.q_min[lost], self.q_max[lost])
            dq[lost] = self.last_dq[lost]
        return q, dq, lost


# joint position limits [rad] of the actuated joints, read from a MuJoCo model file
def joint_limits_from_model(xml_path):
    import mujoco
    model = mujoco.MjModel.from_xml_path(xml_path)
    hinge = model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
    return model.jnt_range[hinge, 0].copy(), model.jnt_range[hinge, 1].copy()


# scipy stores quaternions as [x, y, z, w]; the robot uses [w, x, y, z]
def _from_wxyz(q):
    return R.from_quat([q[1], q[2], q[3], q[0]])

def _to_wxyz(rot):
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z])
