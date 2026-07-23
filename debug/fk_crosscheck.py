##
#
# Forward-kinematics cross-check for the two G1 IMUs.
#
# The pelvis (primary) and torso (secondary) IMUs each run their own onboard
# fusion and drift independently in yaw. But the pelvis->torso transform is
# fully determined by the three waist joint encoders. So we can PREDICT the
# torso orientation from (measured pelvis quat) x FK(waist encoders) and compare
# it against the measured torso quat. The mismatch splits into:
#   - a roughly constant offset  -> IMU mounting / frame-convention difference
#   - a slowly growing component -> torso-IMU yaw drift (the suspected culprit)
#
# FK is done with the same MuJoCo model the controller loads
# (models/g1_29dof_scene.xml), reading the imu_pelvis / imu_torso framequat
# sensors, so the site offsets match training exactly.
#
##

import numpy as np
import mujoco

from utils.math_utils import quat_conjugate, quat_multiply, quat_to_rpy

# waist joint actuator indices within the 29-DoF joint vector (G1JointIndex)
WAIST_JOINT_IDX = (12, 13, 14)  # yaw, roll, pitch

# framequat sensor names in the model
PELVIS_QUAT_SENSOR = "pelvis_imu_quat_sensor"
TORSO_QUAT_SENSOR = "torso_imu_quat_sensor"


class TorsoFK:
    """Predicts torso IMU orientation from pelvis IMU orientation + waist encoders."""

    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # qpos addresses of the three waist joints (base occupies qpos[0:7])
        self.waist_qposadr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
        ]

    # relative rotation pelvis_site -> torso_site for one set of waist angles
    def _rel_quat(self, waist_angles) -> np.ndarray:
        self.data.qpos[:] = 0.0
        self.data.qpos[3] = 1.0  # base quat = identity (w,x,y,z)
        for adr, ang in zip(self.waist_qposadr, waist_angles):
            self.data.qpos[adr] = ang
        mujoco.mj_forward(self.model, self.data)
        pelvis_q = self.data.sensor(PELVIS_QUAT_SENSOR).data.copy()
        torso_q = self.data.sensor(TORSO_QUAT_SENSOR).data.copy()
        # pelvis site is identity here, so this is the pelvis->torso relative rotation
        return quat_multiply(quat_conjugate(pelvis_q), torso_q)

    # predict torso world quat for a whole log
    # pelvis_quat: (T,4) measured pelvis IMU quat [w,x,y,z]
    # q_joints:    (T,29) measured joint encoders
    def predict(self, pelvis_quat: np.ndarray, q_joints: np.ndarray) -> np.ndarray:
        T = pelvis_quat.shape[0]
        pred = np.empty((T, 4), dtype=np.float64)
        for i in range(T):
            waist = q_joints[i, list(WAIST_JOINT_IDX)]
            rel = self._rel_quat(waist)
            pred[i] = quat_multiply(pelvis_quat[i], rel)
        return pred


# divergence between measured and predicted torso quats, expressed as the
# relative rotation rel = predicted^-1 * measured. Returns:
#   angle_rad (T,)  : total geodesic rotation magnitude, 2*arccos(|rel_w|)
#   rpy       (T,3) : roll/pitch/yaw decomposition of rel
#   rel_quat  (T,4) : rel itself, [w,x,y,z] (w forced >= 0 for continuity)
def quat_divergence(measured: np.ndarray, predicted: np.ndarray):
    T = measured.shape[0]
    angle = np.empty(T, dtype=np.float64)
    rpy = np.empty((T, 3), dtype=np.float64)
    rel_quat = np.empty((T, 4), dtype=np.float64)
    for i in range(T):
        rel = quat_multiply(quat_conjugate(predicted[i]), measured[i])
        rel = rel / (np.linalg.norm(rel) + 1e-12)
        if rel[0] < 0:            # resolve q/-q double cover for a continuous plot
            rel = -rel
        angle[i] = 2.0 * np.arccos(np.clip(rel[0], 0.0, 1.0))
        rpy[i] = quat_to_rpy(rel)
        rel_quat[i] = rel
    return angle, rpy, rel_quat
