"""Running KCF contract and command gate.

Feedback Float64MultiArray (110): [policy_time, pose_time, applied_time, valid,
qpos(36), qvel(35), previous_raw_RL_action(29), current_policy_twist(3),
ungated gait sin/cos/period_frames(3)].
qpos: world position (m), body-to-world quaternion wxyz, joints (rad).
qvel: world linear (m/s), BODY angular (rad/s), joints (rad/s).

Command Float64MultiArray (6): [connected, vx_body, vy_body, wz_body,
application_time, feedback_time]. 

Mocap input: nav_msgs/Odometry on MOCAP_TOPIC (or kcf_mocap_topic in the
consumer YAML).
"""

from dataclasses import dataclass, replace
from collections import deque
import math
import numpy as np

FEEDBACK_TOPIC = "/deploy_kcf/feedback"
COMMAND_TOPIC = "/deploy_kcf/kcf_mpc"
SHADOW_TOPIC = "/deploy_kcf/shadow_command"
SIM_STATE_TOPIC = "/deploy_kcf/sim_state"
MOCAP_TOPIC = "/deploy_kcf/mocap"


@dataclass
class Feedback:
    time: float
    pose_time: float
    applied_time: float
    valid: bool
    q: np.ndarray
    v: np.ndarray
    previous_action: np.ndarray
    applied_command: np.ndarray
    policy_clock: np.ndarray

    def pack(self):
        return np.concatenate((
            [self.time, self.pose_time, self.applied_time, float(self.valid)],
            self.q, self.v, self.previous_action, self.applied_command, self.policy_clock,
        )).tolist()

    @classmethod
    def unpack(cls, data, max_pose_age=0.1):
        a = np.asarray(data, dtype=np.float64)
        if a.shape != (110,) or not np.isfinite(a).all():
            raise ValueError("feedback must contain 110 finite values (29 joints and running gait clock)")
        t, pose_t, applied_t, valid = a[:4]
        if valid != 1 or t < 0 or pose_t < 0 or not (-1e-6 <= t - pose_t <= max_pose_age):
            raise ValueError("invalid, future or stale pelvis feedback")
        if applied_t < 0 or not (-1e-6 <= t - applied_t <= 0.1):
            raise ValueError("invalid applied-command timestamp")
        if abs(np.linalg.norm(a[7:11]) - 1.0) > 0.01:
            raise ValueError("pelvis quaternion must be unit wxyz")
        if a[109] not in (37., 43.) or not np.isclose(np.linalg.norm(a[107:109]), 1., atol=1e-5):
            raise ValueError('running requires the ungated unit gait clock and period 37 or 43')
        return cls(t, pose_t, applied_t, True, a[4:40].copy(),
                   a[40:75].copy(), a[75:104].copy(), a[104:107].copy(), a[107:110].copy())


class CommandGate:
    """Manual in control, timed MPC in track, otherwise zero velocity.

    Zero velocity is NOT an emergency stop.
    """

    def __init__(self, timeout=0.1, limits=(2.0, 0.0, 2.0), dt=0.02, command_bands=None):
        self.timeout, self.dt = float(timeout), float(dt)
        self.limits = np.asarray(limits, dtype=np.float64)
        if (not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(dt) or dt <= 0
                or self.limits.shape != (3,) or not np.isfinite(self.limits).all()
                or np.any(self.limits < 0) or not np.any(self.limits > 0)):
            raise ValueError("invalid command gate configuration")
        self.bands = None if command_bands is None else np.asarray(command_bands, dtype=float)
        if self.bands is not None and (self.bands.ndim != 3 or self.bands.shape[1:] != (2, 3)
                or not len(self.bands) or not np.isfinite(self.bands).all()
                or np.any(self.bands[:, 0] > self.bands[:, 1])
                or np.any(np.abs(self.bands) > self.limits + 1e-6)):
            raise ValueError('invalid command bands')
        self.fsm, self.fsm_wall = "init", -math.inf
        self.manual, self.manual_wall = None, -math.inf
        self.candidates, self.last_slot = {}, None

    def set_fsm(self, state, now):
        if state != self.fsm:
            self.candidates.clear()
        self.fsm, self.fsm_wall = str(state), float(now)

    def set_manual(self, data, now):
        a = np.asarray(data, dtype=float)
        self.manual = (a[1:4].copy() if a.ndim == 1 and len(a) >= 4
                       and np.isfinite(a[:4]).all() and a[0] == 1 else None)
        self.manual_wall = float(now)

    def set_mpc(self, data, now):
        a = np.asarray(data, dtype=float)
        if (a.shape != (6,) or not np.isfinite(a).all() or a[0] != 1
                or np.any(np.abs(a[1:4]) > self.limits + 1e-6)
                or a[5] < 0 or not (0 <= a[4] - a[5] <= 0.04 + 1e-6)):
            self.candidates.clear()
            return False
        if self.bands is not None and not np.any(np.all(
                (a[1:4] >= self.bands[:, 0] - 1e-6) & (a[1:4] <= self.bands[:, 1] + 1e-6), axis=1)):
            self.candidates.clear()
            return False
        key = round(a[4] / self.dt)
        self.candidates[key] = (a.copy(), float(now))
        while len(self.candidates) > 8:
            del self.candidates[min(self.candidates)]
        return True

    def select(self, policy_time, now):
        slot = round(policy_time / self.dt)
        if self.last_slot is not None and slot < self.last_slot:
            self.candidates.clear()
        self.last_slot = slot
        for old in [k for k in self.candidates if k < slot]:
            del self.candidates[old]
        operator = (now - self.fsm_wall < self.timeout
                    and now - self.manual_wall < self.timeout and self.manual is not None)
        if not operator:
            self.candidates.clear()
            return np.zeros(3, dtype=np.float32), "operator_unavailable"
        if self.fsm == "control":
            command = np.clip(self.manual, -self.limits, self.limits)
            if self.bands is not None:
                choices = np.clip(command, self.bands[:, 0], self.bands[:, 1])
                command = choices[np.argmin(np.sum((choices - command)**2, axis=1))]
            return command.astype(np.float32), "manual"
        if self.fsm == "track":
            candidate = self.candidates.pop(slot, None)
            if candidate is not None:
                a, received = candidate
                if now - received < self.timeout and abs(policy_time - a[4]) <= 0.003:
                    return a[1:4].astype(np.float32), "kcf"
            return np.zeros(3, dtype=np.float32), "mpc_unavailable"
        self.candidates.clear()
        return np.zeros(3, dtype=np.float32), "fsm_inactive"


def mocap_base(msg):
    # TODO:Edit this to fix the correct mocap structure.
    """Calibrated pelvis Odometry: world pose and child/body-frame twist.
    """
    if msg.header.frame_id != "world" or msg.child_frame_id != "pelvis":
        raise ValueError("mocap must describe world -> pelvis, not a raw marker")
    p, o = msg.pose.pose.position, msg.pose.pose.orientation
    q = np.array([p.x, p.y, p.z, o.w, o.x, o.y, o.z], dtype=float)
    if not np.isfinite(q).all() or abs(np.linalg.norm(q[3:]) - 1) > 0.01:
        raise ValueError("invalid mocap pose")
    w, x, y, z = q[3:]
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    vel = msg.twist.twist.linear
    world_linear = rotation @ np.array([vel.x, vel.y, vel.z])
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    if not np.isfinite(world_linear).all():
        raise ValueError("invalid mocap velocity")
    return stamp, q, world_linear


class StateProvider:
    """Select pelvis feedback without inventing missing proprioception/actions.

    Both sources require complete Feedback from the policy/state provider.
    Mocap supplies a calibrated pelvis pose/velocity in the same clock as that
    feedback; the OptiTrack adapter deploy/mocap/estimator.py publishes this input.
    """

    def __init__(self, source='simulation'):
        if source not in ('simulation', 'mocap'):
            raise ValueError('state source must be simulation or mocap')
        self.source = source
        self.poses = deque(maxlen=8)

    def set_mocap(self, msg, now):
        try:
            pose = mocap_base(msg)
        except (ValueError, TypeError, AttributeError):
            self.poses.clear()
            raise ValueError('invalid mocap measurement') from None
        # Republishing the same measurement must not renew its wall freshness.
        if not any(old[0] == pose[0] for old, _ in self.poses):
            self.poses.append((pose, float(now)))

    def select(self, feedback, now, timeout):
        if self.source == 'simulation':
            return feedback
        fresh = [pose for pose, received in self.poses if now - received < timeout]
        if not fresh:
            raise ValueError('mocap unavailable or stale')
        # Independent ROS subscriptions can deliver a newer pose before an
        # older policy frame. Select the latest non-future measurement instead
        # of spuriously disarming or fusing a pose from the future.
        eligible = [pose for pose in fresh if -1e-6 <= feedback.time - pose[0] <= timeout]
        if not eligible:
            raise ValueError('mocap timestamp is future, stale or on a different clock')
        stamp, base, world_linear = max(eligible, key=lambda pose: pose[0])
        q, v = feedback.q.copy(), feedback.v.copy()
        q[:7], v[:3] = base, world_linear
        return replace(feedback, pose_time=stamp, q=q, v=v)
