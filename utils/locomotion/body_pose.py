##
#
# Go2 body pose command shaping (base height, roll, pitch).
#
# NumPy port of mjlab's velocity_custom.mdp.body_pose.effective_body_pose, the
# rule the Go2-Velocity-Pose-Custom policy was trained under. Every Go2 pose
# control path (simulation, hardware) goes through this one function so the
# commands can never leave the training distribution:
#   - roll and pitch apply only while the twist is ~zero (standing)
#   - while moving, height is clamped to the walking band, and its floor rises
#     linearly with speed (low crouch-walking only at slow speeds)
#   - roll and pitch are scaled down so every foot stays within the legs'
#     height travel at the commanded height
#
##

import numpy as np


# raw [height, roll, pitch] -> the pose the policy is commanded with
def effective_body_pose(raw, twist, p):
    height, roll, pitch = float(raw[0]), float(raw[1]), float(raw[2])
    moving = np.linalg.norm(twist) > p["standing_threshold"]

    if moving:
        speed = float(np.linalg.norm(twist[:2]))
        v0, v1 = p["crouch_speed_range"]
        frac = min(max((speed - v0) / (v1 - v0), 0.0), 1.0)
        walk_lo = p["walking_height_range"][0] + frac * (p["fast_min_height"] - p["walking_height_range"][0])
        height = min(max(height, walk_lo), p["walking_height_range"][1])
        roll, pitch = 0.0, 0.0

    # scale the attitude (in sine space) so every foot can still reach the ground
    lo, hi = p["leg_height_limits"]
    margin = max(min(height - lo, hi - height), 0.0)
    need = p["foot_half_length"] * abs(np.sin(pitch)) + p["foot_half_width"] * abs(np.sin(roll))
    if need > margin:
        scale = margin / max(need, 1e-6)
        roll = float(np.arcsin(scale * np.sin(roll)))
        pitch = float(np.arcsin(scale * np.sin(pitch)))

    return np.array([height, roll, pitch], dtype=np.float32)
