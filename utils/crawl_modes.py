##
#
# Omnidirectional crawl mode selection + command shaping.
#
# Shared by every G1-Crawling-Omni control path (simulation, hardware, and the
# deterministic sim) so the command heuristic can NEVER drift between them.
#
# The omni gait library is three DISJOINT motions plus idle:
#   - forward   : vx in [+0.10, +0.30], with crab vy in +/-0.12 and gentle curve wz in +/-0.12
#   - backward  : vx in [-0.25, -0.05], same crab / curve
#   - turn      : IN PLACE (vx = vy = 0), |wz| in [0.28, 0.60]
#   - idle      : [0, 0, 0]
# Big turns (|wz| > 0.12) exist ONLY at vx = vy = 0, so a naive per-axis box lets
# you command corners the library lacks (e.g. fast-forward + hard-turn); during
# training the nearest reference to such a corner is a pure in-place turn, so the
# policy collapses translation into a spin. To prevent that, each control step we
# (1) SELECT a mode from the commanded vx, then (2) CLAMP the twist to that mode's
# trained range -- keeping every command on the library manifold:
#     vx >= fwd_vx_min  (0.10)         -> forward
#     vx <= bwd_vx_max  (-0.05)        -> backward
#     in between: |wz| > turn_wz_min   -> in-place turn, else idle
#
##

import numpy as np


# select the active crawl mode from the commanded twist. Mode is chosen from vx
# alone (partition of the vx axis); the near-zero-vx band is stationary, and there
# resolves to an in-place turn when |wz| is commanded, otherwise idle.
def select_crawl_mode(vx, wz, *, fwd_vx_min, bwd_vx_max, turn_wz_min):
    if vx >= fwd_vx_min:
        return "forward"
    if vx <= bwd_vx_max:
        return "backward"
    return "turn" if abs(wz) > turn_wz_min else "idle"


# clamp a physical twist to the given mode's trained gait-library range
def shape_crawl_twist(vx, vy, wz, mode, *, fwd_vx, bwd_vx, crab_max, curve_max, turn_wz):
    if mode == "forward":
        return np.array([np.clip(vx, fwd_vx[0], fwd_vx[1]),
                         np.clip(vy, -crab_max, crab_max),
                         np.clip(wz, -curve_max, curve_max)], dtype=np.float32)
    if mode == "backward":
        return np.array([np.clip(vx, bwd_vx[0], bwd_vx[1]),
                         np.clip(vy, -crab_max, crab_max),
                         np.clip(wz, -curve_max, curve_max)], dtype=np.float32)
    if mode == "turn":
        # in-place turn: zero translation, snap |wz| into the trained turn band
        sign = 1.0 if wz >= 0.0 else -1.0
        return np.array([0.0, 0.0, sign * np.clip(abs(wz), turn_wz[0], turn_wz[1])],
                        dtype=np.float32)
    # idle
    return np.zeros(3, dtype=np.float32)


# select + shape in one call. `cfg` is the loaded controller config (dict), which
# must carry the mode thresholds and the per-mode gait-library ranges.
def resolve_crawl_twist(twist, cfg):
    vx, vy, wz = float(twist[0]), float(twist[1]), float(twist[2])
    mode = select_crawl_mode(
        vx, wz,
        fwd_vx_min=cfg["mode_fwd_vx_min"],
        bwd_vx_max=cfg["mode_bwd_vx_max"],
        turn_wz_min=cfg["mode_turn_wz_min"],
    )
    shaped = shape_crawl_twist(
        vx, vy, wz, mode,
        fwd_vx=cfg["fwd_vx_range"],
        bwd_vx=cfg["bwd_vx_range"],
        crab_max=cfg["crab_max"],
        curve_max=cfg["curve_max"],
        turn_wz=cfg["turn_wz_range"],
    )
    return mode, shaped
