##
#
# Upright jogging UNICYCLE: joystick -> twist command shaping.
#
# Shared by every G1-Jog-Unicycle control path (simulation, hardware) so the command
# logic can NEVER drift between them.
#
# The jog gait library is two (vx, wz) BOXES plus idle (mj-nlp examples/g1_mimic_periodic/
# library, families run_fwd + run_bck on a 0.1 m/s x 0.1 rad/s grid, all T = 0.86 s):
#   - forward  : vx in [+0.50, +1.50] m/s  x  wz in [-0.50, +0.50] rad/s
#   - backward : vx in [-1.00, -0.50] m/s  x  wz in [-0.50, +0.50] rad/s
#   - idle     : [0, 0, 0],   the standing pose
# Every clip realizes its vx AND wz at once (an arc: the vx=1.0, wz=0.5 clip turns 0.43 rad
# per stride), so drive and steer are commanded TOGETHER. This is the opposite of the walk
# library (walk_modes.py), whose clips are straight XOR turn-in-place and whose sticks must
# therefore compete: dominant stick, hysteresis, dwell. None of that exists here -- the two
# sticks never fight, every (vx, wz) inside a box is within 0.05 of a trained clip.
# What the library does NOT have: vy (never commanded), a turn in place (wz without vx), and
# the gap 0 < |vx| < 0.5. So every control tick the two sticks become a twist in three steps:
#   1. DEADBAND  drive stick inside `stick_deadband`             -> idle, immediately, whatever
#                the steer stick does (a turn in place is not in the library)
#   2. DRIVE     the drive stick's sign picks forward vs backward; its live range [deadband, 1]
#                maps LINEARLY onto that band: a light touch is the slowest trained jog, full
#                stick the fastest. The gap is never commanded.
#   3. STEER     the steer stick's live range [deadband, 1] maps LINEARLY onto its side of
#                `curve_wz_range` (left +, right -), on top of the drive. wz is the BODY yaw
#                rate in both directions, so in reverse the robot steers like a car backing up:
#                stick left -> nose swings left, tail right.
# Forward <-> backward always passes through the deadband (a stop); within a band the twist
# follows the sticks continuously, matching the continuous (vx, wz) box the sampler drew from.
# The stick remap is the walk library's (walk_modes.remap_stick): a linear stick scale would
# leave the stick below 0.33 dead and never reach the slow end of the backward band.
#
##

import numpy as np

from utils.locomotion.walk_modes import remap_stick


MODES = ("idle", "forward", "backward")


# pick the band the drive stick asks for. The steer stick has no say: it cannot make the
# robot move, and it cannot stop it.
def select_jog_mode(fwd_stick, *, deadband):
    if abs(float(fwd_stick)) < deadband:
        return "idle"
    return "forward" if float(fwd_stick) > 0.0 else "backward"


# the twist for a mode, from the live stick magnitudes remapped onto that band's box
def shape_jog_twist(mode, fwd_stick, turn_stick, *, deadband, fwd_vx, bwd_vx, curve_wz):
    if mode == "idle":
        return np.zeros(3, dtype=np.float32)

    if mode == "forward":
        vx = remap_stick(fwd_stick, deadband, min(fwd_vx), max(fwd_vx))
    else:
        # bwd_vx is negative: the SLOW end is the value nearest zero
        vx = remap_stick(fwd_stick, deadband, max(bwd_vx), min(bwd_vx))

    # steer: each side of the stick maps onto its own side of the curve range (left +)
    if float(turn_stick) >= 0.0:
        wz = remap_stick(turn_stick, deadband, 0.0, max(curve_wz))
    else:
        wz = remap_stick(turn_stick, deadband, 0.0, min(curve_wz))

    return np.array([vx, 0.0, wz], dtype=np.float32)


class JogTwistCommander:
    """Joystick -> unicycle twist shaper. One per control node.

    `cfg` is the loaded controller yaml (stick_deadband, fwd_vx_range, bwd_vx_range,
    curve_wz_range). Call `update(fwd_stick, turn_stick)` once per control tick; it returns
    (mode, twist). Stateless apart from the last mode / twist, kept for logging."""

    def __init__(self, cfg):
        self.deadband = float(cfg["stick_deadband"])
        self.fwd_vx = tuple(float(v) for v in cfg["fwd_vx_range"])
        self.bwd_vx = tuple(float(v) for v in cfg["bwd_vx_range"])
        self.curve_wz = tuple(float(v) for v in cfg["curve_wz_range"])
        assert 0.0 <= self.deadband < 1.0, "stick_deadband must be in [0, 1)"
        assert min(self.fwd_vx) > 0.0 and max(self.bwd_vx) < 0.0, \
            "vx bands must not include 0 (0 is the idle clip)"
        assert min(self.curve_wz) <= 0.0 <= max(self.curve_wz), \
            "curve_wz_range must contain 0 (a straight jog)"
        self.reset()

    def reset(self):
        self.mode = "idle"
        self.twist = np.zeros(3, dtype=np.float32)

    def update(self, fwd_stick, turn_stick):
        self.mode = select_jog_mode(fwd_stick, deadband=self.deadband)
        self.twist = shape_jog_twist(self.mode, fwd_stick, turn_stick,
                                     deadband=self.deadband, fwd_vx=self.fwd_vx,
                                     bwd_vx=self.bwd_vx, curve_wz=self.curve_wz)
        return self.mode, self.twist

    # the stick magnitudes at which each band starts / ends, for start-up printouts
    def describe(self):
        d = self.deadband
        return (f"drive stick inside {d:.2f} -> idle/stand (no turn in place); "
                f"forward  stick [{d:.2f}, 1] -> vx [{min(self.fwd_vx):+.2f}, {max(self.fwd_vx):+.2f}] m/s; "
                f"backward stick [{d:.2f}, 1] -> vx [{max(self.bwd_vx):+.2f}, {min(self.bwd_vx):+.2f}] m/s; "
                f"steer    stick [{d:.2f}, 1] -> wz [0, {max(self.curve_wz):+.2f}] left / "
                f"[0, {min(self.curve_wz):+.2f}] right rad/s, together with the drive")
