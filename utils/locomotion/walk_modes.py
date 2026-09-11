##
#
# Upright walking differential-drive: joystick -> twist command shaping.
#
# Shared by every G1-Standing-DiffDrive control path (simulation, hardware) so the
# command logic can NEVER drift between them.
#
# The walk gait library is three DISJOINT one-axis bands plus idle (mjlab
# tasks/standing_diffdrive, clips from mj-nlp examples/g1_mimic_periodic/library):
#   - forward  : [vx, 0, 0],  vx  in [+0.50, +1.00] m/s
#   - backward : [vx, 0, 0],  vx  in [-0.90, -0.40] m/s
#   - turn     : [0, 0, wz], |wz| in [ 0.50,  1.50] rad/s, in place
#   - idle     : [0, 0, 0],   the standing pose
# The policy was trained ONLY on commands inside those bands, stepping between them:
# never ramping through the gaps, never vx and wz at once, never vy. So every control
# tick the two sticks become a twist in four steps:
#   1. DEADBAND  both sticks inside `stick_deadband`             -> idle, immediately
#   2. MODE      the dominant stick picks straight XOR turn, with a hysteresis margin so
#                a diagonal stick does not chatter between the two; the straight stick's
#                sign picks forward vs backward
#   3. DWELL     a band-to-band change (forward <-> backward <-> turn) waits until the
#                current band has been active for `mode_dwell_time`; the last twist is
#                held meanwhile. Leaving idle and STOPPING are never delayed. Each band
#                change is a clip switch for the policy, and training switched on a
#                3-8 s timer, so this keeps switches from happening every tick.
#   4. REMAP     the live part of the stick, [deadband, 1], maps LINEARLY onto the band:
#                a light touch is the slowest trained gait, full stick the fastest. The
#                gaps (0 < |vx| < 0.4 or 0.5, 0 < |wz| < 0.5) are never commanded.
# The crawl recipe (linear stick scale + a magnitude deadband) would leave half the
# stick dead and could never reach the slow end of the backward band; hence the remap.
#
##

import numpy as np


MODES = ("idle", "forward", "backward", "turn")


# |stick| in [deadband, 1] -> [lo, hi] linearly (lo at the deadband edge, hi at full stick)
def remap_stick(stick, deadband, lo, hi):
    span = max(1.0 - deadband, 1e-6)
    t = (min(abs(float(stick)), 1.0) - deadband) / span
    return lo + (hi - lo) * float(np.clip(t, 0.0, 1.0))


# pick the band the sticks ask for. `current` is the active mode; the hysteresis margin
# favours it when the two sticks are close, so a diagonal stick does not flip modes.
def select_walk_mode(fwd_stick, turn_stick, current, *, deadband, hysteresis):
    a_f, a_t = abs(float(fwd_stick)), abs(float(turn_stick))
    if a_f < deadband and a_t < deadband:
        return "idle"
    straight = "forward" if fwd_stick > 0.0 else "backward"

    if current == "turn":
        # keep turning unless the straight stick is live AND clearly dominant
        want_turn = not (a_f >= deadband and a_f > a_t + hysteresis)
    elif current in ("forward", "backward"):
        # keep going straight unless the turn stick is live AND clearly dominant
        want_turn = a_t >= deadband and a_t > a_f + hysteresis
    else:  # from idle: no preference, the larger stick wins
        want_turn = a_t >= a_f

    if want_turn:
        return "turn" if a_t >= deadband else straight
    return straight if a_f >= deadband else "turn"


# the twist for a mode, from the live stick magnitude remapped onto that band
def shape_walk_twist(mode, fwd_stick, turn_stick, *, deadband, fwd_vx, bwd_vx, turn_wz):
    if mode == "forward":
        vx = remap_stick(fwd_stick, deadband, min(fwd_vx), max(fwd_vx))
        return np.array([vx, 0.0, 0.0], dtype=np.float32)
    if mode == "backward":
        # bwd_vx is negative: the SLOW end is the value nearest zero
        vx = remap_stick(fwd_stick, deadband, max(bwd_vx), min(bwd_vx))
        return np.array([vx, 0.0, 0.0], dtype=np.float32)
    if mode == "turn":
        sign = 1.0 if turn_stick >= 0.0 else -1.0
        wz = remap_stick(turn_stick, deadband, min(turn_wz), max(turn_wz))
        return np.array([0.0, 0.0, sign * wz], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


class WalkTwistCommander:
    """Stateful joystick -> twist shaper (mode + dwell timer). One per control node.

    `cfg` is the loaded controller yaml (stick_deadband, mode_hysteresis, mode_dwell_time,
    fwd_vx_range, bwd_vx_range, turn_wz_range); `dt` the control period. Call
    `update(fwd_stick, turn_stick)` once per control tick; it returns (mode, twist)."""

    def __init__(self, cfg, dt):
        self.deadband = float(cfg["stick_deadband"])
        self.hysteresis = float(cfg["mode_hysteresis"])
        self.dwell = float(cfg["mode_dwell_time"])
        self.fwd_vx = tuple(float(v) for v in cfg["fwd_vx_range"])
        self.bwd_vx = tuple(float(v) for v in cfg["bwd_vx_range"])
        self.turn_wz = tuple(float(v) for v in cfg["turn_wz_range"])
        self.dt = float(dt)
        assert 0.0 <= self.deadband < 1.0, "stick_deadband must be in [0, 1)"
        assert min(self.fwd_vx) > 0.0 and max(self.bwd_vx) < 0.0 and min(self.turn_wz) > 0.0, \
            "band ranges must not include 0 (0 is the idle clip)"
        self.reset()

    def reset(self):
        self.mode = "idle"
        self.time_in_mode = 0.0
        self.twist = np.zeros(3, dtype=np.float32)

    def update(self, fwd_stick, turn_stick):
        want = select_walk_mode(fwd_stick, turn_stick, self.mode,
                                deadband=self.deadband, hysteresis=self.hysteresis)
        if want != self.mode:
            # stopping and starting are immediate; a band-to-band change waits out the dwell
            if want == "idle" or self.mode == "idle" or self.time_in_mode >= self.dwell:
                self.mode = want
                self.time_in_mode = 0.0
            else:
                # change pending: hold the last twist of the current band
                self.time_in_mode += self.dt
                return self.mode, self.twist
        self.twist = shape_walk_twist(self.mode, fwd_stick, turn_stick,
                                      deadband=self.deadband, fwd_vx=self.fwd_vx,
                                      bwd_vx=self.bwd_vx, turn_wz=self.turn_wz)
        self.time_in_mode += self.dt
        return self.mode, self.twist

    # the stick magnitudes at which each band starts / ends, for start-up printouts
    def describe(self):
        d = self.deadband
        return (f"stick deadband {d:.2f} (both sticks inside -> idle/stand); "
                f"forward  stick [{d:.2f}, 1] -> vx [{min(self.fwd_vx):+.2f}, {max(self.fwd_vx):+.2f}] m/s; "
                f"backward stick [{d:.2f}, 1] -> vx [{max(self.bwd_vx):+.2f}, {min(self.bwd_vx):+.2f}] m/s; "
                f"turn     stick [{d:.2f}, 1] -> |wz| [{min(self.turn_wz):.2f}, {max(self.turn_wz):.2f}] rad/s; "
                f"hysteresis {self.hysteresis:.2f}, band-change dwell {self.dwell:.2f} s")
