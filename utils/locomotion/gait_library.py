##
#
# Gait-library clock: commanded twist -> nearest library clip -> per-clip phase clock.
#
# Deploy-side mirror of mjlab's LibraryMotionCommand (crawling_fwd/mdp/commands.py) for the
# policies trained on a twist-labelled gait library. The actor never sees the reference
# motion, but it DOES see two things that depend on which clip the twist snaps to:
#   - motion_phase      (sin, cos) of 2*pi*t/n, where n is the SNAPPED CLIP'S own frame
#                       count, blanked to (0, 0) on the idle clip
#   - commanded_twist   the raw twist, zeroed on the idle clip (the idle "Voronoi cell")
# A single-period library (every clip n = T) reduces to the fixed-T clock the jog / walk
# controllers use. A RAGGED library (G1-WalkJog-Unicycle: walk clips n = 70, jog clips
# n = 43) needs the snap, because the twist decides the period: crossing the walk/jog cut
# switches n, and the frame index is remapped to the same PHASE of the new cycle, exactly as
# mjlab's _phase_remap does in training, so the gait changes stride without a phase jump.
#
# Everything comes from the policy's own metadata (mjlab exports it; policy/
# strip_motion_library.py --attach adds it to older exports):
#   library_clips         [[vx, vy, wz, n_frames], ...] in mjlab's loader (filename) order
#   twist_metric_weights  [wx, wy, wz] for the weighted-L2 nearest-clip snap
#
##

import math

import numpy as np


class GaitLibraryClock:
    """Nearest-clip snap + per-clip phase clock for one control node.

    Per control tick:
        twist_obs = clock.update(twist)   # snap, remap phase on a length change, idle stop
        phase_obs = clock.phase_obs()     # (sin, cos), (0, 0) on the idle clip
        ... run the policy ...
        clock.advance()                   # one frame, wrapping at the clip's own length
    """

    def __init__(self, policy):
        for key in ("library_clips", "twist_metric_weights"):
            assert key in policy.metadata, \
                f"policy has no '{key}' metadata; re-export it from mjlab or attach it with " \
                f"policy/strip_motion_library.py --attach"

        clips = np.asarray(policy.metadata["library_clips"], dtype=np.float64)
        assert clips.ndim == 2 and clips.shape[1] == 4, \
            f"library_clips must be [[vx, vy, wz, n_frames], ...], got shape {clips.shape}"
        self.lib_twists = clips[:, :3]                   # (S, 3)
        self.n_frames = clips[:, 3].astype(np.int64)     # (S,) each clip's own period
        self.weights = np.asarray(policy.metadata["twist_metric_weights"], dtype=np.float64)
        assert self.weights.shape == (3,), "twist_metric_weights must have 3 values"
        assert (self.n_frames > 0).all(), "library_clips has a clip with no frames"

        # the zero-twist idle clip (mjlab: norm of the weighted twist < 1e-4)
        idle = np.nonzero(np.linalg.norm(self.lib_twists * self.weights, axis=-1) < 1e-4)[0]
        self.idle_idx = int(idle[0]) if idle.size else None

        self.num_clips = len(self.n_frames)
        self.periods = sorted(set(int(n) for n in self.n_frames))
        self.ragged = len(self.periods) > 1
        self.reset()

    # stand on the idle clip at frame 0 (or clip 0 if the library has no idle clip)
    def reset(self):
        self.clip = self.idle_idx if self.idle_idx is not None else 0
        self.t = 0
        self.is_idle = self.idle_idx is not None

    # index of the library clip nearest to `twist` under the weighted-L2 twist metric
    def snap(self, twist):
        d = ((self.lib_twists - np.asarray(twist, dtype=np.float64)) ** 2 * self.weights).sum(axis=-1)
        return int(np.argmin(d))

    # snap to the twist's clip, remap the frame on a length change, and return the twist
    # the policy observes (zero on the idle clip)
    def update(self, twist):
        new = self.snap(twist)
        if new != self.clip:
            n_old, n_new = int(self.n_frames[self.clip]), int(self.n_frames[new])
            if n_old != n_new:
                # same PHASE, nearest frame; a phase of ~1 wraps to frame 0 (mjlab _phase_remap,
                # torch.round and np.round both round half to even)
                self.t = int(np.round(self.t / n_old * n_new)) % n_new
            self.clip = new
        self.is_idle = (self.idle_idx is not None) and (self.clip == self.idle_idx)
        if self.is_idle:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(twist, dtype=np.float32)

    # (sin, cos) of 2*pi*t/n for the current clip, blanked to (0, 0) on the idle clip
    def phase_obs(self):
        if self.is_idle:
            return np.zeros(2, dtype=np.float32)
        ang = 2.0 * math.pi * (self.t / int(self.n_frames[self.clip]))
        return np.array([math.sin(ang), math.cos(ang)], dtype=np.float32)

    # advance one frame, wrapping at the current clip's own length
    def advance(self):
        self.t = (self.t + 1) % int(self.n_frames[self.clip])

    # the current clip's period in frames and its twist label
    @property
    def period(self):
        return int(self.n_frames[self.clip])

    @property
    def clip_twist(self):
        return self.lib_twists[self.clip]

    # one-line summary for start-up printouts
    def describe(self):
        counts = {p: int((self.n_frames == p).sum()) for p in self.periods}
        idle = f"idle clip #{self.idle_idx}" if self.idle_idx is not None else "NO idle clip"
        return (f"{self.num_clips} clips, frames -> count {counts}, {idle}, "
                f"snap weights {self.weights.round(4).tolist()}")
