"""Generate samples from the reference selected in kcf_mpc/config/mpc_config.py.
"""

import numpy as np
from kcf_mpc.config.mpc_config import REFERENCE_MODES


def sincos_tracking_reference(reference):
    """Convert [..., px, py, theta, body twist, gravity] targets to ten features.

    Pose inputs and saved scenario settings retain their public nine-value
    format. Only the optimizer reference expands heading into a unit pair.
    """
    reference = np.asarray(reference)
    if reference.ndim == 0 or reference.shape[-1] != 9:
        raise ValueError("Pose/twist reference must end with nine values")
    return np.concatenate(
        (
            reference[..., :2],
            np.sin(reference[..., 2:3]),
            np.cos(reference[..., 2:3]),
            reference[..., 3:],
        ),
        axis=-1,
    )


class PlanarReference:
    """Configured planar pose/twist references, shared by control and plotting."""

    def __init__(self, settings):
        self.state = np.asarray(settings["reference"], dtype=np.float64)
        if (self.state.shape != (9,)
                or not np.isfinite(self.state).all()):
            raise ValueError(
                "MPC reference must contain nine finite planar pose, twist, and gravity values"
            )
        self.mode = settings["reference_mode"]
        if self.mode not in REFERENCE_MODES:
            raise ValueError(f"MPC reference_mode must be one of {REFERENCE_MODES}")
        if self.mode == "lissajous":
            try:
                trajectory = settings["lissajous_reference"]
                raw_trajectory = (
                    trajectory["half_width"], trajectory["half_height"], trajectory["omega"]
                )
                self.half_width, self.half_height, self.omega = map(
                    float, raw_trajectory
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "lissajous_reference requires half_width, half_height, and omega"
                ) from error
            if (any(isinstance(value, bool) for value in raw_trajectory)
                    or not np.isfinite(
                        [self.half_width, self.half_height, self.omega]
                    ).all()
                    or min(self.half_width, self.half_height, self.omega) <= 0.0):
                raise ValueError("Lissajous dimensions and omega must be finite and positive")
        if self.mode == "waypoints":
            self._configure_waypoints(settings.get("waypoint_reference"))
        if self.mode in ("rounded_square", "hairpin", "slalom", "stop_turn_go"):
            self._configure_maneuver(settings.get(f"{self.mode}_reference"))
        startup = settings.get("reference_startup")
        self.startup = None
        self._initial_pose = None
        self._configured_initial_pose = None
        if startup is not None:
            keys = ('hold_seconds', 'transition_seconds', 'maximum_linear_speed', 'maximum_yaw_rate')
            try:
                values = np.asarray([startup[key] for key in keys], dtype=float)
                boolean = any(isinstance(startup[key], (bool, np.bool_)) for key in keys)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError('Invalid reference_startup configuration') from error
            if boolean or not np.isfinite(values).all() or values[0] < 0 or np.any(values[1:] <= 0):
                raise ValueError('Startup hold must be nonnegative and duration/speeds positive and finite')
            self.startup = dict(zip(keys, values))
            if startup.get('initial_pose') is not None:
                self.set_initial_pose(startup['initial_pose'])
                self._configured_initial_pose = self._initial_pose.copy()

    def reset_initial_pose(self):
        self._initial_pose = None
        if self._configured_initial_pose is not None:
            self.set_initial_pose(self._configured_initial_pose)

    def set_initial_pose(self, pose):
        """Anchor startup to measured pose; no global-origin assumption."""
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError('Startup initial_pose must be a finite [x, y, yaw]')
        self._initial_pose = pose.copy()
        if self.startup is None:
            return
        first, _ = self._raw_reference(0.)
        delta = first - pose
        heading_delta = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
        self._startup_yaw_offset = heading_delta - delta[2]
        # Quintic pose steps peak at 1.875 * displacement / duration.
        self._startup_duration = max(self.startup['transition_seconds'],
            1.875 * np.linalg.norm(delta[:2]) / self.startup['maximum_linear_speed'],
            1.875 * abs(heading_delta) / self.startup['maximum_yaw_rate'])

    def _raw_reference(self, t):
        if self.mode == 'lissajous':
            return self.lissajous_reference(t)
        if self.mode == 'waypoints':
            return self.waypoints_reference(t)
        if self.mode == 'stop_turn_go':
            return self.waypoints_reference(t)
        if self.mode in ('rounded_square', 'hairpin', 'slalom'):
            return self.turning_reference(t)
        shape = np.shape(t) + (3,)
        return (np.broadcast_to(self.state[:3], shape).copy(),
                np.broadcast_to(self.state[3:6], shape).copy())

    @property
    def startup_delay(self):
        """Steady path-clock offset after the smooth startup finishes."""
        return (0. if self.startup is None or self._initial_pose is None else
                self.startup['hold_seconds'] + self._startup_duration / 2.)

    @property
    def duration(self):
        """Finite path-clock duration, or None for indefinitely repeating paths."""
        return (float(self.waypoint_times[-1])
                if self.mode in ('waypoints', 'stop_turn_go') else None)

    def path_time(self, t):
        """Saved path clock, used also to align waypoint completion/dwell checks."""
        t = np.asarray(t, dtype=float)
        if self.startup is None or self._initial_pose is None:
            return t
        duration = self._startup_duration
        elapsed = t - self.startup['hold_seconds']
        phase = np.clip(elapsed / duration, 0., 1.)
        integral = phase**4 * (2.5 + phase * (-3. + phase))
        return np.where(elapsed >= duration, elapsed - duration / 2., duration * integral)

    def _configure_waypoints(self, configuration):
        if not isinstance(configuration, dict) or set(configuration) != {"times_s", "poses"}:
            raise ValueError("waypoint_reference requires exactly times_s and poses")
        try:
            times = np.asarray(configuration["times_s"], dtype=np.float64)
            poses = np.asarray(configuration["poses"], dtype=np.float64)
            has_boolean = any(
                isinstance(value, (bool, np.bool_))
                for key in ("times_s", "poses")
                for value in np.asarray(configuration[key], dtype=object).flat
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Waypoint times and poses must be finite numeric arrays") from error
        if (has_boolean or times.ndim != 1 or times.size < 2
                or poses.shape != (times.size, 3)
                or not np.isfinite(times).all() or not np.isfinite(poses).all()
                or times[0] != 0.0 or not np.all(np.diff(times) > 0.0)):
            raise ValueError(
                "Waypoints require at least two finite [x, y, yaw] poses and strictly "
                "increasing finite times_s starting at zero"
            )
        self.waypoint_times = times.copy()
        self.waypoint_poses = poses.copy()
        self.waypoint_poses[:, 2] = np.unwrap(poses[:, 2])

    def _configure_maneuver(self, configuration):
        common = {'start_pose', 'speed'}
        extra = ({'distance', 'yaw_rate', 'dwell_seconds', 'turn_angles'}
                 if self.mode == 'stop_turn_go' else
                 {'turn_radius', 'straight_length', 'clockwise'})
        if self.mode == 'slalom':
            extra = extra | {'heading_amplitude'}
        if not isinstance(configuration, dict) or set(configuration) != common | extra:
            raise ValueError(f'{self.mode}_reference requires exactly {sorted(common | extra)}')

        def number(key, *, allow_zero=False):
            raw = configuration[key]
            try:
                value = float(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(f'{self.mode}_reference {key} must be a finite number') from error
            if (isinstance(raw, (bool, np.bool_)) or not np.isfinite(value)
                    or value < 0. or (not allow_zero and value == 0.)):
                raise ValueError(f'{self.mode}_reference {key} must be finite and '
                                 + ('nonnegative' if allow_zero else 'positive'))
            return value

        try:
            start = np.asarray(configuration['start_pose'], dtype=float).copy()
            boolean = any(isinstance(x, (bool, np.bool_))
                          for x in np.asarray(configuration['start_pose'], dtype=object).flat)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{self.mode}_reference start_pose must be finite [x, y, yaw]') from error
        if start.shape != (3,) or boolean or not np.isfinite(start).all():
            raise ValueError(f'{self.mode}_reference start_pose must be finite [x, y, yaw]')
        self._path_speed = number('speed')
        if self.mode == 'stop_turn_go':
            distance = number('distance')
            yaw_rate = number('yaw_rate')
            dwell = number('dwell_seconds', allow_zero=True)
            try:
                angles = np.asarray(configuration['turn_angles'], dtype=float)
                boolean = any(isinstance(x, (bool, np.bool_))
                              for x in np.asarray(configuration['turn_angles'], dtype=object).flat)
            except (TypeError, ValueError) as error:
                raise ValueError('stop_turn_go turn_angles must be nonzero finite radians') from error
            if (angles.ndim != 1 or not angles.size or boolean or not np.isfinite(angles).all()
                    or np.any(angles == 0.) or np.any(np.abs(angles) > np.pi)):
                raise ValueError('stop_turn_go turn_angles must be nonzero finite radians in [-pi, pi]')
            times, poses = [0.], [start]

            def append(pose, duration):
                if duration > 0.:
                    times.append(times[-1] + duration)
                    poses.append(pose.copy())

            for angle in [*angles, None]:
                pose = poses[-1].copy()
                pose[:2] += distance * np.array([np.cos(pose[2]), np.sin(pose[2])])
                append(pose, 1.875 * distance / self._path_speed)
                append(pose, dwell)
                if angle is not None:
                    pose[2] += angle
                    append(pose, 1.875 * abs(angle) / yaw_rate)
                    append(pose, dwell)
            self._configure_waypoints(dict(times_s=times, poses=poses))
            # Explicit signed turns choose the branch, including exact +/-pi.
            self.waypoint_poses[:, 2] = np.asarray(poses)[:, 2]
            return

        radius, straight = number('turn_radius'), number('straight_length')
        clockwise = configuration['clockwise']
        if not isinstance(clockwise, (bool, np.bool_)):
            raise ValueError(f'{self.mode}_reference clockwise must be boolean')
        direction = -1. if clockwise else 1.
        if self.mode == 'slalom':
            amplitude = number('heading_amplitude')
            if amplitude >= np.pi / 2.:
                raise ValueError('slalom heading_amplitude must be less than pi/2')
            segments = [(0., straight), (direction * amplitude, 0.), (0., straight),
                        (-2. * direction * amplitude, 0.), (0., straight),
                        (direction * amplitude, 0.)]
        else:
            angle, count = (np.pi / 2., 4) if self.mode == 'rounded_square' else (np.pi, 2)
            segments = [(0., straight), (direction * angle, 0.)] * count
        # Fixed-order quadrature integrates forward speed through a quintic
        # heading ramp; no discretized path interpolation or extra dependency.
        nodes, weights = np.polynomial.legendre.leggauss(32)
        self._turn_nodes, self._turn_weights = (nodes + 1.) / 2., weights / 2.
        self._turn_angles = np.array([angle for angle, _ in segments])
        self._turn_durations = np.array([
            1.875 * abs(angle) * radius / self._path_speed if angle else length / self._path_speed
            for angle, length in segments])
        if not np.isfinite(self._turn_durations).all() or np.any(self._turn_durations <= 0.):
            raise ValueError(f'{self.mode}_reference has invalid segment durations')
        self._turn_times = np.r_[0., np.cumsum(self._turn_durations)]
        if not np.isfinite(self._turn_times).all():
            raise ValueError(f'{self.mode}_reference has invalid path duration')
        self._turn_headings = start[2] + np.r_[0., np.cumsum(self._turn_angles[:-1])]
        delta = self._turn_displacement(np.arange(len(segments)), np.ones(len(segments)))
        self._turn_positions = start[:2] + np.vstack((np.zeros(2), np.cumsum(delta, axis=0)[:-1]))
        self.period = float(self._turn_times[-1])
        self._cycle_displacement = delta.sum(axis=0) if self.mode == 'slalom' else np.zeros(2)
        self._cycle_heading = 0. if self.mode == 'slalom' else direction * 2. * np.pi

    def _turn_displacement(self, index, phase):
        s = phase[..., None] * self._turn_nodes
        blend = s**3 * (10. + s * (-15. + 6. * s))
        heading = self._turn_headings[index][..., None] + self._turn_angles[index][..., None] * blend
        factor = self._path_speed * self._turn_durations[index] * phase
        return factor[..., None] * np.stack((
            np.sum(np.cos(heading) * self._turn_weights, axis=-1),
            np.sum(np.sin(heading) * self._turn_weights, axis=-1)), axis=-1)

    def turning_reference(self, t):
        """Continuous forward motion; smooth finite-rate turns, no corner stops."""
        t = np.asarray(t, dtype=float)
        if not np.isfinite(t).all():
            raise ValueError('Turning reference times must be finite')
        active = t >= 0.
        t = np.maximum(t, 0.)
        cycles = np.floor(t / self.period)
        local = np.remainder(t, self.period)
        index = np.minimum(np.searchsorted(self._turn_times, local, side='right') - 1,
                           len(self._turn_durations) - 1)
        phase = (local - self._turn_times[index]) / self._turn_durations[index]
        blend = phase**3 * (10. + phase * (-15. + 6. * phase))
        rate = 30. * phase**2 * (1. - phase)**2 / self._turn_durations[index]
        heading = (self._turn_headings[index] + self._turn_angles[index] * blend
                   + cycles * self._cycle_heading)
        xy = (self._turn_positions[index] + self._turn_displacement(index, phase)
              + cycles[..., None] * self._cycle_displacement)
        twist = np.stack((self._path_speed * np.cos(heading), self._path_speed * np.sin(heading),
                          self._turn_angles[index] * rate), axis=-1)
        return np.concatenate((xy, heading[..., None]), axis=-1), np.where(active[..., None], twist, 0.)

    def reference(self, t):
        """Reference pose and world twist at scalar/array controller time ``t``.

        Returns (q_ref, v_ref), each (..., 3): world pose [px, py, theta]
        and world twist [vx, vy, wz].
        """
        t = np.asarray(t, dtype=float)
        if not np.isfinite(t).all():
            raise ValueError('Reference query times must be finite')
        if self.startup is None or self._initial_pose is None:
            return self._raw_reference(t)
        duration = self._startup_duration
        elapsed = t - self.startup['hold_seconds']
        phase = np.clip(elapsed / duration, 0., 1.)
        blend = phase**3 * (10. + phase * (-15. + 6. * phase))
        derivative = 30. * phase**2 * (1. - phase)**2 / duration
        # Integrate the smooth clock rate: no path jump when startup finishes.
        pose, twist = self._raw_reference(self.path_time(t))
        pose[..., 2] += self._startup_yaw_offset
        delta = pose - self._initial_pose
        return (self._initial_pose + blend[..., None] * delta,
                derivative[..., None] * delta + blend[..., None]**2 * twist)

    def waypoints_reference(self, t):
        """Quintic pose interpolation, zero twist at each node and outside the path.

        Repeating a pose at consecutive times creates an explicit standing dwell.
        Yaw follows the shortest continuous branch between the configured nodes.
        """
        t = np.asarray(t, dtype=np.float64)
        if not np.isfinite(t).all():
            raise ValueError("Waypoint query times must be finite")
        index = np.clip(
            np.searchsorted(self.waypoint_times, t, side="right") - 1,
            0,
            len(self.waypoint_times) - 2,
        )
        duration = self.waypoint_times[index + 1] - self.waypoint_times[index]
        phase = np.clip((t - self.waypoint_times[index]) / duration, 0.0, 1.0)
        blend = phase**3 * (10.0 + phase * (-15.0 + 6.0 * phase))
        blend_rate = 30.0 * phase**2 * (1.0 - phase)**2 / duration
        delta = self.waypoint_poses[index + 1] - self.waypoint_poses[index]
        return (self.waypoint_poses[index] + blend[..., None] * delta,
                blend_rate[..., None] * delta)

    def lissajous_reference(self, t):
        """Baseline Lissajous pose/twist in the world frame, with tangent heading."""
        a, b, w = self.half_width, self.half_height, self.omega
        t = np.asarray(t, dtype=np.float64)
        px, py = a * np.sin(w * t), b * np.sin(2 * w * t)
        vx, vy = a * w * np.cos(w * t), 2 * b * w * np.cos(2 * w * t)
        ax, ay = -a * w**2 * np.sin(w * t), -4 * b * w**2 * np.sin(2 * w * t)
        theta = np.arctan2(vy, vx)
        # Keep diagnostic heading plots continuous. The MPC cost uses the
        # sine/cosine pair and is independent of this choice of angle branch.
        phase = np.remainder(w * t, 2 * np.pi)
        reverse_branch = ((phase >= 3 * np.pi / 4)
                          & (phase <= 5 * np.pi / 4)
                          & (theta > 0.0))
        theta = np.where(reverse_branch, theta - 2 * np.pi, theta)
        wz = (vx * ay - vy * ax) / (vx**2 + vy**2)
        return np.stack([px, py, theta], axis=-1), np.stack([vx, vy, wz], axis=-1)

    def track(self, q_ref, v_ref):
        """Build ten MPC [world position, heading sine/cosine, twist, gravity] targets."""
        cos_yaw, sin_yaw = np.cos(q_ref[..., 2]), np.sin(q_ref[..., 2])
        v_ref_body = np.stack(
            [cos_yaw * v_ref[..., 0] + sin_yaw * v_ref[..., 1],
             -sin_yaw * v_ref[..., 0] + cos_yaw * v_ref[..., 1],
             v_ref[..., 2]],
            axis=-1,
        )
        gravity_ref = np.broadcast_to(self.state[6:9], q_ref.shape)
        return sincos_tracking_reference(
            np.concatenate([q_ref, v_ref_body, gravity_ref], axis=-1)
        )
