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
            self.period = 2. * np.pi / self.omega
            self._configure_cycles(trajectory)
            try:
                self.lissajous_center = np.asarray(trajectory.get('center_xy', [0., 0.]), dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError('lissajous_reference center_xy must be finite [x, y]') from error
            if (self.lissajous_center.shape != (2,) or not np.isfinite(self.lissajous_center).all()
                    or any(isinstance(value, (bool, np.bool_))
                           for value in np.asarray(trajectory.get('center_xy', [0., 0.]), dtype=object).flat)):
                raise ValueError('lissajous_reference center_xy must be finite [x, y]')
        if self.mode == "waypoints":
            self._configure_waypoints(settings.get("waypoint_reference"))
        if self.mode in ("rounded_square", "hairpin", "slalom", "stop_turn_go"):
            self._configure_maneuver(settings.get(f"{self.mode}_reference"))
        if self.mode == "star":
            self._configure_star(settings.get("star_reference"))
        if self.mode == "speed_ramp":
            self._configure_speed_ramp(settings.get("speed_ramp_reference"))
            if self._ramp_speeds.max() > settings.get("input_upper_bounds", (np.inf,))[0]:
                raise ValueError("speed_ramp maximum_speed exceeds the MPC forward command bound")
        self._configure_workspace(settings.get("reference_workspace"))
        startup = settings.get("reference_startup")
        self.startup = None
        self._relative, self._frame = False, None
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
            self._configure_start_pose(startup)
            self._relative = startup.get('relative_start', False)
            if not isinstance(self._relative, (bool, np.bool_)):
                raise ValueError('Startup relative_start must be boolean')
            if self._relative and self.mode == 'fixed' and self._start_pose is None:
                raise ValueError('A relative fixed reference needs start_pose to define the goal relative to it')
            if startup.get('initial_pose') is not None:
                self.set_initial_pose(startup['initial_pose'])
                self._configured_initial_pose = self._initial_pose.copy()

    def _configure_start_pose(self, startup):
        # Optional staging: walk to start_pose and dwell there before the task.
        self._start_pose, self._start_dwell = None, 0.
        if startup.get('start_pose') is None:
            return
        pose = np.asarray(startup['start_pose'], dtype=float)
        dwell = startup.get('start_dwell_seconds', 0.)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError('Startup start_pose must be a finite [x, y, yaw]')
        if isinstance(dwell, (bool, np.bool_)) or not np.isfinite(dwell) or dwell < 0:
            raise ValueError('Startup start_dwell_seconds must be nonnegative and finite')
        self._check_workspace(pose[None, :2], 'Startup start_pose')
        self._start_pose, self._start_dwell = pose, float(dwell)

    def _quintic_seconds(self, delta):
        # Quintic pose steps peak at 1.875 * displacement / duration.
        return max(self.startup['transition_seconds'],
                   1.875 * np.linalg.norm(delta[:2]) / self.startup['maximum_linear_speed'],
                   1.875 * abs(delta[2]) / self.startup['maximum_yaw_rate'])

    def reset_initial_pose(self):
        self._initial_pose, self._frame = None, None
        if self._configured_initial_pose is not None:
            self.set_initial_pose(self._configured_initial_pose)

    def set_initial_pose(self, pose):
        """Anchor startup to measured pose; no global-origin assumption."""
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError('Startup initial_pose must be a finite [x, y, yaw]')
        relative = self.startup is not None and self._relative
        if self.startup is None or (self._start_pose is None and not relative):
            self._check_workspace(pose[None, :2], 'Startup initial_pose')
        self._initial_pose = pose.copy()
        if self.startup is None:
            return
        if relative:
            # Stand here for hold_seconds; anchor_start then fixes the start frame.
            self._task_from, self._stage_seconds = pose.copy(), 0.
            self._task_hold, self._frame = self.startup['hold_seconds'], None
            return
        self._task_from, self._stage_seconds = pose.copy(), 0.
        self._task_hold = self.startup['hold_seconds']
        if self._start_pose is not None:
            # fixed starts at start_pose; every path starts at its own first pose
            start = self._start_pose if self.mode == 'fixed' else self._raw_reference(0.)[0]
            turn = start[2] - pose[2]
            turn = np.arctan2(np.sin(turn), np.cos(turn))
            self._task_from = np.array([*start[:2], pose[2] + turn])
            self._stage_seconds = self._quintic_seconds(self._task_from - pose)
            self._task_hold += self._stage_seconds + self._start_dwell
        self._plan_task_blend()

    def _plan_task_blend(self):
        first, _ = self._task_reference(0.)
        delta = first - self._task_from
        heading_delta = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
        self._startup_yaw_offset = heading_delta - delta[2]
        self._startup_duration = self._quintic_seconds([*delta[:2], heading_delta])

    def needs_anchor(self, t):
        """True on the first tick at or after hold_seconds in relative-start mode."""
        return (self.startup is not None and self._relative and self._initial_pose is not None
                and self._frame is None and t >= self.startup['hold_seconds'])

    def anchor_start(self, pose, t):
        """Record the measured [x, y, yaw] as the start; the task is moved rigidly to begin there.

        fixed: start_pose -> reference is replayed from pose. Paths: their first pose lands on pose.
        """
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError('Relative start pose must be a finite [x, y, yaw]')
        origin = self._start_pose if self.mode == 'fixed' else self._raw_reference(0.)[0]
        turn = pose[2] - origin[2]
        rotation = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
        self._frame = (origin.copy(), pose.copy(), turn, rotation)
        self._task_from, self._task_hold = pose.copy(), max(self.startup['hold_seconds'], float(t))
        self._plan_task_blend()

    def _task_reference(self, t):
        """Configured task, moved into the recorded start frame when relative_start is on."""
        pose, twist = self._raw_reference(t)
        if self._frame is None:
            return pose, twist
        origin, start, turn, rotation = self._frame
        pose = pose.copy()
        pose[..., :2] = start[:2] + (pose[..., :2] - origin[:2]) @ rotation.T
        pose[..., 2] += turn
        twist = twist.copy()
        twist[..., :2] = twist[..., :2] @ rotation.T
        return pose, twist

    def _raw_reference(self, t):
        if self.mode == 'lissajous':
            return self.lissajous_reference(t)
        if self.mode == 'waypoints':
            return self.waypoints_reference(t)
        if self.mode == 'stop_turn_go':
            return self.waypoints_reference(t)
        if self.mode == 'star':
            return self.star_reference(t)
        if self.mode == 'speed_ramp':
            return self.speed_ramp_reference(t)
        if self.mode in ('rounded_square', 'hairpin', 'slalom'):
            return self.turning_reference(t)
        shape = np.shape(t) + (3,)
        return (np.broadcast_to(self.state[:3], shape).copy(),
                np.broadcast_to(self.state[3:6], shape).copy())

    @property
    def task_start(self):
        """Reference time [s] at which the task leaves the start pose (0 without startup)."""
        return (0. if self.startup is None or self._initial_pose is None else self._task_hold)

    @property
    def task_end(self):
        """Reference time [s] when the task reaches its final pose; None if it never ends."""
        if self.startup is None or self._initial_pose is None or (self._relative and self._frame is None):
            return None
        path = 0. if self.mode == 'fixed' else self.duration
        if path is None:
            return None
        return self._task_hold + max(self._startup_duration, self._startup_duration / 2. + path)

    @property
    def startup_delay(self):
        """Steady path-clock offset after the smooth startup finishes."""
        return (0. if self.startup is None or self._initial_pose is None else
                self._task_hold + self._startup_duration / 2.)

    @property
    def duration(self):
        """Finite path-clock duration, or None for indefinitely repeating paths."""
        if getattr(self, '_turn_cycles', None) is not None:
            return float(self._turn_cycles * self.period + .5 * self._turn_stop_seconds)
        if self.mode == 'speed_ramp':
            return float(self._ramp_times[-1])
        return (float(self.waypoint_times[-1])
                if self.mode in ('waypoints', 'stop_turn_go', 'star') else None)

    def path_time(self, t):
        """Saved path clock, used also to align waypoint completion/dwell checks."""
        t = np.asarray(t, dtype=float)
        if self.startup is None or self._initial_pose is None:
            return t
        duration = self._startup_duration
        elapsed = t - self._task_hold
        phase = np.clip(elapsed / duration, 0., 1.)
        integral = phase**4 * (2.5 + phase * (-3. + phase))
        return np.where(elapsed >= duration, elapsed - duration / 2., duration * integral)

    def _configure_workspace(self, configuration):
        """Validate the entire path, including curve extrema, without clipping it.

        Missing/None preserves saved historical runs and unrestricted experiments.
        Startup is a convex pose blend, so an in-bounds initial pose also fits.
        """
        self.workspace_bounds = None
        if configuration is None:
            return
        if not isinstance(configuration, dict) or set(configuration) != {'center_xy', 'size_xy', 'margin'}:
            raise ValueError('reference_workspace requires exactly center_xy, size_xy and margin')
        try:
            center = np.asarray(configuration['center_xy'], dtype=float)
            size = np.asarray(configuration['size_xy'], dtype=float)
            margin = float(configuration['margin'])
            raw = [*np.asarray(configuration['center_xy'], dtype=object).flat,
                   *np.asarray(configuration['size_xy'], dtype=object).flat, configuration['margin']]
        except (TypeError, ValueError) as error:
            raise ValueError('reference_workspace dimensions, center and margin must be numeric') from error
        if (center.shape != (2,) or size.shape != (2,) or not np.isfinite(center).all()
                or not np.isfinite(size).all() or not np.isfinite(margin) or margin < 0.
                or np.any(size <= 2. * margin)
                or any(isinstance(value, (bool, np.bool_)) for value in raw)):
            raise ValueError('reference_workspace needs finite center_xy, positive size_xy and '
                             '0 <= margin < half each side length')
        half = .5 * size - margin
        self.workspace_bounds = np.stack((center - half, center + half))
        if not np.isfinite(self.workspace_bounds).all():
            raise ValueError('reference_workspace bounds must be finite')
        if self.mode == 'lissajous':
            extent = np.array([self.half_width, self.half_height])
            points = self.lissajous_center + np.stack((-extent, extent))
        elif self.mode in ('waypoints', 'stop_turn_go', 'star'):
            points = self.waypoint_poses[:, :2]
        elif self.mode == 'speed_ramp':
            points = self.speed_ramp_reference(np.array([0., self.duration]))[0][:, :2]
        elif self.mode in ('rounded_square', 'hairpin', 'slalom'):
            if self.mode == 'slalom' and self._turn_cycles is None:
                raise ValueError('A workspace-confined slalom requires finite cycles and stop_seconds')
            points = self._turn_extrema()
        else:
            points = self.state[None, :2]
        self._check_workspace(points, f'{self.mode} reference')

    def _check_workspace(self, points, description):
        if self.workspace_bounds is None:
            return
        lower, upper = self.workspace_bounds
        if (not np.isfinite(points).all()
                or np.any(points < lower - 1e-9) or np.any(points > upper + 1e-9)):
            raise ValueError(f'{description} exceeds reference_workspace usable box '
                             f'{lower.tolist()} to {upper.tolist()}; adjust path size/position or workspace')

    def _turn_extrema(self):
        # A monotone heading ramp has x/y extrema where sin/cos(yaw)=0.
        # Invert its quintic blend by bisection, avoiding a sampled bounds check
        # that could miss a boundary excursion between samples.
        points = []
        for index, angle in enumerate(self._turn_angles):
            phases = [0., 1.]
            if angle:
                heading = np.arctan2(np.sin(self._turn_headings[index]), np.cos(self._turn_headings[index]))
                low, high = sorted((heading, heading + angle))
                for multiple in range(int(np.ceil(low / (np.pi / 2.))), int(np.floor(high / (np.pi / 2.))) + 1):
                    target = (multiple * np.pi / 2. - heading) / angle
                    left, right = 0., 1.
                    for _ in range(52):
                        middle = .5 * (left + right)
                        blend = middle**3 * (10. + middle * (-15. + 6. * middle))
                        if blend < target:
                            left = middle
                        else:
                            right = middle
                    phases.append(.5 * (left + right))
            points.extend(self._turn_positions[index] + self._turn_displacement(index, np.array(phases)))
        points = np.asarray(points)
        if self._turn_cycles is not None:
            last_offset = (self._turn_cycles - 1) * self._cycle_displacement
            points = np.vstack((points, points + last_offset))
        return points

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
        optional = {'cycles', 'stop_seconds'} if self.mode != 'stop_turn_go' else set()
        if not isinstance(configuration, dict) or set(configuration) - optional != common | extra:
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
        self._configure_cycles(configuration)

    def _configure_cycles(self, configuration):
        # Optional finite laps: run `cycles` periods, then slow the clock to a stop.
        self._turn_cycles, self._turn_stop_seconds = None, 0.
        optional = {'cycles', 'stop_seconds'}
        if not optional & set(configuration):
            return
        count, stop = configuration.get('cycles'), configuration.get('stop_seconds')
        if (set(configuration) & optional != optional or isinstance(count, (bool, np.bool_))
                or not isinstance(count, (int, np.integer)) or count < 1
                or isinstance(stop, (bool, np.bool_)) or not isinstance(stop, (int, float))
                or not np.isfinite(stop) or stop <= 0.):
            raise ValueError(f'Finite {self.mode} requires positive integer cycles and positive stop_seconds')
        self._turn_cycles, self._turn_stop_seconds = int(count), float(stop)
        if (not np.isfinite(self._turn_cycles * self.period)
                or self._turn_stop_seconds > 2. * self._turn_cycles * self.period):
            raise ValueError(f'{self.mode} stop_seconds cannot exceed twice its moving path duration')

    def _finite_clock(self, t):
        """Path clock and its rate; with finite cycles it eases to a stop at the last lap's end."""
        if self._turn_cycles is None:
            return t, np.ones_like(t)
        stop = self._turn_stop_seconds
        begin = self._turn_cycles * self.period - .5 * stop
        phase = np.clip((t - begin) / stop, 0., 1.)
        rate = 1. - phase**3 * (10. + phase * (-15. + 6. * phase))
        integral = phase**4 * (2.5 + phase * (-3. + phase))
        return np.minimum(t, begin + stop) - stop * integral, rate

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
        t, clock_rate = self._finite_clock(np.maximum(t, 0.))
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
        return (np.concatenate((xy, heading[..., None]), axis=-1),
                np.where(active[..., None], twist * clock_rate[..., None], 0.))

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
        if self._relative and self._frame is None:
            # Before the start is recorded, every horizon sample holds the current pose.
            shape = t.shape + (3,)
            return np.broadcast_to(self._initial_pose, shape).copy(), np.zeros(shape)
        blend, derivative = self._quintic(t - self._task_hold, self._startup_duration)
        # Integrate the smooth clock rate: no path jump when startup finishes.
        pose, twist = self._task_reference(self.path_time(t))
        pose[..., 2] += self._startup_yaw_offset
        delta = pose - self._task_from
        q = self._task_from + blend[..., None] * delta
        v = derivative[..., None] * delta + blend[..., None]**2 * twist
        if self._stage_seconds > 0:
            # Staging: measured pose -> start pose; the dwell is the task's hold.
            blend, derivative = self._quintic(t - self.startup['hold_seconds'], self._stage_seconds)
            stage = self._task_from - self._initial_pose
            staging = (t < self.startup['hold_seconds'] + self._stage_seconds)[..., None]
            q = np.where(staging, self._initial_pose + blend[..., None] * stage, q)
            v = np.where(staging, derivative[..., None] * stage, v)
        return q, v

    @staticmethod
    def _quintic(elapsed, duration):
        phase = np.clip(elapsed / duration, 0., 1.)
        return (phase**3 * (10. + phase * (-15. + 6. * phase)),
                30. * phase**2 * (1. - phase)**2 / duration)

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
        t, clock_rate = self._finite_clock(np.asarray(t, dtype=np.float64))
        px = self.lissajous_center[0] + a * np.sin(w * t)
        py = self.lissajous_center[1] + b * np.sin(2 * w * t)
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
        return (np.stack([px, py, theta], axis=-1),
                np.stack([vx, vy, wz], axis=-1) * clock_rate[..., None])

    def _configure_star(self, configuration):
        keys = {'center_xy', 'radius', 'rotation', 'speed', 'yaw_rate', 'dwell_seconds', 'clockwise'}
        if not isinstance(configuration, dict) or set(configuration) != keys:
            raise ValueError(f'star_reference requires exactly {sorted(keys)}')
        numeric_keys = ('radius', 'rotation', 'speed', 'yaw_rate', 'dwell_seconds')
        try:
            center = np.asarray(configuration['center_xy'], dtype=float)
            radius, rotation, speed, yaw_rate, dwell = values = np.asarray(
                [configuration[key] for key in numeric_keys], dtype=float)
            raw = [*np.asarray(configuration['center_xy'], dtype=object).flat,
                   *[configuration[key] for key in numeric_keys]]
        except (TypeError, ValueError) as error:
            raise ValueError('star_reference requires numeric dimensions, speeds and center_xy') from error
        if (center.shape != (2,) or values.shape != (5,)
                or not np.isfinite(center).all() or not np.isfinite(values).all()
                or any(isinstance(value, (bool, np.bool_)) for value in raw)
                or min(radius, speed, yaw_rate) <= 0. or dwell < 0.
                or not isinstance(configuration['clockwise'], (bool, np.bool_))):
            raise ValueError('star_reference needs finite center_xy/rotation, positive radius/speed/yaw_rate, '
                             'nonnegative dwell_seconds and boolean clockwise')
        direction = -1. if configuration['clockwise'] else 1.
        angles = rotation + direction * 4. * np.pi / 5. * np.arange(5)
        vertices = center + radius * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        vertices = np.vstack((vertices, vertices[0]))
        edge = vertices[1] - vertices[0]
        heading = float(np.arctan2(edge[1], edge[0]))
        times, poses = [0.], [np.r_[vertices[0], heading]]

        def append(pose, duration):
            if duration > 0.:
                times.append(times[-1] + duration)
                poses.append(pose.copy())

        turn = direction * 4. * np.pi / 5.
        for first, second in zip(vertices[:-1], vertices[1:]):
            pose = np.r_[second, heading]
            append(pose, 1.875 * np.linalg.norm(second - first) / speed)
            append(pose, dwell)
            heading += turn
            pose = np.r_[second, heading]
            append(pose, 1.875 * abs(turn) / yaw_rate)
            append(pose, dwell)
        self._configure_waypoints(dict(times_s=times, poses=poses))
        self.waypoint_poses[:, 2] = np.asarray(poses)[:, 2]

    def star_reference(self, t):
        """A finite five-point crossing star with separate forward legs and pivots."""
        return self.waypoints_reference(t)

    def _configure_speed_ramp(self, configuration):
        keys = {'start_pose', 'distance', 'minimum_speed', 'maximum_speed', 'transition_seconds'}
        if not isinstance(configuration, dict) or set(configuration) != keys:
            raise ValueError(f'speed_ramp_reference requires exactly {sorted(keys)}')
        numeric_keys = ('distance', 'minimum_speed', 'maximum_speed', 'transition_seconds')
        try:
            self._ramp_start = np.asarray(configuration['start_pose'], dtype=float).copy()
            distance, low, high, transition = values = np.asarray(
                [configuration[key] for key in numeric_keys], dtype=float)
            raw = [*np.asarray(configuration['start_pose'], dtype=object).flat,
                   *[configuration[key] for key in numeric_keys]]
        except (TypeError, ValueError) as error:
            raise ValueError('speed_ramp_reference requires numeric start_pose, distance, speeds and duration') from error
        if (self._ramp_start.shape != (3,) or values.shape != (4,)
                or not np.isfinite(self._ramp_start).all()
                or not np.isfinite(values).all() or np.any(values <= 0.) or high <= low
                or any(isinstance(value, (bool, np.bool_)) for value in raw)):
            raise ValueError('speed_ramp_reference needs finite start_pose, positive distance/speeds/duration '
                             'and maximum_speed > minimum_speed')
        # Each quintic speed ramp has mean (v_start + v_end)/2.
        # Reserve distance for start/stop, then fit both low<->high ramps exactly.
        ramp_seconds = (distance - low * transition) / (low + high)
        if not np.isfinite(ramp_seconds) or ramp_seconds <= 0.:
            raise ValueError('speed_ramp distance must exceed minimum_speed * transition_seconds')
        self._ramp_times = np.r_[0., np.cumsum([transition, ramp_seconds, ramp_seconds, transition])]
        if not np.isfinite(self._ramp_times).all():
            raise ValueError('speed_ramp_reference has invalid path duration')
        self._ramp_speeds = np.array([0., low, high, low, 0.])
        self._ramp_distances = np.r_[0., np.cumsum(
            .5 * (self._ramp_speeds[:-1] + self._ramp_speeds[1:]) * np.diff(self._ramp_times))]

    def speed_ramp_reference(self, t):
        """Analytic straight-line 0->low->high->low->0 speed profile; hold at end."""
        t = np.asarray(t, dtype=float)
        if not np.isfinite(t).all():
            raise ValueError('Speed ramp query times must be finite')
        index = np.clip(np.searchsorted(self._ramp_times, t, side='right') - 1, 0, 3)
        duration = self._ramp_times[index + 1] - self._ramp_times[index]
        phase = np.clip((t - self._ramp_times[index]) / duration, 0., 1.)
        blend = phase**3 * (10. + phase * (-15. + 6. * phase))
        integral = phase**4 * (2.5 + phase * (-3. + phase))
        first = self._ramp_speeds[index]
        delta = self._ramp_speeds[index + 1] - first
        speed = first + delta * blend
        distance = self._ramp_distances[index] + duration * (first * phase + delta * integral)
        direction = np.array([np.cos(self._ramp_start[2]), np.sin(self._ramp_start[2])])
        pose = np.broadcast_to(self._ramp_start, t.shape + (3,)).copy()
        pose[..., :2] += distance[..., None] * direction
        twist = np.zeros_like(pose)
        twist[..., :2] = speed[..., None] * direction
        return pose, twist

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
