"""ROM plots and recorded-motion playback for deployment runs.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
import time
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kcf_mpc.reference import PlanarReference

COLORS = ("#0b30c6", "#c30d0d")
TITLES = ["px [m]", "py [m]", "sin(theta)", "cos(theta)",
          "vx [m/s]", "vy [m/s]", "wz [rad/s]", "Position error [m]"]
MAX_GAP = .1


def interpolate(times, values, query):
    """Interpolate recorded samples without extending endpoints or long gaps."""
    query = np.asarray(query, dtype=float)
    result = np.stack([np.interp(query, times, values[:, i], left=np.nan, right=np.nan)
                       for i in range(values.shape[1])], axis=-1)
    right = np.clip(np.searchsorted(times, query, side='left'), 1, len(times)-1)
    missing = ((query > times[right-1]+1e-9) & (query < times[right]-1e-9)
               & (times[right]-times[right-1] > MAX_GAP+1e-9))
    return np.where(missing[..., None], np.nan, result)


def broken_lines(times, values):
    gaps = np.flatnonzero(np.diff(times) > MAX_GAP+1e-9)+1
    return (np.insert(times, gaps, (times[gaps-1]+times[gaps])/2),
            np.insert(values, gaps, np.nan, axis=0))


@dataclass
class RecordedRun:
    path: Path
    label: str
    times: np.ndarray
    pose: np.ndarray
    velocity: np.ndarray
    command_times: np.ndarray
    command: np.ndarray
    command_kind: str
    reference_path: object = None
    reference_times: object = None
    reference_values: object = None
    simulated: bool = False

    def reference(self, times):
        if self.reference_path is not None:
            q, v = self.reference_path.reference(times)
            features = self.reference_path.track(q, v)
        elif self.reference_times is not None:
            features = interpolate(self.reference_times, self.reference_values, times)
        else:
            shape = np.shape(times)
            return np.full(shape+(4,), np.nan), np.full(shape+(3,), np.nan)
        return features[..., :4], features[..., 4:7]


def saved_reference(directory, settings=None):
    plan = directory/'trial_plan.json'
    if plan.is_file():
        return PlanarReference(json.loads(plan.read_text())['mpc'])
    runtime = directory/'runtime_reference.json'
    if runtime.is_file() and settings is not None:
        saved = deepcopy(settings)
        if saved.get('reference_startup') is not None:
            saved['reference_startup']['initial_pose'] = json.loads(runtime.read_text())['initial_pose']
        return PlanarReference(saved)
    return None


def model_label(settings, path):
    return {'kcf': 'KCF', 'unicycle': 'Unicycle'}.get(
        settings.get('prediction_model'), path.stem)


def validate_run(run):
    n = len(run.times)
    if (n < 2 or run.pose.shape != (n, 4) or run.velocity.shape != (n, 3)
            or run.command.shape != (len(run.command_times), 3) or not len(run.command_times)
            or np.any(np.diff(run.times) <= 0) or np.any(np.diff(run.command_times) < 0)
            or not all(np.isfinite(a).all() for a in
                       (run.times, run.pose, run.velocity, run.command_times, run.command))
            or np.any(np.linalg.norm(run.pose[:, 2:4], axis=1) < 1e-12)):
        raise ValueError(f'{run.path}: require finite, consistent pose/twist/command data and increasing times')
    return run


def load_rom(path, env_index=0, label=None):
    with np.load(path, allow_pickle=False) as data:
        q, v, u = (data[k] for k in ('q_rom_log', 'v_rom_log', 'cmd_log'))
        if q.ndim != 3 or q.shape[-1] not in (3, 4) or not 0 <= env_index < q.shape[0]:
            raise ValueError(f'{path}: expected (B,T,3/4) ROM poses and a valid env-index')
        if v.shape != (*q.shape[:2], 3) or u.shape != v.shape:
            raise ValueError(f'{path}: ROM velocities/commands must have shape (B,T,3)')
        q, v, u = (a[env_index].astype(float) for a in (q, v, u))
        dt = float(data['log_dt'] if 'log_dt' in data else data['sim_dt'])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f'{path}: log_dt must be positive and finite')
    if q.shape[-1] == 3:
        q = np.column_stack((q[:, :2], np.sin(q[:, 2]), np.cos(q[:, 2])))
    plan = path.parent/'trial_plan.json'
    settings = json.loads(plan.read_text())['mpc'] if plan.is_file() else {}
    reference = saved_reference(path.parent)
    if reference is None:
        warnings.warn(f'{path}: no saved reference; current defaults will not be substituted', stacklevel=2)
    t = np.arange(len(q))*dt
    return validate_run(RecordedRun(path, label or model_label(settings, path), t, q, v, t, u,
                                    'applied', reference_path=reference, simulated=True))


def load_run(value, *, env_index=0, label=None):
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        if (path/'g1_rom_data.npz').is_file():
            return load_rom(path/'g1_rom_data.npz', env_index, label)
        candidates = list(path.glob('mpc_trace_*.npz'))+list(path.glob('mpc_traces/mpc_trace_*.npz'))
        if len(candidates) != 1:
            raise ValueError(f'{path}: select a specific trace; found {len(candidates)} trace files')
        path = candidates[0]
    if path.name == 'g1_rom_data.npz':
        return load_rom(path, env_index, label)
    directory = path.parent.parent if path.parent.name == 'mpc_traces' else path.parent
    result_file = directory/'result.json'
    result = json.loads(result_file.read_text()) if result_file.is_file() else {}
    associated = (path.name in [Path(p).name for p in result.get('traces', [])]
                  or (path.parent.name == 'mpc_traces'
                      and len(list(path.parent.glob('mpc_trace_*.npz'))) == 1))
    if associated and (directory/'g1_rom_data.npz').is_file():
        print(f'Using full recorded motion, including standing holds: {directory / "g1_rom_data.npz"}')
        return load_rom(directory/'g1_rom_data.npz', env_index, label)
    if env_index != 0:
        raise ValueError('Live traces contain one robot; env-index must be zero')
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        settings = meta['settings']
        times = data['state_time'].astype(float)
        measured = data['measured_observation'].astype(float)
        reference = data['reference'].astype(float)
        if (len(times) < 2 or measured.ndim != 2 or measured.shape[0] != len(times)
                or measured.shape[1] not in (85, 103) or reference.ndim != 3
                or reference.shape[0] != len(times) or reference.shape[-1] != 10
                or np.any(np.diff(times) < 0)):
            raise ValueError(f'{path}: expected ordered G1 states and ten reference features')
        runtime_file = directory/'runtime_reference.json'
        runtime = json.loads(runtime_file.read_text()) if runtime_file.is_file() else {}
        origin = runtime.get('t_start', float(times[0]))
        if runtime.get('final_reference_origin_s', origin) != origin:
            warnings.warn(f'{path}: reference clock restarted; using recorded horizons', stacklevel=2)
            runtime = {}
            origin = float(times[0])
        keep = np.r_[np.diff(times) > 0, True]
        pose, velocity = measured[keep, :4], measured[keep, 4:7]
        measured_times = times[keep]-origin
        dt = float(settings.get('prediction_sample_period', .02))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f'{path}: prediction_sample_period must be positive')
        if 'tick_application_time' in data and len(data['tick_application_time']):
            command_times = data['tick_application_time'].astype(float)-origin
            commands = data['tick_command'].astype(float)
        else:
            command_times, commands = times-origin, data['command'].astype(float)
            warnings.warn(f'{path}: no publication ticks; showing recorded solve commands', stacklevel=2)
        reference_times = (data['plan_start_time'][:, None]-origin
                           + dt*np.arange(1, reference.shape[1]+1)).ravel()
        reference_values = reference[..., :7].reshape(-1, 7)
    # Reference rows are future targets: align by plan start + prediction step.
    # For overlapping horizons retain the latest saved target at each time.
    unique, reverse = np.unique(np.round(reference_times[::-1], 9), return_index=True)
    reference_values = reference_values[len(reference_times)-1-reverse]
    valid = np.isfinite(unique) & np.isfinite(reference_values).all(axis=1)
    unique, reference_values = unique[valid], reference_values[valid]
    if len(unique) < 2:
        raise ValueError(f'{path}: no usable saved reference horizon')
    reference_path = saved_reference(directory, settings) if runtime else None
    if command_times[-1] > measured_times[-1]+MAX_GAP:
        warnings.warn(f'{path}: command ticks outlast measured solver states; '
                      'unrecorded standing motion will not be invented', stacklevel=2)
    if meta.get('dropped') or meta.get('dropped_ticks'):
        warnings.warn(f'{path}: trace reports dropped records', stacklevel=2)
    return validate_run(RecordedRun(path, label or model_label(settings, path), measured_times,
        pose, velocity, command_times, commands, 'issued', reference_path=reference_path,
        reference_times=unique, reference_values=reference_values))


def task_start(run):
    """Recorded time [s] when a live run's task leaves its start pose (after staging)."""
    if run.path.suffix != '.npz' or run.path.name == 'g1_rom_data.npz':
        raise ValueError(f'{run.path}: --task-time needs a live MPC trace')
    with np.load(run.path, allow_pickle=False) as data:
        settings = json.loads(str(data['metadata']))['settings']
        heading = float(data['measured_heading_world'][0])
    reference = PlanarReference(settings)
    reference.set_initial_pose([run.pose[0, 0], run.pose[0, 1], heading])
    return reference.task_start


def shift_run(run, offset):
    times = lambda t: None if t is None else t - offset
    return replace(run, times=run.times - offset, command_times=run.command_times - offset,
                   reference_times=times(run.reference_times), reference_path=None)


def comparison_window(runs, start=0., end=None):
    shared_end = min(run.times[-1] for run in runs)
    end = shared_end if end is None else end
    if not np.isfinite([start, end]).all() or not 0 <= start < end <= shared_end+1e-9:
        raise ValueError(f'Require 0 <= start < end <= shared recorded duration ({shared_end:g} s)')
    return float(start), float(min(end, shared_end))


def plot_runs(runs, *, start=0., end=None):
    start, end = comparison_window(runs, start, end)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for run, color in zip(runs, COLORS):
        selected = (run.times >= start) & (run.times <= end+1e-9)
        t = run.times[selected]
        q, v = run.pose[selected], run.velocity[selected]
        q_ref, v_ref = run.reference(t)
        for i, ax in enumerate(axes[0]):
            ax.plot(*broken_lines(t, q[:, i]), color=color, linewidth=1., label=run.label)
            ax.plot(*broken_lines(t, q_ref[:, i]), '--', color=color, alpha=.6,
                    linewidth=.9, label=f'{run.label} reference')
        chosen = (run.command_times >= start) & (run.command_times <= end+1e-9)
        for i, ax in enumerate(axes[1, :3]):
            ax.plot(*broken_lines(t, v[:, i]), color=color, linewidth=1.)
            ax.plot(*broken_lines(run.command_times[chosen], run.command[chosen, i]),
                    ':', color=color, alpha=.65, linewidth=.8)
            ax.plot(*broken_lines(t, v_ref[:, i]), '--', color=color, alpha=.6, linewidth=.9)
        axes[1, 3].plot(*broken_lines(t, np.linalg.norm(q[:, :2]-q_ref[:, :2], axis=1)), color=color)
    clock_label = 'Simulated time [s]' if all(r.simulated for r in runs) else 'Recorded time [s]'
    for ax, title in zip(axes.flat, TITLES):
        ax.set(title=title, xlabel=clock_label, xlim=(start, end)); ax.grid(True, alpha=.25)
    axes[0, 2].set_ylim(-1.1, 1.1); axes[0, 3].set_ylim(-1.1, 1.1)
    axes[0, 0].legend(fontsize=8)
    kinds = '/'.join(dict.fromkeys(r.command_kind for r in runs))
    axes[1, 3].text(.03, .96, 'Solid: measured\nDashed: saved reference\n'
                    f'Dotted: {kinds} command (body twist)',
                    transform=axes[1, 3].transAxes, va='top', fontsize=8)
    fig.suptitle('ROM comparison — model-aligned heading features; world pose / body twist')
    return fig


def triangle_vertices(pose, length=.25, width=.18):
    px, py, sine, cosine = pose
    norm = np.hypot(sine, cosine)
    if not np.isfinite(pose).all() or norm < 1e-12:
        raise ValueError('Cannot draw an invalid pose or zero heading pair')
    sine, cosine = sine/norm, cosine/norm
    body = np.array([[length, 0.], [-length/2, width/2], [-length/2, -width/2]])
    return body @ np.array([[cosine, sine], [-sine, cosine]])+[px, py]


class RecordedPlayback:
    def __init__(self, runs, *, start=0., end=None, overlay=True, speed=1.):
        self.start, self.end = comparison_window(runs, start, end)
        self.runs, self.speed = runs, speed
        self.fig, axes = plt.subplots(1, 1 if overlay else len(runs),
            figsize=(7*(1 if overlay else len(runs)), 6), squeeze=False, constrained_layout=True)
        self.axes = axes[0]
        positions = [r.pose[:, :2] for r in runs]
        for run in runs:
            q, _ = run.reference(run.times)
            positions.append(q[np.isfinite(q).all(axis=1), :2])
        xy = np.concatenate(positions)
        lower, upper = xy.min(axis=0)-.5, xy.max(axis=0)+.5
        self.triangles, self.trails, self.references = [], [], []
        for ax in self.axes:
            ax.set(xlabel='World x [m]', ylabel='World y [m]',
                   xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
            ax.set_aspect('equal', adjustable='box'); ax.grid(True, alpha=.25)
        for i, (run, color) in enumerate(zip(runs, COLORS)):
            ax = self.axes[0 if overlay else i]
            chosen = (run.times >= self.start) & (run.times <= self.end+1e-9)
            q, _ = run.reference(run.times[chosen])
            _, reference_xy = broken_lines(run.times[chosen], q[:, :2])
            ax.plot(reference_xy[:, 0], reference_xy[:, 1], '--', color='gray', linewidth=1.)
            reference = Polygon(triangle_vertices(run.pose[0], .18, .13), facecolor='none',
                edgecolor='gray', linewidth=1.5, label='Reference' if not overlay or i == 0 else None)
            triangle = Polygon(triangle_vertices(run.pose[0]), facecolor=color,
                edgecolor='black', alpha=.85, label=run.label, zorder=3)
            ax.add_patch(reference); ax.add_patch(triangle)
            trail, = ax.plot([], [], color=color, linewidth=1.1, alpha=.8)
            self.triangles.append(triangle); self.trails.append(trail); self.references.append(reference)
            if not overlay:
                ax.set_title(run.label)
        for ax in self.axes:
            ax.legend(loc='best', fontsize=8)
        self.title = self.fig.suptitle('')
        self.update(self.start)

    def update(self, seconds):
        seconds = float(np.clip(seconds, self.start, self.end))
        for i, run in enumerate(self.runs):
            pose = interpolate(run.times, run.pose, seconds)
            if np.isfinite(pose).all() and np.hypot(*pose[2:4]) < 1e-12:
                index = np.clip(np.searchsorted(run.times, seconds), 0, len(run.times)-1)
                pose[2:4] = run.pose[index, 2:4]
            visible = np.isfinite(pose).all() and np.hypot(*pose[2:4]) > 1e-12
            self.triangles[i].set_visible(visible)
            if visible:
                self.triangles[i].set_xy(triangle_vertices(pose))
            chosen = (run.times >= self.start) & (run.times <= seconds+1e-9)
            _, xy = broken_lines(run.times[chosen], run.pose[chosen, :2])
            if visible:
                xy = np.vstack((xy, pose[:2]))
            self.trails[i].set_data(xy[:, 0], xy[:, 1])
            reference, _ = run.reference(seconds)
            visible = np.isfinite(reference).all() and np.hypot(*reference[2:4]) > 1e-12
            self.references[i].set_visible(visible)
            if visible:
                self.references[i].set_xy(triangle_vertices(reference, .18, .13))
        self.title.set_text(f'ROM comparison — t={seconds:.2f} / {self.end:.2f} s — {self.speed:g}× playback')


def play(runs, *, start=0., end=None, speed=1., render_hz=50., overlay=True, loop=True):
    playback = RecordedPlayback(runs, start=start, end=end, overlay=overlay, speed=speed)
    interactive = plt.isinteractive()
    plt.ion(); plt.show(block=False); playback.fig.canvas.draw()
    try:
        while plt.fignum_exists(playback.fig.number):
            began = time.perf_counter()
            while plt.fignum_exists(playback.fig.number):
                frame = time.perf_counter()
                current = min(playback.end, playback.start+(frame-began)*speed)
                playback.update(current)
                playback.fig.canvas.draw_idle(); playback.fig.canvas.flush_events()
                if current >= playback.end:
                    break
                plt.pause(max(.001, min(1/render_hz-(time.perf_counter()-frame),
                                        (playback.end-current)/speed)))
            if not loop:
                break
    finally:
        if not interactive:
            plt.ioff()
    return playback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='*', metavar='RUN', help='one or two traces/run directories/ROM files')
    parser.add_argument('--trace', type=Path, help='trace/run (default: newest in logs/kcf/live_traces)')
    parser.add_argument('--compare', type=Path, help='second trace/run to overlay, e.g. SI')
    parser.add_argument('--labels', nargs='+', help='legend label for each selected run')
    parser.add_argument('--env-index', type=int, default=0)
    parser.add_argument('--out', type=Path, help='state plot filename (default: next to input)')
    parser.add_argument('--output-dir', type=Path, help='save rom_states.png and rom_playback_preview.png')
    parser.add_argument('--task-time', action='store_true',
                        help="t = 0 at each run's task start, after the walk to start_pose")
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--end', type=float)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--render-hz', type=float, default=50.)
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument('--overlay', dest='overlay', action='store_true')
    layout.add_argument('--side-by-side', dest='overlay', action='store_false')
    parser.set_defaults(overlay=True)
    parser.add_argument('--once', action='store_true', help='play once instead of looping')
    parser.add_argument('--no-playback', action='store_true', help='static figures only; works with MPLBACKEND=Agg')
    parser.add_argument('--plan-every', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.runs and args.trace:
            raise ValueError('Use positional runs or --trace, not both')
        inputs = list(args.runs) or ([args.trace] if args.trace else [])
        if not inputs:
            candidates = list((ROOT/'logs/kcf/live_traces').glob('*.npz'))
            if not candidates:
                raise ValueError('No live traces found; select a recording with --trace or RUN')
            inputs = [max(candidates, key=lambda p:p.stat().st_mtime)]
        if args.compare:
            inputs.append(args.compare)
        if not 1 <= len(inputs) <= 2:
            raise ValueError('Select one or two recordings')
        if args.labels and len(args.labels) != len(inputs):
            raise ValueError('Provide one --labels value per recording')
        if not np.isfinite([args.speed,args.render_hz]).all() or min(args.speed,args.render_hz) <= 0:
            raise ValueError('speed and render-hz must be finite and positive')
        runs = [load_run(p, env_index=args.env_index) for p in inputs]
        if args.task_time:
            offsets = [task_start(r) for r in runs]
            runs = [shift_run(r, o) for r, o in zip(runs, offsets)]
            print('Task starts at ' + ', '.join(f'{o:.2f} s' for o in offsets) + ' after t; now t = 0')
        if args.labels:
            runs = [replace(r,label=l) for r,l in zip(runs,args.labels)]
        elif len(runs)==2 and runs[0].label==runs[1].label:
            runs = [replace(r,label=f'{r.label} ({Path(p).name})') for r,p in zip(runs,inputs)]
        start, end = comparison_window(runs, args.start, args.end)
        if not args.no_playback and plt.get_backend().lower() in {'agg','pdf','svg','ps','cairo','template'}:
            raise ValueError('Playback needs an interactive backend; use --no-playback for static export')
        if len(runs)==2:
            if not np.allclose(runs[0].pose[0],runs[1].pose[0],atol=1e-6,rtol=0):
                warnings.warn('Initial poses differ; recorded world coordinates are preserved', stacklevel=2)
            a,_=runs[0].reference(np.linspace(start,end,101)); b,_=runs[1].reference(np.linspace(start,end,101))
            valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            if valid.any() and not np.allclose(a[valid],b[valid],atol=1e-6,rtol=0):
                warnings.warn('Saved references differ; each run retains its own task', stacklevel=2)
    except (ValueError,OSError,KeyError,IndexError) as error:
        parser.error(str(error))
    if args.plan_every is not None:
        warnings.warn('--plan-every is obsolete: the top view shows recorded motion', stacklevel=2)
    static = plot_runs(runs,start=start,end=end)
    output = args.out or (args.output_dir/'rom_states.png' if args.output_dir else
                         Path(str(inputs[0]).rstrip('/')).expanduser().resolve().with_name(
                             Path(inputs[0]).stem+'_plot.png'))
    preview_output = (args.output_dir/'rom_playback_preview.png' if args.output_dir else
                      output.with_name(output.stem+'_topdown.png'))
    output.parent.mkdir(parents=True,exist_ok=True); preview_output.parent.mkdir(parents=True,exist_ok=True)
    static.savefig(output,dpi=160)
    preview = RecordedPlayback(runs,start=start,end=end,overlay=args.overlay,speed=args.speed)
    preview.update(end); preview.fig.savefig(preview_output,dpi=160)
    print(f'Plotting {" and ".join(r.label for r in runs)}: {start:g}–{end:g} recorded seconds')
    print(f'Saved [{output}] and [{preview_output}].')
    if not args.no_playback:
        plt.close(preview.fig)
        print('Recorded-data playback only; close its window to stop. No robot or simulator is started.')
        play(runs,start=start,end=end,speed=args.speed,render_hz=args.render_hz,
             overlay=args.overlay,loop=not args.once)
    if plt.get_backend().lower()=='agg':
        plt.close('all')
    else:
        plt.show()


if __name__ == '__main__':
    main()
