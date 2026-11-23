"""Visible, simulation-only KCF/SI trials using the contained epoch-3240 profile.

Run from deploy_robot with the deploy Conda environment active. A dedicated
localhost ROS domain separates these automated trials from deployment nodes.
The GPU MPC runs in a separate process, retaining its original scheduling.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic, sleep
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
os.environ['DEPLOY_ROOT_DIR'] = str(ROOT)
os.environ['ROS_DOMAIN_ID'] = '91'
os.environ['ROS_LOCALHOST_ONLY'] = '1'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'deploy/kcf'))

MODELS = ('kcf', 'single_integrator')
REFERENCES = ('fixed', 'turn_180', 'lissajous', 'waypoints', 'rounded_square',
              'hairpin', 'slalom', 'stop_turn_go')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare_run(model, directory, initial_xy, initial_yaw, duration, reference='fixed'):
    from kcf_mpc.profiles import load_profile

    directory.mkdir(parents=True, exist_ok=False)
    profile = load_profile('walking')
    profile.validate_assets()
    config = yaml.safe_load((ROOT / 'deploy/configs/g1' /
                            (profile.values['simulation_config'] + '.yaml')).read_text())
    settings = profile.settings(prediction_model=model,
                                reference_mode='fixed' if reference == 'turn_180' else reference)
    if reference == 'turn_180':
        settings['reference'] = (*map(float, initial_xy), float(initial_yaw + np.pi),
                                 *settings['reference'][3:])
    settings['jax']['compilation_cache_dir'] = os.environ.get(
        'JAX_COMPILATION_CACHE_DIR', str(ROOT / '.cache/kcf/jax'))
    settings['live_trace']['directory'] = str(directory / 'mpc_traces')
    settings['live_trace']['capacity'] = max(3000, int((duration + 10) / .02))
    config['home_base_pos'][:2] = list(initial_xy)
    config['home_base_pos'][3:7] = [float(np.cos(initial_yaw / 2)), 0., 0.,
                                      float(np.sin(initial_yaw / 2))]
    label = 'KCF MPC' if model == 'kcf' else 'Single-integrator MPC'
    config['simulation_label'] = f'{label} | {reference} | epoch 3240, 29 DoF'
    (directory / 'simulation.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(directory / 'mpc_settings.json', settings)
    assets = [ROOT / path for path in profile.values['artifact_sha256']]
    manifest = dict(
        model=model, reference=reference, checkpoint_epoch=3240,
        simulation_only=True, ros_domain_id=91, pose_source='MuJoCo ground truth',
        requested_tracking_seconds=duration, initial_xy=list(initial_xy), initial_yaw=initial_yaw,
        asset_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in assets},
        settings_overrides=['selected predictor/reference', 'run-specific trace directory/capacity',
                            'local compilation cache path'],
    )
    sources = [p for directory_name in ('deploy', 'kcf_mpc', 'utils')
               for p in (ROOT / directory_name).rglob('*')
               if p.suffix in ('.py', '.yaml', '.cu', '.cc', '.cuh')]
    manifest['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources}
    write_json(directory / 'manifest.json', manifest)
    return settings


def mpc_worker(settings_file, ready_file):
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from kcf_ros import KCFNode
    from kcf_mpc.profiles import load_profile

    settings = json.loads(settings_file.read_text())

    def factory(profile_name, reference_mode, prediction_model):
        profile = load_profile(profile_name)
        profile.validate_assets()
        source = profile.symbol('runtime_factory')(profile, settings, quiet=True)
        return source, profile.symbol('state_adapter')

    # Keep the ROS context alive while KCF publishes its final zero command
    # and closes the solve trace. Python handles SIGINT via KeyboardInterrupt.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = None
    reference_session = None
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous_signals = {sig: signal.signal(sig, request_stop)
                        for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        args = SimpleNamespace(profile='walking', reference_mode=settings['reference_mode'], shadow=False,
                               state_source='simulation',
                               single_integrator=settings['prediction_model'] == 'single_integrator')
        node = KCFNode(args, controller_factory=factory)
        write_json(ready_file, dict(ready=True, model=settings['prediction_model']))
        while rclpy.ok() and not stop_requested:
            rclpy.spin_once(node, timeout_sec=.005)
            if not node.ready:
                raise RuntimeError('MPC runtime failed; see mpc_console.log')
            if node.active:
                source = node.source
                if reference_session is None:
                    reference_session = dict(t_start=source.t_start,
                                             initial_pose=source.reference_path._initial_pose.tolist())
                elif abs(source.t_start - reference_session['t_start']) > 1e-8:
                    raise RuntimeError('Tracking restarted after feedback/operator loss; repeat this trial')
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            source = node.source
            if source is not None:
                write_json(ready_file.with_name('runtime_reference.json'), dict(
                    **(reference_session or dict(t_start=source.t_start, initial_pose=None)),
                    final_reference_origin_s=source.t_start,
                    solves=source.solve_count, failures=source.failure_count,
                    solver_failures=source.solver_failure_count,
                    deadline_misses=source.deadline_miss_count))
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)


def visible_trial(directory, duration, settle_seconds):
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Float32MultiArray, String
    import mujoco
    from deploy.simulation.g1.control_29dof_kcf import ControlNode
    from deploy.simulation.g1.simulation import SimulationNode
    from kcf_mpc.mpc.mpc_runtime import StartupGCFreeze

    class SimulationOperator(Node):
        def __init__(self):
            super().__init__('fixed_target_simulation_operator')
            self.state = 'control'
            self.fsm = self.create_publisher(String, '/deploy_robot/fsm', 1)
            self.joy = self.create_publisher(Float32MultiArray, '/deploy_robot/joystick', 1)
            self.timer = self.create_timer(.02, self.heartbeat)

        def heartbeat(self):
            self.fsm.publish(String(data=self.state))
            self.joy.publish(Float32MultiArray(data=[1., 0., 0., 0., 0., 0., 0., 0., 0.]))

    stream = (directory / 'mpc_console.log').open('w')
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                              '--worker-settings', str(directory / 'mpc_settings.json'),
                              '--ready-file', str(directory / 'mpc_ready.json')],
                             stdout=stream, stderr=subprocess.STDOUT, env=dict(os.environ))
    executor, nodes, policy, sim, operator = None, [], None, None, None
    gc_guard = None
    rows = []
    requested_start, first_tracking_pose, failure = None, None, None
    sources = {}
    started = monotonic()
    try:
        print(f'Warming GPU MPC for {directory.name}; the viewer opens when ready.', flush=True)
        while not (directory / 'mpc_ready.json').exists():
            if child.poll() is not None:
                raise RuntimeError(f'MPC initialization failed: see {directory / "mpc_console.log"}')
            if monotonic() - started > 180:
                raise TimeoutError('GPU initialization exceeded 180 seconds')
            sleep(.1)
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        executor = SingleThreadedExecutor()
        policy = ControlNode(str(directory / 'simulation.yaml'))
        nodes.append(policy)
        operator = SimulationOperator()
        nodes.append(operator)
        sim = SimulationNode(str(directory / 'simulation.yaml'))
        nodes.append(sim)
        actuator_names = [mujoco.mj_id2name(sim.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                          for i in range(sim.nu)]
        if actuator_names != policy.policy.metadata['joint_names']:
            raise ValueError('Simulation actuator order differs from the hardware walking policy')
        settings = json.loads((directory / 'mpc_settings.json').read_text())
        with sim.viewer.lock():
            mujoco.mjv_initGeom(sim.viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.array([.035, .035, .035]),
                               np.array([*settings['reference'][:2], .035]),
                               np.eye(3).reshape(-1), np.array([1., .2, .2, .8]))
            sim.viewer.user_scn.ngeom = 1
        for node in nodes:
            executor.add_node(node)
        # Keep loaded ROS/ONNX objects out of cyclic-GC scans during the timed loop.
        gc_guard = StartupGCFreeze(True)
        last_sample, last_report = None, -float('inf')
        print(f'VISIBLE RUN: {directory.name}. Settling, then {duration:g} seconds of tracking.', flush=True)
        deadline = monotonic() + 4 * (duration + settle_seconds) + 30
        while sim.viewer.is_running():
            executor.spin_once(timeout_sec=.002)
            if child.poll() is not None:
                raise RuntimeError('GPU MPC worker exited during the trial')
            if monotonic() > deadline:
                raise TimeoutError('Visible simulation did not finish in the wall-time budget')
            if policy.kcf_state is None or policy.last_state_time == last_sample:
                continue
            last_sample = policy.last_state_time
            state = policy.kcf_state.copy()
            rows.append((last_sample, state[1:37], state[37:72],
                         policy.action.copy(), policy.cmd.copy()))
            sources[policy.command_source] = sources.get(policy.command_source, 0) + 1
            if state[3] < .45:
                raise RuntimeError(f'Simulation fell at t={last_sample:.2f}s (height {state[3]:.3f}m)')
            if requested_start is None and last_sample >= settle_seconds:
                requested_start = last_sample
                first_tracking_pose = state[1:8].tolist()
                operator.state = 'track'
                operator.heartbeat()
                print(f'Tracking enabled at simulation t={last_sample:.2f}s; pose={state[1:4]}', flush=True)
            if last_sample - last_report >= 2:
                print(f'{directory.name}: sim={last_sample:.2f}s '
                      f'xy=({state[1]:.3f},{state[2]:.3f}) height={state[3]:.3f} '
                      f'command={policy.cmd.round(3).tolist()} source={policy.command_source}', flush=True)
                last_report = last_sample
            if requested_start is not None and last_sample >= requested_start + duration + .5:
                break
        else:
            raise RuntimeError('Viewer closed before the requested run completed')
        if not sources.get('kcf') or not any(np.linalg.norm(row[4]) > .01 for row in rows):
            raise RuntimeError('No nonzero MPC command reached the simulation policy')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        if operator is not None:
            operator.state = 'control'
            operator.heartbeat()
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=5)
        stream.close()
        if rows:
            np.savez_compressed(directory / 'simulation_states.npz',
                                time=np.array([r[0] for r in rows]),
                                qpos=np.stack([r[1] for r in rows]),
                                qvel=np.stack([r[2] for r in rows]),
                                raw_policy_action=np.stack([r[3] for r in rows]),
                                applied_twist=np.stack([r[4] for r in rows]))
        for node in reversed(nodes):
            if executor is not None:
                executor.remove_node(node)
            node.destroy_node()
        if executor is not None:
            executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        if gc_guard is not None:
            gc_guard.close()
        traces = list((directory / 'mpc_traces').glob('*.npz'))
        if not failure and (child.returncode != 0 or not traces):
            failure = 'MPC worker did not exit cleanly and save its trace'
        result = dict(outcome='failed' if failure else 'complete', error=failure,
                      mpc_returncode=child.returncode, simulation_samples=len(rows),
                      requested_tracking_start_s=requested_start,
                      measured_pose_at_tracking_request=first_tracking_pose,
                      min_base_height_m=min((float(r[1][2]) for r in rows), default=None),
                      command_sources=sources, traces=[str(p) for p in traces],
                      wall_seconds=monotonic() - started)
        if traces:
            with np.load(traces[0], allow_pickle=False) as data:
                t = data['state_time']
                result['trace_time_window_s'] = [float(t[0]), float(t[-1])] if len(t) else []
                result['solve_time_span_seconds'] = float(np.ptp(t)) if len(t) else 0.
                result['recorded_tracking_seconds'] = (float(np.ptp(data['tick_state_time']))
                                                       if len(data['tick_state_time']) else 0.)
                result['dropped_solves'] = json.loads(str(data['metadata']))['dropped']
        runtime_file = directory / 'runtime_reference.json'
        if rows and runtime_file.is_file() and traces:
            from kcf_mpc.state.math_utils import q_to_rom, v_to_rom
            from kcf_mpc.reference import PlanarReference
            runtime = json.loads(runtime_file.read_text())
            settings = json.loads((directory / 'mpc_settings.json').read_text())
            settings['reference_startup']['initial_pose'] = runtime['initial_pose']
            origin = runtime['t_start']
            samples = [r for r in rows if origin <= r[0] <= origin + duration + 1e-8]
            t = np.array([r[0] - origin for r in samples])
            grid = np.arange(int(np.floor(t[-1] / .02 + 1e-6)) + 1) * .02
            q = np.stack([r[1] for r in samples]); v = np.stack([r[2] for r in samples])
            planar = q_to_rom(q)
            pose = np.column_stack((planar[:, :2], np.sin(planar[:, 2]), np.cos(planar[:, 2])))
            velocity = v_to_rom(q, v)
            command = np.stack([r[4] for r in samples])
            interpolate = lambda array: np.column_stack([np.interp(grid, t, array[:, i])
                                                         for i in range(array.shape[1])])
            pose, velocity, command = map(interpolate, (pose, velocity, command))
            np.savez_compressed(directory / 'g1_rom_data.npz', q_rom_log=pose[None],
                                v_rom_log=velocity[None], cmd_log=command[None], log_dt=.02)
            write_json(directory / 'trial_plan.json', dict(mpc=settings, execution=dict(
                reference_time_origin_s=origin, sample_clock='MuJoCo seconds',
                measured_sample_max_gap_s=float(np.max(np.diff(t))))))
            reference_q, _ = PlanarReference(settings).reference(grid)
            position_error = np.linalg.norm(pose[:, :2] - reference_q[:, :2], axis=1)
            yaw = np.arctan2(pose[:, 2], pose[:, 3])
            heading_error = np.arctan2(np.sin(yaw - reference_q[:, 2]),
                                       np.cos(yaw - reference_q[:, 2]))
            result.update(runtime=runtime, position_rmse_m=float(np.sqrt(np.mean(position_error**2))),
                          heading_rmse_deg=float(np.degrees(np.sqrt(np.mean(heading_error**2)))),
                          recorded_rom_seconds=float(grid[-1]))
            if grid[-1] < duration - .02 - 1e-8:
                failure = failure or 'Tracking did not cover the requested uninterrupted duration'
                result.update(outcome='failed', error=failure)
            with np.load(traces[0], allow_pickle=False) as trace:
                result['completed_solves'] = int(trace['completed'].sum())
                result['unsuccessful_solves'] = int((~trace['success']).sum())
                result['compute_deadline_misses'] = int(trace['deadline_missed'].sum())
                result['latency_deadline_misses'] = int(trace['latency_missed'].sum())
                reasons, counts = np.unique(trace['tick_reason'], return_counts=True)
                result['mpc_command_reasons'] = dict(zip(reasons.tolist(), counts.tolist()))
        write_json(directory / 'result.json', result)
        print(json.dumps(result, indent=2), flush=True)
    if failure:
        raise RuntimeError(failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--settle-seconds', type=float, default=2.)
    parser.add_argument('--initial-xy', type=float, nargs=2, default=(0., -.1))
    parser.add_argument('--initial-yaw', type=float, default=0.)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--references', nargs='+', choices=REFERENCES, default=['fixed'])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker-settings', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--ready-file', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_settings:
        mpc_worker(args.worker_settings, args.ready_file)
        return
    if (not np.isfinite([args.duration, args.settle_seconds, *args.initial_xy, args.initial_yaw]).all()
            or args.duration <= 0 or args.settle_seconds < 0):
        parser.error('durations and initial pose must be finite; duration > 0 and settle >= 0')
    stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d_%H%M%S')
    output = (args.output_dir or ROOT / 'logs/simulation' / f'fixed_comparison_{stamp}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f'Saving comparison runs to {output}', flush=True)
    for reference in args.references:
        for model in args.models:
            prepare_run(model, output / reference / model, args.initial_xy, args.initial_yaw,
                        args.duration, reference)
    if args.prepare_only:
        print('Prepared immutable run configs; simulation not started.', flush=True)
        return
    for reference in args.references:
        for model in args.models:
            visible_trial(output / reference / model, args.duration, args.settle_seconds)
    print(f'All selected visible trials finished. Saved runs: {output}', flush=True)


if __name__ == '__main__':
    main()
