"""Model bundles: local assets, numerical settings and explicit adapter plugins.

A new policy/model family selects its own encoder, feedback decoder and runtime
factory. Different state dimensions are never coerced into the walking kernel.
"""

from dataclasses import dataclass
import hashlib
from importlib import import_module
import json
from pathlib import Path
import re

import yaml

DEPLOY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelProfile:
    name: str
    values: dict

    def path(self, relative):
        path = (DEPLOY_ROOT / relative).resolve()
        if not path.is_relative_to(DEPLOY_ROOT):
            raise ValueError('Profile assets must stay inside deploy_robot')
        return path

    def symbol(self, key):
        module, attribute = self.values[key].split(':', 1)
        if not module.startswith(('kcf_mpc.', 'deploy.kcf.')):
            raise ValueError(f'{key} must name a deployment-local plugin')
        result = import_module(module)
        for part in attribute.split('.'):
            result = getattr(result, part)
        return result

    def validate_assets(self):
        for relative, expected in self.values['artifact_sha256'].items():
            path = self.path(relative)
            if not path.is_file():
                raise FileNotFoundError(f'Missing profile asset: {path}. Run git lfs pull if needed.')
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != expected:
                raise ValueError(f'Profile asset checksum mismatch: {relative}. Check the profile or run git lfs pull.')
        saved = json.loads((self.path(self.values['results_dir'])/'config.json').read_text())
        for key in ('state_dim', 'input_dim', 'nH', 'state_schema'):
            if saved[key] != self.values[key]:
                raise ValueError(f'Checkpoint/profile {key} mismatch')
        for key in ('policy_config', 'simulation_config'):
            config_path = DEPLOY_ROOT/'deploy/configs/g1'/f'{self.values[key]}.yaml'
            config = yaml.safe_load(config_path.read_text())
            if config.get('kcf_profile') != self.name:
                raise ValueError(f'{key} does not select profile {self.name}')
            if (DEPLOY_ROOT/'policy/g1'/config['policy_path']).resolve() != self.path(self.values['policy_path']):
                raise ValueError(f'{key} policy does not match the model profile')
            if (DEPLOY_ROOT/'models'/config['xml_path']).resolve() != self.path(self.values['simulation_xml']):
                raise ValueError(f'{key} robot XML does not match the model profile')

    def settings(self, *, reference_mode=None, prediction_model='kcf'):
        settings = self.symbol('mpc_settings_factory')(prediction_model=prediction_model)
        settings['results_dir'] = str(self.path(self.values['results_dir']))
        settings['checkpoint'] = self.values['checkpoint']
        if reference_mode is not None:
            from .config.mpc_config import REFERENCE_MODES
            if reference_mode not in REFERENCE_MODES:
                raise ValueError(f'reference_mode must be one of {REFERENCE_MODES}')
            settings['reference_mode'] = reference_mode
        # optional kcf.twist_limits [vx, vy, wz]: tighten the MPC input bounds, so it only plans
        # (and issues) twists inside them; clipping its output instead would cancel its committed plan
        limits = self.values.get('twist_limits')
        if limits is not None:
            limits = [float(x) for x in limits]
            if len(limits) != 3 or not all(0 < x < float('inf') for x in limits):
                raise ValueError('kcf.twist_limits must be three positive numbers [vx, vy, wz]')
            settings['input_lower_bounds'] = tuple(max(lo, -lim) for lo, lim in zip(settings['input_lower_bounds'], limits))
            settings['input_upper_bounds'] = tuple(min(hi, lim) for hi, lim in zip(settings['input_upper_bounds'], limits))
        return settings


def load_profile(name='walking'):
    if not isinstance(name, str) or re.fullmatch(r'[A-Za-z0-9_-]+', name) is None:
        raise ValueError('Profile must be a local profile name')
    matches = []
    for path in sorted((DEPLOY_ROOT/'deploy/configs/g1').glob('*.yaml')):
        config = yaml.safe_load(path.read_text())
        if isinstance(config, dict) and config.get('kcf_profile') == name:
            matches.append((path, config))
    if not matches:
        raise ValueError(f'Unknown model profile {name!r}; add a complete KCF deployment YAML under deploy/configs/g1')
    if len(matches) != 1:
        raise ValueError(f'Profile {name!r} must select exactly one deployment YAML, found: '
                         + ', '.join(path.name for path, _ in matches))
    path, config = matches[0]
    values = config.get('kcf')
    if not isinstance(values, dict) or values.get('version') != 1:
        raise ValueError(f'Invalid model profile: {path}')
    required = ('runtime_factory', 'state_adapter', 'feedback_decoder', 'state_encoder',
                'heading_encoder', 'robot_config_factory', 'mpc_settings_factory',
                'results_dir', 'checkpoint', 'state_source', 'state_dim', 'input_dim', 'nH',
                'joint_count', 'recurrent_state_dim', 'state_schema', 'artifact_sha256')
    if any(key not in values for key in required) or any(key not in config for key in ('policy_path', 'xml_path')):
        raise ValueError(f'Incomplete model profile: {path}')
    if values['state_source'] not in ('simulation', 'mocap'):
        raise ValueError(f'kcf.state_source must be simulation or mocap: {path}')
    # Derive shared assets/routing from this file, not a second configuration.
    values = dict(values, name=name, policy_config=path.stem, simulation_config=path.stem,
                  policy_path=str(Path('policy/g1')/config['policy_path']),
                  simulation_xml=str(Path('models')/config['xml_path']))
    return ModelProfile(name, values)
