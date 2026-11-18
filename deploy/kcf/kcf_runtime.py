"""Construct a repository-local model profile; no parent-checkout imports."""

from pathlib import Path
import sys
import numpy as np

# Only add this deployment checkout, never a sibling research repository.
DEPLOY_ROOT = Path(__file__).resolve().parents[2]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from kcf_mpc.profiles import load_profile


def load_controller(profile_name='walking', reference_mode=None, prediction_model='kcf'):
    if sys.version_info[:2] != (3, 11) or int(np.__version__.split('.')[0]) != 2:
        raise RuntimeError('Run conda activate deploy: KCF ROS requires Python 3.11 and NumPy 2')
    profile = load_profile(profile_name)
    profile.validate_assets()
    # prediction_model 'kcf' (learned model) or 'single_integrator' (ideal twist), each with its own gains
    settings = profile.settings(reference_mode=reference_mode, prediction_model=prediction_model)
    return (profile.symbol('runtime_factory')(profile, settings, quiet=True),
            profile.symbol('state_adapter'))
