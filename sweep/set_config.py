##
#
# Write a sweep config by copying the base mimic config and swapping only the
# active `policy_path:` / `motion_path:` lines.
#
# The base config (deploy/configs/g1_29dof_mimic.yaml) is never modified: we
# read it and write a separate sweep copy (default g1_29dof_mimic_sweep.yaml)
# into the same directory so the control/sim nodes can still resolve it by name.
# Comments, gains, and home pose in the base file are preserved verbatim.
#
##

import argparse
import os
import re

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT_DIR, "deploy", "configs")

BASE_CONFIG = "g1_29dof_mimic.yaml"
SWEEP_CONFIG = "g1_29dof_mimic_sweep.yaml"


# replace the value on the first active (non-comment) `key: ...` line
def _replace_active(lines, key, value):
    pat = re.compile(rf'^(\s*){re.escape(key)}\s*:.*$')
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if pat.match(line):
            indent = pat.match(line).group(1)
            lines[i] = f'{indent}{key}: "{value}"\n'
            return True
    return False


# write a sweep config with policy_path/motion_path swapped
def write_sweep_config(policy_path, motion_path,
                       base=BASE_CONFIG, out=SWEEP_CONFIG):
    base_path = os.path.join(CONFIG_DIR, base)
    out_path = os.path.join(CONFIG_DIR, out)

    with open(base_path, "r") as f:
        lines = f.readlines()

    if not _replace_active(lines, "policy_path", policy_path):
        raise RuntimeError(f"No active 'policy_path:' line found in {base_path}.")
    if not _replace_active(lines, "motion_path", motion_path):
        raise RuntimeError(f"No active 'motion_path:' line found in {base_path}.")

    with open(out_path, "w") as f:
        f.writelines(lines)

    return out_path


def main():
    p = argparse.ArgumentParser(description="Write a sweep config with a swapped policy/motion.")
    p.add_argument("--policy", required=True, help='policy_path relative to policy/ (e.g. "ablation/kino_backflip_1.onnx").')
    p.add_argument("--motion", required=True, help='motion_path relative to motions/ (e.g. "ablation/kino_backflip.npz").')
    p.add_argument("--base", default=BASE_CONFIG, help=f"Base config filename (default: {BASE_CONFIG}).")
    p.add_argument("--out", default=SWEEP_CONFIG, help=f"Output config filename (default: {SWEEP_CONFIG}).")
    args = p.parse_args()

    out_path = write_sweep_config(args.policy, args.motion, args.base, args.out)
    print(f"[set-config] wrote {out_path}")
    print(f"[set-config]   policy_path: {args.policy}")
    print(f"[set-config]   motion_path: {args.motion}")


if __name__ == "__main__":
    main()
