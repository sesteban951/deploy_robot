#!/usr/bin/env python
"""Pull the trained ONNX policy for each seed-sweep run from W&B.

For each matching run this grabs the latest (datetime-named) ONNX checkpoint
and writes it to the local policy directory, renamed after the run so the file
name is meaningful instead of a raw timestamp.

Runs are matched by W&B group, same as scripts/sweep/pull_curves.py in
unitree_rl_mjlab. By default it grabs every run in the `ablation` group of the
`mjlab` project and writes to policy/ablation/.

Naming: the group token is stripped from the run name, so a policy from run
`kino_traj_ablation_0` lands as `kino_traj_0.onnx`.

Usage:
    python sweep/pull_policies.py                       # all ablation-group runs -> policy/ablation/
    python sweep/pull_policies.py --outdir /tmp/policies
    python sweep/pull_policies.py --group ablation
    python sweep/pull_policies.py --list                # just list matching runs, don't download
"""

import argparse
import os
import sys
import tempfile

import wandb

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_GROUP = "ablation"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default=None, help="W&B entity (default: your default entity).")
    p.add_argument("--project", default="mjlab", help="W&B project (default: mjlab).")
    p.add_argument(
        "--group",
        default=DEFAULT_GROUP,
        help=f"W&B group to match (default: {DEFAULT_GROUP}).",
    )
    p.add_argument(
        "--outdir",
        default=os.path.join(ROOT_DIR, "policy", "ablation"),
        help="Directory to write ONNX policies into (default: policy/ablation).",
    )
    p.add_argument("--list", action="store_true", help="List matching runs and exit (no download).")
    p.add_argument("--overwrite", action="store_true", help="Re-download runs whose policy already exists.")
    return p.parse_args()


def matching_runs(api, path, group):
    runs = list(api.runs(path, filters={"group": group}))
    # Stable, human-friendly ordering.
    runs.sort(key=lambda r: r.name)
    return runs


def policy_name(run_name, group):
    """Strip the group token from the run name: kino_traj_ablation_0 -> kino_traj_0."""
    name = run_name.replace(f"_{group}_", "_").replace(f"_{group}", "").replace(group, "")
    name = name.strip("_") or run_name
    return f"{name}.onnx"


def latest_onnx(run):
    """Return the name of the latest (datetime-sorted) .onnx file in the run, or None."""
    onnx_files = sorted(f.name for f in run.files() if f.name.endswith(".onnx"))
    return onnx_files[-1] if onnx_files else None


def pull_run(run, checkpoint, out_path):
    """Download `checkpoint` from the run and place it at `out_path`."""
    with tempfile.TemporaryDirectory() as tmp:
        run.file(checkpoint).download(tmp, replace=True)
        src = os.path.join(tmp, checkpoint)
        os.replace(src, out_path)


def main():
    args = parse_args()

    api = wandb.Api()
    entity = args.entity or api.default_entity
    path = f"{entity}/{args.project}"
    print(f"[pull-policies] project: {path}")
    print(f"[pull-policies] group: {args.group}")

    runs = matching_runs(api, path, args.group)
    if not runs:
        print("[pull-policies] no matching runs found.", file=sys.stderr)
        sys.exit(1)

    print(f"[pull-policies] {len(runs)} matching runs:")
    for r in runs:
        print(f"    {r.name:<32} {r.state:<10} id={r.id}")

    if args.list:
        return

    os.makedirs(args.outdir, exist_ok=True)
    for r in runs:
        out_path = os.path.join(args.outdir, policy_name(r.name, args.group))
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[pull-policies] skip (exists): {out_path}  (use --overwrite to refresh)")
            continue

        checkpoint = latest_onnx(r)
        if checkpoint is None:
            print(f"[pull-policies] no .onnx file in run {r.name}, skipping.", file=sys.stderr)
            continue

        print(f"[pull-policies] downloading {r.name}: {checkpoint} -> {os.path.basename(out_path)} ...", flush=True)
        pull_run(r, checkpoint, out_path)
        print(f"[pull-policies]   -> {out_path}")

    print(f"[pull-policies] done. Policies in: {args.outdir}")


if __name__ == "__main__":
    main()
