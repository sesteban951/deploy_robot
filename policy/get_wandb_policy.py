##
#
# Download the latest ONNX policy checkpoint from a W&B run.
#
##

# standard library
import argparse
from pathlib import Path

# for W&B API
import wandb

# directory imports
import os
ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")

############################################################################
# MAIN FUNCTION
############################################################################

def main(args=None):

    # parse the input
    parser = argparse.ArgumentParser(description="Download the latest ONNX policy checkpoint from a W&B run.")
    parser.add_argument("wandb_run_path", help="W&B run path (entity/project/run_id)")
    args = parser.parse_args()

    # use the W&B API to find the latest ONNX file in the run
    api = wandb.Api()
    run = api.run(args.wandb_run_path)
    onnx_files = [f.name for f in run.files() if f.name.endswith(".onnx")]

    if not onnx_files:
        raise RuntimeError(f"No .onnx files found in run {args.wandb_run_path}")

    checkpoint = onnx_files[-1]
    print(f"Found: [{len(onnx_files)}] onnx files.")
    print(f"Downloading: [{checkpoint}].")

    # download the checkpoint to the local policy directory
    output_dir = Path(ROOT_DIR) / "policy"
    output_dir.mkdir(parents=True, exist_ok=True)
    run.file(checkpoint).download(str(output_dir), replace=True)
    print(f"Downloaded to: [{output_dir / checkpoint}].")


if __name__ == "__main__":
    main()
