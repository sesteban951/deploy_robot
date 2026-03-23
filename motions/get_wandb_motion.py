##
#
# Download motion files from a W&B registry artifact.
#
##

# standard library
import argparse
import shutil
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
    parser = argparse.ArgumentParser(description="Download motion files from a W&B registry artifact.")
    parser.add_argument("artifact_path", help="W&B artifact path with tag (e.g., wandb-registry-Motions/walk1_subject1:latest)")
    args = parser.parse_args()

    # fetch the artifact from W&B
    api = wandb.Api()
    artifact_full = args.artifact_path
    print(f"Fetching artifact: [{artifact_full}]")
    artifact = api.artifact(artifact_full)

    # extract the motion name from the artifact path (e.g., "walk1_subject1" from "wandb-registry-Motions/walk1_subject1:latest")
    motion_name = args.artifact_path.split("/")[-1].split(":")[0]

    # download to a temp location, then copy npz files with the motion name
    output_dir = Path(ROOT_DIR) / "motions"
    output_dir.mkdir(parents=True, exist_ok=True)
    download_path = artifact.download(root=str(output_dir))

    # copy npz files with the motion name
    npz_files = list(Path(download_path).glob("*.npz"))
    for f in npz_files:
        dest = output_dir / f"{motion_name}.npz"
        shutil.copy2(f, dest)
        print(f"Saved: [{dest}]")


if __name__ == "__main__":
    main()
