import argparse
import os
import subprocess
import sys
from pathlib import Path

# Path to the expert-demonstration generation script
REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = REPO_ROOT / "metaworld" / "data" / "training_data_gen.py"

# ---- CONFIG ----
TASKS = [
    'box-close-v2',
    'hand-insert-v2',
    'handle-pull-v2',
    'pick-place-v2',
    'push-v2',
    'shelf-place-v2',
]

NUM_TRAJ = 10
NOISE = 0.0
RES = 224
CAMERA = "corner"
DATA_DIR = "plex_data"
WRITE_VIDEO = False
INCLUDE_DEPTH = False
USE_PLAY_POLICY = False

parser = argparse.ArgumentParser(description="Generate expert demonstrations for the initial training sets.")
parser.add_argument("--tasks", type=str, default=",".join(TASKS), help="Comma-separated task names")
parser.add_argument("--num_traj", type=int, default=NUM_TRAJ, help="Expert trajectories per task")
parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Output directory")
args = parser.parse_args()

# ---- Build command ----
cmd = [
    sys.executable, str(SCRIPT_PATH),
    "--tasks", args.tasks,
    "--num_traj", str(args.num_traj),
    "--noise", str(NOISE),
    "--res", str(RES),
    "--camera", CAMERA,
    "--data_dir_path", args.data_dir,
]

if WRITE_VIDEO:
    cmd.append("--write_video")
else:
    cmd.append("--nowrite_video")

if INCLUDE_DEPTH:
    cmd.append("--include_depth")
else:
    cmd.append("--noinclude_depth")

if USE_PLAY_POLICY:
    cmd.append("--use_play_policy")
else:
    cmd.append("--use_expert_policy")


env = os.environ.copy()
env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

print("CWD:", REPO_ROOT)
print("PYTHONPATH:", env["PYTHONPATH"])
print("Running:\n", " ".join(cmd), "\n")

subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
