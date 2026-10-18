# DOCI: Deployment-Time Online Imitation Learning from Corrective Demonstrations

This repository contains the code for our paper **DOCI: Deployment-Time Online Imitation Learning from Corrective Demonstrations**.

Experiments are run on [Meta-World](https://github.com/Farama-Foundation/Metaworld) manipulation tasks (a modified copy is vendored in `metaworld/`). For each task, the framework:

1. trains an initial policy from a small set of expert demonstrations,
2. deploys the policy and collects corrective expert demonstrations on the episodes where it fails,
3. updates the policy online with those demonstrations, and
4. evaluates the policy after every update.

## Repository layout

| Path | Description |
|---|---|
| `run.py` | Experiment driver (initial fit, corrective-demo collection, online updates, evaluation) |
| `run.sh` | Launch script for six tasks (`box-close`, `hand-insert`, `pick-place`, `push`, `shelf-place`, `handle-pull`) |
| `plex_data_gen.py` | Generates the initial expert demonstration datasets |
| `models/DOCI.py` | The DOCI policy |
| `utils.py` | Trajectory container and HDF5 dataset loader |
| `metaworld/` | Vendored Meta-World (v2 environments, scripted expert policies, data tools) |

## Setup

### 1. System packages

Compiling `mujoco-py` and rendering offscreen images need a C compiler and OpenGL development files:

```bash
sudo apt-get install -y gcc patchelf libgl1-mesa-dev libglew-dev libosmesa6-dev
```

Without root access, install GLEW into the conda environment instead (after step 3):
`conda install -c conda-forge glew`.

### 2. MuJoCo 2.1 binaries

`mujoco-py` requires the MuJoCo 2.1.0 binaries in `~/.mujoco/mujoco210`:

```bash
mkdir -p ~/.mujoco
wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco

export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
```

Add the two `export` lines to your shell profile so they persist.

### 3. Python environment

```bash
conda create -n doci python=3.10 -y
conda activate doci

# PyTorch (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Remaining dependencies (Cython is pinned below 3 because mujoco-py does not build with Cython 3)
pip install -r requirements.txt

# Install the vendored Meta-World package and this repository in editable mode
pip install -e .
```

The first time `mujoco_py` is imported it compiles a small extension, which takes a minute.

### 4. Check the installation

```bash
python -c "import metaworld; env = metaworld.mw_gym_make('pick-place-v2', cam_height=224, cam_width=224, depth=False); obs = env.reset(); print(obs['image'].shape, obs['full_state'].shape)"
```

This should print `(224, 224, 3) (39,)`.

The DINOv2 weights used by DOCI are downloaded from the Hugging Face hub on first use, so the first run needs internet access.

## Generating data

Initial expert demonstrations are generated with the scripted Meta-World policies (10 noiseless trajectories per task, 224x224 `corner` camera):

```bash
# the six tasks in run.sh
python plex_data_gen.py

# or selected tasks only
python plex_data_gen.py --tasks pick-place-v2 --num_traj 10
```

Options: `--tasks` (comma-separated, default: the six tasks above), `--num_traj` (default 10) and `--data_dir` (default `plex_data`).

Datasets are written to `plex_data/<task>/Sawyer/noise0.0/*.hdf5`.

## Running experiments

Run all six tasks with the default settings in `run.sh`:

```bash
bash run.sh
```

Or run a single task:

```bash
python run.py \
    --model_name doci \
    --tasks pick-place-v2 \
    --horizons 100 \
    --eval_episodes 200 \
    --steps 5 \
    --demos_per_step 5 \
    --dataset_root plex_data \
    --log_dir logs
```

For each task, `run.py` fits DOCI on the initial demonstrations and evaluates it. It then repeats `--steps` times: roll out the current policy, record a corrective expert demonstration from the same start state for each failed episode (`--demos_per_step` per step), update the policy, and evaluate again.

Main arguments:

| Argument | Default | Meaning |
|---|---|---|
| `--model_name` | required | Policy to run (`doci`) |
| `--tasks`, `--horizons` | required | Tasks and their episode horizons |
| `--eval_episodes` | 200 | Evaluation episodes after each update |
| `--steps` | 5 | Number of online update rounds |
| `--demos_per_step` | 5 | Corrective demonstrations collected per round |
| `--eval_seed`, `--collect_seed` | 42, 123 | Seeds for evaluation and demonstration collection |
| `--dataset_root` | `plex_data` | Where the initial datasets are read from |
| `--log_dir` | `logs` | Where results are written |


## Outputs

Results are written to `logs/<model_name>/<task>/`:

| File | Content |
|---|---|
| `config.json` | Run configuration |
| `eval.csv` | Success rate, action deviation and timing after each update |
| `train.csv` | Time taken by the initial fit and each update |
| `collect.csv` | Corrective-demonstration collection statistics |
| `success_rate.png`, `action_deviation.png` | Curves against the number of added trajectories |

## Acknowledgements

The environments, scripted policies and data tooling in `metaworld/` are adapted from [Meta-World](https://github.com/Farama-Foundation/Metaworld) (MIT license).
