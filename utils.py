import os
import glob
import h5py
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class MWTrajectory:
    actions: np.ndarray          # (T, A)
    depths: np.ndarray           # (T, H, W) or empty (0,) if not stored
    full_states: np.ndarray      # (T, S_full)
    observations: np.ndarray     # (T, H, W, 3) uint8
    proprio_states: np.ndarray   # (T, S_prop)
    rewards: np.ndarray          # (T,)
    terminals: np.ndarray        # (T,) bool
    infos: Dict[str, np.ndarray] # {"goal": (T,) bool, ...}

    @property
    def T(self) -> int:
        if self.actions is not None and self.actions.ndim >= 1:
            return int(self.actions.shape[0])
        return 0

    @property
    def action_dim(self) -> int:
        if self.actions is None or self.actions.ndim != 2:
            return 0
        return int(self.actions.shape[1])

    @property
    def full_state_dim(self) -> int:
        if self.full_states is None or self.full_states.ndim != 2:
            return 0
        return int(self.full_states.shape[1])

    @property
    def proprio_dim(self) -> int:
        if self.proprio_states is None or self.proprio_states.ndim != 2:
            return 0
        return int(self.proprio_states.shape[1])


def _find_single_hdf5(task_name: str, noise: float, root: str) -> str:
    """
    Find the .hdf5 file inside:
      {root}/{task_name}/Sawyer/noise{noise}/
    """
    noise_dir = os.path.join(root, task_name, "Sawyer", f"noise{noise}")
    if not os.path.isdir(noise_dir):
        raise FileNotFoundError(f"Directory not found: {noise_dir}")

    files = sorted(glob.glob(os.path.join(noise_dir, "*.hdf5")))
    if len(files) == 0:
        raise FileNotFoundError(f"No .hdf5 files found in: {noise_dir}")
    if len(files) > 1:
        raise RuntimeError(
            f"Expected exactly 1 .hdf5 file in {noise_dir}, found {len(files)}:\n"
            + "\n".join(files)
        )
    return files[0]


def load_mw_dataset(task_name: str, root: str = "plex_data", noise: float = 0.0) -> List[MWTrajectory]:
    """
    Loads the MetaWorld dataset.
    """
    path = _find_single_hdf5(task_name=task_name, noise=noise, root=root)

    out: Dict[str, MWTrajectory] = {}

    def _np(dset: h5py.Dataset) -> np.ndarray:
        return dset[()]

    with h5py.File(path, "r") as f:
        # load groups that look like trajectories
        traj_keys = [k for k in f.keys() if k.startswith("traj_") and isinstance(f[k], h5py.Group)]
        traj_keys = sorted(traj_keys, key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x)

        if not traj_keys:
            raise RuntimeError(f"No trajectory groups (traj_*) found in: {path}")

        for tk in traj_keys:
            g = f[tk]
            
            actions = _np(g["actions"]) if "actions" in g else None
            depths = _np(g["depths"]) if "depths" in g else np.array([], dtype=np.uint16)
            full_states = _np(g["full_states"]) if "full_states" in g else None
            observations = _np(g["observations"]) if "observations" in g else None
            proprio_states = _np(g["proprio_states"]) if "proprio_states" in g else None
            rewards = _np(g["rewards"]) if "rewards" in g else None
            terminals = _np(g["terminals"]) if "terminals" in g else None

            infos: Dict[str, np.ndarray] = {}
            if "infos" in g and isinstance(g["infos"], h5py.Group):
                ig = g["infos"]
                for ik in ig.keys():
                    infos[ik] = _np(ig[ik])

            out[tk] = MWTrajectory(
                actions=actions,
                depths=depths,
                full_states=full_states,
                observations=observations,
                proprio_states=proprio_states,
                rewards=rewards,
                terminals=terminals,
                infos=infos,
            )

    return list(out.values())
