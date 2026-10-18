from collections import deque
from typing import TYPE_CHECKING, Optional, Tuple, Deque, List

import timm
import torch
import torch.nn.functional as F
import numpy as np

if TYPE_CHECKING:
    from utils import MWTrajectory


# ============================================================
#                        NumPy utilities
# ============================================================

def _spectral_radius(
    W: np.ndarray,
    n_iter: int = 20,
    n_vecs: int = 64,
    eps: float = 1e-12,
    reduction: str = "median",  # "median" or "max"
    seed: int = 0,
) -> float:
    """Approximate spectral radius of W via power iteration (NumPy version)."""
    d = W.shape[1]

    rng = np.random.default_rng(seed)
    V = rng.standard_normal((d, n_vecs), dtype=np.float64)
    V = V.astype(W.dtype, copy=False)
    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + eps)

    for _ in range(int(n_iter)):
        V = W @ V
        V = V / (np.linalg.norm(V, axis=0, keepdims=True) + eps)

    WV = W @ V
    rhos = np.linalg.norm(WV, axis=0)

    if reduction == "median":
        rho_hat = np.median(rhos)
    else:
        rho_hat = np.max(rhos)

    return float(rho_hat if np.isfinite(rho_hat) and rho_hat > eps else 1.0)


# ============================================================
#                  Online Ridge Regression
# ============================================================

class OnlineRidge:
    """Incremental ridge regression matching the centered NumPy solution."""

    def __init__(self, dim_in: int, dim_out: int, alpha: float, device=None):
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.alpha = float(alpha)

        self.n = 0
        self.sum_x = np.zeros((self.dim_in,), dtype=np.float64)
        self.sum_y = np.zeros((self.dim_out,), dtype=np.float64)
        self.sum_xx = np.zeros((self.dim_in, self.dim_in), dtype=np.float64)
        self.sum_xy = np.zeros((self.dim_in, self.dim_out), dtype=np.float64)

    def update(self, X: np.ndarray, Y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if X.ndim != 2 or X.shape[1] != self.dim_in:
            raise ValueError(f"Expected X shape (N, {self.dim_in}), got {tuple(X.shape)}")
        if Y.ndim != 2 or Y.shape[1] != self.dim_out:
            raise ValueError(f"Expected Y shape (N, {self.dim_out}), got {tuple(Y.shape)}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Batch size mismatch between X and Y")

        self.n += X.shape[0]
        self.sum_x += X.sum(axis=0)
        self.sum_y += Y.sum(axis=0)
        self.sum_xx += X.T @ X
        self.sum_xy += X.T @ Y

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n == 0:
            raise RuntimeError("No data has been added.")

        n = float(self.n)
        mu_x = self.sum_x / n
        mu_y = self.sum_y / n

        XtX_c = self.sum_xx - n * np.outer(mu_x, mu_x)
        XtY_c = self.sum_xy - n * np.outer(mu_x, mu_y)

        I = np.eye(self.dim_in, dtype=np.float64)
        W = np.linalg.solve(XtX_c + self.alpha * I, XtY_c)
        b = mu_y.reshape(1, -1) - mu_x.reshape(1, -1) @ W
        return W, b


# ============================================================
#                           Policy
# ============================================================

class ESNRidgeBC:
    """
    ESN + ridge BC policy with action chunking and temporal ensemble.

    Pipeline:
      obs, state -> x
      x + ESN memory -> z
      z -> action chunk
      buffer of chunks -> temporal ensemble -> action

    Continual learning:
      - fit(...) initializes the random features / ESN / ridge head
      - update(...) applies online updates the same ridge head
    """

    RELU_CLIP = 10.0

    def __init__(
        self,
        encoder_name: str = "vit_base_patch14_dinov2.lvd142m",
        img_size: int = 224,
        enc_batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        m_img: int = 2000,
        m_state: int = 2000,
        m_fuse: int = 4000,
        m_mem: int = 1000,
        m_ridge: int = 5000,
        ridge_alpha: float = 1,
        chunk_len: int = 5,
        activation: str = "tanh",   # "tanh" or "relu"
        spectral_radius: float = 0.9,
        leak: float = 0.5,
        seed: int = 0,
    ):
        if activation not in ("tanh", "relu"):
            raise ValueError("activation must be 'tanh' or 'relu'")
        if chunk_len < 1:
            raise ValueError("chunk_len must be >= 1")

        self.device = torch.device(device)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.m_img = int(m_img)
        self.m_state = int(m_state)
        self.m_fuse = int(m_fuse)
        self.m_mem = int(m_mem)
        self.m_ridge = int(m_ridge)

        self.spectral_radius = float(spectral_radius)
        self.leak = float(leak)
        self.ridge_alpha = float(ridge_alpha)

        act_tolerance = 1e-5
        self.action_clip = 1 - act_tolerance
        self.enc_batch_size = int(enc_batch_size)
        self.chunk_len = int(chunk_len)
        self.activation = activation

        # Frozen encoder
        # img_size makes timm resample the pretrained position embeddings to the
        # input resolution, so frames are encoded at their native size.
        self.encoder = timm.create_model(
            encoder_name, pretrained=True, num_classes=0, img_size=img_size
        ).to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        cfg = timm.data.resolve_data_config({}, model=self.encoder)
        # resolve_data_config reports the checkpoint's default size, so use img_size directly
        self.enc_input_size = (3, int(img_size), int(img_size))  # (3,H,W)
        self.enc_mean = torch.tensor(cfg["mean"], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.enc_std = torch.tensor(cfg["std"], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)

        # Known after fit-init
        self.d_state: Optional[int] = None
        self.d_action: Optional[int] = None
        self.d_enc: Optional[int] = None
        self.chunk_dim: Optional[int] = None  # chunk_len * d_action

        # State normalization stats
        self.state_mean: Optional[np.ndarray] = None
        self.state_std: Optional[np.ndarray] = None

        # Random parameters
        self.R_img: Optional[np.ndarray] = None
        self.R_state: Optional[np.ndarray] = None
        self.W_fuse: Optional[np.ndarray] = None
        self.W_x2mem: Optional[np.ndarray] = None
        self.R_a2mem: Optional[np.ndarray] = None
        self.W_res: Optional[np.ndarray] = None
        self.W_in: Optional[np.ndarray] = None
        self.W_ridge: Optional[np.ndarray] = None

        # Ridge head: z -> flattened chunk
        self.W_out: Optional[np.ndarray] = None
        self.b_out: Optional[np.ndarray] = None
        self._policy_ridge: Optional[OnlineRidge] = None

        self._clear_runtime()

    # ------------------------------------------------------------------
    # checks
    # ------------------------------------------------------------------

    def _check_dims_inited(self) -> None:
        if self.d_state is None or self.d_action is None or self.d_enc is None:
            raise RuntimeError("Dims not initialized. Call fit(...) first.")
        if self.chunk_dim is None:
            raise RuntimeError("Chunk dim not initialized. Call fit(...) first.")
        if any(
            x is None for x in (
                self.R_img, self.R_state, self.W_fuse, self.W_x2mem,
                self.R_a2mem, self.W_res, self.W_in, self.W_ridge
            )
        ):
            raise RuntimeError("Random parameters missing. Call fit(...) first.")

    def _check_stats_ready(self) -> None:
        if self.state_mean is None or self.state_std is None:
            raise RuntimeError("State norm stats missing. fit(...) must set them before computing x.")

    def _check_trained(self) -> None:
        self._check_dims_inited()
        if self.W_out is None or self.b_out is None or self._policy_ridge is None:
            raise RuntimeError("Policy ridge head not trained. Call fit(...) first.")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(x)
        return np.clip(x, 0.0, self.RELU_CLIP)

    def _clear_runtime(self) -> None:
        self.m = np.zeros((self.m_mem,), dtype=np.float64)
        self._x_curr: Optional[np.ndarray] = None
        self._z_curr: Optional[np.ndarray] = None
        self._last_action: Optional[np.ndarray] = None
        self._chunk_buffer: Deque[np.ndarray] = deque(maxlen=self.chunk_len)

    def _randn64(self, shape, scale: float) -> np.ndarray:
        return (self.rng.standard_normal(shape) * float(scale)).astype(np.float64)

    def _init_random_params(self, d_state: int, d_action: int, d_enc: int) -> None:
        self.d_state = int(d_state)
        self.d_action = int(d_action)
        self.d_enc = int(d_enc)
        self.chunk_dim = int(self.chunk_len * self.d_action)

        self.R_img = self._randn64((self.m_img, self.d_enc), scale=1.0 / np.sqrt(self.d_enc))
        self.R_state = self._randn64((self.m_state, self.d_state), scale=1.0 / np.sqrt(self.d_state))
        self.R_a2mem = self._randn64((self.m_mem, self.d_action), scale=1.0 / np.sqrt(self.d_action))

        fuse_in = self.m_img + self.m_state
        self.W_fuse = self._randn64((self.m_fuse, fuse_in), scale=1.0 / np.sqrt(fuse_in))
        self.W_x2mem = self._randn64((self.m_mem, self.m_fuse), scale=1.0 / np.sqrt(self.m_fuse))

        ridge_in = self.m_fuse + self.m_mem
        self.W_ridge = self._randn64((self.m_ridge, ridge_in), scale=1.0 / np.sqrt(ridge_in))

        W_res = self._randn64((self.m_mem, self.m_mem), scale=1.0 / np.sqrt(self.m_mem))
        rho_hat = _spectral_radius(W_res, seed=self.seed)
        scale = self.spectral_radius / (rho_hat if rho_hat > 1e-12 else 1.0)
        self.W_res = W_res * float(scale)

        self.W_in = self._randn64((self.m_mem, self.m_mem), scale=1.0 / np.sqrt(self.m_mem))

    # ------------------------------------------------------------------
    # encoder / features
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_images(self, obs_uint8: np.ndarray) -> np.ndarray:
        """
        Batched image encoding (processed in chunks of enc_batch_size).
        obs_uint8: (N, H, W, 3) uint8
        Returns (N, d_enc) float64.
        """
        obs_uint8 = np.ascontiguousarray(obs_uint8)
        _, Ht, Wt = self.enc_input_size

        out = []
        for i in range(0, obs_uint8.shape[0], self.enc_batch_size):
            x = torch.from_numpy(obs_uint8[i:i + self.enc_batch_size]).to(self.device)
            x = x.permute(0, 3, 1, 2).float() / 255.0
            x = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)
            x = (x - self.enc_mean) / self.enc_std

            feat = self.encoder.forward_features(x)
            if isinstance(feat, (tuple, list)):
                feat = feat[-1]
            if feat.ndim == 3:
                feat = feat.mean(dim=1)
            out.append(feat.to(dtype=torch.float64).detach().cpu().numpy())

        return np.concatenate(out, axis=0).astype(np.float64)

    def encode_image(self, obs_uint8: np.ndarray) -> np.ndarray:
        """
        Single-image encoding.
        Returns (d_enc,) float64.
        """
        return self.encode_images(np.asarray(obs_uint8)[None])[0]

    def norm_state(self, s: np.ndarray) -> np.ndarray:
        self._check_stats_ready()
        return (s - self.state_mean) / (self.state_std + 1e-6)

    def compute_x(self, obs_uint8: np.ndarray, full_state: np.ndarray) -> np.ndarray:
        self._check_dims_inited()
        self._check_stats_ready()

        s = np.asarray(full_state, dtype=np.float64).reshape(-1)
        s = self.norm_state(s)

        e = self.encode_image(obs_uint8)

        phi_img = self._act(self.R_img @ e)
        phi_state = self._act(self.R_state @ s)
        x_in = np.concatenate([phi_img, phi_state], axis=0)
        x = self._act(self.W_fuse @ x_in)
        return x

    def compute_x_batch(self, obs_uint8: np.ndarray, full_states: np.ndarray) -> np.ndarray:
        """
        Batched version of compute_x.
        obs_uint8: (N, H, W, 3), full_states: (N, d_state). Returns (N, m_fuse).
        """
        self._check_dims_inited()
        self._check_stats_ready()

        S = np.asarray(full_states, dtype=np.float64).reshape(len(full_states), -1)
        S = self.norm_state(S)

        E = self.encode_images(obs_uint8)

        phi_img = self._act(E @ self.R_img.T)
        phi_state = self._act(S @ self.R_state.T)
        x_in = np.concatenate([phi_img, phi_state], axis=1)
        return self._act(x_in @ self.W_fuse.T)

    def compute_u(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        self._check_dims_inited()
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        return self._act(self.W_x2mem @ x + self.R_a2mem @ a)

    def esn_update(self, m: np.ndarray, u: np.ndarray) -> np.ndarray:
        self._check_dims_inited()

        m = np.asarray(m, dtype=np.float64).reshape(-1)
        u = np.asarray(u, dtype=np.float64).reshape(-1)

        pre = self.W_res @ m + self.W_in @ u
        m_new = self._act(pre)
        return (1.0 - self.leak) * m + self.leak * m_new if self.leak < 1.0 else m_new

    def compute_z(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        self._check_dims_inited()

        x = np.asarray(x, dtype=np.float64).reshape(-1)
        m = np.asarray(m, dtype=np.float64).reshape(-1)

        z_in = np.concatenate([x, m], axis=0)
        return self._act(self.W_ridge @ z_in)

    # ------------------------------------------------------------------
    # action chunking
    # ------------------------------------------------------------------

    def chunk_from_z(self, z: np.ndarray) -> np.ndarray:
        """
        Predict action chunk from z.
        z: (m_ridge,)
        returns: (chunk_len, d_action)
        """
        self._check_trained()
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        out = z @ self.W_out + self.b_out.reshape(-1)
        return out.reshape(self.chunk_len, self.d_action)

    @staticmethod
    def ensemble_from_buffer(buf: Deque[np.ndarray], chunk_len: int) -> np.ndarray:
        """
        Temporal ensemble from chunk buffer.
        If newest chunk is at position 0 in the deque, then for offset k
        we use the k-th action from the k-th most recent chunk.
        """
        votes = []
        for k, ch in enumerate(buf):
            if k >= chunk_len:
                break
            votes.append(ch[k])
        return np.stack(votes, axis=0).mean(axis=0)

    def _clip_action(self, a: np.ndarray) -> np.ndarray:
        a = np.clip(a, -self.action_clip, self.action_clip)
        return a

    # ------------------------------------------------------------------
    # public runtime API
    # ------------------------------------------------------------------

    def reset(self, obs0: Optional[np.ndarray] = None, full_state0: Optional[np.ndarray] = None) -> None:
        self._clear_runtime()
        if obs0 is None or full_state0 is None:
            return

        self._check_trained()

        s0 = np.asarray(full_state0)
        if s0.ndim != 1 or s0.shape[0] != self.d_state:
            raise ValueError(f"Expected state shape ({self.d_state},), got {s0.shape}")

        x0 = self.compute_x(obs0, s0)
        z0 = self.compute_z(x0, self.m)
        self._x_curr, self._z_curr = x0, z0

    def predict_action(self) -> np.ndarray:
        """
        Predict action at cached state.
        Returns the temporal ensemble over chunk predictions in the buffer.
        """
        self._check_trained()
        if self._z_curr is None:
            raise RuntimeError("Call reset(...) or step(...) first.")

        chunk_now = self.chunk_from_z(self._z_curr)
        self._chunk_buffer.appendleft(chunk_now)

        a = self.ensemble_from_buffer(self._chunk_buffer, self.chunk_len)
        a = self._clip_action(a)

        self._last_action = a.copy()
        return a.astype(np.float64)

    def step(self, obs: np.ndarray, full_state: np.ndarray) -> None:
        """Advance by one real environment transition."""
        self._check_trained()

        if self._x_curr is None or self._z_curr is None:
            raise RuntimeError("Call reset(obs0, state0) before step().")
        if self._last_action is None:
            raise RuntimeError("Call predict_action() before step().")

        s_next = np.asarray(full_state)
        if s_next.ndim != 1 or s_next.shape[0] != self.d_state:
            raise ValueError(f"Expected state shape ({self.d_state},), got {s_next.shape}")

        # Update memory using the actually executed action
        u = self.compute_u(self._x_curr, self._last_action)
        self.m = self.esn_update(self.m, u)

        # Cache next state
        x_next = self.compute_x(obs, s_next)
        z_next = self.compute_z(x_next, self.m)
        self._x_curr, self._z_curr = x_next, z_next
        self._last_action = None

    def get_latent_state(self) -> Tuple[np.ndarray, np.ndarray, Deque[np.ndarray]]:
        if self._x_curr is None:
            raise RuntimeError("Call reset(...) or step(...) first.")
        return (
            self._x_curr.copy(),
            self.m.copy(),
            deque([x.copy() for x in self._chunk_buffer], maxlen=self.chunk_len),
        )

    # ------------------------------------------------------------------
    # training pair construction
    # ------------------------------------------------------------------

    def compute_x_trajectory(self, obs: np.ndarray, states: np.ndarray) -> np.ndarray:
        """
        Compute x_t for a whole trajectory with batched image encoding.
        """
        self._check_dims_inited()
        self._check_stats_ready()

        obs = np.ascontiguousarray(obs)
        states = np.asarray(states)

        if obs.shape[0] != states.shape[0]:
            raise ValueError("Trajectory obs/states length mismatch.")

        return self.compute_x_batch(obs, states)

    def _build_training_pairs(self, trajectories: List["MWTrajectory"]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build:
          X: (N, m_ridge) latent z_t
          Y: (N, chunk_dim) flattened target chunk at each t
        """
        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []

        for tr in trajectories:
            obs = tr.observations
            states = tr.full_states.astype(np.float64)
            actions = tr.actions.astype(np.float64)

            T = actions.shape[0]
            if T == 0:
                continue
            if obs.shape[0] != T or states.shape[0] != T:
                raise ValueError("Trajectory arrays length mismatch.")

            x_seq = self.compute_x_trajectory(obs, states)
            m = np.zeros((self.m_mem,), dtype=np.float64)

            # t = 0
            x_t = x_seq[0]
            z_t = self.compute_z(x_t, m)
            X_list.append(z_t)

            y0 = np.zeros((self.chunk_len, self.d_action), dtype=np.float64)
            for k in range(self.chunk_len):
                idx = min(k, T - 1)
                y0[k] = actions[idx]
            Y_list.append(y0.reshape(-1))

            # t >= 1
            for t in range(1, T):
                u_prev = self.compute_u(x_t, actions[t - 1])
                m = self.esn_update(m, u_prev)

                x_t = x_seq[t]
                z_t = self.compute_z(x_t, m)
                X_list.append(z_t)

                yt = np.zeros((self.chunk_len, self.d_action), dtype=np.float64)
                for k in range(self.chunk_len):
                    idx = min(t + k, T - 1)
                    yt[k] = actions[idx]
                Y_list.append(yt.reshape(-1))

        if len(X_list) == 0:
            raise ValueError("No training pairs could be built.")

        X = np.stack(X_list, axis=0)
        Y = np.stack(Y_list, axis=0)
        return X, Y

    # ------------------------------------------------------------------
    # training API
    # ------------------------------------------------------------------

    def fit(self, trajectories: List["MWTrajectory"]) -> None:
        """
        Fit the BC model on successful trajectories.
        """
        if len(trajectories) == 0:
            raise ValueError("Empty trajectory list.")

        tr0 = trajectories[0]
        d_state = int(tr0.full_states.shape[1])
        d_action = int(tr0.actions.shape[1])
        d_enc = int(self.encode_image(tr0.observations[0]).size)

        self._init_random_params(d_state=d_state, d_action=d_action, d_enc=d_enc)

        # State normalization stats
        states_cat = np.concatenate([tr.full_states.astype(np.float64) for tr in trajectories], axis=0)
        self.state_mean = states_cat.mean(axis=0)
        self.state_std = states_cat.std(axis=0)

        # Training pairs
        X, Y = self._build_training_pairs(trajectories)

        # Ridge fit
        self._policy_ridge = OnlineRidge(
            dim_in=self.m_ridge,
            dim_out=self.chunk_dim,
            alpha=self.ridge_alpha,
            device=None,
        )
        self._policy_ridge.update(X, Y)
        self.W_out, self.b_out = self._policy_ridge.solve()

        self._clear_runtime()
        print("BC training complete.")

    def update(self, trajectories: List["MWTrajectory"]) -> None:
        """
        Incremental update on additional successful trajectories.
        """
        self._check_trained()

        if len(trajectories) == 0:
            return

        X, Y = self._build_training_pairs(trajectories)

        self._policy_ridge.update(X, Y)
        self.W_out, self.b_out = self._policy_ridge.solve()