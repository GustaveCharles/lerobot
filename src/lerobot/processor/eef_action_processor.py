"""EEF delta action processor for A5 (state-free end-effector policy).

Replaces joint-space relative actions with chunk-wise EEF delta actions:
  action = [eef_pos_delta (3), eef_rot_6d_delta (6), gripper (1)]  dim=10

The input processor loads precomputed absolute EEF poses from a numpy sidecar
(eef_poses.npy, shape (N_total, 12): [pos(3), rotmat_flat(9)]) indexed by the
global dataset row index (batch["index"]).  It then computes chunk-wise deltas,
applies MEAN_STD normalization for pos/rot and MIN_MAX for gripper, and replaces
batch["action"] with the 10-D EEF action tensor.

The output processor inverts the normalization (for inference).  IK is NOT
performed — the output is un-normalized EEF deltas, not joint angles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.configs.types import FeatureType
from lerobot.processor.pipeline import ProcessorStep
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION


# ── Rotation helpers ──────────────────────────────────────────────────────────

def rotmat_to_6d(R: torch.Tensor) -> torch.Tensor:
    """Extract 6D representation from rotation matrix.

    Args:
        R: (..., 3, 3) rotation matrix.
    Returns:
        (..., 6) = [col0 (3), col1 (3)].
    """
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def sixd_to_rotmat(v: torch.Tensor) -> torch.Tensor:
    """Reconstruct rotation matrix from 6D via Gram-Schmidt.

    Args:
        v: (..., 6) = [col0_approx (3), col1_approx (3)].
    Returns:
        (..., 3, 3) rotation matrix.
    """
    a1 = v[..., :3]
    a2 = v[..., 3:6]
    eps = 1e-8
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + eps)
    b2 = a2 - (a2 * b1).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + eps)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack([b1, b2, b3], dim=-1)  # columns → (..., 3, 3)


# ── Input processor (pre-normalization) ──────────────────────────────────────

class EEFActionProcessorStep(ProcessorStep):
    """Replace joint actions with normalized chunk-wise EEF delta actions.

    Must run BEFORE NormalizerProcessorStep (NormalizerProcessorStep should use
    NormalizationMode.IDENTITY for ACTION when this step is active).

    Indexing:
        batch["index"][b] is the global dataset row index of observation frame b.
        The action chunk for sample b covers rows index[b] .. index[b]+horizon-1.
        This aligns with how LeRobot builds batches (action_delta_indices=[0..31]).
    """

    def __init__(self, eef_poses_path: str, eef_stats_path: str, horizon: int = 32):
        self.horizon = horizon

        # Load precomputed poses once at init (small: ~16 MB for 460K rows)
        poses_np = np.load(eef_poses_path).astype(np.float32)
        self._eef_poses = torch.from_numpy(poses_np)  # (N, 12): [pos(3), rotmat(9)]

        with open(eef_stats_path) as f:
            stats = json.load(f)

        self._pos_mean = torch.tensor(stats["eef_pos_delta"]["mean"], dtype=torch.float32)
        self._pos_std  = torch.tensor(stats["eef_pos_delta"]["std"],  dtype=torch.float32)
        self._rot_mean = torch.tensor(stats["eef_rot_delta"]["mean"], dtype=torch.float32)
        self._rot_std  = torch.tensor(stats["eef_rot_delta"]["std"],  dtype=torch.float32)
        g = stats["gripper"]
        self._g_min = float(g["min"])
        self._g_max = float(g["max"])

    def _to_device(self, device: torch.device, dtype: torch.dtype) -> None:
        self._eef_poses = self._eef_poses.to(device=device, dtype=dtype)
        self._pos_mean  = self._pos_mean.to(device=device, dtype=dtype)
        self._pos_std   = self._pos_std.to(device=device, dtype=dtype)
        self._rot_mean  = self._rot_mean.to(device=device, dtype=dtype)
        self._rot_std   = self._rot_std.to(device=device, dtype=dtype)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        global_indices = comp.get("index")  # (B,) int tensor

        action = transition.get(TransitionKey.ACTION)
        if global_indices is None or action is None:
            return transition

        device = action.device
        dtype  = action.dtype
        self._to_device(device, dtype)

        B = global_indices.shape[0]
        T = self.horizon

        # Gather all chunk frames: (B, T) global indices
        offsets = torch.arange(T, device=device, dtype=global_indices.dtype)
        chunk_idx = global_indices.unsqueeze(1) + offsets.unsqueeze(0)  # (B, T)
        chunk_idx = chunk_idx.clamp(0, self._eef_poses.shape[0] - 1)

        # Look up poses: (B, T, 12)
        poses = self._eef_poses[chunk_idx]

        pos     = poses[..., :3]                           # (B, T, 3)
        rotmats = poses[..., 3:].reshape(B, T, 3, 3)      # (B, T, 3, 3)

        # Chunk-wise position delta
        pos_ref  = pos[:, 0:1]                            # (B, 1, 3)
        delta_pos = pos - pos_ref                         # (B, T, 3)

        # Chunk-wise rotation delta: R_delta[k] = R_ref.T @ R[k]
        R_ref   = rotmats[:, 0]                           # (B, 3, 3)
        R_ref_T = R_ref.transpose(-2, -1).unsqueeze(1)   # (B, 1, 3, 3)
        R_delta = R_ref_T @ rotmats                       # (B, T, 3, 3)
        delta_rot = rotmat_to_6d(R_delta)                 # (B, T, 6)

        # Absolute gripper (joint index 5 from original action)
        gripper = action[..., 5:6]                        # (B, T, 1)

        # Normalize
        delta_pos_norm = (delta_pos - self._pos_mean) / (self._pos_std + 1e-8)
        delta_rot_norm = (delta_rot - self._rot_mean) / (self._rot_std + 1e-8)
        g_denom = self._g_max - self._g_min
        gripper_norm = 2.0 * (gripper - self._g_min) / (g_denom + 1e-8) - 1.0

        eef_action = torch.cat([delta_pos_norm, delta_rot_norm, gripper_norm], dim=-1)  # (B, T, 10)

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = eef_action
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # Replace ACTION feature with 10-D EEF action
        if PipelineFeatureType.ACTION in features:
            features[PipelineFeatureType.ACTION] = {
                ACTION: PolicyFeature(shape=(self.horizon, 10), type=FeatureType.ACTION)
            }
        return features

    def get_config(self) -> dict[str, Any]:
        return {"type": "EEFActionProcessorStep", "horizon": self.horizon}


# ── Output processor (post un-normalization) ─────────────────────────────────

class EEFUnnormalizeProcessorStep(ProcessorStep):
    """Un-normalize 10-D EEF delta action.

    Input:  normalized [pos_delta(3), rot_6d_delta(6), gripper(1)]
    Output: un-normalized same layout.

    NOTE: does NOT convert to joint angles — IK is a separate step needed
    for robot execution.
    """

    def __init__(self, eef_stats_path: str):
        with open(eef_stats_path) as f:
            stats = json.load(f)

        self._pos_mean = torch.tensor(stats["eef_pos_delta"]["mean"], dtype=torch.float32)
        self._pos_std  = torch.tensor(stats["eef_pos_delta"]["std"],  dtype=torch.float32)
        self._rot_mean = torch.tensor(stats["eef_rot_delta"]["mean"], dtype=torch.float32)
        self._rot_std  = torch.tensor(stats["eef_rot_delta"]["std"],  dtype=torch.float32)
        g = stats["gripper"]
        self._g_min = float(g["min"])
        self._g_max = float(g["max"])

    def _to_device(self, device: torch.device, dtype: torch.dtype) -> None:
        for attr in ("_pos_mean", "_pos_std", "_rot_mean", "_rot_std"):
            setattr(self, attr, getattr(self, attr).to(device=device, dtype=dtype))

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        self._to_device(action.device, action.dtype)

        pos_norm  = action[..., :3]
        rot_norm  = action[..., 3:9]
        grip_norm = action[..., 9:10]

        pos    = pos_norm * (self._pos_std + 1e-8) + self._pos_mean
        rot_6d = rot_norm * (self._rot_std + 1e-8) + self._rot_mean
        g_denom = self._g_max - self._g_min
        gripper = (grip_norm + 1.0) / 2.0 * (g_denom + 1e-8) + self._g_min

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = torch.cat([pos, rot_6d, gripper], dim=-1)
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features  # output shape matches input shape

    def get_config(self) -> dict[str, Any]:
        return {"type": "EEFUnnormalizeProcessorStep"}
