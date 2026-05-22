#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path

import numpy as np

# gripper_frame_link index for so101_new_calib.urdf (matches precompute_eef_sidecars.py)
DEFAULT_SO101_EEF_FRAME_ID = 15


class PinocchioKinematics:
    """Pinocchio FK/IK for SO-101 deploy (no placo — avoids broken dylib on macOS)."""

    def __init__(
        self,
        urdf_path: str | Path,
        frame_id: int = DEFAULT_SO101_EEF_FRAME_ID,
        n_arm: int = 5,
    ):
        import pinocchio as pin

        self._pin = pin
        self.frame_id = frame_id
        self.n_arm = n_arm
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()

    def fk(self, joint_angles_deg: np.ndarray) -> np.ndarray:
        """End-effector pose (4x4) from joint angles in degrees (first n_arm arm joints)."""
        pin = self._pin
        q = np.zeros(self.model.nq)
        q[: self.n_arm] = np.deg2rad(joint_angles_deg[: self.n_arm])
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.frame_id]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = oMf.rotation
        T[:3, 3] = oMf.translation
        return T

    def inverse_kinematics(
        self,
        joint_angles_deg: np.ndarray,
        target_pose: np.ndarray,
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
        max_iter: int = 200,
        tol: float = 1e-4,
        dt: float = 0.2,
        damp: float = 1e-4,
    ) -> np.ndarray:
        """Damped least-squares IK; returns joint angles in degrees (same layout as input)."""
        pin = self._pin
        q = np.zeros(self.model.nq)
        q[: self.n_arm] = np.deg2rad(joint_angles_deg[: self.n_arm])
        oMdes = pin.SE3(target_pose[:3, :3].astype(np.float64), target_pose[:3, 3].astype(np.float64))
        w = np.diag([position_weight] * 3 + [orientation_weight] * 3).astype(np.float64)

        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            oMf = self.data.oMf[self.frame_id]
            iMd = oMf.actInv(oMdes)
            err = pin.log(iMd).vector
            err_w = w @ err
            if np.linalg.norm(err_w) < tol:
                break
            J = pin.computeFrameJacobian(self.model, self.data, q, self.frame_id, pin.LOCAL)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            J_w = w @ J
            v = -J_w.T @ np.linalg.solve(J_w @ J_w.T + damp * np.eye(6), err_w)
            q = pin.integrate(self.model, q, v * dt)

        out = joint_angles_deg.copy().astype(np.float64)
        out[: self.n_arm] = np.rad2deg(q[: self.n_arm])
        return out
