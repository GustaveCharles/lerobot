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

import numpy as np


def constrain_gripper_approach(
    R: np.ndarray,
    *,
    forward_axis: np.ndarray,
    max_forward_dot: float,
    approach_axis: int = 2,
    down_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Keep gripper approach axis out of a cone around horizontal 'forward'."""
    z = np.asarray(R[:, approach_axis], dtype=np.float64)
    zn = np.linalg.norm(z)
    if zn < 1e-8:
        return R
    z = z / zn

    f = np.asarray(forward_axis, dtype=np.float64).copy()
    f[2] = 0.0
    fn = np.linalg.norm(f)
    if fn < 1e-8:
        return R
    f = f / fn

    if float(np.dot(z, f)) <= max_forward_dot:
        return R

    z_pf = z - np.dot(z, f) * f
    z_pf_n = np.linalg.norm(z_pf)
    sin_max = np.sqrt(max(0.0, 1.0 - max_forward_dot**2))
    if z_pf_n < 1e-8:
        down = np.array([0.0, 0.0, -1.0]) if down_axis is None else np.asarray(down_axis, dtype=np.float64)
        dn = np.linalg.norm(down)
        z_new = down / dn if dn > 1e-8 else np.array([0.0, 0.0, -1.0])
    else:
        z_new = max_forward_dot * f + sin_max * (z_pf / z_pf_n)

    if down_axis is not None:
        down = np.asarray(down_axis, dtype=np.float64)
        dn = np.linalg.norm(down)
        if dn > 1e-8 and float(np.dot(z_new, down / dn)) < 0.2:
            down = down / dn
            z_new = 0.75 * down + 0.25 * z_new
            z_new /= np.linalg.norm(z_new)

    hint_x = np.asarray(R[:, 0], dtype=np.float64)
    x = hint_x - np.dot(hint_x, z_new) * z_new
    if np.linalg.norm(x) < 1e-6:
        down_ref = np.array([0.0, 0.0, -1.0]) if down_axis is None else np.asarray(down_axis, dtype=np.float64)
        x = np.cross(z_new, down_ref)
        if np.linalg.norm(x) < 1e-6:
            x = np.cross(z_new, f)
    x = x / np.linalg.norm(x)
    y = np.cross(z_new, x)
    cols: list[np.ndarray | None] = [None, None, None]
    cols[approach_axis] = z_new
    other = [i for i in range(3) if i != approach_axis]
    cols[other[0]] = x
    cols[other[1]] = y
    return np.column_stack(cols)
