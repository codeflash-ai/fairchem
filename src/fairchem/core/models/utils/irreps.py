"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

# Cache for cg_change_mat tensors to avoid recomputation and device transfer
_CG_MAT_CACHE: dict[str, dict[int, torch.Tensor]] = {}


def cg_change_mat(ang_mom: int, device: str = "cpu") -> torch.tensor:
    if ang_mom not in [2]:
        raise NotImplementedError

    global _CG_MAT_CACHE
    if device not in _CG_MAT_CACHE:
        _CG_MAT_CACHE[device] = {}

    if ang_mom in _CG_MAT_CACHE[device]:
        return _CG_MAT_CACHE[device][ang_mom]

    if ang_mom == 2:
        # Precompute all the constants to avoid repeated computation
        sqrt3_inv = 3 ** (-0.5)
        sqrt2_inv = 2 ** (-0.5)
        sqrt_05 = 0.5**0.5
        sqrt6_inv = 6 ** (-0.5)
        two_sqrt6_inv = 2 * (6 ** (-0.5))

        change_mat = torch.tensor(
            [
                [sqrt3_inv, 0, 0, 0, sqrt3_inv, 0, 0, 0, sqrt3_inv],
                [0, 0, 0, 0, 0, sqrt2_inv, 0, -sqrt2_inv, 0],
                [0, 0, -sqrt2_inv, 0, 0, 0, sqrt2_inv, 0, 0],
                [0, sqrt2_inv, 0, -sqrt2_inv, 0, 0, 0, 0, 0],
                [0, 0, sqrt_05, 0, 0, 0, sqrt_05, 0, 0],
                [0, sqrt2_inv, 0, sqrt2_inv, 0, 0, 0, 0, 0],
                [
                    -sqrt6_inv,
                    0,
                    0,
                    0,
                    two_sqrt6_inv,
                    0,
                    0,
                    0,
                    -sqrt6_inv,
                ],
                [0, 0, 0, 0, 0, sqrt2_inv, 0, sqrt2_inv, 0],
                [-sqrt2_inv, 0, 0, 0, 0, 0, 0, 0, sqrt2_inv],
            ],
            device=device,
        ).detach()
        _CG_MAT_CACHE[device][ang_mom] = change_mat
        return change_mat

    # This code path should never be reached, but included for completeness
    raise NotImplementedError


def irreps_sum(ang_mom: int) -> int:
    """
    Returns the sum of the dimensions of the irreps up to the specified angular momentum.

    :param ang_mom: max angular momenttum to sum up dimensions of irreps
    """
    # Use closed-form formula: sum_{i=0}^n (2i+1) = (n+1)**2
    return (ang_mom + 1) ** 2
