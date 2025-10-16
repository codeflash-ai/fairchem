"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

_CG_CHANGE_MAT_CACHE = {}


def cg_change_mat(ang_mom: int, device: str = "cpu") -> torch.tensor:
    if ang_mom not in [2]:
        raise NotImplementedError

    if ang_mom == 2:
        cache_key = device
        # Cache per device
        if cache_key not in _CG_CHANGE_MAT_CACHE:
            mat = torch.tensor(
                [
                    [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
                    [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
                    [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
                    [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
                    [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
                    [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
                    [
                        -(6 ** (-0.5)),
                        0,
                        0,
                        0,
                        2 * 6 ** (-0.5),
                        0,
                        0,
                        0,
                        -(6 ** (-0.5)),
                    ],
                    [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
                    [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
                ],
                device=device,
            ).detach()
            _CG_CHANGE_MAT_CACHE[cache_key] = mat
        else:
            mat = _CG_CHANGE_MAT_CACHE[cache_key]
        change_mat = mat
    return change_mat


def irreps_sum(ang_mom: int) -> int:
    """
    Returns the sum of the dimensions of the irreps up to the specified angular momentum.

    :param ang_mom: max angular momenttum to sum up dimensions of irreps
    """
    # sum_{i=0}^{ang_mom} (2i+1) = (ang_mom+1)^2
    return (ang_mom + 1) ** 2
