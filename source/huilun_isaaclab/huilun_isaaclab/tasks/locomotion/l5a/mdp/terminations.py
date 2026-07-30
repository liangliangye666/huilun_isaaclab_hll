# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""L5A task termination helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_out_of_limits(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Terminate environments whose raw policy action contains an extreme value."""
    # RslRlVecEnvWrapper clips to the same threshold before ActionManager sees
    # the action, so equality must count as a violation.
    return torch.max(torch.abs(env.action_manager.action), dim=1).values >= threshold
