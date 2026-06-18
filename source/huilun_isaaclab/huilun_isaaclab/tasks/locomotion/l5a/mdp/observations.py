# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def velocity_height_commands(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float,
) -> torch.Tensor:
    """Return velocity command plus the fixed base-height target used by balance rewards."""
    velocity_command = env.command_manager.get_command(command_name)
    height_command = torch.full_like(velocity_command[:, :1], target_height)
    return torch.cat((velocity_command, height_command), dim=1)
