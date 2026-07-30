# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""L5A-specific action terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RandomizedDefaultJointPositionAction(JointPositionAction):
    """Position action with a persistent per-environment joint-zero error.

    The same sampled offset is used by the position target, relative joint
    observations, and reset defaults. This preserves the old L5A semantics
    without letting these three references silently diverge.
    """

    cfg: RandomizedDefaultJointPositionActionCfg

    def __init__(self, cfg: RandomizedDefaultJointPositionActionCfg, env: ManagerBasedEnv):
        if not cfg.use_default_offset:
            raise ValueError(
                "RandomizedDefaultJointPositionAction requires use_default_offset=True "
                "so action targets, observations, and reset defaults share one reference."
            )
        super().__init__(cfg, env)
        lower, upper = cfg.default_offset_range
        if lower > upper:
            raise ValueError(f"Invalid default_offset_range: {cfg.default_offset_range}")

        zero_error = torch.empty_like(self._offset).uniform_(lower, upper)
        self._offset.add_(zero_error)
        self._asset.data.default_joint_pos[:, self._joint_ids] = self._offset
        self.default_offset_error = zero_error


@configclass
class RandomizedDefaultJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for persistent L5A joint-zero randomization."""

    class_type: type[ActionTerm] = RandomizedDefaultJointPositionAction
    default_offset_range: tuple[float, float] = (-0.05, 0.05)
