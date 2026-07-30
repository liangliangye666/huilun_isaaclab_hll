# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Implicit PD actuator with per-environment randomized command delay."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions


class DelayedImplicitActuator(ImplicitActuator):
    """Delay position, velocity, and effort targets before implicit PhysX PD."""

    cfg: DelayedImplicitActuatorCfg

    def __init__(self, cfg: DelayedImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._position_delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._velocity_delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._effort_delay = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)

    def reset(self, env_ids: Sequence[int] | slice | None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            num_envs = self._num_envs
        elif isinstance(env_ids, slice):
            num_envs = len(range(*env_ids.indices(self._num_envs)))
        else:
            num_envs = len(env_ids)
        delays = torch.randint(
            self.cfg.min_delay,
            self.cfg.max_delay + 1,
            (num_envs,),
            dtype=torch.long,
            device=self._device,
        )
        for buffer in (self._position_delay, self._velocity_delay, self._effort_delay):
            buffer.set_time_lag(delays, env_ids)
            buffer.reset(env_ids)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        if control_action.joint_positions is not None:
            control_action.joint_positions = self._position_delay.compute(control_action.joint_positions)
        if control_action.joint_velocities is not None:
            control_action.joint_velocities = self._velocity_delay.compute(control_action.joint_velocities)
        if control_action.joint_efforts is not None:
            control_action.joint_efforts = self._effort_delay.compute(control_action.joint_efforts)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for :class:`DelayedImplicitActuator`."""

    class_type: type = DelayedImplicitActuator
    min_delay: int = 0
    max_delay: int = 0
