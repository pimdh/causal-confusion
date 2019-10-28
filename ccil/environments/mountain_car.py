import torch
import numpy as np
from gym.envs.classic_control import MountainCarEnv
from gym.envs.registration import register
from gym import Wrapper

from ccil.utils.utils import data_root_path


class MCRichDenseEnv(Wrapper):
    """Richer initial conditions + dense rewards for easy training of expert."""
    def __init__(self):
        super().__init__(MountainCarEnv())

    def reset(self):
        self.env.state = np.array([
            self.np_random.uniform(low=-1, high=0.5),
            self.unwrapped.np_random.randn() * 0.07,
        ])
        return np.array(self.env.state)

    def step(self, action):
        state, reward, done, info = super().step(action)

        reward = self.env._height(self.env.state[0]) * 0.5 - 1

        return state, reward, done, info


class MCRichEnv(Wrapper):
    """Richer initial conditions."""

    def __init__(self):
        super().__init__(MountainCarEnv())

    def reset(self):
        self.env.state = np.array([
            self.np_random.uniform(low=-1, high=0.5),
            self.unwrapped.np_random.randn() * 0.07,
            ])
        return np.array(self.env.state)


register(
    id="MountainCarRichDense-v0",
    entry_point="ccil.environments.mountain_car:MCRichDenseEnv",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="MountainCarRich-v0",
    entry_point="ccil.environments.mountain_car:MCRichEnv",
    max_episode_steps=200,
    reward_threshold=-110.0,
)


class MountainCarStateEncoder:
    """
    Map batch from TransitionDataset or Trajectory into state vector.
    """
    def __init__(self, random):
        """
        :param random: Whether to use random action.
        """
        self.random = random

    def batch(self, batch):
        assert batch.states.shape[1] >= 2
        x = batch.states[:, -1, :]

        if self.random:
            prev_action = torch.randint(0, 3, (x.shape[0], 1), device=x.device, dtype=torch.float)
        else:
            prev_action = batch.actions[:, -2]

        return torch.cat([x.float(), prev_action.float()], 1)

    def step(self, state, trajectory):
        if trajectory and not self.random:
            prev_action = trajectory.actions[-1]
        else:
            prev_action = np.atleast_1d(np.random.randint(0, 3))
        x = np.concatenate([state, prev_action])
        return x


class MountainExpertCarStateEncoder:
    """
    Map batch from TransitionDataset or Trajectory into state vector.
    """
    def batch(self, batch):
        return batch.states[:, -1, :]

    def step(self, state, trajectory):
        return state


class MountainCarExpert:
    def __init__(self):
        expert_path = data_root_path / "experts/mountaincar_deepq_custom.pickle"
        from baselines import deepq
        self.expert = deepq.load_act(expert_path)

    def __call__(self, state):
        return self.expert(state)

