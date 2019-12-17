"""
Classes for running policies in environment.
"""
import numpy as np
from typing import Callable, List

import torch
from gym import Env
from torch.distributions import Categorical
from tqdm import tqdm

from ccil.utils.data import Trajectory
from ccil.utils.models import PolicyModel
from ccil.utils.state_encoder import StateEncoder
from ccil.utils.utils import random_mask_from_state


class PolicyRunner:
    """
    Execute policy in environment.
    """

    def __init__(
        self,
        env: Env,
        agent: Callable[[np.ndarray], np.ndarray],
        state_encoder: StateEncoder,
    ):
        self.env = env
        self.agent = agent
        self.state_encoder = state_encoder

    def run_episode(self) -> Trajectory:
        """
        Create trajectory until end of episode.
        :return:
        """
        state, done = self.env.reset(), False
        trajectory = None
        while not done:
            x = self.state_encoder.step(state, trajectory)
            action = self.agent(x).item()

            prev_action, prev_state = action, state
            state, rew, done, info = self.env.step(action)

            trajectory = Trajectory.add_step(
                trajectory, prev_state, prev_action, rew, None, info=info
            )
        trajectory.finished()
        return trajectory

    def run_num_steps(self, num_steps: int, verbose=False) -> List[Trajectory]:
        """
        Create list of trajectories with in total num_steps steps.
        :param num_steps:
        :param verbose: whether to show progress bar
        :return:
        """
        progress_bar = tqdm(total=num_steps, disable=not verbose)
        steps = 0
        trajectories = []
        while True:
            trajectory = self.run_episode()
            steps += len(trajectory)
            progress_bar.update(len(trajectory))
            trajectories.append(trajectory)
            if steps >= num_steps:
                break

        progress_bar.close()
        return trajectories

    def run_num_episodes(self, num_episodes: int, verbose=False) -> List[Trajectory]:
        """
        Create multiple trajectories.
        :param num_episodes:
        :param verbose:
        :return:
        """
        trajectories = []
        for _ in tqdm(range(num_episodes), disable=not verbose):
            trajectory = self.run_episode()
            trajectories.append(trajectory)
        return trajectories


def run_fixed_mask(
    env: Env,
    policy_model: PolicyModel,
    state_encoder: StateEncoder,
    mask: np.ndarray,
    num_episodes,
) -> List[Trajectory]:
    """
    Run policy model with fixed mask.
    :param env:
    :param policy_model:
    :param state_encoder:
    :param mask: int array.
    :param num_episodes:
    :return:
    """
    agent = FixedMaskPolicyAgent(policy_model, mask)
    runner = PolicyRunner(env, agent, state_encoder)
    trajectories = runner.run_num_episodes(num_episodes)
    return trajectories


def hard_discrete_action(output):
    return output.argmax(-1)


def sample_discrete_action(output):
    return Categorical(logits=output).sample()


class RandomMaskPolicyAgent:
    """
    Turn PolicyModel into callable agent. Sample random policy at each step.
    """

    def __init__(self, policy: PolicyModel, output_transformation=hard_discrete_action):
        self.policy = policy
        self.device = next(policy.parameters()).device
        self.output_transformation = output_transformation

    def __call__(self, state: np.ndarray) -> np.ndarray:
        x = torch.tensor(state, device=self.device, dtype=torch.float)[None]
        mask = random_mask_from_state(x)
        output = self.policy.forward(x, mask)
        action = self.output_transformation(output)
        return action


class FixedMaskPolicyAgent:
    """
    Turn PolicyModel into callable agent. Use one fixed mask.
    """

    def __init__(
        self,
        policy: PolicyModel,
        mask: np.ndarray,
        output_transformation=hard_discrete_action,
    ):
        self.policy = policy
        self.device = next(policy.parameters()).device
        self.mask = torch.tensor(mask, device=self.device, dtype=torch.float)
        self.output_transformation = output_transformation

    def __call__(self, state: np.ndarray) -> np.ndarray:
        x = torch.tensor(state, device=self.device, dtype=torch.float)[None]
        output = self.policy.forward(x, self.mask)
        action = self.output_transformation(output)
        return action
