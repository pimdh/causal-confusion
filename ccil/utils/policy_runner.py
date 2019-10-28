import torch
from torch.distributions import Categorical
from tqdm import tqdm

from ccil.utils.data import Trajectory
from ccil.utils.utils import random_mask_from_state


class PolicyRunner:
    def __init__(self, env, agent, state_encoder):
        self.env = env
        self.agent = agent
        self.state_encoder = state_encoder

    def run_episode(self):
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

    def run_num_steps(self, num_steps, verbose=False):
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

    def run_num_episodes(self, num_episodes, verbose=False):
        trajectories = []
        for _ in tqdm(range(num_episodes), disable=not verbose):
            trajectory = self.run_episode()
            trajectories.append(trajectory)
        return trajectories


def run_fixed_mask(env, policy_model, state_encoder, mask, num_episodes):
    agent = FixedMaskPolicyAgent(policy_model, mask)
    runner = PolicyRunner(env, agent, state_encoder)
    trajectories = runner.run_num_episodes(num_episodes)
    return trajectories


def hard_discrete_action(output):
    return output.argmax(-1)


def sample_discrete_action(output):
    return Categorical(logits=output).sample()


class RandomMaskPolicyAgent:
    def __init__(self, policy, output_transformation=hard_discrete_action):
        self.policy = policy
        self.device = next(policy.parameters()).device
        self.output_transformation = output_transformation

    def __call__(self, state):
        x = torch.tensor(state, device=self.device, dtype=torch.float)[None]
        mask = random_mask_from_state(x)
        output = self.policy.forward(x, mask)
        action = self.output_transformation(output)
        return action


class FixedMaskPolicyAgent:
    def __init__(self, policy, mask, output_transformation=hard_discrete_action):
        self.policy = policy
        self.device = next(policy.parameters()).device
        self.mask = torch.tensor(mask, device=self.device, dtype=torch.float)
        self.output_transformation = output_transformation

    def __call__(self, state):
        x = torch.tensor(state, device=self.device, dtype=torch.float)[None]
        output = self.policy.forward(x, self.mask)
        action = self.output_transformation(output)
        return action
