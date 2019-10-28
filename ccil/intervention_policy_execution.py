from pprint import pprint

import gym
import numpy as np
import argparse
from time import perf_counter

import torch
from sklearn.linear_model import Ridge
from torch.distributions import Bernoulli

from ccil.environments.mountain_car import MountainCarStateEncoder
from ccil.utils.data import Trajectory
from ccil.utils.policy_runner import PolicyRunner, FixedMaskPolicyAgent, run_fixed_mask
from ccil.utils.utils import data_root_path


def sample(weights, temperature):
    return Bernoulli(logits=torch.from_numpy(weights) / temperature).sample().long().numpy()


def linear_regression(masks, rewards, alpha=1.0):
    model = Ridge(alpha).fit(masks, rewards)
    return model.coef_, model.intercept_


class SoftQAlgo:
    def __init__(
            self,
            num_dims,
            reward_fn,
            its,
            temperature=1.0,
            device=None,
            evals_per_it=1,
    ):
        self.num_dims = num_dims
        self.reward_fn = reward_fn
        self.its = its
        self.device = device
        self.temperature = lambda t: temperature
        self.evals_per_it = evals_per_it

    def run(self):
        t = self.temperature(0)
        weights = np.zeros(self.num_dims)

        trace = []
        masks = []
        rewards = []
        for it in range(self.its):
            start = perf_counter()
            mask = sample(weights, t)
            reward = np.mean([self.reward_fn(mask) for _ in range(self.evals_per_it)])
            masks.append(mask)
            rewards.append(reward)

            weights, _ = linear_regression(masks, rewards, alpha=1.0)

            trace.append(
                {
                    "it": it,
                    "reward": reward,
                    "mask": mask,
                    "weights": weights,
                    "mode": (np.sign(weights).astype(np.int64) + 1) // 2,
                    "time": perf_counter() - start,
                    "past_mean_reward": np.mean(rewards),
                }
            )
            pprint(trace[-1])

        return trace


def intervention_policy_execution(args):
    policy_save_dir = data_root_path / 'policies'
    if args.policy_name:
        policy_path = policy_save_dir / f"{args.policy_name}.pkl"
    else:
        policy_paths = policy_save_dir.glob('confounded_uniform*.pkl')
        if not policy_paths:
            raise RuntimeError("No policy found")
        policy_path = next(iter(sorted(policy_paths, reverse=True)))
    policy_model = torch.load(policy_path)
    print(f"Loaded policy from {policy_path}")

    env = gym.make("MountainCar-v0")
    state_encoder = MountainCarStateEncoder(random=False)

    def run_step(mask):
        trajectories = run_fixed_mask(env, policy_model, state_encoder, mask, 1)
        return Trajectory.reward_sum_mean(trajectories)

    trace = SoftQAlgo(3, run_step, args.num_its, temperature=10).run()

    best_mask = trace[-1]['mode']
    trajectories = run_fixed_mask(env, policy_model, state_encoder, best_mask, 20)
    print(f"Final mask {best_mask.tolist()}")
    print(f"Final reward {Trajectory.reward_sum_mean(trajectories)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_name', help="Policy save filename")
    parser.add_argument('--num_its', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=10)
    intervention_policy_execution(parser.parse_args())


if __name__ == '__main__':
    main()
