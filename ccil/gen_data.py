import argparse
from pathlib import Path

import gym
import torch

from ccil.environments.mountain_car import MountainExpertCarStateEncoder, MountainCarExpert
from ccil.utils.data import TransitionDataset, Trajectory
from ccil.utils.policy_runner import PolicyRunner
from ccil.utils.utils import data_root_path


def gen_data(args):
    if args.env == 'mountain_car':
        expert = MountainCarExpert()
        expert_state_encode = MountainExpertCarStateEncoder()
        env = gym.make("MountainCarRich-v0")
    else:
        raise ValueError()

    runner = PolicyRunner(env, expert, expert_state_encode)
    trajectories = runner.run_num_steps(args.num_steps, True)
    dataset = TransitionDataset.from_trajectories(trajectories, stack_size=2, expert_trajectories=True)

    if args.save_path is None:
        save_path = data_root_path / 'demonstrations' / f'{args.env}.pkl'
    else:
        save_path = Path(args.save_path)

    save_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(dataset, save_path)
    print(f'Mean reward: {Trajectory.reward_sum_mean(trajectories)}')
    print('Done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='mountain_car')
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--save_path')
    gen_data(parser.parse_args())


if __name__ == "__main__":
    main()
