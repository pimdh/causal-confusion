from datetime import datetime
import argparse
from functools import partial

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader

from ccil.environments.mountain_car import MountainCarStateEncoder
from ccil.utils.data import random_split, batch_cat, DataLoaderRepeater, Trajectory
from ccil.utils.models import SimplePolicy, MLP, UniformMaskPolicy
from ccil.utils.policy_runner import PolicyRunner, RandomMaskPolicyAgent, FixedMaskPolicyAgent
from ccil.utils.utils import random_mask_from_state, data_root_path, mask_idx_to_mask


def train_step(engine, batch, state_encoder, policy_model, optimizer, criterion, device):
    x, y = state_encoder.batch(batch), batch.labels()
    x, y = x.to(device), y.to(device)

    mask = random_mask_from_state(x)
    output = policy_model.forward(x, mask)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def inference_step(engine, batch, state_encoder, policy_model, device):
    x, y = state_encoder.batch(batch), batch.labels()
    x, y = x.to(device), y.to(device)
    mask = random_mask_from_state(x)
    output = policy_model.forward(x, mask)
    return output, y


def print_metrics(engine, trainer, evaluator_name):
    print(
        f"Epoch: {trainer.state.epoch:> 3} {evaluator_name.title(): <5} "
        f"loss={engine.state.metrics['loss']:.4f} "
        f"acc={engine.state.metrics['acc']:.4f}")


def run_simple(policy_model, state_encoder):
    """
    Run the policy in environment.
    """
    env = gym.make("MountainCar-v0")
    agent = RandomMaskPolicyAgent(policy_model)
    runner = PolicyRunner(env, agent, state_encoder)
    trajectories = runner.run_num_episodes(20)
    print(f'Mean reward: {Trajectory.reward_sum_mean(trajectories)}')


def run_uniform(policy_model, state_encoder):
    """
    Run all 8 policies in environment.
    """
    env = gym.make("MountainCar-v0")
    for mask_idx in range(8):
        agent = FixedMaskPolicyAgent(policy_model, mask_idx_to_mask(3, mask_idx))
        runner = PolicyRunner(env, agent, state_encoder)
        trajectories = runner.run_num_episodes(20)
        mask = mask_idx_to_mask(3, mask_idx).tolist()
        print(f'Mean reward mask {mask}: {Trajectory.reward_sum_mean(trajectories)}')


def imitate(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = torch.load(data_root_path / 'demonstrations' / 'mountain_car.pkl')
    train_dataset, test_dataset = random_split(dataset, [args.num_samples, args.num_samples], args.data_seed)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=batch_cat),
        'test': DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=batch_cat),
    }
    # So that 1 train epoch has fixed number of samples (500 batches) regardless of dataset size
    dataloaders['train_repeated'] = DataLoaderRepeater(dataloaders['train'], 500)

    if args.network == 'simple':
        policy_model = SimplePolicy(MLP([3, 50, 50, 3])).to(device)
        max_epochs = 10
    elif args.network == 'uniform':
        policy_model = UniformMaskPolicy(MLP([6, 50, 50, 50, 3])).to(device)
        max_epochs = 20
    else:
        raise ValueError()

    optimizer = torch.optim.Adam(policy_model.parameters())

    def criterion(x, y):
        return F.cross_entropy(x, y[:, 0])

    metrics = {
        'loss': Loss(F.cross_entropy, output_transform=lambda x: (x[0], x[1][:, 0])),
        'acc': Accuracy(output_transform=lambda x: (x[0], x[1][:, 0])),
    }

    state_encoder = MountainCarStateEncoder(args.input_mode == 'original')

    trainer = Engine(partial(
        train_step, state_encoder=state_encoder, policy_model=policy_model,
        optimizer=optimizer, criterion=criterion, device=device
    ))
    evaluators = {
        name: Engine(partial(
            inference_step, state_encoder=state_encoder, policy_model=policy_model, device=device))
        for name in ['train', 'test']}
    for evaluator_name, evaluator in evaluators.items():
        for name, metric in metrics.items():
            metric.attach(evaluator, name)
        evaluator.add_event_handler(Events.COMPLETED, print_metrics, evaluator_name=evaluator_name, trainer=trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_eval(_trainer):
        for name, evaluator in evaluators.items():
            evaluator.run(dataloaders[name])

    trainer.run(dataloaders['train_repeated'], max_epochs=max_epochs)
    print("Trained")

    # Run policies in environment
    run_fn = dict(simple=run_simple, uniform=run_uniform)[args.network]
    run_fn(policy_model, state_encoder)

    if args.save:
        name = args.name or f"{args.input_mode}_{args.network}_{datetime.now():%Y%m%d-%H%M%S}"
        save_dir = data_root_path / 'policies'
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{name}.pkl"
        torch.save(policy_model, path)
        print(f"Policy saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_mode', choices=['original', 'confounded'])
    parser.add_argument('network', choices=['simple', 'uniform'])
    parser.add_argument('--data_seed', type=int, help="Seed for splitting train/test data. Default=random")
    parser.add_argument('--num_samples', type=int, default=300)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--name', help="Policy save filename")
    imitate(parser.parse_args())


if __name__ == '__main__':
    main()
