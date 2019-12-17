"""
Main behaviour cloning training loop.
"""
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
from ccil.utils.state_encoder import StateEncoder
from ccil.utils.data import random_split, batch_cat, DataLoaderRepeater, Trajectory, Batch
from ccil.utils.graph_distribution import UniformDistribution, CombinatorialGumbelDistribution, sparse_prior_logits, \
    GraphDistribution
from ccil.utils.models import SimplePolicy, MLP, MaskPolicy, PolicyModel
from ccil.utils.policy_runner import PolicyRunner, RandomMaskPolicyAgent, FixedMaskPolicyAgent
from ccil.utils.utils import data_root_path, mask_idx_to_mask


def train_step(
        engine: Engine, batch: Batch, state_encoder: StateEncoder, policy_model: PolicyModel, optimizer, criterion,
        device, graph_distribution: GraphDistribution):
    """
    PyTorch ignite training step.
    :param engine:
    :param batch:
    :param state_encoder:
    :param policy_model:
    :param optimizer:
    :param criterion:
    :param device:
    :param graph_distribution:
    :return:
    """
    x, y = state_encoder.batch(batch), batch.labels()
    x, y = x.to(device), y.to(device)

    mask, sample_data = graph_distribution.rsample(len(x), x.device)
    output = policy_model.forward(x, mask)
    loss = criterion(output, y)
    loss = graph_distribution.regularize_loss(loss, x, mask, output, sample_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def inference_step(
        engine: Engine, batch: Batch, state_encoder: StateEncoder, policy_model: PolicyModel, device,
        graph_distribution: GraphDistribution):
    """
    PyTorch ignite inference step.
    :param engine:
    :param batch:
    :param state_encoder:
    :param policy_model:
    :param device:
    :param graph_distribution:
    :return:
    """
    x, y = state_encoder.batch(batch), batch.labels()
    x, y = x.to(device), y.to(device)
    mask, _ = graph_distribution.rsample(len(x), x.device)
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


def run_graphs(policy_model, state_encoder):
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
    """
    Learn imitator.
    Uses PyTorch Ignite.

    :param args: See Parser below.
    """
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
        graph_distribution = UniformDistribution(3)
        max_epochs = 10
    elif args.network == 'uniform':
        policy_model = MaskPolicy(MLP([6, 50, 50, 50, 3])).to(device)
        graph_distribution = UniformDistribution(3)
        max_epochs = 20
    elif args.network == 'combinatorial':
        prior_logits = sparse_prior_logits(3, 0.6)
        graph_distribution = CombinatorialGumbelDistribution(0.5, 3, beta=0.05, prior_logits=prior_logits).to(device)
        policy_model = MaskPolicy(MLP([6, 50, 50, 50, 3])).to(device)
        max_epochs = 20
    else:
        raise ValueError()

    optimizer = torch.optim.Adam(list(policy_model.parameters()) + list(graph_distribution.parameters()))

    def criterion(x, y):
        return F.cross_entropy(x, y[:, 0])

    metrics = {
        'loss': Loss(F.cross_entropy, output_transform=lambda x: (x[0], x[1][:, 0])),
        'acc': Accuracy(output_transform=lambda x: (x[0], x[1][:, 0])),
    }

    state_encoder = MountainCarStateEncoder(args.input_mode == 'original')

    trainer = Engine(partial(
        train_step, state_encoder=state_encoder, policy_model=policy_model,
        optimizer=optimizer, criterion=criterion, device=device,
        graph_distribution=graph_distribution,
    ))
    evaluators = {
        name: Engine(partial(
            inference_step,
            state_encoder=state_encoder,
            policy_model=policy_model,
            device=device,
            graph_distribution=graph_distribution))
        for name in ['train', 'test']}
    for evaluator_name, evaluator in evaluators.items():
        for name, metric in metrics.items():
            metric.attach(evaluator, name)
        evaluator.add_event_handler(Events.COMPLETED, print_metrics, evaluator_name=evaluator_name, trainer=trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_eval(_trainer):
        for name, evaluator in evaluators.items():
            evaluator.run(dataloaders[name])
        if not isinstance(graph_distribution, UniformDistribution):
            print(graph_distribution)

    trainer.run(dataloaders['train_repeated'], max_epochs=max_epochs)
    print("Trained")

    # Run policies in environment
    run_fn = run_simple if args.network == 'simple' else run_graphs
    run_fn(policy_model, state_encoder)

    if args.save:
        name = args.name or f"{args.input_mode}_{args.network}_{datetime.now():%Y%m%d-%H%M%S}"
        save_dir = data_root_path / 'policies'
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{name}.pkl"
        probs_path = save_dir / f"{name}-probs.pkl"
        torch.save(policy_model, path)
        torch.save(graph_distribution.probs.cpu().detach(), probs_path)
        print(f"Saved to dir: {save_dir}")
        print(f"Under name:   {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_mode', choices=['original', 'confounded'],
        help="Whether to add random action [original] or previous [confounded].")
    parser.add_argument(
        'network', choices=['simple', 'uniform', 'combinatorial'],
        help=("What kind of graph distribution to use. Simple: mask nothing, Uniform: sample uniformly. "
              "Combinatorial: learn categorical over all 2**N graphs.")
    )
    parser.add_argument('--data_seed', type=int, help="Seed for splitting train/test data. Default=random")
    parser.add_argument('--num_samples', type=int, default=300, help="Num samples used for training.")
    parser.add_argument('--save', action='store_true', help="Whether to save learned policy.")
    parser.add_argument('--name', help="Policy save filename")
    imitate(parser.parse_args())


if __name__ == '__main__':
    main()
