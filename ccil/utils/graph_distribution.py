"""
Model distributions over causal graphs as distributions over binary masks.
"""
from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, RelaxedOneHotCategorical

from ccil.utils.utils import onehot_to_mask_idx_tensor, print_array, all_masks_tensor


class GraphDistribution(nn.Module):
    """
    Abstract distribution over graphs.
    """

    def regularize_loss(self, loss, x, mask, output, sample_data):
        """
        Add to loss terms regarding mask learning.
        :param loss: behaviour cloning loss
        :param x: encoded state
        :param mask: mask tensor
        :param output: output of PolicyModel
        :param sample_data: 2nd output of rsample
        :return: new loss
        """
        raise NotImplementedError

    def rsample(self, n, device=None) -> Tuple[torch.Tensor, Any]:
        """
        Sample n graphs with reparametrization.
        :param n:
        :param device:
        :return:
        """
        raise NotImplementedError

    @property
    def probs(self):
        """
        Return probabilities of categories.
        :return:
        """
        raise NotImplementedError

    def forward(self, input):
        """
        Dummy method, unused.
        :param input:
        :return:
        """
        pass


class UniformDistribution(GraphDistribution):
    def __init__(self, num_vars):
        super().__init__()
        self.num_vars = num_vars

    def regularize_loss(self, loss, x, mask, output, sample_data):
        return loss

    def rsample(self, n, device=None):
        return torch.randint(0, 2, size=(n, self.num_vars), device=device), {}

    @property
    def probs(self):
        return torch.ones(2 ** self.num_vars) / 2 ** self.num_vars


def categorical_cross_entropy(distr_a: Categorical, distr_b: Categorical):
    return -torch.sum(distr_a.probs * F.log_softmax(distr_b.logits, 0))


def sparse_prior_logits(n, strength):
    masks = all_masks_tensor(n)
    num_zeros = n - masks.sum(1)
    logits = strength * num_zeros.float()
    return logits


class CombinatorialGumbelDistribution(GraphDistribution):
    def __init__(self, temperature, num_vars, beta=1.0, prior_logits=None):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))
        self.logits = nn.Parameter(torch.zeros(2 ** num_vars))
        self.register_buffer(
            "prior_logits",
            torch.zeros(2 ** num_vars) if prior_logits is None else prior_logits,
        )
        self.beta = beta

    @property
    def distr(self):
        return RelaxedOneHotCategorical(
            temperature=self.temperature, logits=self.logits
        )

    @property
    def hard_distr(self):
        return Categorical(logits=self.logits)

    @property
    def prior(self):
        return Categorical(logits=self.prior_logits)

    @property
    def probs(self):
        return self.hard_distr.probs

    def regularize_loss(self, loss, x, mask, output, sample_data):
        kl = (
            categorical_cross_entropy(self.hard_distr, self.prior)
            - self.hard_distr.entropy()
        )
        return loss + kl * self.beta

    def rsample(self, n, device=None):
        onehot = self.distr.rsample((n,))
        mask = onehot_to_mask_idx_tensor(onehot)

        return mask, dict(log_prob=self.distr.log_prob(onehot))

    def __str__(self):
        return f"<CombinatorialGumbelDistribution p={print_array(self.distr.probs.detach().cpu(), precision=3)}>"
