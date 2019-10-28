from os import environ
from pathlib import Path

import numpy as np

import torch

data_root_path = Path(environ.get('DATA_PATH', './data'))


def random_mask_from_state(x):
    return torch.randint(0, 2, size=x.shape, device=x.device)


def mask_idx_to_mask(n, i):
    i = np.asarray(i)
    assert np.all(i < 2**n)
    r = 2 ** np.arange(n - 1, -1, -1)
    x = (i[..., None] % (2 * r)) // r
    return x


def mask_to_mask_idx(mask):
    mask = np.asarray(mask)
    n = mask.shape[-1]
    return (mask * 2**np.arange(n - 1, -1, -1)).sum(-1)


def test_mask_idx_to_mask():
    assert mask_idx_to_mask(3, 0).tolist() == [0, 0, 0]
    assert mask_idx_to_mask(3, 1).tolist() == [0, 0, 1]
    assert mask_idx_to_mask(3, 2).tolist() == [0, 1, 0]
    assert mask_idx_to_mask(3, 7).tolist() == [1, 1, 1]
    assert mask_idx_to_mask(3, [1, 2]).tolist() == [[0, 0, 1], [0, 1, 0]]


def test_mask_to_mask_idx():
    assert mask_to_mask_idx([0, 0, 0]).tolist() == 0
    assert mask_to_mask_idx([[1, 0, 0], [1, 1, 1]]).tolist() == [4, 7]



