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


def mask_idx_to_mask_tensor(n, i):
    assert torch.all(i < 2**n)
    r = 2 ** torch.arange(n - 1, -1, -1, device=i.device)
    x = (i[..., None] % (2 * r)) // r
    return x


def mask_to_mask_idx(mask):
    mask = np.asarray(mask)
    n = mask.shape[-1]
    r = 2 ** np.arange(n - 1, -1, -1)
    return (mask * r).sum(-1)


def mask_to_mask_idx_tensor(mask):
    n = mask.shape[-1]
    r = 2 ** torch.arange(n - 1, -1, -1, device=mask.device)
    return (mask * r).sum(-1)


def all_masks_tensor(n, device=None):
    r = 2 ** torch.arange(n - 1, -1, -1, device=device)
    masks = torch.arange(2**n, device=device)[:, None] % (2*r) // r
    return masks


def onehot_to_mask_idx_tensor(onehot):
    n = int(np.log2(onehot.shape[-1]))
    assert 2 ** n == onehot.shape[-1]
    masks = all_masks_tensor(n, onehot.device).to(onehot)
    return torch.einsum('...i,ij->...j', onehot, masks)


def test_mask_idx_to_mask():
    assert mask_idx_to_mask(3, 0).tolist() == [0, 0, 0]
    assert mask_idx_to_mask(3, 1).tolist() == [0, 0, 1]
    assert mask_idx_to_mask(3, 2).tolist() == [0, 1, 0]
    assert mask_idx_to_mask(3, 7).tolist() == [1, 1, 1]
    assert mask_idx_to_mask(3, [1, 2]).tolist() == [[0, 0, 1], [0, 1, 0]]


def test_mask_to_mask_idx():
    assert mask_to_mask_idx([0, 0, 0]).tolist() == 0
    assert mask_to_mask_idx([[1, 0, 0], [1, 1, 1]]).tolist() == [4, 7]


def print_array(arr, **opts):
    original = np.get_printoptions()
    np.set_printoptions(**opts)
    out = str(np.asarray(arr))
    np.set_printoptions(**original)
    return out

