import numpy as np
import torch

from ccil.utils.data import Batch, Trajectory


class StateEncoder:
    """
    Encode
    """
    def batch(self, batch: Batch) -> torch.Tensor:
        """
        Encode batch of transitions.
        :param batch:
        :return: encoded tensor of shape [len(batch), ...]
        """
        raise NotImplementedError

    def step(self, state: np.ndarray, trajectory: Trajectory) -> np.ndarray:
        """
        Encode single state transition.
        :param state: current observation
        :param trajectory: trajectory history before current observation
        :return: encoded state as np array
        """
        raise NotImplementedError