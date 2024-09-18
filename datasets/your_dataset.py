import utils
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset
from datasets.core import TrajectoryDataset


class YourTrajectoryDataset(TensorDataset, TrajectoryDataset):
    def __init__(self, data_directory):
        data_directory = Path(data_directory)

    def get_seq_length(self, idx):
        raise NotImplementedError

    def get_frames(self, idx, frames):
        raise NotImplementedError
        # return obs / 255.0, actions, masks

    def __getitem__(self, idx):
        T = self.get_seq_length(idx)
        return self.get_frames(idx, range(T))
