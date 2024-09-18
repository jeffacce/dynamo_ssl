import utils
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset
from datasets.core import TrajectoryDataset


class SimKitchenTrajectoryDataset(TensorDataset, TrajectoryDataset):
    def __init__(self, data_directory, onehot_goals=False):
        data_directory = Path(data_directory)
        states = torch.from_numpy(np.load(data_directory / "observations_seq.npy"))
        actions = torch.from_numpy(np.load(data_directory / "actions_seq.npy"))
        masks = torch.from_numpy(np.load(data_directory / "existence_mask.npy"))
        goals = torch.load(data_directory / "onehot_goals.pth")
        # The current values are in shape T x N x Dim, move to N x T x Dim
        states, actions, masks, goals = utils.transpose_batch_timestep(
            states, actions, masks, goals
        )
        observations = torch.load(data_directory / "observations_seq_img_multiview.pth")
        self.masks = masks
        tensors = [observations, actions, masks]
        if onehot_goals:
            tensors.append(goals)
        TensorDataset.__init__(self, *tensors)
        self.actions = self.tensors[1]
        self.states = states

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        result = [x[idx, frames] for x in self.tensors]
        result[0] = result[0] / 255.0
        return result

    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return self.get_frames(idx, range(T))
