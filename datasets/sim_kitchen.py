import utils
import torch
import numpy as np
from pathlib import Path
from datasets.core import TrajectoryDataset


class SimKitchenTrajectoryDataset(TrajectoryDataset):
    def __init__(self, data_directory, prefetch=True, onehot_goals=False):
        self.data_directory = Path(data_directory)
        states = torch.from_numpy(np.load(self.data_directory / "observations_seq.npy"))
        actions = torch.from_numpy(np.load(self.data_directory / "actions_seq.npy"))
        goals = torch.load(self.data_directory / "onehot_goals.pth")
        # The current values are in shape T x N x Dim, move to N x T x Dim
        self.states, self.actions, self.goals = utils.transpose_batch_timestep(
            states, actions, goals
        )
        self.Ts = np.load(self.data_directory / "existence_mask.npy").sum(axis=0).astype(int).tolist()
        
        self.prefetch = prefetch
        if self.prefetch:
            self.obses = []
            for i in range(len(self.Ts)):
                self.obses.append(torch.load(self.data_directory / "obses" / f"{i:03d}.pth"))
        self.onehot_goals = onehot_goals

    def get_seq_length(self, idx):
        return self.Ts[idx]

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.Ts)):
            T = self.Ts[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        # obs, act, mask / obs, act, mask, goal
        if self.prefetch:
            obs = self.obses[idx][frames]
        else:
            obs = torch.load(self.data_directory / "obses" / f"{idx:03d}.pth")[frames]
        obs = obs / 255.0
        act = self.actions[idx, frames]
        mask = torch.ones((len(frames)))
        if self.onehot_goals:
            goal = self.goals[idx, frames]
            return obs, act, mask, goal
        else:
            return obs, act, mask

    def __getitem__(self, idx):
        T = self.Ts[idx]
        return self.get_frames(idx, range(T))
    
    def __len__(self):
        return len(self.Ts)