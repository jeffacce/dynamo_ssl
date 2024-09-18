import os
import torch
import einops
import numpy as np
from pathlib import Path
from typing import Optional
from datasets.core import TrajectoryDataset


class PushMultiviewTrajectoryDataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        onehot_goals=False,
        subset_fraction: Optional[float] = None,
        prefetch: bool = False,
    ):
        self.data_directory = Path(data_directory)
        self.states = np.load(self.data_directory / "multimodal_push_observations.npy")
        self.actions = np.load(self.data_directory / "multimodal_push_actions.npy")
        self.masks = np.load(self.data_directory / "multimodal_push_masks.npy")

        self.subset_fraction = subset_fraction
        if self.subset_fraction:
            assert self.subset_fraction > 0 and self.subset_fraction <= 1
            n = int(len(self.states) * self.subset_fraction)
        else:
            n = len(self.states)
        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.masks = self.masks[:n]

        self.states = torch.from_numpy(self.states).float()
        self.actions = torch.from_numpy(self.actions).float() / 0.03
        self.masks = torch.from_numpy(self.masks).bool()
        self.prefetch = prefetch
        if self.prefetch:
            self.obses = []
            for i in range(n):
                vid_path = self.data_directory / "obs_multiview" / f"{i:03d}.pth"
                self.obses.append(torch.load(vid_path))
        self.onehot_goals = onehot_goals
        if self.onehot_goals:
            self.goals = torch.load(self.data_directory / "onehot_goals.pth").float()
            self.goals = self.goals[:n]

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        if self.prefetch:
            obs = self.obses[idx][frames]
        else:
            obs = torch.load(self.data_directory / "obs_multiview" / f"{idx:03d}.pth")[
                frames
            ]
        obs = einops.rearrange(obs, "T V H W C -> T V C H W") / 255.0
        act = self.actions[idx, frames]
        mask = self.masks[idx, frames]
        if self.onehot_goals:
            goal = self.goals[idx, frames]
            return obs, act, mask, goal
        else:
            return obs, act, mask

    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return self.get_frames(idx, range(T))

    def __len__(self):
        return len(self.states)
