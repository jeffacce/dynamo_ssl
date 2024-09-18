import torch
import einops
import pickle
from pathlib import Path
from typing import Optional
from datasets.core import TrajectoryDataset


class PushTDataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory,
        subset_fraction: Optional[float] = None,
        relative=False,
    ):
        self.data_directory = Path(data_directory)
        self.relative = relative
        self.states = torch.load(self.data_directory / "states.pth")
        if relative:
            self.actions = torch.load(self.data_directory / "rel_actions.pth")
        else:
            self.actions = torch.load(self.data_directory / "abs_actions.pth")
        with open(self.data_directory / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)

        self.subset_fraction = subset_fraction
        if self.subset_fraction:
            assert self.subset_fraction > 0 and self.subset_fraction <= 1
            n = int(len(self.states) * self.subset_fraction)
        else:
            n = len(self.states)
        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]

        for i in range(n):
            T = self.seq_lengths[i]
            self.actions[i, T:] = 0  # redo zero padding

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        vid_dir = self.data_directory / "obses"
        obs = torch.load(str(vid_dir / f"episode_{idx:03d}.pth"))
        obs = obs[frames]  # THWC
        obs = einops.rearrange(obs, "T H W C -> T 1 C H W") / 255.0  # T V C H W, 1 view
        act = self.actions[idx, frames]
        mask = torch.ones(len(act)).bool()
        return obs, act, mask

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)
