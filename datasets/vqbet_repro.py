import abc
import utils
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Callable


class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories.
    TrajectoryDataset[i] returns: (observations, actions, mask)
        observations: Tensor[T, ...], T frames of observations
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: 0: invalid; 1: valid
    """

    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError


class TrajectorySlicerDataset(TrajectoryDataset):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        window: int,
        action_window: int,
        vqbet_get_future_action_chunk: bool = True,
        future_conditional: bool = False,
        min_future_sep: int = 0,
        future_seq_len: Optional[int] = None,
        only_sample_tail: bool = False,
        transform: Optional[Callable] = None,
        use_libero_goal: bool = False,
    ):
        if future_conditional:
            assert future_seq_len is not None, "must specify a future_seq_len"
        self.dataset = dataset
        self.window = window
        self.action_window = action_window
        self.vqbet_get_future_action_chunk = vqbet_get_future_action_chunk
        self.future_conditional = future_conditional
        self.min_future_sep = min_future_sep
        self.future_seq_len = future_seq_len
        self.only_sample_tail = only_sample_tail
        self.transform = transform
        self.slices = []
        self.use_libero_goal = use_libero_goal
        min_seq_length = np.inf
        if vqbet_get_future_action_chunk:
            min_window_required = window + action_window
        else:
            min_window_required = max(window, action_window)
        for i in range(len(self.dataset)):  # type: ignore
            T = self.dataset.get_seq_length(i)  # avoid reading actual seq (slow)
            min_seq_length = min(T, min_seq_length)
            if T - min_window_required < 0:
                print(
                    f"Ignored short sequence #{i}: len={T}, window={min_window_required}"
                )
            else:
                self.slices += [
                    (i, 0, end + 1) for end in range(window - 1)
                ]  # slice indices follow convention [start, end)
                self.slices += [
                    (i, start, start + window)
                    for start in range(T - min_window_required)
                ]  # slice indices follow convention [start, end)

        if min_seq_length < min_window_required:
            print(
                f"Ignored short sequences. To include all, set window <= {min_seq_length}."
            )

    def get_seq_length(self, idx: int) -> int:
        if self.future_conditional:
            return self.future_seq_len + self.window
        else:
            return self.window

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        if end - start < self.window:
            obs, act, *others = self.dataset[i]
            obs = utils.inference.repeat_start_to_length(
                obs[start:end], self.window, dim=0
            )
            act = utils.inference.repeat_start_to_length(
                act[start : end - 1 + self.action_window],
                self.window + self.action_window - 1,
                dim=0,
            )
            values = [obs, act]
        else:
            values = [
                self.dataset[i][0][start:end],
                self.dataset[i][1][start : end - 1 + self.action_window],
            ]

        if self.use_libero_goal:
            goals = self.dataset[i][2][start:end]
            if end - start < self.window:
                goals = utils.inference.repeat_start_to_length(
                    goals, self.window, dim=0
                )
            values.append(goals)

        # optionally apply transform
        if self.transform is not None:
            values = self.transform(values)
        if len(values) == 2:  # placeholder goal
            values.append(torch.ones([1, 1, 1]))
        return tuple(values)
