import abc
import utils
import torch
import numpy as np
from torch import default_generator, randperm
from torch.utils.data import Dataset, Subset
from typing import Callable, Optional, Sequence, List, Any
from torch.nn.utils.rnn import pad_sequence


# Taken from python 3.5 docs
def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories.
    TrajectoryDataset[i] returns: (observations, actions, mask)
        observations: Tensor[T, ...], T frames of observations
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: False: invalid; True: valid
    """

    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_frames(self, idx, frames):
        """
        Returns the frames from the idx-th trajectory at the specified frames.
        Used to speed up slicing.
        """
        raise NotImplementedError


class TrajectorySubset(TrajectoryDataset, Subset):
    """
    Subset of a trajectory dataset at specified indices.

    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: TrajectoryDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    def get_all_actions(self):
        return self.dataset.get_all_actions()

    def get_frames(self, idx, frames):
        return self.dataset.get_frames(self.indices[idx], frames)


class TrajectorySlicerDataset:
    def __init__(
        self,
        dataset: TrajectoryDataset,
        window: int,
        future_conditional: bool = False,
        min_future_sep: int = 0,
        future_seq_len: Optional[int] = None,
        only_sample_tail: bool = False,
        transform: Optional[Callable] = None,
        num_extra_predicted_actions: Optional[int] = None,
        frame_step: int = 1,
        repeat_first_frame: bool = False,
    ):
        """
        Slice a trajectory dataset into unique (but overlapping) sequences of length `window`.

        dataset: a trajectory dataset that satisfies:
            dataset.get_seq_length(i) is implemented to return the length of sequence i
            dataset[i] = (observations, actions, mask)
            observations: Tensor[T, ...]
            actions: Tensor[T, ...]
            mask: Tensor[T]
                False: invalid
                True: valid
        window: int
            number of timesteps to include in each slice
        future_conditional: bool = False
            if True, observations will be augmented with future observations sampled from the same trajectory
        min_future_sep: int = 0
            minimum number of timesteps between the end of the current sequence and the start of the future sequence
            for the future conditional
        future_seq_len: Optional[int] = None
            the length of the future conditional sequence;
            required if future_conditional is True
        only_sample_tail: bool = False
            if True, only sample future sequences from the tail of the trajectory
        transform: function (observations, actions, mask[, goal]) -> (observations, actions, mask[, goal])
        """
        if future_conditional:
            assert future_seq_len is not None, "must specify a future_seq_len"
        self.dataset = dataset
        self.window = window
        self.future_conditional = future_conditional
        self.min_future_sep = min_future_sep
        self.future_seq_len = future_seq_len
        self.only_sample_tail = only_sample_tail
        self.transform = transform
        self.num_extra_predicted_actions = num_extra_predicted_actions or 0
        self.slices = []
        self.frame_step = frame_step
        min_seq_length = np.inf
        if num_extra_predicted_actions:
            window = window + num_extra_predicted_actions
        for i in range(len(self.dataset)):  # type: ignore
            T = self.dataset.get_seq_length(i)  # avoid reading actual seq (slow)
            min_seq_length = min(T, min_seq_length)
            if T - window < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={window}")
            else:
                if repeat_first_frame:
                    self.slices += [(i, 0, end + 1) for end in range(window - 1)]
                window_len_with_step = (window - 1) * frame_step + 1
                last_start = T - window_len_with_step
                self.slices += [
                    (i, start, start + window_len_with_step)
                    for start in range(last_start)
                ]  # slice indices follow convention [start, end)

        if min_seq_length < window:
            print(
                f"Ignored short sequences. To include all, set window <= {min_seq_length}."
            )

    def get_seq_length(self, idx: int) -> int:
        if self.future_conditional:
            return self.future_seq_len + self.window
        else:
            return self.window

    def get_all_actions(self) -> torch.Tensor:
        return self.dataset.get_all_actions()

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        T = self.dataset.get_seq_length(i)

        if (
            self.num_extra_predicted_actions is not None
            and self.num_extra_predicted_actions != 0
        ):
            assert self.frame_step == 1, "NOT TESTED"
            if self.future_conditional:
                raise NotImplementedError(
                    "num_extra_predicted_actions with future_conditional not implemented"
                )
            assert end <= T, f"end={end} > T={T}"
            observations, actions, mask = self.dataset.get_frames(i, range(start, end))
            observations = observations[: self.window]

            values = [observations, actions, mask.bool()]
        else:
            if self.future_conditional:
                assert self.frame_step == 1, "NOT TESTED"
                valid_start_range = (
                    end + self.min_future_sep,
                    self.dataset.get_seq_length(i) - self.future_seq_len,
                )
                if valid_start_range[0] < valid_start_range[1]:
                    if self.only_sample_tail:
                        future_obs_range = range(T - self.future_seq_len, T)
                    else:
                        future_start = np.random.randint(*valid_start_range)
                        future_end = future_start + self.future_seq_len
                        future_obs_range = range(future_start, future_end)
                    obs, actions, mask = self.dataset.get_frames(
                        i, list(range(start, end)) + list(future_obs_range)
                    )
                    future_obs = obs[end - start :]
                    obs = obs[: end - start]
                    actions = actions[: end - start]
                    mask = mask[: end - start]
                else:
                    # zeros placeholder T x obs_dim
                    obs, actions, mask = self.dataset.get_frames(i, range(start, end))
                    obs_dims = obs.shape[1:]
                    future_obs = torch.zeros((self.future_seq_len, *obs_dims))

                # [observations, actions, mask, future_obs (goal conditional)]
                values = [obs, actions, mask.bool(), future_obs]
            else:
                observations, actions, mask = self.dataset.get_frames(
                    i, range(start, end, self.frame_step)
                )
                values = [observations, actions, mask.bool()]

        if end - start < self.window + self.num_extra_predicted_actions:
            # this only happens for repeating the very first frames
            values = [
                utils.inference.repeat_start_to_length(
                    x, self.window + self.num_extra_predicted_actions, dim=0
                )
                for x in values
            ]
            values[0] = values[0][: self.window]

        # optionally apply transform
        if self.transform is not None:
            values = self.transform(values)
        return tuple(values)


class TrajectoryEmbeddingDataset(TrajectoryDataset):
    def __init__(
        self,
        model,
        dataset: TrajectoryDataset,
        device="cpu",
        embed_goal=False,
    ):
        self.data = utils.inference.embed_trajectory_dataset(
            model,
            dataset,
            obs_only=False,
            device=device,
            embed_goal=embed_goal,
        )
        assert len(self.data) == len(dataset)

        self.seq_lengths = [len(x[0]) for x in self.data]
        self.on_device_data = []
        n_tensors = len(self.data[0])
        for i in range(n_tensors):
            self.on_device_data.append(
                pad_sequence([x[i] for x in self.data], batch_first=True).to(device)
            )
        self.data = self.on_device_data

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        return torch.cat([x[1] for x in self.data], dim=0)

    def get_frames(self, idx, frames):
        return [x[idx, frames] for x in self.data]

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


def get_train_val_sliced(
    traj_dataset: TrajectoryDataset,
    train_fraction: float = 0.9,
    random_seed: int = 42,
    window_size: int = 10,
    future_conditional: bool = False,
    min_future_sep: int = 0,
    future_seq_len: Optional[int] = None,
    only_sample_tail: bool = False,
    transform: Optional[Callable[[Any], Any]] = None,
    num_extra_predicted_actions: Optional[int] = None,
    frame_step: int = 1,
):
    train, val = split_traj_datasets(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    traj_slicer_kwargs = {
        "window": window_size,
        "future_conditional": future_conditional,
        "min_future_sep": min_future_sep,
        "future_seq_len": future_seq_len,
        "only_sample_tail": only_sample_tail,
        "transform": transform,
        "num_extra_predicted_actions": num_extra_predicted_actions,
        "frame_step": frame_step,
    }

    train_slices = TrajectorySlicerDataset(train, **traj_slicer_kwargs)
    val_slices = TrajectorySlicerDataset(val, **traj_slicer_kwargs)
    return train_slices, val_slices


def random_split_traj(
    dataset: TrajectoryDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
) -> List[TrajectorySubset]:
    """
    (Modified from torch.utils.data.dataset.random_split)

    Randomly split a trajectory dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split_traj(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (TrajectoryDataset): TrajectoryDataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [
        TrajectorySubset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set
