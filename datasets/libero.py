import torch
import einops
import numpy as np
from pathlib import Path
from typing import Optional
from torch.nn.utils.rnn import pad_sequence
from datasets.core import TrajectoryDataset


class LiberoGoalDataset(TrajectoryDataset):
    # data structure:
    # libero_goal
    #      task_name
    #          demo_{i}
    #              agentview_image.mp4
    #              robot0_eye_in_hand_image.mp4
    #              robot0_joint_pos.npy
    #              robot0_eef.npy
    #              robot0_gripper_qpos.npy
    #              object_states.npy
    #              actions.npy
    def __init__(self, data_directory, subset_fraction: Optional[float] = None):
        self.dir = Path(data_directory) / "libero_goal"
        self.task_names = list(self.dir.iterdir())
        self.task_names.sort()
        self.demos = []
        for task_name in self.task_names:
            self.demos += list(task_name.iterdir())

        self.subset_fraction = subset_fraction
        if self.subset_fraction:
            assert 0 < self.subset_fraction <= 1
            n = int(len(self.demos) * self.subset_fraction)
            self.demos = self.demos[:n]

        # prefetch all npy data
        self.joint_pos = []
        self.eef = []
        self.gripper_qpos = []
        self.object_states = []
        self.states = []
        self.actions = []
        for demo in self.demos:
            self.joint_pos.append(np.load(demo / "robot0_joint_pos.npy"))
            self.eef.append(np.load(demo / "robot0_eef.npy"))
            self.gripper_qpos.append(np.load(demo / "robot0_gripper_pos.npy"))
            self.object_states.append(np.load(demo / "object_states.npy"))
            state = np.concatenate(
                [
                    self.joint_pos[-1],
                    self.eef[-1],
                    self.gripper_qpos[-1],
                    self.object_states[-1],
                ],
                axis=1,
            )
            act = np.load(demo / "actions.npy")
            self.states.append(torch.from_numpy(state))
            self.actions.append(torch.from_numpy(act))

        # pad state dimension to same length for linear probe diagnostics
        MAX_DIM = 128
        for i in range(len(self.states)):
            self.states[i] = torch.cat(
                [
                    self.states[i],
                    torch.zeros(
                        self.states[i].shape[0], MAX_DIM - self.states[i].shape[1]
                    ),
                ],
                dim=1,
            )
        # pad states and actions to the same time length
        self.states = pad_sequence(self.states, batch_first=True).float()
        self.actions = pad_sequence(self.actions, batch_first=True).float()

        # last frame goal
        self.goals = None
        goals = []
        for i in range(0, 500, 50):
            last_obs, _, _ = self.get_frames(i, [-1])  # 1 V C H W
            goals.append(last_obs)
        self.goals = goals

    def __len__(self):
        return len(self.demos)

    def get_frames(self, idx, frames):
        demo = self.demos[idx]
        agentview_obs = torch.load(
            str(demo / "agentview_image.pth"),
        )
        robotview_obs = torch.load(
            str(demo / "robot0_eye_in_hand_image.pth"),
        )
        agentview = agentview_obs[frames]
        robotview = robotview_obs[frames]
        obs = torch.stack([agentview, robotview], dim=1)
        obs = einops.rearrange(obs, "T V H W C -> T V C H W") / 255.0
        act = self.actions[idx][frames]

        if self.goals is not None:
            task_idx = idx // 50
            goal = self.goals[task_idx].repeat(len(frames), 1, 1, 1, 1)
            return obs, act, goal
        else:
            return obs, act, None

    def __getitem__(self, idx):
        return self.get_frames(idx, range(len(self.joint_pos[idx])))

    def get_seq_length(self, idx):
        return len(self.joint_pos[idx])

    def get_all_actions(self):
        actions = []
        for i in range(len(self.demos)):
            T = self.get_seq_length(i)
            actions.append(self.actions[i][:T])
        return torch.cat(actions, dim=0)
