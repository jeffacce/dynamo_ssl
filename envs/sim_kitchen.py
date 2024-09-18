"""Environments using kitchen and Franka robot."""

import logging
import einops
import gym
import numpy as np
import torch
from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1
from gym.envs.registration import register
from dm_control.mujoco import engine

OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}
BONUS_THRESH = 0.3

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w",
)


class KitchenBase(KitchenTaskRelaxV1):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.
    TASK_ELEMENTS = []
    ALL_TASKS = [
        "bottom burner",
        "top burner",
        "light switch",
        "slide cabinet",
        "hinge cabinet",
        "microwave",
        "kettle",
    ]
    REMOVE_TASKS_WHEN_COMPLETE = True
    TERMINATE_ON_TASK_COMPLETE = True
    TERMINATE_ON_WRONG_COMPLETE = False
    COMPLETE_IN_ANY_ORDER = (
        True  # This allows for the tasks to be completed in arbitrary order.
    )

    def __init__(
        self, dataset_url=None, ref_max_score=None, ref_min_score=None, **kwargs
    ):
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        self.all_completions = []
        self.completion_ids = []
        self.goal_masking = True
        super(KitchenBase, self).__init__(**kwargs)

    def set_goal_masking(self, goal_masking=True):
        """Sets goal masking for goal-conditioned approaches (like RPL)."""
        self.goal_masking = goal_masking

    def _get_task_goal(self, task=None, actually_return_goal=False):
        if task is None:
            task = ["microwave", "kettle", "bottom burner", "light switch"]
        new_goal = np.zeros_like(self.goal)
        if self.goal_masking and not actually_return_goal:
            return new_goal
        for element in task:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal

        return new_goal

    def reset_model(self):
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        self.all_completions = []
        self.completion_ids = []
        return super(KitchenBase, self).reset_model()

    def set_task_goal(self, one_hot_indices):
        """Sets the goal for the robot to complete the given tasks."""
        self.tasks_to_complete = []
        for i, idx in enumerate(one_hot_indices):
            if idx == 1:
                self.tasks_to_complete.append(self.ALL_TASKS[i])
        logging.info("Setting task goal to {}".format(self.tasks_to_complete))
        self.TASK_ELEMENTS = self.tasks_to_complete
        self.goal = self._get_task_goal(task=self.tasks_to_complete)

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
        reward = 0.0
        next_q_obs = obs_dict["qp"]
        next_obj_obs = obs_dict["obj_qp"]
        next_goal = self._get_task_goal(
            task=self.TASK_ELEMENTS, actually_return_goal=True
        )  # obs_dict['goal']
        idx_offset = len(next_q_obs)
        completions = []
        all_completed_so_far = True
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] - next_goal[element_idx]
            )
            complete = distance < BONUS_THRESH
            condition = (
                complete and all_completed_so_far
                if not self.COMPLETE_IN_ANY_ORDER
                else complete
            )
            if condition:  # element == self.tasks_to_complete[0]:
                logging.info("Task {} completed!".format(element))
                completions.append(element)
                self.all_completions.append(element)
                self.completion_ids.append(self.ALL_TASKS.index(element))
            all_completed_so_far = all_completed_so_far and complete
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        reward_dict["bonus"] = bonus
        reward_dict["r_total"] = bonus
        score = bonus
        return reward_dict, score

    def step(self, a, b=None):
        obs, reward, done, env_info = super(KitchenBase, self).step(a, b=b)
        if self.TERMINATE_ON_TASK_COMPLETE:
            done = not self.tasks_to_complete
        if self.TERMINATE_ON_WRONG_COMPLETE:
            all_goal = self._get_task_goal(task=self.ALL_TASKS)
            for wrong_task in list(set(self.ALL_TASKS) - set(self.TASK_ELEMENTS)):
                element_idx = OBS_ELEMENT_INDICES[wrong_task]
                distance = np.linalg.norm(obs[..., element_idx] - all_goal[element_idx])
                complete = distance < BONUS_THRESH
                if complete:
                    done = True
                    break
        env_info["all_completions"] = self.all_completions
        env_info["all_completions_ids"] = self.completion_ids
        env_info["image"] = self.render(mode="rgb_array")
        return obs, reward, done, env_info

    def get_goal(self):
        """Loads goal state from dataset for goal-conditioned approaches (like RPL)."""
        raise NotImplementedError

    def _split_data_into_seqs(self, data):
        """Splits dataset object into list of sequence dicts."""
        seq_end_idxs = np.where(data["terminals"])[0]
        start = 0
        seqs = []
        for end_idx in seq_end_idxs:
            seqs.append(
                dict(
                    states=data["observations"][start : end_idx + 1],
                    actions=data["actions"][start : end_idx + 1],
                )
            )
            start = end_idx + 1
        return seqs

    def render(self, mode="human", size=(224, 224), distance=2.5):
        if mode == "rgb_array":
            camera = engine.MovableCamera(self.sim, *size)
            camera.set_pose(
                distance=distance, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-35
            )
            img = camera.render()
            return img
        else:
            super(KitchenTaskRelaxV1, self).render()


class KitchenAllV0(KitchenBase):
    TASK_ELEMENTS = KitchenBase.ALL_TASKS


class KitchenWrapper(gym.Wrapper):
    def __init__(self, env, id):
        super(KitchenWrapper, self).__init__(env)
        self.id = id
        self.env = env

    def reset(self, goal_idx=None, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        return_obs = self.render(mode="rgb_array", size=(224, 224), distance=2.5)
        return self.preprocess_img(return_obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return_obs = self.render(mode="rgb_array", size=(224, 224), distance=2.5)
        return self.preprocess_img(return_obs), reward, done, info

    def preprocess_img(self, img):
        img_tensor = torch.from_numpy(np.array(img))
        img_tensor = einops.rearrange(img_tensor, "H W C -> 1 C H W")
        return img_tensor / 255.0


register(
    id="kitchen-v0",
    entry_point="envs.sim_kitchen:KitchenAllV0",
    max_episode_steps=280,
    reward_threshold=4.0,
)
