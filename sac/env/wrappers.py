import random
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

import torch
import gym
from gym.spaces import Box

from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class PyTorchWrapper:
    def __init__(self, env: Union[gym.Env, VecEnv], device: str = "cpu"):
        self.env = env
        self.vec_env = isinstance(env, VecEnv)
        self.num_envs = 1 if not isinstance(env, VecEnv) else env.num_envs

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_high = env.action_space.high
        self.action_low = env.action_space.low
        self.action_space.high = np.ones_like(self.action_high)
        self.action_space.low = -np.ones_like(self.action_low)

        self.is_3d_observation = (
            isinstance(self.observation_space, Box)
            and len(self.observation_space.shape) == 3  # type: ignore
        )
        if self.is_3d_observation and isinstance(self.observation_space, Box):
            self.observation_space = Box(
                low=0,
                high=255,
                shape=(
                    self.observation_space.shape[2],  # type: ignore
                    self.observation_space.shape[0],  # type: ignore
                    self.observation_space.shape[1],  # type: ignore
                ),
                dtype=np.uint8,
            )

        self.device = device
        self._max_episode_steps = 1000

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if not self.vec_env:
            action = action.squeeze()
        action = action * (self.action_space.high - self.action_space.low) / 2 + (
            (self.action_space.high + self.action_space.low) / 2
        )
        obs, reward, done, info = self.env.step(action)
        if self.is_3d_observation:
            obs = wrap_3d_obs(obs)
        info = {
            k: to_torch(v).to(self.device)
            for k, v in (
                info if isinstance(info, dict) else stack_list_dict(info)
            ).items()
        }
        if "task" not in info.keys():
            info["task"] = torch.zeros((self.num_envs, 1), dtype=torch.int64).to(
                self.device
            )

        if self.vec_env:
            obs = to_torch(obs).to(self.device)
            reward = to_torch(reward).to(self.device)
            done = to_torch(done).to(self.device)
        else:
            obs = expand_dim(to_torch(obs)).to(self.device)
            reward = expand_dim(torch.Tensor([reward])).to(self.device)
            done = expand_dim(torch.Tensor([done])).to(self.device)
        if self.is_3d_observation:
            obs = obs.to(torch.uint8)
        reward = reward.float()
        done = done.bool()
        return (obs, reward, done, info)

    def reset(self) -> torch.Tensor:
        obs = self.env.reset()
        if self.is_3d_observation:
            obs = to_torch(wrap_3d_obs(obs))
            obs = obs.to(torch.uint8)
        else:
            obs = to_torch(obs)
        if self.vec_env:
            return obs.to(self.device)
        else:
            return expand_dim(obs).to(self.device)

    def render(self, mode="human"):
        self.env.render(mode=mode)

    def reset_task(self, task_id: Union[List[int], int]):
        if isinstance(self.env, MetaworldWrapper):
            self.env.reset_task(task_id)
        else:
            pass


class BraxWrapper:
    def __init__(self, env: gym.Env, device: str = "cpu", num_envs: int = 1) -> None:
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.device = device
        self.num_envs = num_envs

        self._max_episode_steps = 1000

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode="human"):
        self.env.render(mode=mode)

    def reset_task(self, task_id):
        pass


class MetaworldWrapper:
    def __init__(self, suite, suite_id: str):
        self.suite = suite
        self.suite_id = suite_id
        self.env: Optional[gym.Env] = None
        self.task = random.choice(self.suite.train_tasks)

    def step(self, action):
        if self.env is None:
            raise RuntimeError("Environment not initialized")
        return self.env.step(action)

    def reset(self):
        self.env = self.suite.train_classes[self.suite_id]()
        self.env.set_task(self.task)
        return self.env.reset()

    def render(self, mode="human"):
        if self.env is None:
            raise RuntimeError("Environment not initialized")
        self.env.render(mode=mode)

    def reset_task(self, task_id=None):
        if task_id is None:
            self.task = random.choice(self.suite.train_tasks)
        else:
            self.task = self.suite.train_tasks[task_id]


def wrap_3d_obs(obs):
    obs = obs.transpose(2, 0, 1)
    return obs


def expand_dim(x: torch.Tensor):
    return x.unsqueeze(0)


def stack_list_dict(d: List[Dict[str, np.ndarray]]):
    """
    Stack a list of dicts of numpy arrays into a single dict of numpy arrays.
    """
    stacked = {}
    for k in d[0].keys():
        stacked[k] = np.stack([d_[k] for d_ in d])
    return stacked


def to_torch(x: Union[Any, np.ndarray]):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.copy()).float()
    elif isinstance(x, Sequence):
        return torch.Tensor(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        return torch.Tensor([x])
