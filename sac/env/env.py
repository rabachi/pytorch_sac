import functools
from typing import Any, Dict, Optional, Type, Union

import gym
from brax import envs as brax_envs
from brax.envs import to_torch
import torch

# import metaworld
# import distracting_dmc2gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from bayesian_daml.sac.env.wrappers import (
    BraxWrapper,
    MetaworldWrapper,
    PyTorchWrapper,
)


if torch.cuda.is_available():
    v = torch.ones(1, device="cuda")  # init torch cuda before jax


def setup_environment(
    type: str,
    env_id: str,
    n_envs: int,
    device: str = "cpu",
    seed: Optional[int] = None,
    **kwargs,
):
    if type == "brax":
        return setup_brax_env(env_id, n_envs, device)
    elif type == "gym":
        base_env = gym.make(env_id)
    elif type == "dm_control":
        base_env = distracting_dmc2gym.make(
            domain_name=kwargs["domain_name"],
            task_name=kwargs["task_name"],
            seed=seed,
            difficulty=kwargs["difficulty"],
            background_dataset_path=kwargs["background_dataset_path"],
            background_dataset_videos=kwargs["background_dataset_videos"],
            background_kwargs=kwargs["background_kwargs"],
            camera_kwargs=kwargs["camera_kwargs"],
            render_kwargs=kwargs["render_kwargs"],
            pixels_only=kwargs["pixels_only"],
            pixels_observation_key=kwargs["pixels_observation_key"],
            height=kwargs["height"],
            width=kwargs["width"],
            camera_id=kwargs["camera_id"],
            frame_skip=kwargs["frame_skip"],
            environment_kwargs=kwargs["environment_kwargs"],
            channels_first=False,
            episode_length=kwargs["episode_length"],
            time_limit=kwargs["time_limit"],
        )
    elif type == "metaworld":
        benchmark = kwargs.pop("benchmark", None)
        if benchmark == "ML1":
            multi_task_suite = metaworld.ML1(
                env_id
            )  # Construct the benchmark, sampling tasks
        else:
            raise ValueError(f"Unknown benchmark {benchmark}")
        return PyTorchWrapper(MetaworldWrapper(multi_task_suite, env_id))
    else:
        raise ValueError(f"Unknown environment type: {type}")
    if n_envs == 1:
        return PyTorchWrapper(base_env, device=device)
    elif n_envs > 1:
        return PyTorchWrapper(
            make_vec_env(base_env, n_envs, seed=seed, vec_env_cls=SubprocVecEnv),
            device=device,
        )
    else:
        raise ValueError(f"n_envs must be > 0, got {n_envs}")


def setup_brax_env(
    env_id: str, n_envs: int, device: str = "cpu", seed: Optional[int] = None
):
    env_name = "brax-" + env_id + "-v0"
    entry_point = functools.partial(brax_envs.create_gym_env, env_name=env_id)
    gym.register(env_name, entry_point=entry_point)

    # create a gym environment that contains 4096 parallel ant environments
    gym_env = gym.make(env_name, batch_size=n_envs)

    # wrap it to interoperate with torch data structures
    gym_env = to_torch.JaxToTorchWrapper(gym_env, device=device)  # type: ignore

    return BraxWrapper(gym_env, device=device)


def make_vec_env(
    env_id: gym.Env,
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: the environment ID or the environment class
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :return: The wrapped environment
    """
    assert not isinstance(
        env_id, MetaworldWrapper
    ), "Vectorizing Metaworld is not supported"

    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs

    def make_env(rank):
        def _init():
            env = env_id
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            return env

        return _init

    if vec_env_cls is None:
        vec_env_cls = DummyVecEnv

    return vec_env_cls(
        [make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs
    )
