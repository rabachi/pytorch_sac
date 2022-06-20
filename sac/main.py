import argparse
import datetime
import os
import gym
import numpy as np
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
import hydra
from sac.env.env import setup_environment

from sac.replay_memory import DataBuffer, ReplayMemory
from sac.viper_sac import ViperSAC


@hydra.main(config_path="../config", config_name="sac_test")
def main(args):
    print(args)

    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    env = gym.make(args.env_name)
    # env = setup_environment(
    #     type="brax",
    #     env_id="ant",
    #     n_envs=1,
    #     device="cuda",
    #     seed=4,
    # )

    # env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    if args.algorithm == "viper_sac":
        agent = ViperSAC(env.observation_space.shape[0], env.action_space, args)
    else:
        raise ValueError("Unknown algorithm")

    # Tesnorboard
    writer = SummaryWriter(
        "{}_{}_{}_{}".format(
            args.algorithm, args.env_name, args.policy, args.num_ensemble
        )
    )

    # Memory
    # memory = ReplayMemory(args.replay_size, args.seed)
    memory = DataBuffer(args.replay_size, env, device="cuda" if args.cuda else "cpu")

    if os.path.exists("train.meta"):
        agent.load_checkpoint(args.env_name)
        memory.load(".")
        with open("train.meta", "r") as f:
            total_numsteps = int(f.read())
            updates = args.updates_per_step * total_numsteps
    else:
        total_numsteps = 0
        updates = 0

    # Training Loop
    total_numsteps = 0
    updates = 0

    # torch.autograd.set_detect_anomaly(True)

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    (
                        critic_1_loss,
                        critic_2_loss,
                        policy_loss,
                        ent_loss,
                        alpha,
                    ) = agent.update_parameters(memory, args.batch_size, updates)

                    if total_numsteps + 1 % args.roll_freq == 0:
                        print("rolled")
                        agent.roll()

                    writer.add_scalar("loss/critic_1", critic_1_loss, updates)
                    # writer.add_scalar("loss/critic_2", critic_2_loss, updates)
                    writer.add_scalar("loss/policy", policy_loss, updates)
                    writer.add_scalar("loss/entropy_loss", ent_loss, updates)
                    writer.add_scalar("entropy_temperature/alpha", alpha, updates)
                    writer.add_scalar(
                        "actions/action_norm",
                        # torch.norm(action.detach().cpu()).item(),
                        np.linalg.norm(action),
                        updates,
                    )
                    updates += 1

            with torch.no_grad():
                next_state, reward, done, _ = env.step(action)  # Step
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(
                state, action, reward, next_state, mask
            )  # Append transition to memory

            state = next_state

        if total_numsteps > args.num_steps:
            break

        writer.add_scalar("reward/train", episode_reward, total_numsteps)
        print(
            "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
                i_episode,
                total_numsteps,
                episode_steps,
                round(episode_reward, 2),
                # round(episode_reward.cpu().item(), 2),
            )
        )

        if i_episode % 50 == 0:
            agent.save_checkpoint(args.env_name)
            memory.save(".")
            with open("train.meta", "w") as f:
                f.write(str(total_numsteps))
            print("checkpoint done")
            if args.eval is True:
                with torch.no_grad():
                    avg_reward = 0.0
                    episodes = 10
                    for _ in range(episodes):
                        state = env.reset()
                        episode_reward = 0
                        done = False
                        while not done:
                            action = agent.select_action(state, evaluate=True)

                            next_state, reward, done, _ = env.step(action)
                            episode_reward += reward

                            state = next_state
                        avg_reward += episode_reward
                    avg_reward /= episodes

                writer.add_scalar("avg_reward/test", avg_reward, total_numsteps)

                print("----------------------------------------")
                print(
                    "Test Episodes: {}, Avg. Reward: {}".format(
                        episodes,
                        round(avg_reward, 2)
                        # round(avg_reward.cpu().item(), 2)
                    )
                )
                print("----------------------------------------")

    env.close()


if __name__ == "__main__":
    main()
