import os
import random
import numpy as np
import torch


class DataBuffer:
    def __init__(
        self,
        capacity: int,
        env,
        device="cpu",
    ):
        self.device = device

        obs_space = env.observation_space
        act_space = env.action_space

        self.obs_shape = obs_space.shape
        self.act_shape = act_space.shape
        # print(*obs_space.shape)
        self.s = torch.zeros((capacity, *obs_space.shape[0:]), dtype=torch.float).to(
            self.device
        )
        self.s_n = torch.zeros((capacity, *obs_space.shape[0:]), dtype=torch.float).to(
            self.device
        )
        self.a = torch.zeros((capacity, *act_space.shape[0:]), dtype=torch.float).to(
            self.device
        )
        self.r = torch.zeros((capacity, 1)).to(self.device)
        self.d = torch.zeros((capacity, 1)).to(self.device)
        self.t = torch.zeros((capacity, 1)).to(self.device)

        self.capacity = capacity
        self.fill_counter = 0
        self.full = False

    def push(self, state, action, reward, next_step, done):
        self.fill_counter += 1
        if self.fill_counter == self.capacity:
            self.fill_counter = 0
            self.full = True
        if isinstance(state, torch.Tensor):
            self.s[self.fill_counter] = state
            self.a[self.fill_counter] = action
            self.r[self.fill_counter] = reward
            self.d[self.fill_counter] = done
            self.s_n[self.fill_counter] = next_step
        else:
            self.s[self.fill_counter] = torch.from_numpy(state).to(self.device)
            self.a[self.fill_counter] = torch.from_numpy(action).to(self.device)
            self.r[self.fill_counter] = torch.Tensor([reward]).to(self.device)
            self.d[self.fill_counter] = torch.Tensor([done]).to(self.device)
            self.s_n[self.fill_counter] = torch.from_numpy(next_step).to(self.device)

    def sample(self, batch_size):
        if len(self) < batch_size:
            return None
        indices = np.random.randint(0, len(self), batch_size)
        return (
            self.s[indices],
            self.a[indices],
            self.r[indices],
            self.s_n[indices],
            self.d[indices],
        )

    def __len__(self):
        if self.full:
            return self.capacity
        else:
            return self.fill_counter

    def save(self, path):
        print("Saving buffer")
        torch.save(self.s, os.path.join(path, "s.torch"))
        torch.save(self.s_n, os.path.join(path, "s_n.torch"))
        torch.save(self.a, os.path.join(path, "a.torch"))
        torch.save(self.r, os.path.join(path, "r.torch"))
        torch.save(self.d, os.path.join(path, "d.torch"))
        with open(os.path.join(path, "meta.buffer"), "w") as f:
            f.write(str(self.fill_counter) + "\n")
            f.write(str(self.full))

    def load(self, path):
        self.s = torch.load(os.path.join(path, "s.torch"))
        self.s_n = torch.load(os.path.join(path, "s_n.torch"))
        self.a = torch.load(os.path.join(path, "a.torch"))
        self.r = torch.load(os.path.join(path, "r.torch"))
        self.d = torch.load(os.path.join(path, "d.torch"))
        with open(os.path.join(path, "meta.buffer"), "r") as f:
            self.fill_counter = int(f.readline())
            self.full = "True" == f.readline()

        print(f"reset buffer to step {self.fill_counter} and full {self.full}")

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists("checkpoints/"):
            os.makedirs("checkpoints/")

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print("Saving buffer to {}".format(save_path))

        with open(save_path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print("Loading buffer from {}".format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
