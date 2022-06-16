import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac.utils import soft_update, hard_update
from sac.model import EnsembleGaussianPolicy, EnsembleQNetwork


class SunriseSAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.temperature = args.temperature
        self.num_ensemble = args.num_ensemble
        action_shape = action_space.shape[0]

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.update_idx = 0
        self.dropout_matrix = (torch.ones((args.num_ensemble, 1, 1))).to(self.device)

        print("Buidling criti")
        self.critic = EnsembleQNetwork(
            num_inputs, action_shape, args.hidden_size, args.num_ensemble
        ).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        print("Buidling critic_target")
        self.critic_target = EnsembleQNetwork(
            num_inputs, action_shape, args.hidden_size, args.num_ensemble
        ).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(
                torch.Tensor(action_space.shape).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        print("Buidling policy")
        self.policy = EnsembleGaussianPolicy(
            num_inputs, action_shape, args.hidden_size, args.num_ensemble
        ).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.unsqueeze(torch.from_numpy(state).float().cuda(), 0)
            next_state_action, _, mean_action = self.policy.sample(state)
            idx = torch.randint(self.update_idx, high=self.num_ensemble, size=(1,))
            if evaluate:
                return np.array((mean_action[idx][0]).squeeze().cpu())
            action = (next_state_action[idx][0]).squeeze()
            return np.array(action.cpu())

    def roll(self):
        with torch.no_grad():
            if self.update_idx < self.num_ensemble - 1:
                self.dropout_matrix[self.update_idx] = 0
                self.update_idx += 1
            else:
                self.critic.roll()
                self.critic_target.roll()
                self.policy.roll()

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            mask_batch,
        ) = memory.sample(batch_size=batch_size)

        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # action_batch = torch.FloatTensor(action_batch).to(self.device)
        # reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        # mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            next_q1_value = reward_batch + mask_batch * self.gamma * qf1_next_target
            next_q2_value = reward_batch + mask_batch * self.gamma * qf2_next_target
            next_q_value = (
                torch.min(next_q1_value, next_q2_value) - self.alpha * next_state_log_pi
            )
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = torch.mean(torch.sum((qf1 - next_q_value) ** 2, dim=-1))
        qf2_loss = torch.mean(torch.sum((qf2 - next_q_value) ** 2, dim=-1))
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (self.dropout_matrix * ((self.alpha * log_pi) - qf_pi)).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
            0.0,
        )

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists("checkpoints/"):
            os.makedirs("checkpoints/")
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print("Saving models to {}".format(ckpt_path))
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            },
            ckpt_path,
        )

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
