from typing import List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m, multiplier=5.0, random_bias=False):
    print(m)
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, EnsembleLinearLayer):
            torch.nn.init.xavier_uniform_(m.weight)
            m.weight = nn.Parameter(m.weight * multiplier)
            if random_bias:
                torch.nn.init.xavier_uniform_(m.bias)
                m.bias = nn.Parameter(m.bias * multiplier)
            else:
                torch.nn.init.constant_(m.bias, 0)


class MLP(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class TwinnedMLP(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(TwinnedMLP, self).__init__()

        self.linear1_1 = nn.Linear(num_inputs, hidden_dim)
        self.linear1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear1_3 = nn.Linear(hidden_dim, 1)

        self.linear2_1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2_3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x1 = F.relu(self.linear1_1(state))
        x1 = F.relu(self.linear1_2(x1))
        x1 = self.linear1_3(x1)

        x2 = F.relu(self.linear2_1(state))
        x2 = F.relu(self.linear2_2(x2))
        x2 = self.linear2_3(x2)

        return x1, x2


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.net1 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NormLayer(),
            nn.Linear(hidden_dim, 1),
        )

        # Q2 architecture
        self.net2 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NormLayer(),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        q1 = self.net1(xu)
        q2 = self.net2(xu)
        return q1, q2


class EnsembleQNetwork(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        ensemble_members,
        add_random_priors=False,
    ):
        super(EnsembleQNetwork, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.num_ensemble_members = ensemble_members
        self.add_random_priors = add_random_priors

        self.net1 = nn.Sequential(
            nn.Linear(num_inputs+ num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # NormLayer(),
        )
        self.net1_final = EnsembleLinearLayer(ensemble_members, hidden_dim, 1)

        self.net2 = nn.Sequential(
            nn.Linear(num_inputs+ num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
            # NormLayer(),
        )
        self.net2_final = EnsembleLinearLayer(ensemble_members, hidden_dim, 1)

        if add_random_priors:
            self.net_offset = nn.Sequential(
                EnsembleLinearLayer(
                    ensemble_members, num_inputs + num_actions, hidden_dim
                ),
                nn.ReLU(),
                EnsembleLinearLayer(ensemble_members, hidden_dim, hidden_dim),
                nn.ReLU(),
                EnsembleLinearLayer(ensemble_members, hidden_dim, 1),
            )
            # for layer in self.net_offset:
            #     weights_init_(layer, multiplier=1.0, random_bias=True)
            self.net_offset.requires_grad_(False)

        self.apply(weights_init_)

    def roll(self):
        self.net1_final.roll()
        self.net2_final.roll()

    def forward(self, state, action):
        if state.dim() == action.dim():
            xu = torch.cat([state, action], 1)
        else:
            xu = torch.cat(
                [state.unsqueeze(0).expand(self.num_ensemble_members, -1, -1), action],
                -1,
            )
        out1 = self.net1_final(self.net1(xu))
        out2 = self.net2_final(self.net2(xu))
        if self.add_random_priors:
            with torch.no_grad():
                out_offset = self.net_offset(xu)
                out1 += out_offset
                out2 += out_offset
        return out1, out2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x) + epsilon
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = F.softplus(log_std)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class EnsembleGaussianPolicy(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        ensemble_members,
        action_space=None,
    ):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.ensemble_members = ensemble_members

        self.feature_net1 = EnsembleLinearLayer(ensemble_members, num_inputs, hidden_dim)
        self.feature_net2 = EnsembleLinearLayer(ensemble_members, hidden_dim, hidden_dim)
        # self.feature_net = nn.Sequential(
        #     EnsembleLinearLayer(ensemble_members, num_inputs, hidden_dim),
        #     nn.ReLU(),
        #     EnsembleLinearLayer(ensemble_members, hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     # self.norm = NormLayer()
        # )
        self.mean_head1 = EnsembleLinearLayer(ensemble_members, hidden_dim, hidden_dim)
        self.mean_head2 = EnsembleLinearLayer(ensemble_members, hidden_dim, num_actions)

        # self.mean_head = nn.Sequential(
        #     EnsembleLinearLayer(ensemble_members, hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     EnsembleLinearLayer(ensemble_members, hidden_dim, num_actions),
        # )

        self.log_std_head1 = EnsembleLinearLayer(ensemble_members, hidden_dim, hidden_dim)
        self.log_std_head2 = EnsembleLinearLayer(ensemble_members, hidden_dim, num_actions)
        # self.log_std_head = nn.Sequential(
        #     EnsembleLinearLayer(ensemble_members, hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     EnsembleLinearLayer(ensemble_members, hidden_dim, num_actions),
        # )
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def roll(self):
        self.feature_net1.roll()
        self.feature_net2.roll()
        self.mean_head1.roll()
        self.mean_head2.roll()
        self.log_std_head1.roll()
        self.log_std_head2.roll()

    def forward(self, state):
        x1 = F.relu(self.feature_net1(state))
        x2 = F.relu(self.feature_net2(x1))
        # x = self.feature_net(state)
        # x = self.norm(x)
        mean1 = F.relu(self.mean_head1(x2))
        mean = self.mean_head2(mean1)

        log_ = F.relu(self.log_std_head1(x2))
        log_std = torch.exp(self.log_std_head2(log_)) + epsilon
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = F.softplus(log_std)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0.0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.0), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class EnsembleLinearLayer(nn.Module):
    """
    Efficient linear layer for ensemble models.
    Taken from https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/models/util.py
    """

    def __init__(
        self, num_members: int, in_size: int, out_size: int, bias: bool = True
    ):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(
            torch.rand(self.num_members, self.in_size, self.out_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.rand(self.num_members, 1, self.out_size))
            self.use_bias = True
            torch.nn.init.zeros_(self.bias)
        else:
            self.use_bias = False

    def roll(self):
        with torch.no_grad():
            self.weight = nn.Parameter(torch.cat([torch.clone(self.weight[1:]), torch.clone(self.weight[-1:])], dim=0))
            self.bias = nn.Parameter(torch.cat([torch.clone(self.bias[1:]), torch.clone(self.bias[-1:])], dim=0))

    def forward(self, x):
        if x.dim() == 2:
            xw = x.matmul(self.weight)
        else:
            xw = torch.bmm(x, self.weight)
            # xw = torch.einsum("ebd,edm->ebm", x, self.weight)
        if self.use_bias:
            return xw + self.bias
        else:
            return xw


class NormLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.norm(x, dim=-1, keepdim=True)
