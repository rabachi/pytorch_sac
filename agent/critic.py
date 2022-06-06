import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

import utils

class VIPCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, device, num_Qs=3) -> None:
        super().__init__()
        
        trunk_output_dim = math.ceil(hidden_dim/2)
        self.Q_trunk = utils.mlp(obs_dim + action_dim, hidden_dim, trunk_output_dim, hidden_depth)
        # self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth) #for double Q-learning trick
        self.num_Qs = num_Qs
        Q_heads = []
        # self.device = torch.device(device)
        for q_idx in range(num_Qs):
            Q_heads.append(utils.mlp(trunk_output_dim, hidden_dim, 1, 1))#.to(self.device)
        self.Q_heads = nn.ModuleList(Q_heads)
        self.outputs = {}
        self.apply(utils.weight_init)

    def forward_trunk(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        trunk_out = self.Q_trunk(obs_action)

        return trunk_out
    
    def forward(self, obs, action, q_idx):
        trunk_out = self.forward_trunk(obs, action)
        head_out = self.Q_heads[q_idx](trunk_out)

        # assert obs.size(0) == action.size(0)
        # obs_action = torch.cat([obs, action], dim=-1)
        # q2 = self.Q2(obs_action)
        self.outputs[f'q{q_idx}'] = head_out #don't save
        #     outputs.append(self.Q_heads[q_idx](trunk_out))
        # #torch.stack for outputs 
        # # return self.outputs.values() #is this a "good" idea?
        # return torch.stack(outputs)
        return head_out#, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step) #can't log this because didn't save the outputs
        # to_check_len = list(self.Q_heads.values())
        # for v in to_check_len[1:]:
        #     assert len(v) == len(to_check_len[0])
        #     assert type(v) == type(to_check_len[0])
            
        # for k, v in self.Q_heads.items():
        #     if type(v) is nn.Linear:
        #         for i in range(len(v)):
        #             logger.log_param(f'train_critic/q{k}_fc{i}', v, step)


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)
