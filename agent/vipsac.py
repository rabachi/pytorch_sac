import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from agent import Agent
import agent
import utils

import hydra


class VIPSACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, num_agents):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.num_agents = num_agents #number of separate SAC actors that should be learned. after some number of steps, actors get frozen one by one while their corresponding critics are updated, eventually the actors are cycled(?) so that the first frozen one gets updated to the most recent one while the second frozen one becomes the first frozen and so on. 
        # self.critics = {} #should I use torch tensors for these instead of dicts? (is that possible)
        # self.critic_targets = {}
        self.actors = {}
        self.log_alphas = {}

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        for agent_idx in range(self.num_agents):
            self.actors[agent_idx] = hydra.utils.instantiate(actor_cfg).to(self.device)
            self.log_alphas[agent_idx] = torch.tensor(np.log(init_temperature)).to(self.device)
            self.log_alphas[agent_idx].requires_grad = True

        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizers = {}
        self.log_alpha_optimizers = {}

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)
        for agent_idx in range(self.num_agents):
            self.actor_optimizers[agent_idx] = torch.optim.Adam(self.actors[agent_idx].parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)
            self.log_alpha_optimizers[agent_idx] = torch.optim.Adam([self.log_alphas[agent_idx]],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        for agent_idx in range(self.num_agents):
            self.actors[agent_idx].train(training)
        self.critic.train(training)

    # @property #is it a bad idea to get rid of this
    def alpha(self, agent_idx):
        return self.log_alphas[agent_idx].exp()

    def act(self, obs, agent_idx, sample=False): #changed this method's signature!
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actors[agent_idx](obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step):
        critic_loss = 0
        #vectorize the loop below
        for agent_idx in range(self.num_agents):
            with torch.no_grad():
                dist = self.actors[agent_idx](next_obs)
                #why are they using rsample if don't need gradients ... 
                next_action = dist.rsample() #need to have separate samples for each policy
                log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
                target_Q1 = self.critic_target(next_obs, next_action, agent_idx) 
                target_V = target_Q1 - self.alpha(agent_idx).detach() * log_prob
                target_Q = reward + (not_done * self.discount * target_V)
                target_Q = target_Q.detach()

            # get current Q estimates
            current_Q1 = self.critic(obs, action, agent_idx)
            critic_loss += F.mse_loss(current_Q1, target_Q)
        
        logger.log('train_critic/loss', critic_loss, step)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step() #this step is done self.num_agents times for the trunk! This could lead to the improvement we may observe (should do the same for baseline for fair comparison. Or should we update just once with the average of all gradients? Accumulate the gradients for all self.num_agents critics and do a single grad step?) (Edit: changed it to the latter)
        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step, agent_idx):
        dist = self.actors[agent_idx](obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1 = self.critic(obs, action, agent_idx) #had double_Q here

        # actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha(agent_idx).detach() * log_prob - actor_Q1).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizers[agent_idx].zero_grad()
        actor_loss.backward()
        self.actor_optimizers[agent_idx].step()

        self.actors[agent_idx].log(logger, step)

        if self.learnable_temperature: #where does this come from? how it affects results to set it to true/false?
            self.log_alpha_optimizers[agent_idx].zero_grad()
            alpha_loss = (self.alpha(agent_idx) *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha(agent_idx), step)
            alpha_loss.backward()
            self.log_alpha_optimizers[agent_idx].step()

    def update(self, replay_buffer, logger, step, actors_to_update):
        '''
        actors_to_update: List[int] indices of actors that need to be updated
        '''
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                           logger, step)

        if step % self.actor_update_frequency == 0:
            for actor_idx in actors_to_update:
                self.update_actor_and_alpha(obs, logger, step, actor_idx)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
