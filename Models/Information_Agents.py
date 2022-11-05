import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RolloutStorage:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_storage(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=1),
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, rollout_storage):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        if rollout_storage:
            rollout_storage.states.append(state)
            rollout_storage.actions.append(action)
            rollout_storage.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class Demon:
    def __init__(
        self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip
    ):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, rollout_storage):
        return self.policy_old.act(state, rollout_storage)

    def update(self, rollout_storage):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(rollout_storage.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.stack(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = rewards.squeeze(-1)

        # convert list to tensor
        old_states = torch.squeeze(
            torch.stack(rollout_storage.states).to(device), 1
        ).detach()
        old_actions = torch.squeeze(
            torch.stack(rollout_storage.actions).to(device), 1
        ).detach()
        old_logprobs = (
            torch.squeeze(torch.stack(rollout_storage.logprobs), 1).to(device).detach()
        )

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            try:
                state_values = state_values[
                    :-1, :
                ]  # reward is computed as the mutual info between consequenct mem state,
                # therefore n-1 values only.
                ratios = ratios[:-1, :]  # the same for ratio
                dist_entropy = dist_entropy[:-1, :]  # the same for entropy
                advantages = rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages
                )
                # Finding Surrogate Loss:
                loss = (
                    -torch.min(surr1, surr2)
                    + 0.5 * self.MseLoss(state_values, rewards)
                    - 0.01 * dist_entropy
                )
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
            except Exception:
                # Do thing for the sequences of lentgh 1.
                loss = torch.zeros_like(rewards).to(device)
                continue

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss


############################################
# Mutual information Estimator Network######
############################################


def linear_reset(module, gain=1.0):
    assert isinstance(module, torch.nn.Linear)
    init.xavier_uniform_(module.weight, gain=gain)
    s = module.weight.size(1)
    if module.bias is not None:
        module.bias.data.zero_()


class FNet(nn.Module):
    """
    Monte-Carlo estimators for Mutual Information Known as MINE.
    Mine produces estimates that are neither an upper or lower bound on MI.
    Other ZNet can be Introduced to address the problem of building bounds with finite samples (unlike Monte Carlo)
    """

    def __init__(self):
        super(FNet, self).__init__()

    def reset_parameters(self):
        for module in self.lstm:
            if isinstance(module, torch.nn.Linear):
                linear_reset(module, gain=init.calculate_gain("relu"))

        for module in self.hidden2f:
            if isinstance(module, torch.nn.Linear):
                linear_reset(module, gain=init.calculate_gain("relu"))

    def init(self, input_size):
        self.lstm = nn.Sequential(nn.LSTM(input_size, 32, batch_first=True))
        self.hidden2f = nn.Sequential(nn.Linear(32, 1))
        self.reset_parameters()

    def forward(self, data):
        output, (hn, cn) = self.lstm(data)
        output = F.elu(output)
        fvals = self.hidden2f(output)
        return fvals


class ZNet(nn.Module):
    def __init__(self):
        super(ZNet, self).__init__()

    def reset_parameters(self):
        for module in self.lstm:
            if isinstance(module, torch.nn.Linear):
                linear_reset(module, gain=init.calculate_gain("relu"))

        for module in self.hidden2z:
            if isinstance(module, torch.nn.Linear):
                linear_reset(module, gain=init.calculate_gain("relu"))

    def init(self, input_size):
        self.lstm = nn.Sequential(nn.LSTM(input_size, 32, batch_first=True))
        self.hidden2z = nn.Sequential(nn.Linear(32, 1))
        self.reset_parameters()

    def forward(self, data):
        output, (hn, cn) = self.lstm(data)
        output = F.elu(output)
        zvals = self.hidden2z(output)
        return F.softplus(zvals)
