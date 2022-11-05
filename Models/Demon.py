import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.distributions import Normal

from collections import namedtuple

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6

SavedAction = namedtuple("SavedAction", ["action", "log_prob", "mean"])


def linear_reset(module, gain=1.0):
    assert isinstance(module, torch.nn.Linear)
    init.xavier_uniform_(module.weight, gain=gain)
    s = module.weight.size(1)
    if module.bias is not None:
        module.bias.data.zero_()


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
        zvals = self.hidden2z(output)
        return F.softplus(zvals)


class FNet(nn.Module):
    def __init__(self):
        super(FNet, self).__init__()

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
        fvals = self.hidden2z(output)
        return fvals


class Demon(torch.nn.Module):
    """
    Demon manipulates the external memory of DNC.
    """

    def __init__(self, layer_sizes=[]):
        super(Demon, self).__init__()
        self.layer_sizes = layer_sizes
        self.action_scale = torch.tensor(1)
        self.action_bias = torch.tensor(0.0)
        self.saved_actions = []

    def get_output_size(self):
        return self.layer_sizes[-1]

    def reset_parameters(self):
        for module in self.model:
            if isinstance(module, torch.nn.Linear):
                linear_reset(module, gain=init.calculate_gain("relu"))
        linear_reset(self.embed_mean, gain=init.calculate_gain("relu"))
        linear_reset(self.embed_log_std, gain=init.calculate_gain("relu"))

    def init(self, input_size, output_size):
        # Xavier init: input to all gates is layers_sizes[i-1] + layer_sizes[i] + input_size -> layer_size big.
        # So use xavier init according to this.
        self.input_sizes = [input_size] + self.layer_sizes[:-1]
        layers = []
        for i, size in enumerate(self.layer_sizes):
            layers.append(nn.Linear(self.input_sizes[i], self.layer_sizes[i]))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)
        self.embed_mean = nn.Linear(self.layer_sizes[-1], output_size)
        self.embed_log_std = nn.Linear(self.layer_sizes[-1], output_size)

        self.reset_parameters()

    def forward(self, data):
        x = self.model(data)
        x = F.relu(x)
        mean, log_std = self.embed_mean(x), self.embed_log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = torch.exp(log_std)
        return mean, std

    def act(self, data):
        """
        pathwise derivative estimator for taking actions.
        :param data:
        :return:
        """
        mean, std = self.forward(data)
        normal = Normal(mean, std)
        x = normal.rsample()

        y = torch.softmax(x, dim=1)

        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(action)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.softmax(mean, dim=1) * self.action_scale + self.action_bias
        self.saved_actions.append(SavedAction(action, log_prob, mean))

        return mean
