import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, observation_dim, action_dim, num_layers=3, hidden_dim=128):
        super(DQN, self).__init__()
        layers = []
        input_dim = observation_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Mixer(nn.Module):
    def __init__(self, num_agents, observation_dim, hidden_dim):
        super(Mixer, self).__init__()
        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim

        self.hyper_w1 = nn.Linear(observation_dim, self.hidden_dim)
        self.hyper_b1 = nn.Linear(observation_dim, self.hidden_dim)
        self.hyper_w2 = nn.Linear(self.hidden_dim, 1)
        self.hyper_b2 = nn.Linear(self.hidden_dim, 1)

    def forward(self, q_values, observation):
        batch_size = observation.size(0)
        q_values = q_values.view(batch_size, self.num_agents, 1)

        # using hypernetwork weights
        w1 = self.hyper_w1(observation).view(batch_size, self.num_agents, self.hidden_dim)
        b1 = self.hyper_b1(observation).view(batch_size, self.num_agents, self.hidden_dim)

        hidden = F.relu(torch.matmul(q_values.transpose(1, 2), w1) + b1)

        w2 = self.hyper_w2(hidden).view(batch_size, self.num_agents, 1)
        b2 = self.hyper_b2(hidden).view(batch_size, self.num_agents, 1)

        output = torch.sum(w2 + b2, dim=1)
        return output

class ObservationEncoder(nn.Module):
    def __init__(self, num_agents, obs_dim, state_dim):
        super().__init__()
        self.encoder = nn.Linear(num_agents * obs_dim, state_dim)

    def forward(self, observations):
        return self.encoder(observations.view(observations.size(0), -1))
