import torch
import random

class ReplayMemory:
    def __init__(self, capacity, observation_dim, num_agents):
        self.capacity = capacity
        self.idx = 0
        self.size = 0
        self.num_agents = num_agents
        self.observation_dim = observation_dim

        self.observations = torch.zeros((capacity, num_agents, observation_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, num_agents), dtype=torch.int64)
        self.rewards = torch.zeros((capacity, num_agents), dtype=torch.float32)
        self.next_observations = torch.zeros((capacity, num_agents, observation_dim), dtype=torch.float32)
        self.dones = torch.zeros((capacity, num_agents), dtype=torch.bool)

    def add(self, observations, actions, rewards, next_observations, dones):
        self.observations[self.idx] = observations
        self.actions[self.idx] = actions
        self.rewards[self.idx] = rewards
        self.next_observations[self.idx] = next_observations
        self.dones[self.idx] = dones

        # Update the circular buffer index
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # Randomly sample a batch of transitions
        indices = random.sample(range(self.size), batch_size)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size
