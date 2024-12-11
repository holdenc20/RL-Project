import torch
import numpy as np
from collections import namedtuple
import random


Batch = namedtuple(
    'Batch', ('observations', 'actions', 'rewards', 'next_states', 'dones')
)


class ReplayMemory:
    def __init__(self, max_size, state_size, agents=1):
        """Replay memory implemented as a circular buffer."""
        self.max_size = int(max_size)
        self.state_size = state_size

        self.observations = torch.zeros((max_size, state_size), dtype=torch.float32)
        self.actions = torch.zeros((max_size, agents), dtype=torch.long)
        self.rewards = torch.zeros((max_size, agents), dtype=torch.float32)
        self.next_states = torch.zeros((max_size, state_size), dtype=torch.float32)
        self.dones = torch.zeros((max_size, agents), dtype=torch.bool)

        self.idx = 0  # Pointer to the current location in the circular buffer
        self.size = 0  # Indicates the number of transitions currently stored in the buffer

    def __len__(self):
        return self.size

    def add(self, observations, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.observations[self.idx] = torch.as_tensor(observations, dtype=torch.float32)
        self.actions[self.idx] = torch.as_tensor(action, dtype=torch.long).unsqueeze(0)
        self.rewards[self.idx] = torch.as_tensor(reward, dtype=torch.float32)
        self.next_states[self.idx] = torch.as_tensor(next_state, dtype=torch.float32)
        self.dones[self.idx] = torch.as_tensor(done, dtype=torch.bool)

        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size) -> Batch:
        """Sample a batch of experiences."""
        batch_idx = np.random.choice(self.size, size=min(self.size, batch_size), replace=False)

        batch = Batch(
            observations=self.observations[batch_idx],
            actions=self.actions[batch_idx],
            rewards=self.rewards[batch_idx],
            next_states=self.next_states[batch_idx],
            dones=self.dones[batch_idx]
        )

        return batch

    def populate(self, env, num_steps):
        """Populate this replay memory with `num_steps` from the random policy."""
        observations, _ = env.reset()

        for _ in range(num_steps):
            actions = {}
            for agent in env.agents:
                action_space = env.action_spaces[agent]
                actions[agent] = action_space.sample()  # Random action

            next_observations, rewards, terminations, truncations, _ = env.step(actions)

            # Add the experience (state, action, reward, next_state, done) to the memory buffer
            for agent in env.agents:
                done = terminations[agent] or truncations[agent]
                self.add(
                    observations[agent],
                    actions[agent],
                    rewards[agent],
                    next_observations[agent],
                    done
                )

            observations = next_observations
