import torch
import numpy as np
from collections import namedtuple
import random


Batch = namedtuple(
    'Batch', ('states', 'actions', 'rewards', 'next_states', 'dones')
)


class ReplayMemory:
    def __init__(self, max_size, state_size):
        """Replay memory implemented as a circular buffer."""
        self.max_size = int(max_size)
        self.state_size = state_size

        self.observations = torch.empty((max_size, state_size), dtype=torch.float32)
        self.actions = torch.empty((max_size, 1), dtype=torch.long)
        self.rewards = torch.empty((max_size, ), dtype=torch.float32)
        self.next_states = torch.empty((max_size, state_size), dtype=torch.float32)
        self.dones = torch.empty((max_size, ), dtype=torch.bool)

        self.idx = 0 # Pointer to the current location in the circular buffer

        self.size = 0 # Indicates number of transitions currently stored in the buffer

    def add(self, observations, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.observations[self.idx] = torch.tensor(observations, dtype=torch.float32)
        self.actions[self.idx] = torch.tensor(action, dtype=torch.long).unsqueeze(0)
        self.rewards[self.idx] = torch.tensor(reward, dtype=torch.float32)
        self.next_states[self.idx] = torch.tensor(next_state, dtype=torch.float32)
        self.dones[self.idx] = torch.tensor(done, dtype=torch.bool)

        # Circulate the pointer to the next position
        self.idx = (self.idx + 1) % self.max_size
        # Update the current buffer size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size) -> Batch:
        """Sample a batch of experiences."""
        batch_idx = np.random.choice(self.size, size=min(self.size, batch_size), replace=False)

        #'Batch', ('states', 'actions', 'rewards', 'next_states', 'dones')
        batch = Batch(
            observations =self.observations[batch_idx],
            actions      =self.actions[batch_idx],
            rewards      =self.rewards[batch_idx],
            next_states  =self.next_states[batch_idx],
            dones        = self.dones[batch_idx]
        )

        return batch

    def populate(self, env, num_steps):
        """Populate this replay memory with `num_steps` from the random policy."""
        #observations, infos = 
        env.reset()
        for _ in range(num_steps):
            actions = {}
            for agent in env.agents:
                action_space = env.action_space(agent)
                #actions[agent] = action_space.sample()

            next_observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}#env.step(actions)
            """
            for agent in env.agents:
                done = terminations[agent] or truncations[agent]
                self.add(
                    observations[agent],
                    actions[agent],
                    rewards[agent],
                    next_observations[agent],
                    done
                )"""
    
            if all(terminations.values()) or all(truncations.values()):
                env.reset()
            else:
                observations = next_observations