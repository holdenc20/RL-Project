import torch
import numpy as np
from collections import namedtuple
import random


Batch = namedtuple(
    'Batch', ('observations', 'actions', 'rewards', 'next_states', 'dones')
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

    def __len__(self):
        return self.size

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


    @staticmethod
    def populate(env, num_steps, agent_memory, memory):
        """Populate this replay memory with `num_steps` from the random policy."""
        # Reset the environment to start collecting data
        observations, _ = env.reset()  # This returns the initial observations for all agents
        
        for _ in range(num_steps):
            actions = {}
            for agent in env.agents:
                # Sample a random action from the agent's action space
                action_space = env.action_spaces[agent]
                actions[agent] = action_space.sample()  # Random action
                
            # Take a step in the environment with the chosen actions
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Add the experience (state, action, reward, next_state, done) to the memory buffer
            for agent in env.agents:
                done = terminations[agent] or truncations[agent]  # Check if the episode is done
                if agent == 'agent_0':
                    agent_memory.add(
                        observations[agent],  # Current state of the agent
                        actions[agent],       # Action taken by the agent
                        rewards[agent],       # Reward received by the agent
                        next_observations[agent],  # Next state of the agent
                        done                  # Whether the episode is done for the agent
                    )
                else:
                    memory.add(
                        observations[agent],  # Current state of the agent
                        actions[agent],       # Action taken by the agent
                        rewards[agent],       # Reward received by the agent
                        next_observations[agent],  # Next state of the agent
                        done                  # Whether the episode is done for the agent
                    )

            # If all agents are done (terminated or truncated), reset the environment
            if all(terminations.values()) or all(truncations.values()):
                observations, _ = env.reset()
            else:
                observations = next_observations  # Update observations for the next step
