from pandas import Categorical
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gymnasium as gym
import random
import numpy as np
import copy

from DQN import DQN
from ReplayMemory import ReplayMemory

def train_dqn_batch(optimizer, batch, dqn_model, dqn_target, gamma) -> float:
    """Perform a single batch-update step on the given DQN model."""
    states, actions, rewards, next_states, dones = batch
    
    # Get the Q-values for each agent
    q_values = dqn_model(states)  # Shape: [batch_size, num_actions]
    
    # Gather the Q-values for selected actions
    values = q_values.gather(1, actions)  # actions should be [batch_size, 1]
    values = values.squeeze(1)  # Remove extra dimension, values shape will be [batch_size]

    with torch.no_grad():
        # Get the next Q-values from the target model
        next_q_values = dqn_target(next_states).detach()  # Shape: [batch_size, num_actions]
        max_next_q_values = next_q_values.max(dim=1)[0]  # Get max Q-values across actions, shape: [batch_size]
        # Compute target Q-values

        dones = dones.squeeze(1)
        rewards = rewards.squeeze(1)
        target_values = rewards + gamma * (1 - dones.float()) * max_next_q_values  # Shape: [batch_size]

    # Calculate the loss between the predicted Q-values and target Q-values
    loss = F.mse_loss(values, target_values)

    # Perform the optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()



def train_iql(env, num_steps, *, num_saves=5, replay_size, replay_prepopulate_steps=0, batch_size, exploration, gamma):
    env.reset()
    num_agents = len(env.agents)
    num_observations = env.observation_spaces[env.agents[0]].shape[0]
    num_actions = env.action_spaces[env.agents[0]].n

    print("Initializing IQL models for agents...")
    print(f"Observations: {num_observations}, Actions: {num_actions}, Agents: {num_agents}")

    agent_models = {
        agent: {
            "dqn_model": DQN(num_observations, num_actions, num_layers=3, hidden_dim=128),
            "dqn_target": DQN(num_observations, num_actions, num_layers=3, hidden_dim=128),
            "optimizer": torch.optim.RMSprop(DQN(num_observations, num_actions, num_layers=3, hidden_dim=128).parameters(), lr=5e-5, alpha=0.99, eps=1e-5),
            "replay_memory": ReplayMemory(replay_size, num_observations),
        }
        for agent in env.agents
    }

    # Prepopulate replay memories
    for agent in env.agents:
        agent_models[agent]["replay_memory"].populate(env, replay_prepopulate_steps)

    losses = {agent: [] for agent in env.agents}
    total_rewards = []
    t_total = 0
    steps = 0
    for eps in tqdm(range(int(num_steps))):
        observations, _ = env.reset()
        t_episode = 0
        steps += 1

        while t_episode < 26 and env.agents:
            eps_value = exploration.value(t_total)
            actions = {}

            for agent in env.agents:
                observation = observations[agent]
                observation = torch.tensor(observation, dtype=torch.float32)

                if random.random() > eps_value:
                    # Exploitation
                    q_values = agent_models[agent]["dqn_model"](observation.unsqueeze(0))
                    action = torch.argmax(q_values, dim=-1).item()
                else:
                    # Exploration
                    action = env.action_spaces[agent].sample()

                actions[agent] = action

            next_observations, rewards, terminations, truncations, _ = env.step(actions)

            for agent in env.agents:
                agent_models[agent]["replay_memory"].add(
                    observations[agent],
                    actions[agent],
                    rewards[agent],
                    next_observations[agent],
                    terminations[agent] or truncations[agent],
                )

                # Train the agent if replay memory has enough samples
                if len(agent_models[agent]["replay_memory"]) >= batch_size:
                    batch = agent_models[agent]["replay_memory"].sample(batch_size)
                    loss = train_dqn_batch(
                        agent_models[agent]["optimizer"],
                        batch,
                        agent_models[agent]["dqn_model"],
                        agent_models[agent]["dqn_target"],
                        gamma,
                    )
                    print(f"Agent {agent} loss: {loss}")
                    #losses[agent].append(loss)

                # Update the target network periodically
                if t_total % 500 == 0:
                    agent_models[agent]["dqn_target"].load_state_dict(agent_models[agent]["dqn_model"].state_dict())

                observations[agent] = next_observations[agent]
            t_total += 1
            t_episode += 1

        if steps % 50 == 0:
            test_iql_rewards = test_iql(env, agent_models, num_episodes=50)
            total_rewards.append(test_iql_rewards)
            print(f"Episode: {eps}, Test reward: {test_iql_rewards}")

    env.close()
    return agent_models, losses, total_rewards



def test_iql(env, agent_models, num_episodes=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_rewards = 0
    for _ in range(num_episodes):
        observations, _ = env.reset()
        done = False
        steps = 0

        while not done and env.agents and steps < 26:
            actions = {}
            for agent in env.agents:
                observation = torch.tensor(observations[agent], dtype=torch.float32, device=device)
                q_values = agent_models[agent]["dqn_model"](observation.unsqueeze(0).to(device))
                action = torch.argmax(q_values, dim=-1).item()
                actions[agent] = action

            observations, rewards, term, trunc, _ = env.step(actions)
            total_rewards += sum(rewards.values())

            done = any(term.values()) or any(trunc.values())

    return total_rewards / num_episodes
