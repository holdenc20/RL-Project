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
    q_values = dqn_model(states)
    values = q_values.gather(1, actions).squeeze(-1)

    next_q_values = dqn_target(next_states).detach()
    max_next_q_values = next_q_values.max(dim=1)[0]

    target_values = rewards + gamma * max_next_q_values * (1 - dones.float())
    target_values = target_values.detach()

    loss = F.mse_loss(values, target_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train_playing_dqn(env, num_steps, *, num_saves=5, replay_size, replay_prepopulate_steps=0, batch_size, exploration, gamma):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reset the environment
    env.reset(seed=42)
    num_agents = len(env.agents)
    num_observations = env.observation_spaces[env.agents[0]].shape[0]
    num_actions = env.action_spaces[env.agents[0]].n

    # Initialize the DQN model, optimizer, and replay memory
    print("Initializing DQN model...")
    print(num_observations)
    print(num_actions)


    dqn_model = DQN(num_observations, num_actions, num_layers=4, hidden_dim=256).to(device)


    num_observations_agent = env.observation_spaces['agent_0'].shape[0]
    num_actions_agent = env.action_spaces['agent_0'].n
    dqn_agent_model = DQN(num_observations_agent, num_actions_agent, num_layers=4, hidden_dim=256).to(device)

    
    optimizer = torch.optim.Adam(dqn_model.parameters())
    optimizer_agent = torch.optim.Adam(dqn_agent_model.parameters())

    memory = ReplayMemory(replay_size, num_observations)    

    memory_agent = ReplayMemory(replay_size, num_observations_agent)

    # Populate the replay memory with random data
    ReplayMemory.populate(env, replay_prepopulate_steps, memory_agent, memory)
    ReplayMemory.populate(env, replay_prepopulate_steps, memory_agent, memory)


    losses_agent = []
    losses = []

    rewards = []
    t_total = 0

    # Training Loop
    for eps in tqdm(range(int(num_steps))):
        observations, _ = env.reset()  # Reset the environment at the start of each episode
        t_episode = 0
        while t_episode < 1000 and env.agents:
            done = False
            eps = exploration.value(t_total)  # Get the exploration value (epsilon-greedy)
            actions = {}  # Dictionary to store actions for each agent

            for agent in env.agents:
                #print("Agent:", agent)
                observation = observations[agent]
                observation = torch.tensor(observation, dtype=torch.float32, device=device, requires_grad=False)

                if random.random() > eps:
                    # Use DQN to choose an action (exploitation)
                    if agent == 'agent_0':
                        q_values = dqn_agent_model(observation.unsqueeze(0).to(device))
                    else:
                        q_values = dqn_model(observation.unsqueeze(0).to(device))  # Feed observation to model
                    policy = F.softmax(q_values, dim=-1).detach()
                    action = torch.argmax(policy, dim=-1).item()
                else:
                    # Random action for exploration
                    action = env.action_spaces[agent].sample()  # Randomly sample an action
                    #print(f"Random action for {agent}: {action}")


                #print("Observations:", len(observation))
                actions[agent] = action  # Store the action for the agent

            observations, rewards, terminations, truncations, _= env.step(actions)


            for agent in env.agents:
                if agent == 'agent_0':
                    memory_agent.add(observations[agent], actions[agent], rewards[agent], observations[agent], terminations[agent] or truncations[agent])
                else:
                    memory.add(observations[agent], actions[agent], rewards[agent], observations[agent], terminations[agent] or truncations[agent])

            if len(memory) >= batch_size:
                #model
                batch = memory.sample(batch_size)
                loss = train_dqn_batch(optimizer, batch, dqn_model, dqn_model, gamma)
                losses.append(loss)

            if len(memory_agent) >= batch_size:
                #agent model
                batch_agent = memory_agent.sample(batch_size)
                loss = train_dqn_batch(optimizer_agent, batch_agent, dqn_agent_model, dqn_agent_model, gamma)
                losses_agent.append(loss)

            # Track and log rewards, terminations, and truncations
            #print("------------------------------")
            #print("Iteration:", t_total)
            for agent in env.agents:
                reward = rewards[agent]
                termination = terminations[agent]
                truncation = truncations[agent]
                #print(f"Agent {agent} - Reward: {reward}, Termination: {termination}, Truncation: {truncation}")
            
                if terminations[agent] or truncations[agent]:
                    done = True
                if rewards[agent] == -10:
                    done = True

            t_total += 1
            t_episode += 1

    env.close()
    return dqn_model, dqn_agent_model, losses_agent, losses

def test(env, current_dqn_model, old_dqn_model):
    total = 0
    for i in range(100):
        env.reset(seed=42)
        done = False
        total_reward = 0
        while not done:
            q_values = (
                current_dqn_model(torch.tensor(state, dtype=torch.float32))
                if env.current_player == 0
                else old_dqn_model(torch.tensor(state, dtype=torch.float32))
            )

            possible_actions = env.possible_actions()
            if len(possible_actions) == 0:
                break
            num_actions = q_values.size(0)
            mask = torch.full((num_actions,), float('-inf'))
            mask[possible_actions] = 0

            masked_q_values = q_values + mask
            action = torch.argmax(masked_q_values).item()

            state, reward, done, _ = env.step(action)
            total_reward += reward

        total += total_reward

    return total / 10