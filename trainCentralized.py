import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ReplayMemory import ReplayMemory
from DQN import DQN

def train_centralized_batch(optimizer, batch, q_network, target_network, gamma):
    states, actions, rewards, next_states, dones = batch

    joint_states = states.view(states.shape[0], -1)
    joint_next_states = next_states.view(next_states.shape[0], -1)
    joint_actions = actions.view(actions.shape[0], -1).sum(dim=1, keepdim=True)

    q_values = q_network(joint_states)
    q_values_selected = q_values.gather(1, joint_actions)

    with torch.no_grad():
        next_q_values = target_network(joint_next_states)
        max_next_actions = next_q_values.argmax(dim=1, keepdim=True)
        max_next_q_values = next_q_values.gather(1, max_next_actions)
        target_q_values = rewards.mean(dim=1, keepdim=True) + gamma * (1 - dones.float()) * max_next_q_values

    loss = F.mse_loss(q_values_selected, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def train_centralized(env, num_steps, *, num_saves=5, replay_size, replay_prepopulate_steps=0, batch_size, exploration, gamma, target_update_interval=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env.reset()
    num_agents = len(env.agents)
    num_observations = env.observation_spaces[env.agents[0]].shape[0]
    num_actions = env.action_spaces[env.agents[0]].n

    joint_obs_dim = num_observations * num_agents
    joint_action_dim = num_actions * num_agents
    q_network = DQN(joint_obs_dim, joint_action_dim, num_layers=4, hidden_dim=256).to(device)
    target_network = DQN(joint_obs_dim, joint_action_dim, num_layers=4, hidden_dim=256).to(device)

    target_network.load_state_dict(q_network.state_dict())

    optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-4, eps=1e-5) # MAYBE CHANGE THIS
    memory = ReplayMemory(replay_size, joint_obs_dim, num_agents)

    # Prepopulate replay memory
    env.reset()
    for _ in range(replay_prepopulate_steps):
        observations, _ = env.reset()
        actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        next_observations, rewards, terminations, truncations, _ = env.step(actions)

        if env.agents:
            memory.add(
                torch.cat(
                    [torch.tensor(observations[agent], dtype=torch.float32).flatten() for agent in env.agents], 
                    dim=0
                ),
                torch.tensor([actions[agent] for agent in env.agents], dtype=torch.int64),
                torch.tensor([rewards[agent] for agent in env.agents], dtype=torch.float32),
                torch.cat(
                    [torch.tensor(next_observations[agent], dtype=torch.float32).flatten() for agent in env.agents], 
                    dim=0
                ),
                torch.tensor([terminations[agent] or truncations[agent] for agent in env.agents], dtype=torch.bool)
            )

    losses = []
    returns = []
    t_total = 0

    steps = 0
    for eps in tqdm(range(num_steps)):
        observations, _ = env.reset()
        t_episode = 0
        steps += 1

        while t_episode < 26 and env.agents:
            current_eps = exploration.value(t_total)
            actions = {}

            # Process joint observations into a single flattened tensor
            joint_observations = torch.cat(
                [torch.tensor(observations[agent], dtype=torch.float32).flatten() for agent in env.agents], 
                dim=0
            ).to(device)

            if random.random() > current_eps:
                q_values = q_network(joint_observations.unsqueeze(0))  # Shape: [1, joint_action_dim]
                joint_action = torch.argmax(q_values, dim=-1).item()   # Select joint action
                for i, agent in enumerate(env.agents):
                    actions[agent] = (joint_action // num_actions ** i) % num_actions
            else:
                actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}

            next_observations, rewards, terminations, truncations, _ = env.step(actions)

            if env.agents:
                memory.add(
                    torch.cat(
                        [torch.tensor(observations[agent], dtype=torch.float32).flatten() for agent in env.agents], 
                        dim=0
                    ),  # Flatten observations
                    torch.tensor([actions[agent] for agent in env.agents], dtype=torch.int64),
                    torch.tensor([rewards[agent] for agent in env.agents], dtype=torch.float32),
                    torch.cat(
                        [torch.tensor(next_observations[agent], dtype=torch.float32).flatten() for agent in env.agents], 
                        dim=0
                    ),
                    torch.tensor([terminations[agent] or truncations[agent] for agent in env.agents], dtype=torch.bool)
                )

            if len(memory) >= batch_size:
                batch = memory.sample(batch_size)
                loss = train_centralized_batch(optimizer, batch, q_network, target_network, gamma)
                losses.append(loss)

            t_total += 1
            t_episode += 1
            observations = next_observations

            if t_total % target_update_interval == 0:
                target_network.load_state_dict(q_network.state_dict())

        if steps % 50 == 0:
            #print(f"Episode: {eps}, Loss: {loss:.4f}")
            reward_test = test_centralized(env, q_network, num_episodes=50)
            returns.append(reward_test)
            #print(f"Reward: {reward_test:.4f}")

    return q_network, losses, returns

def test_centralized(env, q_network, num_episodes = 5):
    total_reward = 0
    for _ in range(num_episodes):
        steps = 0
        observations, _ = env.reset()
        while env.agents and steps < 26:
            steps += 1
            joint_observations = torch.cat([torch.tensor(observations[agent], dtype=torch.float32) for agent in env.agents], dim=0)
            q_values = q_network(joint_observations.unsqueeze(0))
            joint_action = torch.argmax(q_values, dim=-1).item()
            actions = {}
            for i, agent in enumerate(env.agents):
                actions[agent] = (joint_action // env.action_spaces[agent].n ** i) % env.action_spaces[agent].n
            observations, rewards, _, _, _ = env.step(actions)
            total_reward += sum(rewards.values())

    return total_reward / num_episodes