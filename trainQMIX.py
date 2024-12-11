from pandas import Categorical
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gymnasium as gym
import random

from QMIX import Mixer, DQN
from ReplayMemoryQMIX import ReplayMemory

def train_vdn_batch(optimizer, batch, dqn_models, dqn_target_models, gamma):
    states, actions, rewards, next_states, dones = batch
    num_agents = len(dqn_models)

    q_values = torch.stack([dqn(states[:, i, :]) for i, dqn in enumerate(dqn_models)], dim=1)
    q_values_selected = torch.gather(q_values, dim=2, index=actions.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_q_values = torch.stack([dqn_target(next_states[:, i, :]) for i, dqn_target in enumerate(dqn_target_models)], dim=1)
        max_next_q_values = next_q_values.max(dim=2)[0]
        target_q_values = rewards + gamma * (1 - dones.float()) * max_next_q_values

    global_q_values = q_values_selected.sum(dim=1, keepdim=True)
    target_global_q_values = target_q_values.sum(dim=1, keepdim=True)

    loss = F.mse_loss(global_q_values, target_global_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_qmix_batch(optimizer, batch, dqn_models, mixer_model, dqn_target_models, mixer_target, gamma):
    states, actions, rewards, next_states, dones = batch
    num_agents = len(dqn_models)

    q_values = torch.stack([dqn(states[:, i, :]) for i, dqn in enumerate(dqn_models)], dim=1)
    q_values_selected = torch.gather(q_values, dim=2, index=actions.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_q_values = torch.stack([dqn_target(next_states[:, i, :]) for i, dqn_target in enumerate(dqn_target_models)], dim=1)
        max_next_q_values = next_q_values.max(dim=2)[0]

        target_global_q_values = mixer_target(max_next_q_values, next_states) 
        target_global_q_values = rewards.sum(dim=1, keepdim=True) + gamma * (1 - dones.float()) * target_global_q_values

    # Compute predicted global Q-values
    global_q_values = mixer_model(q_values_selected, states)

    target_global_q_values = target_global_q_values.sum(dim=1, keepdim=True)

    # Check shapes - was running into issues with this
    assert global_q_values.shape == target_global_q_values.shape, \
        f"Shape mismatch: {global_q_values.shape} vs {target_global_q_values.shape}"

    loss = F.mse_loss(global_q_values, target_global_q_values)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(mixer_model.parameters()), 10.0)
    optimizer.step()

    return loss.item()


def train_qmix(env, num_steps, *, num_saves=5, replay_size, replay_prepopulate_steps=0, batch_size, exploration, gamma, target_update_interval=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env.reset()
    num_agents = len(env.agents)
    num_observations = env.observation_spaces[env.agents[0]].shape[0]
    num_actions = env.action_spaces[env.agents[0]].n

    dqn_models = [DQN(num_observations, num_actions, num_layers=3, hidden_dim=128).to(device) for _ in range(num_agents)]
    dqn_target_models = [DQN(num_observations, num_actions, num_layers=3, hidden_dim=128).to(device) for _ in range(num_agents)]

    for target, source in zip(dqn_target_models, dqn_models):
        target.load_state_dict(source.state_dict())

    mixer_model = Mixer(num_agents, observation_dim=num_observations, hidden_dim=128).to(device)
    mixer_target = Mixer(num_agents, observation_dim=num_observations, hidden_dim=128).to(device)
    mixer_target.load_state_dict(mixer_model.state_dict())

    parameters = list(mixer_model.parameters()) + sum([list(dqn_model.parameters()) for dqn_model in dqn_models], [])
    optimizer = torch.optim.RMSprop(parameters, lr=5e-5, alpha=0.99, eps=1e-5)

    memory = ReplayMemory(replay_size, num_observations, num_agents)

    # Prepopulate replay memory
    env.reset()
    for _ in range(replay_prepopulate_steps):
        observations, _ = env.reset()
        actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        next_observations, rewards, terminations, truncations, _ = env.step(actions)

        if env.agents:
            memory.add(
                torch.stack([torch.tensor(observations[agent], dtype=torch.float32) for agent in env.agents], dim=0),
                torch.tensor([actions[agent] for agent in env.agents], dtype=torch.int64),
                torch.tensor([rewards[agent] for agent in env.agents], dtype=torch.float32),
                torch.stack([torch.tensor(next_observations[agent], dtype=torch.float32) for agent in env.agents], dim=0),
                torch.tensor([terminations[agent] or truncations[agent] for agent in env.agents], dtype=torch.bool)
            )

    losses = []
    returns = []
    t_total = 0
    step = 0
    for eps in tqdm(range(int(num_steps))):
        step += 1
        observations, _ = env.reset()
        t_episode = 0

        while t_episode < 26 and env.agents:
            current_eps = exploration.value(t_total)
            actions = {}

            for i, agent in enumerate(env.agents):
                observation = torch.tensor(observations[agent], dtype=torch.float32, device=device)
                if random.random() > current_eps:
                    q_values = dqn_models[i](observation.unsqueeze(0))
                    action = torch.argmax(q_values, dim=-1).item()
                else:
                    action = env.action_spaces[agent].sample()
                actions[agent] = action

            next_observations, rewards, terminations, truncations, _ = env.step(actions)

            if env.agents:
                memory.add(
                    torch.stack([torch.tensor(observations[agent], dtype=torch.float32) for agent in env.agents], dim=0),
                    torch.tensor([actions[agent] for agent in env.agents], dtype=torch.int64),
                    torch.tensor([rewards[agent] for agent in env.agents], dtype=torch.float32),
                    torch.stack([torch.tensor(next_observations[agent], dtype=torch.float32) for agent in env.agents], dim=0),
                    torch.tensor([terminations[agent] or truncations[agent] for agent in env.agents], dtype=torch.bool)
                )

            if len(memory) >= batch_size:
                batch = memory.sample(batch_size)
                #loss = train_qmix_batch(optimizer, batch, dqn_models, mixer_model, dqn_target_models, mixer_target, gamma)
                loss = train_vdn_batch(optimizer, batch, dqn_models, dqn_target_models, gamma)
                losses.append(loss)

            t_total += 1
            t_episode += 1
            observations = next_observations

            if t_total % target_update_interval == 0:
                for target, source in zip(dqn_target_models, dqn_models):
                    target.load_state_dict(source.state_dict())
                mixer_target.load_state_dict(mixer_model.state_dict())

        if step % 50 == 0:
            #print(f"Episode: {eps}, Loss: {loss:.4f}")
            reward_test = test(env, dqn_models, mixer_model, num_episodes=50)
            #print(f"Reward: {reward_test:.4f}")
            returns.append(reward_test)

    env.close()
    return dqn_models, mixer_model, losses, returns




def test(env, dqn_models, mixer, num_episodes=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_rewards = 0
    for _ in range(num_episodes):
        observations, _ = env.reset()
        done = False
        steps = 0

        while env.agents and steps < 26:
            steps += 1
            actions = {}
            for i, agent in enumerate(env.agents):
                observation = torch.tensor(observations[agent], dtype=torch.float32, device=device)
                q_values = dqn_models[i](observation.unsqueeze(0).to(device))
                action = torch.argmax(q_values, dim=-1).item()
                actions[agent] = action

            observations, rewards, _, _, _ = env.step(actions)
            total_rewards += sum(rewards.values())

    return total_rewards / num_episodes
