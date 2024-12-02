from pettingzoo.mpe import simple_tag_v3

import torch
from DQN import DQN

env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=4, num_obstacles=3, max_cycles=250, continuous_actions=False, render_mode="human")

dqn_model = DQN(20, 5, num_layers=4, hidden_dim=256)
dqn_agent_model = DQN(18, 5, num_layers=4, hidden_dim=256)


dqn_agent_model.load_state_dict(torch.load('dqn_agent_model.pth'))
dqn_model.load_state_dict(torch.load('dqn_model.pth'))

observations, _ = env.reset()
for i in range(1000):
    actions = {}

    for agent in env.agents:
        obs = torch.tensor(observations[agent], dtype=torch.float32).unsqueeze(0)
        if agent == 'agent_0':
            q_values = dqn_agent_model(obs)
        else:
            q_values = dqn_model(obs)

        action = torch.argmax(q_values).item()
        actions[agent] = action


    observations, rewards, terminations, truncations, _ = env.step(actions)

    print(rewards)

    if all(terminations.values()) or all(truncations.values()):
        break
env.close()