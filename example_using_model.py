from pettingzoo.mpe import simple_spread_v3

import torch
from DQN import DQN

env = simple_spread_v3.parallel_env(N=4, max_cycles=250, continuous_actions=False, render_mode="human")

dqn_model1 = DQN(24, 5, num_layers=3, hidden_dim=128)

dqn_model1.load_state_dict(torch.load('QMIX_models/model_QMIX0.pth'))

dqn_model2 = DQN(24, 5, num_layers=3, hidden_dim=128)

dqn_model2.load_state_dict(torch.load('QMIX_models/model_QMIX1.pth'))

dqn_model3 = DQN(24, 5, num_layers=3, hidden_dim=128)

dqn_model3.load_state_dict(torch.load('QMIX_models/model_QMIX2.pth'))

observations, _ = env.reset()
for i in range(50):
    actions = {}

    for agent in env.agents:
        obs = torch.tensor(observations[agent], dtype=torch.float32).unsqueeze(0)
        if agent == 'agent_0':
            q_values = dqn_model1(obs)
        elif agent == 'agent_1':
            q_values = dqn_model2(obs)
        elif agent == 'agent_2':
            q_values = dqn_model3(obs)

        action = torch.argmax(q_values).item()
        actions[agent] = action


    observations, rewards, terminations, truncations, _ = env.step(actions)

    print(rewards)

    if all(terminations.values()) or all(truncations.values()):
        break
env.close()