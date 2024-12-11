from pettingzoo.mpe import simple_spread_v3

import torch
from QMIX import DQN

num_adv = 1

env = simple_spread_v3.parallel_env(N=4, max_cycles=250, continuous_actions=False, render_mode="human")

dqn_models = [DQN(24, 5, num_layers=3, hidden_dim=128) for _ in range(num_adv)]


for i, model in enumerate(dqn_models):
    model.load_state_dict(torch.load(f'QMIX_models/model_QMIX{i}.pth'))



observations, _ = env.reset()
for i in range(25):
    actions = {}

    for agent in env.agents:
        obs = torch.tensor(observations[agent], dtype=torch.float32).unsqueeze(0)
        q_values = dqn_model(obs)

        action = torch.argmax(q_values).item()
        actions[agent] = action


    observations, rewards, terminations, truncations, _ = env.step(actions)

    print(rewards)

    if all(terminations.values()) or all(truncations.values()):
        break
env.close()