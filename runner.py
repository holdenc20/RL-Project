import torch
from ExponentialSchedule import ExponentialSchedule
from DQN import DQN
from train import train_playing_dqn
import numpy as np

from pettingzoo.mpe import simple_tag_v3

env = simple_tag_v3.env(num_good=1, num_adversaries=4, num_obstacles=3, max_cycles=25, continuous_actions=False, render_mode="human")
gamma = 0.99

num_steps = 5000
num_saves = 20

replay_size = 2000
replay_prepopulate_steps = 500

batch_size = 32
exploration = ExponentialSchedule(1.0, 0.05, 5000)

dqn_models, returns, lengths, losses, testing_returns = train_playing_dqn(
    env,
    num_steps,
    num_saves=num_saves,
    replay_size=replay_size,
    replay_prepopulate_steps=replay_prepopulate_steps,
    batch_size=batch_size,
    exploration=exploration,
    gamma=gamma,
)

#assert len(dqn_models) == num_saves
#assert all(isinstance(value, DQN) for value in dqn_models.values())

checkpoint = {key: dqn.custom_dump() for key, dqn in dqn_models.items()}
torch.save(checkpoint, f'model_testLargerNET.pth')
#checkpoint = torch.load(f'checkpoint_{env.spec.id}.pt')

import matplotlib.pyplot as plt

np.save('testing_returns2.npy', testing_returns)

linspace_testing_returns = np.linspace(0, len(testing_returns)-1, len(testing_returns), endpoint=True)

window_size = 10
averaged_testing_returns = np.mean(testing_returns[:len(testing_returns) - len(testing_returns) % window_size].reshape(-1, window_size), axis=1)

linspace_averaged = np.linspace(0, len(averaged_testing_returns)-1, len(averaged_testing_returns), endpoint=True)

plt.plot(linspace_averaged, averaged_testing_returns)
plt.savefig('testing_returns_averaged2.png')