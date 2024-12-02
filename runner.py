import torch
from ExponentialSchedule import ExponentialSchedule
from DQN import DQN
from train import train_playing_dqn
import numpy as np

from pettingzoo.mpe import simple_tag_v3

env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=4, num_obstacles=3, max_cycles=250, continuous_actions=False, render_mode="rgb_array")
gamma = 0.99

num_steps = 5000
num_saves = 20

replay_size = 2000
replay_prepopulate_steps = 500

batch_size = 32
exploration = ExponentialSchedule(1.0, 0.05, 0.9 * num_steps)

#dqn_models, returns, lengths, losses, testing_returns = 
dqn_model, dqn_agent_model, losses_agent, losses_other = train_playing_dqn(
    env,
    num_steps,
    num_saves=num_saves,
    replay_size=replay_size,
    replay_prepopulate_steps=replay_prepopulate_steps,
    batch_size=batch_size,
    exploration=exploration,
    gamma=gamma,
)

torch.save(dqn_model.state_dict(), 'dqn_model.pth')
torch.save(dqn_agent_model.state_dict(), 'dqn_agent_model.pth')

#checkpoint = {key: dqn.custom_dump() for key, dqn in dqn_models.items()}
#torch.save(checkpoint, f'model_1.pth')
#checkpoint = torch.load(f'checkpoint_{env.spec.id}.pt')

#import matplotlib.pyplot as plt

np.save('losses_agent.npy', losses_agent)
np.save('losses_other.npy', losses_other)
