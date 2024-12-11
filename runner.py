import torch
from ExponentialSchedule import ExponentialSchedule, LinearSchedule
from DQN import DQN
from train import train_iql
from trainQMIX import train_qmix
from trainCentralized import train_centralized
import numpy as np

from pettingzoo.mpe import simple_spread_v3

model = "ALL" #ALL, QMIX, CENTRAL, DQN

num_adv = 3

env = simple_spread_v3.parallel_env(N=num_adv, max_cycles=25, continuous_actions=False, render_mode="rgb_array")
gamma = 0.99

num_steps = 1000 #make 10000
num_saves = 20

replay_size = 10000
replay_prepopulate_steps = 1000

batch_size = 256
exploration = ExponentialSchedule(1.0, 0.05, 0.9 * num_steps)
#exploration = LinearSchedule(1.0, 0.05, num_steps)

#dqn_models, returns, lengths, losses, testing_returns = 

if model == "ALL":
    returns_list = []
    
    for i in range(2):
        qmix_model, mixer_model, losses, reward = train_qmix(
            env,
            num_steps,
            num_saves=num_saves,
            replay_size=replay_size,
            replay_prepopulate_steps=replay_prepopulate_steps,
            batch_size=batch_size,
            exploration=exploration,
            gamma=gamma,
        )
        returns_list.append(reward)

    average_return = np.mean(returns_list, axis=0)
    agent_file = 'reward_agent_QMIX_final' + str(num_adv) + '.npy'
    np.save(agent_file, average_return)

    returns_list = []
    for i in range(2):
        central_model, losses_agent, reward = train_centralized(
            env,
            num_steps,
            num_saves=num_saves,
            replay_size=replay_size,
            replay_prepopulate_steps=replay_prepopulate_steps,
            batch_size=batch_size,
            exploration=exploration,
            gamma=gamma,
        )
        returns_list.append(reward)
    
    average_return = np.mean(returns_list, axis=0)
    agent_file = 'reward_agent_CENTRAL_final' + str(num_adv) + '.npy'
    np.save(agent_file, average_return)

    returns_list = []
    for i in range(2):
        dqn_models, losses_agent, reward = train_iql(
            env,
            num_steps,
            num_saves=num_saves,
            replay_size=replay_size,
            replay_prepopulate_steps=replay_prepopulate_steps,
            batch_size=batch_size,
            exploration=exploration,
            gamma=gamma,
        )
        returns_list.append(reward)
    
    average_return = np.mean(returns_list, axis=0)
    agent_file = 'reward_agent_DQN_final' + str(num_adv) + '.npy'
    np.save(agent_file, average_return)



elif model == "QMIX":
    qmix_model, mixer_model, losses, returns = train_qmix(
        env,
        num_steps,
        num_saves=num_saves,
        replay_size=replay_size,
        replay_prepopulate_steps=replay_prepopulate_steps,
        batch_size=batch_size,
        exploration=exploration,
        gamma=gamma,
    )
    for i in range(num_adv):
        qmix_model_file = 'QMIX_models/model_' + model + str(i) + '.pth'

        torch.save(qmix_model[i].state_dict(), qmix_model_file)

    agent_file = 'loss_agent_' + model + str(num_adv) + '.npy'

    np.save(agent_file, losses)

    agent_file = 'reward_agent_' + model + str(num_adv) + '.npy'
    np.save(agent_file, returns)
elif model == "CENTRAL":
    central_model, losses_agent, reward = train_centralized(
        env,
        num_steps,
        num_saves=num_saves,
        replay_size=replay_size,
        replay_prepopulate_steps=replay_prepopulate_steps,
        batch_size=batch_size,
        exploration=exploration,
        gamma=gamma,
    )

    agent_file = 'model_' + model + str(num_adv) + '.pth'

    torch.save(central_model.state_dict(), agent_file)

    agent_file = 'loss_agent_' + model + str(num_adv) + '.npy'

    np.save(agent_file, losses_agent)

    agent_file = 'reward_agent_' + model + str(num_adv) + '.npy'

    np.save(agent_file, reward)
elif model == "DQN":
    dqn_models, losses_agent, reward = train_iql(
        env,
        num_steps,
        num_saves=num_saves,
        replay_size=replay_size,
        replay_prepopulate_steps=replay_prepopulate_steps,
        batch_size=batch_size,
        exploration=exploration,
        gamma=gamma,
    )

    env.reset()

    print(reward)

    for agent in env.agents:
        m = dqn_models[agent]["dqn_model"]
        agent_file = 'model_' + model + str(num_adv) + '.pth'
        torch.save(m.state_dict(), agent_file)




        agent_file = 'loss_agent_' + model + str(num_adv) + f"_{agent}" + '.npy'
        np.save(agent_file, losses_agent)
        agent_file = 'reward_agent_' + model + str(num_adv)  + f"_{agent}" + '.npy'

        np.save(agent_file, reward)

    print(reward[0])
    print(reward[num_steps // 10])
    print(reward[num_steps // 5])
    print(reward[num_steps // 2])
    print(reward[num_steps - 1])