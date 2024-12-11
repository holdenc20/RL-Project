import numpy as np
import matplotlib.pyplot as plt

str = 'reward_agent_QMIX_final1'
str2 = 'reward_agent_CENTRAL_final1'
str3 = 'reward_agent_DQN_final1'

testing_returns1 = np.load(str + '.npy')
testing_returns2 = np.load(str2 + '.npy')
testing_returns3 = np.load(str3 + '.npy')

linspace_testing_returns1 = np.linspace(0, len(testing_returns1) - 1, len(testing_returns1), endpoint=True)
linspace_testing_returns2 = np.linspace(0, len(testing_returns2) - 1, len(testing_returns2), endpoint=True)
linspace_testing_returns3 = np.linspace(0, len(testing_returns3) - 1, len(testing_returns3), endpoint=True)

window_size = len(testing_returns1) // 40
window_size = 1
averaged_testing_returns1 = np.mean(testing_returns1[:len(testing_returns1) - len(testing_returns1) % window_size]
                                    .reshape(-1, window_size), axis=1)
averaged_testing_returns2 = np.mean(testing_returns2[:len(testing_returns2) - len(testing_returns2) % window_size]
                                    .reshape(-1, window_size), axis=1)
averaged_testing_returns3 = np.mean(testing_returns3[:len(testing_returns3) - len(testing_returns3) % window_size]
                                    .reshape(-1, window_size), axis=1)

linspace_averaged1 = np.linspace(0, len(averaged_testing_returns1) - 1, len(averaged_testing_returns1), endpoint=True) * 40
linspace_averaged2 = np.linspace(0, len(averaged_testing_returns2) - 1, len(averaged_testing_returns2), endpoint=True) * 40
linspace_averaged3 = np.linspace(0, len(averaged_testing_returns3) - 1, len(averaged_testing_returns3), endpoint=True) * 40

plt.figure(figsize=(10, 6))
plt.plot(linspace_averaged1, averaged_testing_returns1, label="QMIX", color="blue")
plt.plot(linspace_averaged2, averaged_testing_returns2, label="Centralized", color="orange")
plt.plot(linspace_averaged3, averaged_testing_returns3, label="DQN", color="green")
plt.title('Agent Rewards over Episodes (Agent#=3)', fontsize=14)
plt.xlabel('Episodes', fontsize=12)
plt.ylabel('Rewards', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("reward_comparison_plot.png")
plt.show()
