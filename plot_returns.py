import numpy as np
import matplotlib.pyplot as plt

str = 'losses_other'

testing_returns = np.load(str + '.npy')


linspace_testing_returns = np.linspace(0, len(testing_returns)-1, len(testing_returns), endpoint=True)

window_size = 1
averaged_testing_returns = np.mean(testing_returns[:len(testing_returns) - len(testing_returns) % window_size].reshape(-1, window_size), axis=1)

linspace_averaged = np.linspace(0, len(averaged_testing_returns)-1, len(averaged_testing_returns), endpoint=True)

plt.plot(linspace_averaged, averaged_testing_returns)
plt.savefig(str + '.png')