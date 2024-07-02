import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
from copy import copy

read_dictionary = np.load('2D_chains.npy',allow_pickle='TRUE').item()

colours = ['deepskyblue','mediumseagreen','orange','hotpink','mediumorchid','mediumvioletred', 'red']
labels = ['MH', 'AMH', 'DR MH', 'DRAM', 'pCN', 'EnKF', 'Baby']
fig, ax = plt.subplots(nrows=7, ncols =2, sharex= 'col')

for i, q in zip(read_dictionary[0], range(7)):
    ax[q][0].plot(range(len(i)), i, alpha = 0.8, c = colours[q], label = labels[q])
    ax[q][0].axhline(10, alpha = 0.8, c = 'k', linestyle='dashed')
    ax[q][0].set_ylim([0,35])


for i, q in zip(read_dictionary[1],  range(7)):
    ax[q][1].plot(range(len(i)), i,  alpha = 0.8, c = colours[q])
    ax[q][1].axhline(0.3, alpha = 0.8, c = 'k', linestyle='dashed')
    ax[q][1].set_ylim([0,0.5])
    ax[q][1].yaxis.set_label_position("right")
    ax[q][1].yaxis.tick_right()


ax[1][1].axhline(0.3, alpha = 0.8, c = 'k', linestyle='dashed', label = 'True Value')
ax[0][0].set_title("Youngs Modulus: 10")
ax[0][1].set_title("Poisson's Ratio: 0.3")

plt.tight_layout()
plt.subplots_adjust(bottom = 0.17)
fig.legend(loc='lower center', ncol = 4, prop={'size': 15})

plt.show()