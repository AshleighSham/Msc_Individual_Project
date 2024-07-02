import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
from copy import copy

#read_dictionary = np.load('4D_chains.npy',allow_pickle='TRUE').item()
read_dictionary = np.load('4D_chains_beam.npy',allow_pickle='TRUE').item()

colours = ['mediumvioletred','k', 'red']
labels = ['EnKF', 'k', 'Baby']
fig, ax = plt.subplots(nrows=4, ncols =2, sharex= 'col')
for i, q in zip(read_dictionary[0], [0, 2]):
    ax[q][0].plot(range(len(i)), i, alpha = 0.8, c = colours[q], label = labels[q])
    ax[q][0].axhline(10, alpha = 0.8, c = 'k', linestyle='dashed')
    ax[q][0].set_ylim([-1,35])
    ax[q][0].set_ylabel('$E_I$')

for i, q in zip(read_dictionary[1], [1, 3]):
    ax[q][0].plot(range(len(i)), i, alpha = 0.8, c = colours[(q - 1)])
    ax[q][0].axhline(1, alpha = 0.8, c = 'k', linestyle='dashed')
    ax[q][0].set_ylim([-1,35])
    ax[q][0].set_ylabel('$E_M$')
    
for i, q in zip(read_dictionary[2], [0, 2]):
    ax[q][1].plot(range(len(i)), i, alpha = 0.8, c = colours[q])
    ax[q][1].axhline(0.3, alpha = 0.8, c = 'k', linestyle='dashed')
    ax[q][1].set_ylim([-0.1,0.6])
    ax[q][1].yaxis.set_label_position("right")
    ax[q][1].yaxis.tick_right()
    ax[q][1].set_ylabel('$v_I$')

for i, q in zip(read_dictionary[3], [1, 3]):
    ax[q][1].plot(range(len(i)), i, alpha = 0.8, c = colours[(q-1)])
    ax[q][1].axhline(0.3, alpha = 0.8, c = 'k', linestyle='dashed')
    ax[q][1].set_ylim([-0.1,0.6])
    ax[q][1].yaxis.set_label_position("right")
    ax[q][1].yaxis.tick_right()
    ax[q][1].set_ylabel('$v_M$')

ax[0][0].axhline(10, alpha = 0.8, c = 'k', linestyle='dashed', label = 'True Value')
ax[3][1].set_xlabel('Number of Samples')
ax[3][0].set_xlabel('Number of Samples')



plt.tight_layout()
plt.subplots_adjust(bottom = 0.17)
fig.legend(loc='lower center', ncol = 4, prop={'size': 15})

plt.show()