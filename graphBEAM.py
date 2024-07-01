import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
from copy import copy

read_dictionary = np.load('my_fileBEAM.npy',allow_pickle='TRUE').item()

colours = ['deepskyblue','mediumseagreen','orange','hotpink','mediumorchid','mediumvioletred', 'red']
labels = ['MH', 'AMH', 'DR MH', 'DRAM', 'pCN', 'EnKF', 'Baby']
fig, ax = plt.subplots(2,2, figsize = (10,10) )

read_dictionary['graphs'][0][-2][1] = np.array(read_dictionary['graphs'][0][-2][1]) /6
read_dictionary['graphs'][1][-2][1] = np.array(read_dictionary['graphs'][1][-2][1]) /2.6
read_dictionary['graphs'][2][-2][1] = np.array(read_dictionary['graphs'][2][-2][1]) /2.4
read_dictionary['graphs'][3][-2][1] = np.array(read_dictionary['graphs'][3][-2][1]) /2.2


for i, j, q in zip(read_dictionary['graphs'][0], read_dictionary['values'][0], range(7)):
    a = 0.8
    if q == 6:
        a = 1
    ax[0][0].plot(i[0], i[1], alpha = a, c = colours[q])
    ax[0][0].plot([j], [-0.1], c = colours[q], marker = "|", markersize=18, markeredgewidth = 3)

for i, j, q in zip(read_dictionary['graphs'][1], read_dictionary['values'][1], range(7)):
    a = 0.8
    if q == 6:
        a = 1
    ax[1][0].plot(i[0], i[1], alpha = a, c = colours[q])
    ax[1][0].plot([j], [-0.05], c = colours[q], marker = "|", markersize=18, markeredgewidth = 3)

for i, j, q in zip(read_dictionary['graphs'][2], read_dictionary['values'][2], range(7)):
    a = 0.8
    if q == 6:
        a = 1
    ax[0][1].plot(i[0], i[1], alpha = a, c = colours[q])
    ax[0][1].plot([j], [-1], c = colours[q], marker = "|", markersize=18, markeredgewidth = 3)

for i, j, q in zip(read_dictionary['graphs'][3], read_dictionary['values'][3], range(7)):
    a = 0.8
    if q == 6:
        a = 1
    ax[1][1].plot(i[0], i[1], alpha = a, c = colours[q])
    ax[1][1].plot([j], [-1], c = colours[q], marker = "|", markersize=18, markeredgewidth = 3, label = labels[q])

ax[0][0].plot([10], [-0.1], c = 'black', marker = "o", markersize=5, markeredgewidth = 3, label = 'True Value', alpha = 0.6)
ax[0][1].plot([0.3], [-1], c = 'black', marker = "o", markersize=5, markeredgewidth = 3, alpha = 0.6)
ax[1][0].plot([1], [-0.05], c = 'black', marker = "o", markersize=5, markeredgewidth = 3, alpha = 0.6)
ax[1][1].plot([0.3], [-1], c = 'black', marker = "o", markersize=5, markeredgewidth = 3, alpha = 0.6)
plt.tight_layout()
plt.subplots_adjust(bottom = 0.17)
fig.legend(loc='lower center', ncol = 8, prop={'size': 15})
ax[0][0].grid()
ax[0][1].grid()
ax[1][0].grid()
ax[1][1].grid()
ax[0][0].set(xlabel = r"Inclusion Young's Modulus, $E$ GPa", ylabel = 'Density')
ax[0][1].set(xlabel = r"Inclusion Poisson's Ratio, $\nu$", ylabel = 'Density')
ax[1][0].set(xlabel = r"Matrix Young's Modulus, $E$ GPa", ylabel = 'Density')
ax[1][1].set(xlabel = r"Matrix Poisson's Ratio, $\nu$", ylabel = 'Density')

plt.show()