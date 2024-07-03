import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
from copy import copy

read_dictionary = np.load('my_file.npy',allow_pickle='TRUE').item()

colours = ['deepskyblue','mediumseagreen','orange','hotpink','mediumorchid','mediumvioletred', 'red']
labels = ['MH', 'AMH', 'DR MH', 'DRAM', 'pCN', 'EnKF', 'Baby']
# fig, ax = plt.subplots(1, 2, figsize = (10, 5) )

# for i, j, q in zip(read_dictionary['graphs'][0], read_dictionary['values'][0], range(7)):
#     ax[0].plot(i[0], i[1], alpha = 0.8, c = colours[q])
#     ax[0].plot([j], [-0.02], c = colours[q], marker = "|", markersize=18, markeredgewidth = 3)

# for i, j, q in zip(read_dictionary['graphs'][1], read_dictionary['values'][1], range(7)):
#     ax[1].plot(i[0], i[1], alpha = 0.8, c = colours[q])
#     ax[1].plot([j], [-1], c = colours[q], marker = "|", markersize=18, markeredgewidth = 3, label = labels[q])

# ax[0].plot([10], [-0.02], c = 'black', marker = "o", markersize=5, markeredgewidth = 3, label = 'True Value', alpha = 0.6)
# ax[1].plot([0.3], [-1], c = 'black', marker = "o", markersize=5, markeredgewidth = 3, alpha = 0.6)
# plt.tight_layout()
# plt.subplots_adjust(bottom = 0.17)
# fig.legend(loc='lower center', ncol = 8, prop={'size': 15})
# ax[0].grid()
# ax[0].set(xlabel = r"Young's Modulus, $E$ GPa", ylabel = 'Density')
# ax[1].set(xlabel = r"Poisson's Ratio, $\nu$", ylabel = 'Density')
# ax[1].grid()
# plt.show()

fig, ax = plt.subplots(nrows=2, ncols =1, sharex= 'col')

for i, q in zip(read_dictionary['graphs'][0], range(7)):
    ax[0].boxplot(i[1])
    ax[0].grid()
    ax[0].set(xlabel = 'E', ylabel = 'PDF')
    ax[0].axvline(10, c = 'k', linestyle='dashed')

for i, q in zip(read_dictionary['graphs'][1], range(7)):
    ax[1].boxplot(i[1])
    ax[1].grid()
    ax[1].set(xlabel = 'E', ylabel = 'PDF')
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].axvline(0.3, alpha = 0.8, c = 'k', linestyle='dashed')
    

ax[0][1].axvline(0.3, alpha = 0.8, c = 'k', linestyle='dashed', label = 'True Value')
plt.subplots_adjust(bottom = 0.17)
fig.legend(loc='lower center', ncol = 8, prop={'size': 15})
plt.tight_layout()
plt.show()
