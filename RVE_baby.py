import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
from copy import copy

#read_dictionary = np.load('RVE_Baby.npy',allow_pickle='TRUE').item()
read_dictionary = np.load('RVE_EnKF.npy',allow_pickle='TRUE').item()
print('Number of Runs: ', len(read_dictionary['Median'][0]))
fig, ax = plt.subplots(4, 1)
Tru = [10, 1, 0.3, 0.3]
ecol = ['palevioletred', 'lightseagreen', 'cornflowerblue', 'mediumpurple']
col = ['mediumvioletred', 'teal','royalblue','rebeccapurple']
lab = ['$E_I$', '$E_M$', '$v_I$', '$v_M$']

for i in range(4):
    ax[i].scatter(range(len(read_dictionary['Median'][i])),read_dictionary['Initial'][i], marker = 'x', c= 'k', s = 15, alpha = 0.6, linewidth = 1.5)
    ax[i].errorbar(range(len(read_dictionary['Median'][i])),read_dictionary['Median'][i], yerr=read_dictionary['Uncertainty'][i], color = col[i], ecolor = ecol[i], marker ='o', linestyle = '', markersize = 5, zorder = 10, capsize = 3)
    ax[i].axhline(np.mean(read_dictionary['Median'][i]), color = col[i], linestyle = 'dashed', alpha = 0.7)
    ax[i].axhline(Tru[i], color = 'black', linestyle = 'dashed')
    ax[i].set_ylabel(f'{lab[i]}')
plt.show()