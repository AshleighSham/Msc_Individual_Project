import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
from copy import copy

read_dictionary = np.load('3D_chains_plas_4.npy',allow_pickle='TRUE').item()

colours = ['deepskyblue','mediumseagreen','orange','hotpink','mediumorchid','mediumvioletred', 'red']
labels = ['MH', 'AMH', 'DR MH', 'DRAM', 'pCN', 'EnKF', 'Baby']
fig, ax = plt.subplots(nrows = 7, ncols =4, sharex= 'col')

for i, q in zip(read_dictionary[0], range(7)):
    ax[q][0].plot(range(len(i)), i, alpha = 0.8, c = colours[q], label = labels[q])
    ax[q][0].axhline(210, alpha = 0.8, c = 'k', linestyle=(0,(5,5)))

for i, q in zip(read_dictionary[1], range(7)):
    ax[q][1].plot(range(len(i)), i, alpha = 0.8, c = colours[q])
    ax[q][1].axhline(0.3, alpha = 0.8, c = 'k', linestyle=(0,(5,5)))

for i, q in zip(read_dictionary[2],  range(7)):
    ax[q][2].plot(range(len(i)), i,  alpha = 0.8, c = colours[q])
    ax[q][2].axhline(0.2, alpha = 0.8, c = 'k', linestyle=(0,(5,5)))


for i, q in zip(read_dictionary[3],  range(7)):
    ax[q][3].plot(range(len(i)), i,  alpha = 0.8, c = colours[q])
    ax[q][3].axhline(0.3, alpha = 0.8, c = 'k', linestyle=(0,(5,5)))



ax[0][0].axhline(210, alpha = 0.8, c = 'k', linestyle=(0,(5,5)), label = 'True Value')
ax[0][0].set_title("$E$")
ax[0][1].set_title("$v$")
ax[0][2].set_title("$\sigma_Y$")
ax[0][3].set_title("$H$")

# ax[0][0].set_title("Young's Modulus")
# ax[0][1].set_title("Poisson's Ratio")

plt.tight_layout()
plt.subplots_adjust(bottom = 0.17)
fig.legend(loc='lower center', ncol = 4, prop={'size': 15})

# for i, j in zip(read_dictionary[0], read_dictionary[1]):
#     print('------')
#     print(np.median(i), np.median(j))
#     print(np.sqrt(np.var(i)), np.sqrt(np.var(j)))
#     print('-------')

fig, ax = plt.subplots(nrows=4, ncols =1,  sharex= True)
violin_parts0 = ax[0].violinplot(read_dictionary[0], showmedians=True)
ax[0].grid()
ax[0].set(ylabel = '$E$')


for pc, color in zip(violin_parts0['bodies'], colours):
    pc.set_facecolor(color)

for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts0[partname]
    vp.set_edgecolor(colours)
    vp.set_facecolor(colours)
    vp.set_linewidth(2.5)

ax[0].set_xticks([y + 1 for y in range(len(read_dictionary[0]))],
                  labels=labels)
    #ax[0].axvline(10, c = 'k', linestyle='dashed')
ax[0].axhline(210, alpha = 0.7, c = 'k', linestyle=(0,(5,5)))
violin_parts1 = ax[1].violinplot(read_dictionary[1], showmedians=True)
ax[1].grid()
ax[1].set(ylabel = '$v$')
ax[1].axhline(0.3, alpha = 0.7, c = 'k', linestyle=(0,(5,5)), label = 'True Value')

ax[1].set_xticks([y + 1 for y in range(len(read_dictionary[0]))],
                  labels=labels)

for pc, color in zip(violin_parts1['bodies'], colours):
    pc.set_facecolor(color)
for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts1[partname]
    vp.set_edgecolor(colours)
    vp.set_facecolor(colours)
    vp.set_linewidth(2.5)


violin_parts2 = ax[2].violinplot(read_dictionary[2], showmedians=True)
ax[2].grid()
ax[2].set(ylabel = '$\sigma_Y$')


for pc, color in zip(violin_parts2['bodies'], colours):
    pc.set_facecolor(color)

for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts2[partname]
    vp.set_edgecolor(colours)
    vp.set_facecolor(colours)
    vp.set_linewidth(2.5)

# ax[2].set_xticks([y + 1 for y in range(len(read_dictionary[0]))],
#                   labels=labels)
ax[2].axhline(0.2, alpha = 0.7, c = 'k', linestyle=(0,(5,5)))

violin_parts3 = ax[3].violinplot(read_dictionary[3], showmedians=True)
ax[3].grid()
ax[3].set(ylabel = '$H$')


for pc, color in zip(violin_parts3['bodies'], colours):
    pc.set_facecolor(color)

for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts3[partname]
    vp.set_edgecolor(colours)
    vp.set_facecolor(colours)
    vp.set_linewidth(2.5)

#ax[3].set_xticks([y + 1 for y in range(len(read_dictionary[0]))],labels=labels)
ax[3].axhline(0.3, alpha = 0.7, c = 'k', linestyle=(0,(5,5)))
    
plt.subplots_adjust(bottom = 0.17)
fig.legend(loc='lower center', ncols = 4, prop={'size': 15})
plt.tight_layout()
plt.show()