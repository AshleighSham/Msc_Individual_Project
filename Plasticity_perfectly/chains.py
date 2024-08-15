import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
from copy import copy
import pandas as pd

MH_data = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_boi\MH.csv')
AMH_data = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_boi\AMH.csv')
DR_data = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_boi\MH_DR.csv')
DRAM_data = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_boi\DRAM.csv')
ENKF_data = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_boi\EnKF.csv')

algorithms = [MH_data, AMH_data, DR_data, DRAM_data, ENKF_data]


E = [data['E'] for data in algorithms]
V = [data['v'] for data in algorithms]
sy = [data['sy'] for data in algorithms]
H = [data['H'] for data in algorithms]

colours = ['deepskyblue','mediumseagreen','orange','hotpink','mediumvioletred']
labels = ['MH', 'AMH', 'DR MH', 'DRAM','EnKF']
fig, ax = plt.subplots(nrows = 5, ncols =4, sharex= 'col')

# for i, q in zip(E, range(5)):
#     ax[q][0].plot(range(len(i)), i, alpha = 0.8, c = colours[q], label = labels[q])
#     ax[q][0].axhline(206.9, alpha = 0.8, c = 'k', linestyle=(0,(5,5)))
#     ax[q][0].set_ylim([10,240])

# for i, q in zip(V, range(5)):
#     ax[q][1].plot(range(len(i)), i, alpha = 0.8, c = colours[q])
#     ax[q][1].axhline(0.29, alpha = 0.8, c = 'k', linestyle=(0,(5,5)))
#     ax[q][1].set_ylim([-0.1,0.6])

# for i, q in zip(sy,  range(5)):
#     ax[q][2].plot(range(len(i)), i,  alpha = 0.8, c = colours[q])
#     ax[q][2].axhline(0.45, alpha = 0.8, c = 'k', linestyle=(0,(5,5)))
#     ax[q][2].set_ylim([-0.1,0.6])


# for i, q in zip(H,  range(5)):
#     ax[q][3].plot(range(len(i)), i,  alpha = 0.8, c = colours[q])
#     ax[q][3].axhline(0.20, alpha = 0.8, c = 'k', linestyle=(0,(5,5)))
#     ax[q][3].set_ylim([-0.1,0.6])

# ax[0][0].axhline(206.9, alpha = 0.8, c = 'k', linestyle='dashed', label = 'True Value')
# ax[0][0].set_title("$E$ (GPa)")
# ax[0][1].set_title("$v$")
# ax[0][2].set_title("$\sigma_y$ (GPa)")
# ax[0][3].set_title("$H$ (GPa)")

# plt.tight_layout()
# plt.subplots_adjust(bottom = 0.17)
# fig.legend(loc='lower center', ncol = 8, prop={'size': 15})

# plt.show()
# for i, j in zip(read_dictionary[0], read_dictionary[1]):
#     print('------')
#     print(np.median(i), np.median(j))
#     print(np.sqrt(np.var(i)), np.sqrt(np.var(j)))
#     print('-------')

fig, ax = plt.subplots(nrows=4, ncols =1,  sharex= True)
violin_parts0 = ax[0].violinplot(E, showmedians=True)
ax[0].grid()
ax[0].set(ylabel = '$E$')


for pc, color in zip(violin_parts0['bodies'], colours):
    pc.set_facecolor(color)

for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts0[partname]
    vp.set_edgecolor(colours)
    vp.set_facecolor(colours)
    vp.set_linewidth(2.5)

ax[0].set_xticks([y + 1 for y in range(len(E))],
                  labels=labels)
    #ax[0].axvline(10, c = 'k', linestyle='dashed')
ax[0].axhline(206.9, alpha = 0.7, c = 'k', linestyle=(0,(5,5)))
violin_parts1 = ax[1].violinplot(V, showmedians=True)
ax[1].grid()
ax[1].set(ylabel = '$v$')
ax[1].axhline(0.29, alpha = 0.7, c = 'k', linestyle=(0,(5,5)), label = 'True Value')

ax[1].set_xticks([y + 1 for y in range(len(E))],
                  labels=labels)

for pc, color in zip(violin_parts1['bodies'], colours):
    pc.set_facecolor(color)
for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts1[partname]
    vp.set_edgecolor(colours)
    vp.set_facecolor(colours)
    vp.set_linewidth(2.5)


violin_parts2 = ax[2].violinplot(sy, showmedians=True)
ax[2].grid()
ax[2].set(ylabel = '$v_I$')


for pc, color in zip(violin_parts2['bodies'], colours):
    pc.set_facecolor(color)

for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts2[partname]
    vp.set_edgecolor(colours)
    vp.set_facecolor(colours)
    vp.set_linewidth(2.5)

ax[2].set_xticks([y + 1 for y in range(len(sy))],
                  labels=labels)
ax[2].axhline(0.45, alpha = 0.7, c = 'k', linestyle=(0,(5,5)))

violin_parts3 = ax[3].violinplot(H, showmedians=True)
ax[3].grid()
ax[3].set(ylabel = '$v_M$')


for pc, color in zip(violin_parts3['bodies'], colours):
    pc.set_facecolor(color)

for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts3[partname]
    vp.set_edgecolor(colours)
    vp.set_facecolor(colours)
    vp.set_linewidth(2.5)

ax[3].set_xticks([y + 1 for y in range(len(H))],labels=labels)
ax[3].axhline(0.2, alpha = 0.7, c = 'k', linestyle=(0,(5,5)))
    
plt.subplots_adjust(bottom = 0.17)
fig.legend(loc='lower center', ncols = 4, prop={'size': 15})
plt.tight_layout()
plt.show()