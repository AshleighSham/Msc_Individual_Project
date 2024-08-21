import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_context('talk')
from copy import copy

 # Figure width in inches, approximately A4-width - 2*1.25in margin
plt.rcParams.update({    # 4:3 aspect ratio
    'font.size' : 13,                   # Set font size to 11pt
    'axes.labelsize': 15,               # -> axis labels
    'legend.fontsize': 15,              # -> legends
    'font.family': 'charter',
    'text.usetex': True,
    'text.latex.preamble': (            # LaTeX preamble
        r'\usepackage[T1]{fontenc}'
    ),
    "font.weight": 'bold',
    "axes.labelweight": 'bold',
    'axes.titlesize' : 15
})
true_vals = [10, 0.3, 0.005, 0.001, 1, 0.3, 0.007, 0.002]
graph_labs = [r'$E_I$ (GPa)', r'$\nu_I$', r'$\sigma_{yI}$ (GPa)',r'$H_I$ (GPa)', r'$E_M$ (GPa)', r'$\nu_M$', r'$\sigma_{yM}$ (GPa)', r'$H_M$ (GPa)']
lims = [[-2,40], [-0.1, 0.6], [-0.005, 0.025], [-0.02,0.12], [-3,40], [-0.1, 0.6], [-0.005, 0.025], [-0.02,0.12],]

# true_vals = [10, 0.3]
# graph_labs = [r'$E$ (GPa)', r'$\nu$']
# lims = [[4.5,11], [-0.1,0.6]]

read_dictionary = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}

labs = ['Ei', 'vi', 'syi', 'Hi', 'Em', 'vm', 'sym', 'Hm']

option = 0

if option == 0:
    data1 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE_2\corner\MH.csv')
    data2 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE_2\corner\AMH.csv')
    data3 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE_2\corner\MH_DR.csv')
    data4 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE_2\corner\DRAM.csv')
    data5 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE_2\corner\EnKF.csv')
elif option == 1: 
    data1 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE\test\MH.csv')
    data2 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE\test\AMH.csv')
    data3 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE\test\MH_DR.csv')
    data4 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE\test\DRAM.csv')
    data5 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE\test\EnKF.csv')
else: 
    data1 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE_2\side\MH.csv')
    data2 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE_2\side\AMH.csv')
    data3 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE_2\side\MH_DR.csv')
    data4 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE_2\side\DRAM.csv')
    data5 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE_2\side\EnKF.csv')

for i in range(len(labs)):
    read_dictionary[i].append(data1[labs[i]])
    #print(labs[i], 'MH', np.median(data1[labs[i]]), 2*np.sqrt(np.var(data1[labs[i]])))    
    read_dictionary[i].append(data2[labs[i]])
    #print(labs[i], 'AMH', np.median(data2[labs[i]]), 2*np.sqrt(np.var(data2[labs[i]])))
    read_dictionary[i].append(data3[labs[i]])
    #print(labs[i], 'MH_DR', np.median(data3[labs[i]]), 2*np.sqrt(np.var(data3[labs[i]])))
    read_dictionary[i].append(data4[labs[i]])
    #print(labs[i], 'DRAM', np.median(data4[labs[i]])), 2*np.sqrt(np.var(data4[labs[i]]))
    read_dictionary[i].append(data5[labs[i]])
    #print(labs[i], 'eNkf', np.median(data5[labs[i]])), 2*np.sqrt(np.var(data5[labs[i]]))


#read_dictionary = np.load('4D_chains_BEAM_10000.npy',allow_pickle='TRUE').item()


colours = ['deepskyblue','mediumseagreen','orange','hotpink','mediumvioletred']
labels = ['MH', 'AMH', 'DR MH', 'DRAM', 'EnKF']
fig, ax = plt.subplots(nrows = len(true_vals), ncols =len(labels))

for j in range(len(true_vals)):
    for i, q in zip(read_dictionary[j], range(len(labels))):
        ax[0][q].set_title(labels[q])
        ax[j][q].plot(range(len(i)), i, alpha = 0.8, c = colours[q])
        ax[j][q].axhline(true_vals[j], alpha = 0.6, c = 'k', linestyle=(0,(5,5)))
        ax[j][q].set_ylim(lims[j])
        ax[j][q].set_xticks([]) 
        if j == len(true_vals) - 1:
            ax[j][q].set_xticks([1500])
        if j % 2 == 0:
            if q != 0:
                ax[j][q].set_yticks([]) 
            ax[j][0].set_ylabel(graph_labs[j])
        else:
            if q != (len(labels) - 1):
                ax[j][q].set_yticks([]) 
            ax[j][len(labels) - 1].yaxis.tick_right()
            ax[j][len(labels) - 1].yaxis.set_label_position("right")
            ax[j][len(labels) - 1].set_ylabel(graph_labs[j])

fig.text(0.50, 0.07, "Number of Samples", horizontalalignment='center',
        fontsize = 'large')
ax[0][0].axhline(true_vals[0], alpha = 0.5, c = 'k', linestyle=(0,(5,5)), label= 'True Value')
#ax[0][0].axhline(10, alpha = 0.8, c = 'k', linestyle='dashed', label = 'True Value')
# ax[0][0].set_ylabel(graph_labs[0])
# ax[1][0].set_ylabel(graph_labs[1])
#ax[2][0].set_ylabel(graph_labs[2])
#ax[3][0].set_ylabel(graph_labs[3])

plt.tight_layout()
plt.subplots_adjust(wspace=0.0, top=0.95, bottom=0.133, left=0.25, right=0.75, hspace = 0.2)
fig.legend(loc='lower center', ncol = 1)

plt.show()
# for i, j in zip(read_dictionary[0], read_dictionary[1]):
#     print('------')
#     print(np.median(i), np.median(j))
#     print(np.sqrt(np.var(i)), np.sqrt(np.var(j)))
#     print('-------')

fig, ax = plt.subplots(nrows=len(true_vals)//2, ncols =2)


for j in range(len(true_vals)):
    violin_parts0 = ax[j % 4][j // 4].violinplot(read_dictionary[j], showmedians=True)
    ax[j % 4][j // 4].grid()
    ax[j % 4][j // 4].set(ylabel = graph_labs[j])


    for pc, color in zip(violin_parts0['bodies'], colours):
        pc.set_facecolor(color)

    for partname in ('cbars','cmins','cmaxes','cmedians'):
        vp = violin_parts0[partname]
        vp.set_edgecolor(colours)
        vp.set_facecolor(colours)
        vp.set_linewidth(2.5)
    if j % 4 != 3:
        ax[j % 4][j // 4].set_xticks([])
        #ax[0].axvline(10, c = 'k', linestyle='dashed')
    if j % 4 + j // 4 != 0:
        ax[j % 4][j // 4].axhline(true_vals[j], alpha = 0.7, c = 'k', linestyle=(0,(5,5)))
    if j // 4 == 1:
        ax[j % 4][j // 4].yaxis.set_label_position("right")
        ax[j % 4][j // 4].yaxis.tick_right()
# violin_parts1 = ax[1].violinplot(read_dictionary[1], showmedians=True)
# ax[1].grid()
# ax[1].set(ylabel = graph_labs[1])
# ax[1].axhline(true_vals[1], alpha = 0.7, c = 'k', linestyle=(0,(5,5)), label = 'True Value')

# ax[1].set_xticks([])

# for pc, color in zip(violin_parts1['bodies'], colours):
#     pc.set_facecolor(color)
# for partname in ('cbars','cmins','cmaxes','cmedians'):
#     vp = violin_parts1[partname]
#     vp.set_edgecolor(colours)
#     vp.set_facecolor(colours)
#     vp.set_linewidth(2.5)


# violin_parts2 = ax[2].violinplot(read_dictionary[2], showmedians=True)
# ax[2].grid()
# ax[2].set(ylabel =graph_labs[2])


# for pc, color in zip(violin_parts2['bodies'], colours):
#     pc.set_facecolor(color)

# for partname in ('cbars','cmins','cmaxes','cmedians'):
#     vp = violin_parts2[partname]
#     vp.set_edgecolor(colours)
#     vp.set_facecolor(colours)
#     vp.set_linewidth(2.5)

# ax[2].set_xticks([])
# ax[2].axhline(true_vals[2], alpha = 0.7, c = 'k', linestyle=(0,(5,5)))

# violin_parts3 = ax[3].violinplot(read_dictionary[3], showmedians=True)
# ax[3].grid()
# ax[3].set(ylabel = graph_labs[3])


# for pc, color in zip(violin_parts3['bodies'], colours):
#     pc.set_facecolor(color)

# for partname in ('cbars','cmins','cmaxes','cmedians'):
#     vp = violin_parts3[partname]
#     vp.set_edgecolor(colours)
#     vp.set_facecolor(colours)
#     vp.set_linewidth(2.5)

ax[3][0].set_xticks([y + 1 for y in range(len(read_dictionary[0]))],labels=labels)
ax[3][1].set_xticks([y + 1 for y in range(len(read_dictionary[0]))],labels=labels)

ax[0][0].axhline(true_vals[0], alpha = 0.7, c = 'k', linestyle=(0,(5,5)), label = 'True Value')
plt.tight_layout()    
plt.subplots_adjust(wspace=0.044, top=0.94, bottom=0.11, left=0.2, right=0.8, hspace = 0.1)
fig.legend(loc='lower center', ncols = 1)
plt.show()