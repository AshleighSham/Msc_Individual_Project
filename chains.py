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
true_vals = [10, 1, 0.3, 0.3]
graph_labs = [r'$E_I$ (GPa)', r'$E_M$ (GPa)', r'$\nu_I$', r'$\nu_M$']
lims = [[4.5,15], [-1, 23], [-0.1, 0.6], [-0.1,0.6]]

# true_vals = [10, 0.3]
# graph_labs = [r'$E$ (GPa)', r'$\nu$']
# lims = [[4.5,11], [-0.1,0.6]]

#read_dictionary = {0:[], 1:[], 2:[]}

#labs = ['E', 'v']
# data1 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_boii\CURRENT_CHAINS\MH.csv')
# data2 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_boii\CURRENT_CHAINS\AMH.csv')
# data3 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_boii\CURRENT_CHAINS\MH_DR.csv')
# data4 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_boii\CURRENT_CHAINS\DRAM.csv')
# data5 = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_boii\CURRENT_CHAINS\EnKF.csv')



# for i in range(3):
#     read_dictionary[i].append(data1[labs[i]])
#     #print(labs[i], 'MH', np.median(data1[labs[i]][500:]), 2*np.sqrt(np.var(data1[labs[i]][500:])))    
#     read_dictionary[i].append(data2[labs[i]])
#     #print(labs[i], 'AMH', np.median(data2[labs[i]][500:]), 2*np.sqrt(np.var(data2[labs[i]][500:])))
#     read_dictionary[i].append(data3[labs[i]])
#     #print(labs[i], 'MH_DR', np.median(data3[labs[i]][500:]), 2*np.sqrt(np.var(data3[labs[i]][500:])))
#     read_dictionary[i].append(data4[labs[i]])
#     #print(labs[i], 'DRAM', np.median(data4[labs[i])), 2*np.sqrt(np.var(data4[labs[i]][500:])))
#     read_dictionary[i].append(data5[labs[i]])
#     #print(labs[i], 'eNkf', np.median(data5[labs[i])), 2*np.sqrt(np.var(data5[labs[i]])))


read_dictionary = np.load('4D_chains_BEAM_10000.npy',allow_pickle='TRUE').item()


colours = ['deepskyblue','mediumseagreen','orange','hotpink','mediumvioletred']
labels = ['MH', 'AMH', 'DR MH', 'DRAM', 'EnKF']
fig, ax = plt.subplots(nrows = len(true_vals), ncols =len(labels))

for j in range(len(true_vals)):
    for i, q in zip(read_dictionary[j], range(len(labels))):
        ax[j][0].set_ylabel(graph_labs[j])
        ax[0][q].set_title(labels[q])
        ax[j][q].plot(range(len(i)), i, alpha = 0.8, c = colours[q])
        ax[j][q].axhline(true_vals[j], alpha = 0.6, c = 'k', linestyle=(0,(5,5)))
        ax[j][q].set_ylim(lims[j])
        ax[j][q].set_xticks([]) 
        if q != 0:
            ax[j][q].set_yticks([]) 
        if j == len(true_vals) - 1:
            ax[j][q].set_xticks([5000])

# for i, q in zip(read_dictionary[1], range(5)):
#     ax[1][q].plot(range(len(i)), i, alpha = 0.8, c = colours[q])
#     ax[1][q].axhline(true_vals[1], alpha = 0.6, c = 'k', linestyle=(0,(5,5)))
#     ax[1][q].set_ylim([-0.1,0.6])
#     ax[1][q].set_xticks([1500]) 
#     if q != 0:
#         ax[1][q].set_yticks([]) 

# for i, q in zip(read_dictionary[2],  range(6)):
#     ax[2][q].plot(range(len(i)), i,  alpha = 0.8, c = colours[q])
#     ax[2][q].axhline(true_vals[2], alpha = 0.6, c = 'k', linestyle=(0,(5,5)))
#     ax[2][q].set_ylim([-0.1,0.6])
#     ax[2][q].set_xticks([1500]) 
#     if q != 0:
#         ax[2][q].set_yticks([]) 


# for i, q in zip(read_dictionary[3],  range(6)):
#     ax[3][q].plot(range(len(i)), i,  alpha = 0.8, c = colours[q])
#     ax[3][q].axhline(true_vals[3], alpha = 0.6, c = 'k', linestyle=(0,(5,5)))
#     ax[3][q].set_ylim([-0.1,0.6])
#     ax[3][q].set_xticks([1500])
#     if q != 0:
#         ax[3][q].set_yticks([]) 

fig.text(0.50, 0.07, "Number of Samples", horizontalalignment='center',
        fontsize = 'large')
ax[0][0].axhline(true_vals[0], alpha = 0.5, c = 'k', linestyle=(0,(5,5)), label= 'True Value')
#ax[0][0].axhline(10, alpha = 0.8, c = 'k', linestyle='dashed', label = 'True Value')
# ax[0][0].set_ylabel(graph_labs[0])
# ax[1][0].set_ylabel(graph_labs[1])
#ax[2][0].set_ylabel(graph_labs[2])
#ax[3][0].set_ylabel(graph_labs[3])

plt.tight_layout()
plt.subplots_adjust(wspace=0.0, top=0.91, bottom=0.15, left=0.2, right=0.8, hspace = 0.2)
fig.legend(loc='lower center', ncol = 1)

plt.show()
# for i, j in zip(read_dictionary[0], read_dictionary[1]):
#     print('------')
#     print(np.median(i), np.median(j))
#     print(np.sqrt(np.var(i)), np.sqrt(np.var(j)))
#     print('-------')

fig, ax = plt.subplots(nrows=len(true_vals), ncols =1)
violin_parts0 = ax[0].violinplot(read_dictionary[0], showmedians=True)
ax[0].grid()
ax[0].set(ylabel = graph_labs[0])


for pc, color in zip(violin_parts0['bodies'], colours):
    pc.set_facecolor(color)

for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts0[partname]
    vp.set_edgecolor(colours)
    vp.set_facecolor(colours)
    vp.set_linewidth(2.5)

ax[0].set_xticks([])
    #ax[0].axvline(10, c = 'k', linestyle='dashed')
ax[0].axhline(true_vals[0], alpha = 0.7, c = 'k', linestyle=(0,(5,5)))
violin_parts1 = ax[1].violinplot(read_dictionary[1], showmedians=True)
ax[1].grid()
ax[1].set(ylabel = graph_labs[1])
ax[1].axhline(true_vals[1], alpha = 0.7, c = 'k', linestyle=(0,(5,5)), label = 'True Value')

ax[1].set_xticks([])

for pc, color in zip(violin_parts1['bodies'], colours):
    pc.set_facecolor(color)
for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts1[partname]
    vp.set_edgecolor(colours)
    vp.set_facecolor(colours)
    vp.set_linewidth(2.5)


violin_parts2 = ax[2].violinplot(read_dictionary[2], showmedians=True)
ax[2].grid()
ax[2].set(ylabel =graph_labs[2])


for pc, color in zip(violin_parts2['bodies'], colours):
    pc.set_facecolor(color)

for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts2[partname]
    vp.set_edgecolor(colours)
    vp.set_facecolor(colours)
    vp.set_linewidth(2.5)

ax[2].set_xticks([])
ax[2].axhline(true_vals[2], alpha = 0.7, c = 'k', linestyle=(0,(5,5)))

violin_parts3 = ax[3].violinplot(read_dictionary[3], showmedians=True)
ax[3].grid()
ax[3].set(ylabel = graph_labs[3])


for pc, color in zip(violin_parts3['bodies'], colours):
    pc.set_facecolor(color)

for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts3[partname]
    vp.set_edgecolor(colours)
    vp.set_facecolor(colours)
    vp.set_linewidth(2.5)

ax[-1].set_xticks([y + 1 for y in range(len(read_dictionary[0]))],labels=labels)

ax[3].axhline(true_vals[3], alpha = 0.7, c = 'k', linestyle=(0,(5,5)))
plt.tight_layout()    
plt.subplots_adjust(wspace=0.15, top=0.94, bottom=0.11, left=0.2, right=0.8, hspace = 0.1)
fig.legend(loc='lower center', ncols = 1)
plt.show()