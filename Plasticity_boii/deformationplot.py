from config import config
import numpy as np
import utilities
from MH import MH_mcmc
from EnKF import EnKF_mcmc
from DRAM import DRAM_algorithm
from AMH import AMH_mcmc
from MH_DR import MH_DR_mcmc
from crank import Crank_mcmc
from baby import Baby_mcmc
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')

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

inp = {}

# data = np.load('RVE_EnKF2D.npy', allow_pickle=True)
# data = data.tolist()

# range of the parameters based on the prior density
inp['range']=np.array([[config['Imposed Limits']['Youngs Modulus'][0], config['Imposed Limits']['Poissons Ratio'][0], config['Imposed Limits']['Yield Stress'][0], config['Imposed Limits']['Hardening Modulus'][0]], 
                       [config['Imposed Limits']['Youngs Modulus'][1], config['Imposed Limits']['Poissons Ratio'][1], config['Imposed Limits']['Yield Stress'][1], config['Imposed Limits']['Hardening Modulus'][1]]])    
inp['s'] = config['s']                      

# number of iteration in MCMC
inp['nsamples']=config['Number of samples']

# initial covariance                          
inp['icov']=np.array([[config['Initial Variance']['Youngs Modulus'], 0, 0, 0],[0, config['Initial Variance']['Poissons Ratio'], 0, 0],
                      [0, 0, config['Initial Variance']['Yield Stress'], 0], [0, 0, 0, config['Initial Variance']['Hardening Modulus']]])                                   

# initial guesss of the parameters based on the prior
inp['theta0']=np.array([[config['Initial Material Parameters']['Youngs Modulus']],[config['Initial Material Parameters']['Poissons Ratio']],
                        [config['Initial Material Parameters']['Yield Stress']], [config['Initial Material Parameters']['Hardening Modulus']]])  

# std                               
inp['sigma']=config["Standard Deviation"]

# The starting point of the Kalman MCMC           
inp['Kalmans']= config['Starting Kalman point']                    

# assumed measurement error for Kalman MCMC
inp['me']=config['Measurement error for Kalman']

#adaption setp size
inp['adapt'] = config['Adaption Step Size']

#mesh set up
inp['mesh'] = [config['Mesh grid']['quad'], 
               config['Mesh grid']['sf'],
               config['Mesh grid']['Nodal Coordinates'],
               config['Mesh grid']['Element Node Numbers'],
               config['Mesh grid']['Number of elements'],
               config['Mesh grid']['Force Magnitude'],
               config['Mesh grid']['Force Nodes'],
               config['Mesh grid']['Fixed Nodes'],
               config['Mesh grid']['Element ID'],
               config['Mesh grid']['thickness']]

ini = np.array([[config['True Material Parameters']['Youngs Modulus']], [config['True Material Parameters']['Poissons Ratio']], [config['True Material Parameters']['Yield Stress']],[config['True Material Parameters']['Hardening Modulus']]])

d=utilities.forward_model(ini, inp['mesh'])

XYZ = np.array([[ 0.,  0.],
       [ 1.,  0.],
       [ 2.,  0.],
       [ 3.,  0.],
       [ 4.,  0.],
       [ 5.,  0.],
       [ 0.,  1.],
       [ 1.,  1.],
       [ 2.,  1.],
       [ 3.,  1.],
       [ 4.,  1.],
       [ 5.,  1.],
       [ 0.,  2.],
       [ 1.,  2.],
       [ 2.,  2.],
       [ 3.,  2.],
       [ 4.,  2.],
       [ 5.,  2.],
       [ 0.,  3.],
       [ 1.,  3.],
       [ 2.,  3.],
       [ 3.,  3.],
       [ 4.,  3.],
       [ 5.,  3.],
       [ 0.,  4.],
       [ 1.,  4.],
       [ 2.,  4.],
       [ 3.,  4.],
       [ 4.,  4.],
       [ 5.,  4.],
       [ 0.,  5.],
       [ 1.,  5.],
       [ 2.,  5.],
       [ 3.,  5.],
       [ 4.,  5.],
       [ 5.,  5.],
       [ 0.,  6.],
       [ 1.,  6.],
       [ 2.,  6.],
       [ 3.,  6.],
       [ 4.,  6.],
       [ 5.,  6.],
       [ 0.,  7.],
       [ 1.,  7.],
       [ 2.,  7.],
       [ 3.,  7.],
       [ 4.,  7.],
       [ 5.,  7.],
       [ 0.,  8.],
       [ 1.,  8.],
       [ 2.,  8.],
       [ 3.,  8.],
       [ 4.,  8.],
       [ 5.,  8.],
       [ 0.,  9.],
       [ 1.,  9.],
       [ 2.,  9.],
       [ 3.,  9.],
       [ 4.,  9.],
       [ 5.,  9.],
       [ 0., 10.],
       [ 1., 10.],
       [ 2., 10.],
       [ 3., 10.],
       [ 4., 10.],
       [ 5., 10.],
       [ 0., 11.],
       [ 1., 11.],
       [ 2., 11.],
       [ 3., 11.],
       [ 4., 11.],
       [ 5., 11.],
       [ 0., 12.],
       [ 1., 12.],
       [ 2., 12.],
       [ 3., 12.],
       [ 4., 12.],
       [ 5., 12.],
       [ 0., 13.],
       [ 1., 13.],
       [ 2., 13.],
       [ 3., 13.],
       [ 4., 13.],
       [ 5., 13.],
       [ 0., 14.],
       [ 1., 14.],
       [ 2., 14.],
       [ 3., 14.],
       [ 4., 14.],
       [ 5., 14.],
       [ 0., 15.],
       [ 1., 15.],
       [ 2., 15.],
       [ 3., 15.],
       [ 4., 15.],
       [ 5., 15.]])

CON = np.array([[ 1,  2,  8,  7],
       [ 2,  3,  9,  8],
       [ 3,  4, 10,  9],
       [ 4,  5, 11, 10],
       [ 5,  6, 12, 11],
       [ 7,  8, 14, 13],
       [ 8,  9, 15, 14],
       [ 9, 10, 16, 15],
       [10, 11, 17, 16],
       [11, 12, 18, 17],
       [13, 14, 20, 19],
       [14, 15, 21, 20],
       [15, 16, 22, 21],
       [16, 17, 23, 22],
       [17, 18, 24, 23],
       [19, 20, 26, 25],
       [20, 21, 27, 26],
       [21, 22, 28, 27],
       [22, 23, 29, 28],
       [23, 24, 30, 29],
       [25, 26, 32, 31],
       [26, 27, 33, 32],
       [27, 28, 34, 33],
       [28, 29, 35, 34],
       [29, 30, 36, 35],
       [31, 32, 38, 37],
       [32, 33, 39, 38],
       [33, 34, 40, 39],
       [34, 35, 41, 40],
       [35, 36, 42, 41],
       [37, 38, 44, 43],
       [38, 39, 45, 44],
       [39, 40, 46, 45],
       [40, 41, 47, 46],
       [41, 42, 48, 47],
       [43, 44, 50, 49],
       [44, 45, 51, 50],
       [45, 46, 52, 51],
       [46, 47, 53, 52],
       [47, 48, 54, 53],
       [49, 50, 56, 55],
       [50, 51, 57, 56],
       [51, 52, 58, 57],
       [52, 53, 59, 58],
       [53, 54, 60, 59],
       [55, 56, 62, 61],
       [56, 57, 63, 62],
       [57, 58, 64, 63],
       [58, 59, 65, 64],
       [59, 60, 66, 65],
       [61, 62, 68, 67],
       [62, 63, 69, 68],
       [63, 64, 70, 69],
       [64, 65, 71, 70],
       [65, 66, 72, 71],
       [67, 68, 74, 73],
       [68, 69, 75, 74],
       [69, 70, 76, 75],
       [70, 71, 77, 76],
       [71, 72, 78, 77],
       [73, 74, 80, 79],
       [74, 75, 81, 80],
       [75, 76, 82, 81],
       [76, 77, 83, 82],
       [77, 78, 84, 83],
       [79, 80, 86, 85],
       [80, 81, 87, 86],
       [81, 82, 88, 87],
       [82, 83, 89, 88],
       [83, 84, 90, 89],
       [85, 86, 92, 91],
       [86, 87, 93, 92],
       [87, 88, 94, 93],
       [88, 89, 95, 94],
       [89, 90, 96, 95]])

CON = CON - 1

colour = 'palevioletred'

label = 'Nodal Displacements'

ls = 'solid'

ch = 0.9
 
fig, ax = plt.subplots()
ccc1=np.array(XYZ[:,0])
ccc2=np.array(d[0:len(d):2]).reshape(-1)
ccc= np.array(ccc1+ccc2) 

ddd1=np.array(XYZ[:,1])
ddd2=np.array(d[1:len(d):2]).reshape(-1)
ddd= np.array(ddd1+ddd2)

#figure = plt.figure()
ax.plot(XYZ[:, 1], XYZ[:,0],'ok', markersize='8', zorder = 1, alpha = 0.8, label = 'Nodal Coordinates')
ax.scatter(XYZ[:,1] + d[1:len(d):2].reshape(-1), XYZ[:,0] + d[0:len(d):2].reshape(-1), c = 'mediumvioletred' ,s=50, label = label, zorder = 11, alpha = ch)
#plt.title(title)

for i in range(len(CON)):
    ax.fill( XYZ[CON[i, :], 1],XYZ[CON[i, :], 0], edgecolor='k', fill=False, linestyle = (0,(2,2)), zorder = 1, linewidth = 3)
    ax.fill( XYZ[CON[i, :], 1] + ddd2[(CON[i, :])], XYZ[CON[i, :], 0] + ccc2[(CON[i, :])],edgecolor = colour, linestyle = ls, fill=False, zorder = 5, alpha = ch, linewidth = 3)

ax.set_ylim([-1, 6])
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_xlim([-1, 16])
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

ax.set(ylabel = 'y (m)', xlabel = 'x (m)')
ax.set_aspect('equal')
ax.legend(loc = 'lower center', ncol = 2)
plt.subplots_adjust( bottom=0.15)
plt.show()